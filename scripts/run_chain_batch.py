"""Run five M1 -> M2 -> M3/M4 chains, with optional live selection and mapping.

Each downstream scenario is derived from independently verified upstream results.
Live mode sends business descriptions through JEV, Chroma and DeepSeek before
using the validated mapping to build and solve. Expected IDs are checked locally,
never supplied as forced model choices to the selector.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from html import escape
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as pipeline_app
from main import read_json, write_json, run_build
from orllm.data import load_base

DEFAULT_INSTANCES = [
    "G20-5-1a__Q70", "G20-5-1b__Q150", "G50-5-1ab__Q70",
    "G50-5-2ab__Q150", "G100-10-3ab__Q70",
]
MODELS = ("M1", "M2", "M3", "M4")
STATUS = {"optimal": "达到最优性容差", "feasible_limit": "限时可行",
          "partial": "部分可行", "infeasible": "不可行", "no_solution": "未找到可行解",
          "error": "执行错误", "running": "正在运行", "pending": "等待前序结果"}
FIELDS = [("case", "组别"), ("instance_id", "数据实例"), ("model_id", "模型"),
          ("status_label", "求解状态"), ("objective", "模型目标值"),
          ("max_subproblem_gap", "最大子问题gap"), ("aggregate_gap", "汇总界差距"),
          ("runtime_seconds", "求解耗时/秒"), ("opened_depots_text", "开放仓库"),
          ("customer_count", "客户数"), ("vehicle_count", "路线数"),
          ("subproblem_count", "子模型数"), ("independent_checks", "独立检查通过"),
          ("report_path", "报告路径"), ("jev_confidence", "JEV选型置信度"),
          ("llm_model", "LLM模型"), ("retrieval_count", "检索知识条数")]


def fmt(value, key=""):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if isinstance(value, float):
        if "gap" in key:
            return f"{value:.3%}"
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def derive_decisions(summary, base, model_id):
    """Only use a complete, decoded, independently checked assignment solution."""
    if not summary["complete_solution"] or not summary["cost_breakdown_complete"]:
        raise ValueError(f"{model_id} has no complete, verified business solution")
    business = summary["business"]
    opened = [d["depot_id"] for d in business["depots"] if d["opened"]]
    entries = business["assignments"]
    assignment = {r["customer_id"]: r["depot_id"] for r in entries}
    if len(assignment) != len(entries) or set(assignment) != set(base["C"]):
        raise ValueError("Upstream customer coverage is incomplete or duplicated")
    if any(d not in opened for d in assignment.values()):
        raise ValueError("Upstream assignment uses a closed depot")
    for d in opened:
        load = sum(base["q"][c] for c in base["C"] if assignment[c] == d)
        if load > base["U"][d] + 1e-6:
            raise ValueError("Upstream depot capacity is violated")
    return opened, assignment


def problem_text(scenario):
    lines = [f"使用数据实例 {scenario['instance_id']}。"]
    if scenario["model_id"] == "M1":
        lines += ["选择开放哪些候选仓库，并将每个客户完整分配给一个已开放仓库。",
                  "满足仓库容量，最小化仓库启用成本与整单直送成本，不要求安排串点路线。"]
    else:
        lines.append("已经确定开放的仓库：" + "、".join(scenario["opened_depots"]) + "。")
        if scenario["model_id"] == "M2":
            lines.append("固定上述仓库，只优化客户归属，满足仓库容量，最小化整单直送成本。")
        else:
            for d in scenario["opened_depots"]:
                clients = [c for c, depot in scenario["assignment"].items() if depot == d]
                lines.append(f"{d} 服务的客户：{'、'.join(clients) if clients else '无'}。")
            lines += ["保持客户归属，安排每辆车从所属仓库出发并返回该仓库的路线和访问顺序。",
                      "每个客户只访问一次且不拆单，满足车辆载重容量，不设置客户时间窗。"]
            if scenario["model_id"] == "M3":
                lines.append("采用普通车辆，最小化出车固定成本与行驶成本。")
            else:
                lines += ["采用电动车，出发时满电，途中不充电，满足每条路线的电池续航约束。",
                          "最小化出车固定成本与用电成本；电池、电价和电耗率读取所选数据实例。",
                          f"距离换算：每坐标单位 {scenario['distance_scale_km']} 公里。",
                          "换算依据：" + scenario["distance_scale_source"]]
    return "\n".join(lines) + "\n"


def validate_live_request(expected, actual):
    """Reject semantic mapping changes before solving; never patch missing values."""
    for field in ("instance_id", "model_id", "model_version", "assignment", "distance_scale_km"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"LLM 映射与业务描述不一致：{field}")
    expected_opened, actual_opened = expected.get("opened_depots"), actual.get("opened_depots")
    if (expected_opened is None) != (actual_opened is None) or (
            expected_opened is not None and set(expected_opened) != set(actual_opened)):
        raise ValueError("LLM 映射改变了前序确定的开放仓库")
    if expected["model_id"] == "M4" and not actual.get("distance_scale_source"):
        raise ValueError("LLM 映射丢失了距离单位换算依据")


def map_live_request(args, expected, folder, manifest, output):
    from orllm.rag import retrieve_for_mapping as retrieve
    if getattr(args, "resume", False) and all((folder / name).is_file() for name in
            ("selection.json", "retrieval.json", "mapping.json", "pipeline_report.json")):
        selection = read_json(folder / "selection.json")
        if (selection.get("provider") != "typesafe" or selection.get("status") != "selected"
                or selection.get("model_id") != expected["model_id"]
                or selection.get("confidence", -1) < selection.get("threshold", 0.7)):
            raise ValueError("保存的 JEV 结果不能用于继续此任务")
        from orllm.llm import validate_mapping
        mapping = read_json(folder / "mapping.json")
        validate_mapping(ROOT, mapping, expected["model_id"],
                         [x["instance_id"] for x in pipeline_app.manifest()["instances"]])
        scenario = pipeline_app.scenario_from_mapping(mapping)
        if scenario is None:
            raise ValueError("保存的 LLM 映射未就绪，不能续算")
        validate_live_request(expected, scenario)
        scenario["description"] = (folder / "problem.txt").read_text(encoding="utf-8")
        write_json(folder / "scenario.json", scenario)
        pipeline = read_json(folder / "pipeline_report.json")
        pipeline.update(status="ready", stage="solve", mapping_matches_request=True,
                        provider_records_reused=True, api_calls_this_resume=0)
        write_json(folder / "pipeline_report.json", pipeline)
        print(f"REUSE JEV/RAG/DEEPSEEK {expected['model_id']} semantic_checks=passed", flush=True)
        return scenario, pipeline
    live_args = pipeline_app.make_parser().parse_args([
        "run", "--description-file", str(folder / "problem.txt"),
        "--jev-config", args.jev_config, "--config", args.llm_config,
        "--rag-config", args.rag_config, "--output", str(folder)])
    description = pipeline_app.description_from_args(live_args)
    pipeline = {"status": "running", "stage": "selection", "selection_source": "typesafe",
                "expected_model_id": expected["model_id"], "solver": args.solver,
                "llm_model": manifest["settings"]["llm_model"], "api_calls": 0, "artifacts": []}
    started = time.perf_counter()
    def save():
        write_json(folder / "pipeline_report.json", pipeline)
        write_overview(output, manifest)
    try:
        manifest["api_calls"] += 1
        pipeline["api_calls"] += 1
        save()
        selection = pipeline_app.selection_for_description(live_args, description)
        write_json(folder / "selection.json", selection)
        pipeline["artifacts"].append("selection.json")
        pipeline["model_id"] = selection.get("model_id")
        if selection["status"] != "selected":
            raise ValueError(f"JEV 选型未通过：{selection['status']}")
        if selection["model_id"] != expected["model_id"]:
            raise ValueError(f"JEV 选择 {selection['model_id']}，与本组任务预期 {expected['model_id']} 不同；已保留真实结果")
        print(f"JEV selected={selection['model_id']} confidence={selection['confidence']:.3%}", flush=True)
        pipeline["stage"] = "retrieval"
        save()
        options = pipeline_app.retrieval_options(live_args, default_backend="model_pack")
        context = retrieve(ROOT, description, selection["model_id"], version="1.0", **options)
        if not context:
            raise ValueError("RAG 没有返回所选模型的知识")
        write_json(folder / "retrieval.json", {"model_id": selection["model_id"], **options, "context": context})
        pipeline["artifacts"].append("retrieval.json")
        pipeline.update(stage="mapping", retrieval_backend=options["backend"])
        manifest["api_calls"] += 1
        pipeline["api_calls"] += 1
        save()
        mapping = pipeline_app.mapping_for_selection(live_args, description, selection, context)
        write_json(folder / "mapping.json", mapping)
        pipeline["artifacts"].append("mapping.json")
        scenario = pipeline_app.scenario_from_mapping(mapping)
        if scenario is None:
            raise ValueError(f"DeepSeek 映射未就绪：{mapping['status']}，缺失 {mapping['missing_fields']}，不支持 {mapping['unsupported_requirements']}")
        validate_live_request(expected, scenario)
        scenario["description"] = description
        write_json(folder / "scenario.json", scenario)
        pipeline.update(status="ready", stage="solve", mapping_matches_request=True,
                        provider_and_rag_seconds=time.perf_counter() - started)
        save()
        print(f"DEEPSEEK mapping=ready RAG={options['backend']} context={len(context)} semantic_checks=passed", flush=True)
        return scenario, pipeline
    except Exception as exc:
        pipeline.update(status="error", error=f"{type(exc).__name__}: {exc}")
        save()
        pipeline_app.save_run_report(folder, {"status": "error", "errors": [str(exc)], "artifacts": []})
        raise


def route_reference(m3, m4_dir, instances):
    """Reprice M3 routes under M4's energy rule; do not fix M4 route choices."""
    by_depot = {x["depot_id"]: x for x in instances}
    routes = []
    for route in m3["business"]["routes"]:
        x = by_depot[route["depot_id"]]
        load = sum(x["q"][c] for c in route["nodes"][1:-1])
        energy = 0.0
        for a, b in zip(route["nodes"], route["nodes"][1:]):
            energy += x["ell"][a][b] * (x["r0"] + (x["r1"] - x["r0"]) * load / x["Q"])
            if b != x["depot_id"]:
                load -= x["q"][b]
        routes.append({"route_id": route["route_id"], "depot_id": route["depot_id"],
                       "nodes": route["nodes"], "energy": energy, "battery_capacity": x["B"],
                       "battery_feasible": energy <= x["B"] + 1e-6,
                       "m4_reference_cost": x["h"] + x["p"] * energy})
    complete = bool(routes) and m3["complete_solution"] and all(r["battery_feasible"] for r in routes)
    result = {"source": "../M3/summary.json", "description": "M3 路线按 M4 口径重算的参考方案，M4 自由重新优化路线。",
              "complete_feasible_reference": complete, "routes": routes,
              "reference_objective": sum(r["m4_reference_cost"] for r in routes) if complete else None}
    write_json(m4_dir / "previous_routes_reference.json", result)
    return result


def row_for(case, base, folder, report, summary, parent):
    models = summary["models"]
    artifacts = report["artifacts"]
    checks = bool(artifacts) and all(
        read_json(folder / a["solution"]).get("objective_check", {}).get("valid") is True
        and a.get("primal_check", {}).get("valid") is True for a in artifacts)
    bounds = [m.get("mip_dual_bound") for m in models]
    objective = summary["total_objective"]
    aggregate_gap = None
    if objective is not None and bounds and all(isinstance(b, (int, float)) and math.isfinite(b) for b in bounds):
        aggregate_gap = max(0.0, objective - sum(bounds)) / max(1e-12, abs(objective))
    gaps = [m["mip_gap"] for m in models if m.get("mip_gap") is not None]
    opened = report["provenance"]["scenario_values"].get("opened_depots")
    if report["model_id"] == "M1":
        opened = [d["depot_id"] for d in summary["business"]["depots"] if d["opened"]]
    exact = summary["status"] == "optimal" and aggregate_gap is not None and aggregate_gap <= 1e-8
    label = "最优（gap 0）" if exact else STATUS.get(summary["status"], summary["status"])
    return dict(case=case, instance_id=base["instance_id"], model_id=report["model_id"],
                status=summary["status"], status_label=label, objective=objective,
                costs=summary["costs"], max_subproblem_gap=max(gaps) if gaps else None,
                aggregate_gap=aggregate_gap, runtime_seconds=summary["runtime_seconds"],
                opened_depots=opened, opened_depots_text=", ".join(opened or []),
                customer_count=len(base["C"]), vehicle_count=len(summary["business"]["routes"]) or None,
                subproblem_count=len(artifacts), independent_checks=checks,
                report_path=str((folder / "report.html").relative_to(parent)).replace("\\", "/"))


def write_overview(output, manifest):
    rows = manifest["results"]
    write_json(output / "batch_manifest.json", manifest)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(label for _, label in FIELDS)
        writer.writerows([row.get(key) for key, _ in FIELDS] for row in rows)
    notes = [
        "每组依次执行 M1 → M2 → M3 → M4；M2 固定 M1 的开放仓库，M3/M4 固定 M2 的客户归属。",
        ("本批次每个任务实际调用 JEV 选型、Chroma 检索和 DeepSeek 映射，再使用经过一致性检查的 LLM 配置调用本机求解器。"
         if manifest["settings"].get("pipeline") == "live" else
         "这是固定模型的本地求解批次，模型编号已由本次任务指定；没有调用 JEV / DeepSeek。"),
        "M3 路线会按 M4 能耗口径重算并保存为参考方案；M4 仍按电动车目标重新优化路线。",
        manifest["settings"]["distance_scale_source"],
        "M1 含仓库启用与整单直送成本，M2 仅含整单直送成本，M3 含出车与行驶成本，M4 含出车与用电成本。四列目标值不可直接作为同一种总成本排名。",
        "M3/M4 的目标不包含已固定的仓库启用成本。这是顺序决策的演示，不是联合选址与路径规划的全局最优证明。",
        "限时可行方案保留实际 gap；达到设置的间隙容差不等同于 gap 为零。汇总界差距由各仓目标和下界相加计算，不平均各仓 gap。",
    ]
    settings = manifest["settings"]
    notes.append(f"当前设置：M1/M2 每个模型 {settings['assignment_time_limit']:g} 秒、gap 容差 {settings.get('assignment_mip_gap', 0):.1%}；"
                 f"M3/M4 每仓子模型 {settings['route_time_limit']:g} 秒、gap 容差 {settings['route_mip_gap']:.1%}。"
                 "续算复用的结果保留原始耗时，可能超过当前时限；整批耗时还包括多个子模型、API 与校验。")
    verified = sum(r["independent_checks"] and r["status"] in {"optimal", "feasible_limit"} for r in rows)
    elapsed = time.perf_counter() - manifest["_started_perf"]
    public = {k: v for k, v in manifest.items() if not k.startswith("_")}
    public["completed_task_count"] = len(rows)
    public["verified_task_count"] = verified
    public["wall_time_seconds"] = elapsed
    public["notes"] = notes
    write_json(output / "batch_manifest.json", public)
    captions = ["组别 / 数据", "M1 仓库与直送", "M2 固定仓库分配", "M3 普通车辆路线", "M4 电动车路线"]
    body = []
    md = ["# 五组 M1–M4 顺序求解实验", "", f"完成 {len(rows)} / {len(manifest['instances']) * 4} 个任务；已验证 {verified} 个。", "",
          "| " + " | ".join(captions) + " |", "| " + " | ".join("---" for _ in captions) + " |"]
    for number, instance_id in enumerate(manifest["instances"], 1):
        items = [r for r in rows if r["case"] == number]
        cells = [f"<strong>第 {number} 组</strong><br>{escape(instance_id)}"]
        md_cells = [f"第 {number} 组 / {instance_id}"]
        for model in MODELS:
            r = next((x for x in items if x["model_id"] == model), None)
            if r is None:
                cells.append('<span class="muted">等待前序结果</span>')
                md_cells.append("等待前序结果")
                continue
            tag = "ok" if r["independent_checks"] else "bad"
            details = f"{escape(r['status_label'])}<br>最大子问题 gap {fmt(r['max_subproblem_gap'], 'gap')}<br>{fmt(r['runtime_seconds'])} 秒"
            extra = f"<br>{r['vehicle_count']} 条路线" if r["vehicle_count"] is not None else f"<br>开放 {escape(r['opened_depots_text'])}"
            if r.get("jev_confidence") is not None:
                trace_dir = r["report_path"].rsplit("/", 1)[0]
                extra += f"<br>JEV 置信度 {r['jev_confidence']:.2%}<br>DeepSeek 映射通过 · RAG {r['retrieval_count']} 条"
                extra += f'<br><a href="{trace_dir}/selection.json">选型</a> · <a href="{trace_dir}/mapping.json">映射</a> · <a href="{trace_dir}/retrieval.json">知识</a>'
            cells.append(f'<strong class="value">{fmt(r["objective"])}</strong><span class="{tag}">{details}</span>{extra}<br><a href="{escape(r["report_path"], quote=True)}">查看详细报告</a>')
            md_cells.append(f"[{fmt(r['objective'])}]({r['report_path']}) / {r['status_label']} / gap {fmt(r['max_subproblem_gap'], 'gap')}")
        body.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
        md.append("| " + " | ".join(md_cells) + " |")
    md += ["", "## 解释与口径", "", *[f"- {x}" for x in notes], "", "## 每组目录", "",
           "M1–M4 各自包含 problem.txt、scenario.json、report.html、完整模型、解与日志。chain.json 保存上下游依赖及一致性检查，M4/previous_routes_reference.json 保存 M3 路线的电动车参考成本。", ""]
    (output / "summary.md").write_text("\n".join(md), encoding="utf-8")
    head = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五组 M1–M4 顺序求解实验</title><style>
    *{box-sizing:border-box}body{background:#f3f6fa;color:#203148;margin:0;font:15px/1.65 'Microsoft YaHei','Segoe UI',sans-serif}main{max-width:1450px;margin:auto;padding:32px 24px}h1{font-size:28px;margin:4px 0}h2{font-size:20px}p{margin:8px 0}.eyebrow{color:#5b758c;font-size:12px;letter-spacing:2px}.metrics{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}.metric{background:white;border:1px solid #dce5ef;border-radius:10px;flex:1;min-width:160px;padding:16px}.metric strong{display:block;font-size:24px}.metric span{font-size:13px;color:#61748a}.box{background:white;border:1px solid #dce5ef;border-radius:10px;padding:22px;margin:20px 0}.wrap{overflow:auto}table{border-collapse:collapse;width:100%;min-width:940px}th{background:#eaf0f7;text-align:left;white-space:nowrap}td,th{padding:15px 12px;border-bottom:1px solid #e3eaf2;vertical-align:top}td:first-child{width:19%;overflow-wrap:anywhere}td:not(:first-child){width:20%}.value{display:block;font-size:21px}.ok,.bad{display:block;font-size:12px}.ok{color:#316b5b}.bad{color:#a04037}.muted{color:#8292a1}a{color:#245f97}li{margin:8px 0}.notice{padding:15px 18px;border-left:4px solid #ca9534;background:#fff3d9;margin:18px 0}code{background:#edf2f7;padding:2px 5px;border-radius:4px}.footer{color:#77869a;font-size:12px}@media(max-width:700px){main{padding:18px 12px}.box{padding:14px}h1{font-size:23px}}</style></head><body><main>"""
    metric_values = [("任务完成", f"{len(rows)} / {len(manifest['instances']) * 4}"), ("已通过独立校验", str(verified)),
                     ("数据组数", str(len(manifest["instances"]))), ("累计运行时间/秒", fmt(elapsed))]
    if manifest["settings"].get("pipeline") == "live":
        metric_values.append(("API 请求次数", str(manifest["api_calls"])))
    metrics = '<div class="metrics">' + ''.join(f'<div class="metric"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>' for label,value in metric_values) + '</div>'
    subtitle = ('JEV 选型 → Chroma 检索 → DeepSeek 映射 → 校验 → ' + escape(manifest["settings"]["solver"]) + ' 求解；每组继承前序结果。') if manifest["settings"].get("pipeline") == "live" else '仓库选址 → 固定仓库客户分配 → 普通车辆路线 → 电动车路线'
    html = head + '<p class="eyebrow">FIXED MODELS · SEQUENTIAL DECISIONS · VERIFIED RESULTS</p><h1>五组 M1–M4 顺序求解实验</h1><p>' + subtitle + '</p>' + metrics
    html += '<p class="notice">各模型目标的成本口径不同，请结合成本分项阅读；距离换算是明确的演示实验假设。</p>'
    html += '<section class="box"><h2>五组结果与详细报告</h2><div class="wrap"><table><thead><tr>' + ''.join(f'<th>{escape(x)}</th>' for x in captions) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div></section>'
    html += '<section class="box"><h2>实验解释与数据口径</h2><ul>' + ''.join(f'<li>{escape(x)}</li>' for x in notes) + '</ul></section>'
    html += '<section class="box"><h2>结果文件</h2><p><a href="summary.csv">Excel 汇总 CSV</a> · <a href="summary.md">文字汇总</a> · <a href="batch_manifest.json">完整批次记录</a></p><p>每组目录内的 <code>chain.json</code> 记录来源，M1–M4 子目录各自保留业务描述、场景、模型、解和日志。</p></section>'
    html += f'<p class="footer">批次状态：{escape(manifest["status"])} · 本地求解器：{escape(manifest["settings"]["solver"])}</p></main></body></html>'
    (output / "index.html").write_text(html, encoding="utf-8")


def run(args):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    prefix = "live_chain" if args.pipeline == "live" else "chain"
    output = Path(args.output).resolve() if args.output else ROOT / "outputs/experiments" / f"{prefix}_{stamp}"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("请指定新的输出目录，以免覆盖历史批次")
    output.mkdir(parents=True, exist_ok=True)
    scale_source = (f"演示实验假设：1 坐标单位 = {args.distance_scale_km:g} km，r0/r1 按 kWh/km 解释。"
                    "沿用随附 M4 演示的口径，不表示原始 Excel 已确认物理单位。")
    manifest = {"status": "running", "instances": args.instances, "results": [], "chains": [],
                "settings": {"solver": args.solver, "pipeline": args.pipeline, "assignment_time_limit": args.assignment_time_limit,
                             "route_time_limit": args.route_time_limit, "route_mip_gap": args.route_mip_gap,
                             "assignment_mip_gap": args.assignment_mip_gap,
                             "distance_scale_km": args.distance_scale_km, "distance_scale_source": scale_source},
                "api_calls": 0, "_started_perf": time.perf_counter(),
                "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}
    if args.resume:
        manifest = read_json(output / "batch_manifest.json")
        if manifest["instances"] != args.instances:
            raise ValueError("续算实例清单必须与原批次一致")
        for field in ("pipeline", "solver", "distance_scale_km"):
            if manifest["settings"][field] != getattr(args, field):
                raise ValueError(f"续算不能改变已确认任务的 {field}")
        manifest.setdefault("settings_history", []).append({
            "at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "previous_settings": dict(manifest["settings"]),
            "completed_task_count": len(manifest["results"])})
        manifest["settings"].update(assignment_time_limit=args.assignment_time_limit,
                                    assignment_mip_gap=args.assignment_mip_gap,
                                    route_time_limit=args.route_time_limit,
                                    route_mip_gap=args.route_mip_gap)
        if "started_at" not in manifest:
            manifest["started_at"] = datetime.strptime(output.name[-15:], "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone(timedelta(hours=8))).isoformat()
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(manifest["started_at"])).total_seconds()
        manifest.update(status="running", chains=[], _started_perf=time.perf_counter() - elapsed)
        manifest.pop("error", None)
    if args.pipeline == "live":
        pipeline_app.load_local_environment()
        for name, path in (("jev", args.jev_config), ("llm", args.llm_config)):
            config = read_json(path)
            manifest["settings"][name + "_model"] = config["model"]
            manifest["settings"][name + "_endpoint"] = config["base_url"]
    write_overview(output, manifest)
    print(f"OUTPUT={output}", flush=True)
    try:
        for number, instance_id in enumerate(args.instances, 1):
            base = load_base(ROOT, instance_id)
            if not all(key in base["ev"] for key in ("B", "p", "r0", "r1")):
                raise ValueError(f"{instance_id} lacks EV parameters")
            case_dir = output / f"{number:02d}_{instance_id}"
            case_dir.mkdir(exist_ok=args.resume)
            opened = assignment = None
            summaries = {}
            chain = {"case": number, "instance_id": instance_id, "steps": [], "checks": {}}
            for model_id in MODELS:
                folder = case_dir / model_id
                folder.mkdir(exist_ok=args.resume)
                description = {"M1": "仓库与归属联合决策，采用整单独立直送计费。",
                               "M2": "开放仓库来自本组 M1 的已校验解，重新优化客户归属。",
                               "M3": "开放仓库来自本组 M1，客户归属来自本组 M2 的已校验解；优化常规车辆路线。",
                               "M4": "开放仓库来自本组 M1，客户归属来自本组 M2；本组 M3 路线保留作参照，重新优化电动车路线。"}[model_id]
                if model_id == "M4":
                    description += scale_source
                scenario = dict(model_version="1.0", scenario_id=f"chain_{number:02d}_{model_id}_{instance_id}",
                                instance_id=instance_id, model_id=model_id, description=description,
                                opened_depots=opened if model_id != "M1" else None,
                                assignment=assignment if model_id in {"M3", "M4"} else None,
                                distance_scale_km=args.distance_scale_km if model_id == "M4" else None,
                                distance_scale_source=scale_source if model_id == "M4" else None)
                expected_problem = problem_text(scenario)
                problem_path = folder / "problem.txt"
                if args.resume and problem_path.exists() and problem_path.read_text(encoding="utf-8") != expected_problem:
                    raise ValueError("续算的业务描述与已保存任务不同，不能复用映射")
                write_json(folder / "request.json", scenario)
                problem_path.write_text(expected_problem, encoding="utf-8")
                print(f"START case={number} model={model_id} data={instance_id}", flush=True)
                pipeline = None
                old_row = next((r for r in manifest["results"] if r["case"] == number and r["model_id"] == model_id), None)
                required_gap = args.assignment_mip_gap if model_id in {"M1", "M2"} else args.route_mip_gap
                task_converged = (args.resume and old_row is not None and old_row["status"] == "optimal"
                                  and old_row["independent_checks"] and old_row["max_subproblem_gap"] is not None
                                  and old_row["max_subproblem_gap"] <= required_gap + 1e-10)
                if task_converged:
                    report = read_json(folder / "build_report.json")
                    actual_scenario = read_json(folder / "scenario.json")
                    validate_live_request(scenario, actual_scenario)
                    print(f"REUSE TASK case={number} model={model_id}", flush=True)
                else:
                    if args.pipeline == "live":
                        scenario, pipeline = map_live_request(args, scenario, folder, manifest, output)
                    else:
                        write_json(folder / "scenario.json", scenario)
                    report = run_build(scenario, folder, solver=args.solver,
                                       time_limit=args.assignment_time_limit if model_id in {"M1", "M2"} else args.route_time_limit,
                                       mip_gap=required_gap, reuse_verified=args.resume,
                                       reference_routes=summaries["M3"]["business"]["routes"] if model_id == "M4" else None,
                                       reference_assignments=summaries["M1"]["business"]["assignments"] if model_id == "M2" else None)
                summary = read_json(folder / "summary.json")
                summaries[model_id] = summary
                row = row_for(number, base, folder, report, summary, output)
                if pipeline is not None:
                    pipeline.update(status=report["status"], solve_status=report.get("solve_status"),
                                    reports=report.get("reports", {}))
                    if "build_report.json" not in pipeline["artifacts"]:
                        pipeline["artifacts"].append("build_report.json")
                    write_json(folder / "pipeline_report.json", pipeline)
                if args.pipeline == "live":
                    row.update(jev_confidence=read_json(folder / "selection.json")["confidence"],
                               llm_model=manifest["settings"]["llm_model"],
                               retrieval_count=len(read_json(folder / "retrieval.json")["context"]))
                manifest["results"] = [r for r in manifest["results"] if not (r["case"] == number and r["model_id"] == model_id)]
                manifest["results"].append(row)
                manifest["results"].sort(key=lambda r: (r["case"], r["model_id"]))
                chain["steps"].append({"model_id": model_id, "status": summary["status"],
                                       "objective": row["objective"], "scenario": f"{model_id}/scenario.json",
                                       "solution_source": "M1/M1_all.solution.json" if model_id == "M2" else
                                                          "M2/M2_all.solution.json" if model_id in {"M3", "M4"} else None})
                write_json(case_dir / "chain.json", chain)
                write_overview(output, manifest)
                print(f"DONE case={number} model={model_id} status={summary['status']} objective={fmt(row['objective'])} max_gap={fmt(row['max_subproblem_gap'],'gap')} seconds={fmt(row['runtime_seconds'])}", flush=True)
                if report["status"] != "solved" or not row["independent_checks"] or not summary["cost_breakdown_complete"]:
                    raise RuntimeError(f"case {number} {model_id}: no complete independently checked result")
                if model_id == "M1":
                    opened, _ = derive_decisions(summary, base, "M1")
                    chain["opened_depots_from_M1"] = opened
                elif model_id == "M2":
                    opened2, assignment = derive_decisions(summary, base, "M2")
                    chain["checks"]["M2_opened_depots_match_M1"] = set(opened2) == set(opened)
                    if set(opened2) != set(opened):
                        raise ValueError("M2 changed fixed depot set")
                    chain["assignment_from_M2"] = assignment
                    direct = summaries["M1"]["costs"]["delivery_cost"]
                    chain["checks"]["M2_delivery_cost_no_worse_than_M1"] = summary["total_objective"] <= direct + 1e-5
                    if not chain["checks"]["M2_delivery_cost_no_worse_than_M1"]:
                        raise ValueError("M2 is worse than its feasible M1 allocation")
                else:
                    actual = report["provenance"]["scenario_values"]
                    chain["checks"][f"{model_id}_assignment_matches_M2"] = actual["assignment"] == assignment
                    chain["checks"][f"{model_id}_depots_match_M1"] = set(actual["opened_depots"]) == set(opened)
                    if not all(chain["checks"].values()):
                        raise ValueError("Sequential scenario inputs do not match upstream decisions")
                    if model_id == "M4":
                        instances = [read_json(folder / a["instance"]) for a in report["artifacts"]]
                        reference = route_reference(summaries["M3"], folder, instances)
                        chain["M3_routes_as_M4_reference"] = {
                            "feasible": reference["complete_feasible_reference"],
                            "objective_under_M4_rules": reference["reference_objective"],
                            "record": "M4/previous_routes_reference.json"}
                write_json(case_dir / "chain.json", chain)
            manifest["chains"].append(chain)
            write_overview(output, manifest)
        manifest["status"] = "complete"
        write_overview(output, manifest)
        print(f"COMPLETE {len(manifest['results'])} verified tasks. REPORT={output/'index.html'}", flush=True)
        return 0
    except Exception as exc:
        manifest.update(status="error", error=f"{type(exc).__name__}: {exc}")
        write_overview(output, manifest)
        raise


def main():
    p = argparse.ArgumentParser(description="五组 M1–M4 顺序实验，支持 JEV/RAG/DeepSeek 和更长时限续算")
    p.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    p.add_argument("--output")
    p.add_argument("--resume", action="store_true", help="复用原批次 API 记录与已达标解，续算未达标子问题")
    p.add_argument("--pipeline", choices=("fixed", "live"), default="fixed")
    p.add_argument("--jev-config", default=pipeline_app.default_config("jev"))
    p.add_argument("--llm-config", default=pipeline_app.default_config("llm"))
    p.add_argument("--rag-config", default=pipeline_app.default_config("rag"))
    p.add_argument("--solver", choices=("copt",), default="copt")
    p.add_argument("--assignment-time-limit", type=float, default=300)
    p.add_argument("--assignment-mip-gap", type=float, default=0.08)
    p.add_argument("--route-time-limit", type=float, default=300)
    p.add_argument("--route-mip-gap", type=float, default=0.08)
    p.add_argument("--distance-scale-km", type=float, default=0.1)
    a = p.parse_args()
    if a.resume and not a.output:
        p.error("--resume 必须指定原批次的 --output 目录")
    if any(not math.isfinite(x) or x <= 0 for x in (a.assignment_time_limit,a.route_time_limit,a.distance_scale_km)):
        p.error("时间限制与距离换算比例必须为正的有限数值")
    if any(not math.isfinite(x) or not 0 <= x < 1 for x in (a.assignment_mip_gap, a.route_mip_gap)):
        p.error("assignment-mip-gap 和 route-mip-gap 必须在 [0,1) 内")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
