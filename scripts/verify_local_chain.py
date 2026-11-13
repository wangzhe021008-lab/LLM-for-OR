"""Exercise the actual web workspace: local Qwen + live JEV, M1 -> M2 -> M3 -> M4."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from html import escape
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, build_opener, ProxyHandler
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(output, record):
    fields = [("model", "阶段"), ("purpose", "任务"), ("status", "状态"), ("objective", "目标值"),
              ("gap_percent", "最大子问题 gap (%)"), ("verified_count", "校验通过子问题"),
              ("subproblem_count", "子问题数"), ("route_count", "路线数"),
              ("elapsed_seconds", "全流程秒数"), ("solve_seconds", "求解秒数"), ("decisions_preserved", "继承一致"),
              ("result_id", "结果目录")]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(label for _, label in fields)
        for row in record["steps"]:
            writer.writerow(row.get(key) for key, _ in fields)
    save(output / "chain_validation.json", record)
    body = []
    for row in record["steps"]:
        cells = [escape(str(row.get(key, "—"))) for key, _ in fields[:-1]]
        href = record["base_url"] + "/artifact?id=" + row["result_id"] + "&name=report.html"
        cells.append('<a href="' + escape(href, quote=True) + '">详细报告</a>')
        body.append("<tr>" + "".join("<td>" + value + "</td>" for value in cells) + "</tr>")
    caption = f"实例 {record['instance']}；目标 gap {record['target_gap']:.0%}；每个子问题限时 {record['time_limit']:g} 秒。"
    attempts = len(record.get("previous_attempts", []))
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>本地 Qwen 完整链条测试</title>'
            '<style>body{font:15px/1.7 "Microsoft YaHei",sans-serif;margin:32px;color:#24384d;background:#f7f9fc}'
            'table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #dce4ed;padding:12px;text-align:left}'
            'th{background:#eaf0f7}a{color:#245f97}p{max-width:1100px}</style>'
            '<h1>本地 Qwen：选址 → 分配 → 车辆路径 → 电动车路径</h1><p>' + escape(caption) + '</p>'
            '<p>每步由真实工作台完成 Qwen 聊天整理 → JEV 选模 → Chroma 检索 → Qwen 参数映射 → 建模 → COPT 求解 → 独立校验。'
            'M2 固定 M1 仓库，M3/M4 固定 M2 客户归属。M4 距离换算 0.1 km/坐标单位是明确的演示实验假设。'
            '各阶段成本口径不同，目标值不能直接排名；这是顺序决策测试。</p>'
            + f'<p>续测次数：{attempts}。此前失败和修正记录保留在 <a href="chain_validation.json">chain_validation.json</a>，已通过的步骤不重复调用或求解。</p>'
            +
            '<table><thead><tr>' + ''.join('<th>' + escape(label) + '</th>' for _, label in fields)
            + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table><p>批次状态：'
            + escape(record['status']) + '</p><p>' + escape(record.get('error', '')) + '</p></html>')
    (output / "index.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--instance", default="G20-5-1a__Q70")
    parser.add_argument("--gap", type=float, default=0.1)
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--resume", type=Path, help="继续同一测试批次，保留通过阶段和失败尝试记录")
    parser.add_argument("--fresh-session", action="store_true", help="续测时新建对话，仍明确引用已通过的上游结果")
    args = parser.parse_args()
    if not 0 <= args.gap < 1 or not 0 < args.time_limit <= 86400:
        parser.error("gap 需在 [0,1) 范围内，time-limit 需在 (0,86400] 范围内")
    if not args.resume:
        from orllm.data import load_base
        ev = load_base(ROOT, args.instance).get("ev", {})
        missing = [key for key in ("B", "p", "r0", "r1") if key not in ev or ev[key] is None]
        if missing:
            raise ValueError("此 M1–M4 全链条测试需要带电动车参数的实例；当前缺少 " + ", ".join(missing))
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output = args.resume.resolve() if args.resume else ROOT / "outputs/experiments" / ("qwen_web_chain_" + stamp)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    opener = build_opener(ProxyHandler({}))
    def api(path, data=None):
        body = None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
        request = Request(args.base_url + path, data=body, headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=30) as response:
            return json.load(response)
    health = api("/api/health")
    if health.get("kind") != "orllm-chat-workspace-v1":
        raise RuntimeError("地址不是 ORLLM 工作台")
    if args.resume:
        record = json.loads((output / "chain_validation.json").read_text(encoding="utf-8"))
        if record.get("solver") != "copt":
            raise ValueError("此历史批次未记录使用 COPT，请新建 COPT 验证批次，避免混合不同求解器的结果。")
        if record["instance"] != args.instance or record["base_url"] != args.base_url:
            raise ValueError("续测不能更改实例或工作台地址")
        session = api("/api/session?id=" + record["session_id"])
        if args.fresh_session:
            record.setdefault("previous_sessions", []).append(record["session_id"])
            session = api("/api/session", {})
            record["session_id"] = session["id"]
        record.setdefault("previous_attempts", []).append({"status": record["status"], "error": record.get("error"), "at": stamp})
        record.update(status="running", target_gap=args.gap, time_limit=args.time_limit)
        record.pop("error", None)
    else:
        session = api("/api/session", {})
        record = {"status": "running", "base_url": args.base_url, "instance": args.instance,
              "target_gap": args.gap, "time_limit": args.time_limit, "llm_source": "qwen_local", "solver": "copt",
              "session_id": session["id"], "steps": [], "live_jev": True,
              "distance_scale_km": 0.1, "distance_scale_source": "显式演示实验假设"}
    prompts = [
        ("M1", "选址与分配", f"使用数据实例 {args.instance}，选择开放的候选仓库，并将全部客户完整分配到已开放仓库，满足仓库容量。按既定整单独立直送口径，最小化仓库开放成本与配送成本，不要求多客户车辆路线。请实际求解。"),
        ("M2", "固定仓库重新分配", "沿用当前结果中已开放的仓库和相同数据实例，固定开放仓库，只重新优化全部客户的归属，满足仓库容量，仅最小化配送成本。本阶段不计仓库开放费用，不做车辆路径规划。请实际求解。"),
        ("M3", "常规车辆路径", "沿用当前结果的开放仓库、完整客户归属和相同数据实例，不改变任何客户所属仓库。采用普通车辆，规划从所属仓库出发并返回的配送路线，每个客户恰好访问一次、不拆单，满足车辆载重容量，不设置客户时间窗。最小化出车固定成本与行驶成本，请实际求解。"),
        ("M4", "电动车路径", "沿用当前结果的开放仓库、完整客户归属和相同数据实例，不改变客户所属仓库。重新规划电动车配送路线，从所属仓库满电出发并返回，途中不充电，每个客户恰好访问一次、不拆单，满足载重和电池续航约束，无客户时间窗。最小化出车固定成本与用电成本，电池、电价和电耗率读取该实例。明确采用演示实验假设：每坐标单位为 0.1 公里，电耗率按 kWh/km 解释；该换算并非已确认的真实物理单位。请实际求解。"),
    ]
    reference = record["steps"][-1]["result_id"] if record["steps"] else None
    expected_depots = set(record["steps"][0]["opened_depots"]) if record["steps"] else None
    expected_assignment = next((row["assignment"] for row in record["steps"] if row["model"] == "M2"), None)
    print("OUTPUT=" + str(output), flush=True)
    write_report(output, record)
    try:
        for model, purpose, prompt in prompts:
            if any(row["model"] == model for row in record["steps"]):
                print("REUSE verified " + model, flush=True)
                continue
            for name in (model + "_job.json", model + "_result.json"):
                path = output / name
                if path.exists():
                    shutil.copyfile(path, output / (stamp + "_previous_" + name))
            print("START " + model, flush=True)
            started = time.monotonic()
            job = None
            job_path, result_path = output / (model + "_job.json"), output / (model + "_result.json")
            if args.resume and job_path.exists() and result_path.exists():
                candidate = json.loads(job_path.read_text(encoding="utf-8"))
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if (candidate.get("status") == "done" and candidate.get("action") == "run"
                        and previous.get("summary", {}).get("complete_solution")):
                    job = candidate
                    started -= job.get("elapsed_seconds", 0)
                    print("RECHECK completed " + model, flush=True)
            if job is None:
                submitted = api("/api/chat", {"session_id": session["id"], "request_id": uuid.uuid4().hex,
                                "prompt": prompt, "selected_result_id": reference, "llm_source": "qwen_local",
                                "settings": {"solver": "copt", "time_limit": args.time_limit, "mip_gap": args.gap}})
                previous_stage = None
                while True:
                    job = api("/api/job?id=" + submitted["job_id"])
                    if job["stage"] != previous_stage:
                        previous_stage = job["stage"]
                        print(model + " " + job.get("stage_label", previous_stage), flush=True)
                    if job["status"] != "running":
                        break
                    if time.monotonic() - started > 1800:
                        raise TimeoutError("本次任务超过 30 分钟；未重复提交，后台任务仍可从工作台查看")
                    time.sleep(1)
            save(output / (model + "_job.json"), job)
            if job["status"] != "done" or job.get("action") != "run" or not job.get("result_id"):
                raise RuntimeError(model + " 未完成求解：" + str(job.get("error") or job.get("reply")))
            result = api("/api/result?id=" + job["result_id"])
            save(output / (model + "_result.json"), result)
            summary = result["summary"]
            actual_model = result["selection"].get("model_id")
            if actual_model != model:
                raise RuntimeError(f"本步预期 {model}，JEV 实际选型 {actual_model}")
            if not summary.get("complete_solution") or not summary.get("cost_breakdown_complete"):
                raise RuntimeError(model + " 未得到完整且独立校验通过的方案")
            if result.get("llm", {}).get("provider_id") != "qwen_local":
                raise RuntimeError("实际模型来源不是本地 Qwen")
            mapping = result["mapping"]
            if mapping["instance_id"] != args.instance:
                raise RuntimeError("数据实例改变")
            if model == "M4" and (mapping.get("distance_scale_km") != record["distance_scale_km"]
                                  or not mapping.get("distance_scale_source")):
                raise RuntimeError("M4 距离换算或来源与显式实验假设不一致")
            depots = {item["depot_id"] for item in summary["business"]["depots"] if item["opened"]}
            assignment = {item["customer_id"]: item["depot_id"] for item in summary["business"]["assignments"]}
            if model in {"M3", "M4"}:
                # Routing reports store customer visits in routes, while the
                # assignment table is reserved for M1/M2 decision variables.
                assignment = {}
                for route in summary["business"]["routes"]:
                    nodes, depot = route["nodes"], route["depot_id"]
                    if len(nodes) < 3 or nodes[0] != depot or nodes[-1] != depot:
                        raise RuntimeError("路线没有从所属仓库出发并返回")
                    for customer in nodes[1:-1]:
                        if customer in assignment:
                            raise RuntimeError("路线中存在重复服务的客户")
                        assignment[customer] = depot
                mapped = {item["customer_id"]: item["depot_id"] for item in mapping.get("assignment") or []}
                if mapped != expected_assignment:
                    raise RuntimeError("路径映射改变了上游客户归属")
            if model == "M1":
                expected_depots = depots
            elif depots != expected_depots:
                raise RuntimeError("继承的开放仓库改变")
            if model == "M2":
                expected_assignment = assignment
            elif model in {"M3", "M4"} and assignment != expected_assignment:
                raise RuntimeError("继承的客户归属改变")
            quality = result["quality"]
            gap = quality.get("actual_max_gap")
            row = {"model": model, "purpose": purpose, "status": summary.get("status_label", summary["status"]),
                   "objective": summary.get("total_objective"), "gap_percent": None if gap is None else round(gap * 100, 4),
                   "target_met": quality.get("gap_state") == "met", "verified_count": quality["verified_count"],
                   "subproblem_count": quality["total_count"], "route_count": len(summary["business"].get("routes", [])),
                   "elapsed_seconds": round(time.monotonic() - started, 2), "decisions_preserved": True,
                   "solve_seconds": round(summary.get("runtime_seconds") or 0, 4),
                   "result_id": job["result_id"], "opened_depots": sorted(depots), "assignment": assignment,
                   "llm": result["llm"], "jev_model": result["selection"].get("provider_model")}
            record["steps"].append(row)
            write_report(output, record)
            print("DONE " + json.dumps({key: row[key] for key in ["model", "objective", "gap_percent", "verified_count", "route_count", "elapsed_seconds"]}, ensure_ascii=False), flush=True)
            reference = job["result_id"]
        record["status"] = "passed" if all(row["target_met"] for row in record["steps"]) else "verified_feasible_with_gap_warning"
        write_report(output, record)
        print("COMPLETE " + record["status"], flush=True)
        return 0
    except Exception as exc:
        record.update(status="error", error=str(exc))
        write_report(output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
