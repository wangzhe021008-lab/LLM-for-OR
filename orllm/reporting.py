"""Chinese business reports over independently verified fixed-model solutions.

The JSON/LP/solution files remain the audit record. This module decodes only
verified incumbents, and never promotes a partial sum to a complete objective.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import re


STATUS = {
    "optimal": "达到求解器最优性容差", "feasible_limit": "已有可行方案，尚未证明最优",
    "infeasible": "模型不可行", "unbounded": "模型无界", "error": "求解出错",
    "no_solution": "未获得通过检查的可行解", "not_solved": "已建模，尚未求解",
    "invalid_solution": "解校验未通过", "partial": "仅部分子问题获得可行方案",
    "needs_clarification": "需补充业务信息", "invalid": "输入校验未通过",
    "needs_review": "选型待审核", "no_match": "没有匹配模型",
    "infeasible_or_unbounded": "不可行或无界，尚未区分", "unsupported": "当前模型不支持该业务需求",
}
MODEL_NAMES = {"M1": "仓库选址与客户分配", "M2": "固定仓库的客户分配",
               "M3": "固定归属的车辆路径", "M4": "固定归属的电动车辆路径"}
COST_NAMES = {"opening_cost": "仓库启用成本", "delivery_cost": "整单直送成本",
              "vehicle_cost": "车辆启动成本", "travel_cost": "路径行驶成本",
              "energy_cost": "电耗成本"}
TABLES = {
    "model_summary": ("子问题求解情况", [
        ("model_id", "模型"), ("depot_id", "仓库/范围"), ("status_label", "状态"),
        ("solver", "求解器"), ("objective", "已验证目标值"), ("mip_gap", "相对差距"),
        ("mip_dual_bound", "目标下界"), ("runtime_seconds", "耗时/秒"),
        ("customer_count", "客户数"), ("variable_count", "变量数"),
        ("constraint_count", "约束数"), ("check_label", "原始约束检查"),
        ("max_absolute_violation", "最大绝对违反量"), ("objective_check_label", "目标值复核")]),
    "depots": ("仓库与容量", [("model_id", "模型"), ("depot_id", "仓库"),
        ("opened", "启用"), ("customer_count", "客户数"), ("demand", "配送需求"),
        ("capacity", "仓库容量"), ("utilization", "容量利用率"),
        ("opening_cost", "启用成本"), ("delivery_cost", "整单直送成本"),
        ("vehicle_count", "车辆数/路线数")]),
    "assignments": ("客户分配", [("model_id", "模型"), ("customer_id", "客户"),
        ("depot_id", "归属仓库"), ("demand", "需求量"), ("delivery_cost", "整单直送成本")]),
    "routes": ("配送路线", [("model_id", "模型"), ("depot_id", "仓库"),
        ("route_id", "路线"), ("path", "访问顺序"), ("customer_count", "客户数"),
        ("demand", "装载需求"), ("capacity", "车辆容量"), ("utilization", "装载率"),
        ("vehicle_cost", "启动成本"), ("travel_cost", "行驶成本"),
        ("distance_km", "距离/km"), ("energy", "实际能耗"),
        ("battery_capacity", "电池容量"), ("energy_cost", "电耗成本"), ("total_cost", "路线总成本")]),
    "route_arcs": ("路线分段明细", [("model_id", "模型"), ("depot_id", "仓库"),
        ("route_id", "路线"), ("sequence", "段序"), ("from_node", "起点"),
        ("to_node", "终点"), ("load", "该段车载需求"), ("travel_cost", "行驶成本"),
        ("distance_km", "距离/km"), ("energy", "实际能耗"), ("energy_cost", "电耗成本")]),
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _status_label(status, mip_gap=None):
    # Solvers also return "optimal" when the configured nonzero gap is met.
    if status == "optimal" and _finite(mip_gap) and 0 <= mip_gap <= 1e-8:
        return "最优（gap 0）"
    return STATUS.get(status, status)


def _format(value, key=""):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if _finite(value):
        if key == "max_absolute_violation" and value != 0:
            return f"{value:.3g}"
        if key in {"mip_gap", "utilization"}:
            return f"{value:.2%}"
        if key in {"customer_count", "vehicle_count", "variable_count", "constraint_count", "sequence"}:
            return str(int(value))
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _values(solution):
    values = {}
    for row in solution.get("decision_values", []):
        key = (row["family"], tuple(row["indices"]))
        if key in values or not _finite(row["value"]):
            raise ValueError("决策值重复或含非有限数值")
        values[key] = float(row["value"])
    return values


def _assignment(instance, values):
    model_id, D, C = instance["model_id"], instance["D"], instance["C"]
    depots, assignments = [], []
    opening_total = delivery_total = 0.0
    for customer in C:
        assigned = [d for d in D if values["z", (d, customer)] > 0.5]
        if len(assigned) != 1:
            raise ValueError(f"客户 {customer} 未恢复出唯一仓库")
        depot = assigned[0]
        assignments.append(dict(model_id=model_id, customer_id=customer, depot_id=depot,
            demand=instance["q"][customer], delivery_cost=instance["g"][depot][customer]))
    for depot in D:
        opened = values["y", (depot,)] > 0.5 if model_id == "M1" else True
        rows = [r for r in assignments if r["depot_id"] == depot]
        demand = sum(r["demand"] for r in rows)
        opening = instance["F"][depot] * values["y", (depot,)] if model_id == "M1" else None
        delivery = sum(instance["g"][depot][c] * values["z", (depot, c)] for c in C)
        opening_total += opening or 0
        delivery_total += delivery
        depots.append(dict(model_id=model_id, depot_id=depot, opened=opened,
            customer_count=len(rows), demand=demand, capacity=instance["U"][depot],
            utilization=demand / instance["U"][depot], opening_cost=opening,
            delivery_cost=delivery, vehicle_count=None))
    costs = {"delivery_cost": delivery_total}
    if model_id == "M1":
        costs["opening_cost"] = opening_total
    return {"depots": depots, "assignments": assignments, "routes": [], "route_arcs": [], "costs": costs}


def _routing(instance, values):
    model_id, depot, C = instance["model_id"], instance["depot_id"], instance["C"]
    electric = model_id == "M4"
    nodes = [depot] + C
    selected = {(a, b) for a in nodes for b in nodes if a != b and values["x", (a, b)] > 0.5}
    successors = {}
    for a, b in sorted(selected):
        successors.setdefault(a, []).append(b)
    if any(len(successors.get(c, [])) != 1 for c in C):
        raise ValueError("无法从选中弧恢复完整路径")
    used, visited, routes, arc_rows = set(), set(), [], []
    for first in successors.get(depot, []):
        path = [depot, first]
        while path[-1] != depot:
            customer = path[-1]
            if customer in visited or customer not in C:
                raise ValueError("路径包含重复客户或不连通子环")
            visited.add(customer)
            path.append(successors[customer][0])
        route_id = f"{depot}-R{len(routes) + 1:02d}"
        demand = sum(instance["q"][c] for c in path[1:-1])
        travel = energy = distance = 0.0
        for sequence, (a, b) in enumerate(zip(path, path[1:]), 1):
            used.add((a, b))
            load = values["f", (a, b)]
            row = dict(model_id=model_id, depot_id=depot, route_id=route_id,
                sequence=sequence, from_node=a, to_node=b, load=load,
                travel_cost=None, distance_km=None, energy=None, energy_cost=None)
            if electric:
                length = instance["ell"][a][b]
                segment_energy = length * (instance["r0"] * values["x", (a, b)]
                    + (instance["r1"] - instance["r0"]) * load / instance["Q"])
                row.update(distance_km=length, energy=segment_energy,
                           energy_cost=instance["p"] * segment_energy)
                distance += length
                energy += segment_energy
            else:
                row["travel_cost"] = instance["c"][a][b] * values["x", (a, b)]
                travel += row["travel_cost"]
            arc_rows.append(row)
        vehicle_cost = instance["h"] * values["x", (depot, first)]
        energy_cost = instance["p"] * energy if electric else None
        routes.append(dict(model_id=model_id, depot_id=depot, route_id=route_id,
            path=" → ".join(path), nodes=path, customer_count=len(path) - 2,
            demand=demand, capacity=instance["Q"], utilization=demand / instance["Q"],
            vehicle_cost=vehicle_cost, travel_cost=None if electric else travel,
            distance_km=distance if electric else None, energy=energy if electric else None,
            battery_capacity=instance["B"] if electric else None, energy_cost=energy_cost,
            total_cost=vehicle_cost + (energy_cost if electric else travel)))
    if visited != set(C) or used != selected:
        raise ValueError("部分客户或选中弧未形成从仓库出发并返回的路线")
    costs = {"vehicle_cost": sum(r["vehicle_cost"] for r in routes)}
    cost_key = "energy_cost" if electric else "travel_cost"
    costs[cost_key] = sum(r[cost_key] for r in routes)
    demand = sum(instance["q"].values())
    depots = [dict(model_id=model_id, depot_id=depot, opened=True, customer_count=len(C),
        demand=demand, capacity=instance["U"], utilization=demand / instance["U"],
        opening_cost=None, delivery_cost=None, vehicle_count=len(routes))]
    return {"depots": depots, "assignments": [], "routes": routes, "route_arcs": arc_rows, "costs": costs}


def _size(instance, artifact):
    n = len(instance.get("C", []))
    d = len(instance.get("D", []))
    model_id = instance.get("model_id")
    arcs = n * (n + 1)
    defaults = {"M1": (d + d*n, n + d*n + d), "M2": (d*n, n+d),
                "M3": (2*arcs, 4*n+1+arcs), "M4": (2*arcs+n, n*n+5*n+1+arcs)}
    variables, constraints = defaults.get(model_id, (None, None))
    return artifact.get("variable_count", variables), artifact.get("constraint_count", constraints)


def _check_label(check, *, legacy=False):
    if not check:
        return "旧结果未记录" if legacy else "未进行"
    if check.get("valid") is True:
        return "通过"
    if check.get("reason") in {"No complete incumbent", "No validated incumbent"}:
        return "无可行解可检查"
    return "未通过"


def _warning_message(item, summary):
    """Explain preflight warnings in the context of the *current* solve."""
    if not isinstance(item, dict):
        return str(item)
    code, field = item.get("code"), item.get("field", "")
    if code == "FEASIBILITY_NOT_CERTIFIED":
        relevant = [r for r in summary["models"] if r["model_id"] in {"M1", "M2"}]
        if relevant and all(r["has_verified_solution"] for r in relevant):
            return "建模前的容量检查只验证了必要条件；本次求解已获得通过独立约束检查的客户分配，方案可行性已验证。"
        return "仓库总容量充足只是必要条件，不能保证不可拆分的客户需求一定能够分配；仍需获得并校验可行方案。"
    if code == "EXPLICIT_DISTANCE_ASSUMPTION":
        scenario = summary["provenance"].get("scenario_values", {})
        scale, source = scenario.get("distance_scale_km"), scenario.get("distance_scale_source")
        conversion = f"本次采用每坐标单位 {_format(scale)} km" if scale is not None else "本次采用场景中明确记录的公里换算比例"
        note = f"原始数据未注明坐标单位；{conversion}，电耗率按每公里解释。"
        return note + (f"换算依据/假设：{source}" if source else "换算依据应以场景记录为准。")
    if code == "DIRECT_BASELINE_NOT_CERTIFIED":
        match = re.fullmatch(r"instances\[(\d+)\]", str(field))
        index = int(match.group(1)) if match else None
        row = summary["models"][index] if index is not None and index < len(summary["models"]) else None
        prefix = f"仓库 {row['depot_id']}：" if row else ""
        if row and row["has_verified_solution"]:
            return prefix + "逐客户单独往返的基线方案曾超出电池容量；本次优化路线已通过独立约束检查，优化方案的可行性已验证。"
        return prefix + "逐客户单独往返的基线方案超出电池容量；这只能说明该基线未通过，不能据此断言原路径模型不可行。请以求解状态为准。"
    # Unknown diagnostic types remain understandable without rendering raw JSON.
    text = item.get("message", _format(item))
    return f"{field}：{text}" if field else str(text)


def _fallback_message(row):
    attempts = row.get("solver_attempts", [])
    chain = " → ".join(str(a.get("solver", "未知后端")) for a in attempts)
    reason_names = {"dependency_unavailable": "Python 依赖不可用", "license_error": "许可证不适用或不可用", "backend_error": "后端运行出错"}
    reasons = [f"{a.get('solver', '原后端')}：{reason_names.get(a.get('error_type'), '未能完成求解')}"
               for a in attempts[:-1] if a.get("status") == "error"]
    text = f"求解过程中已自动切换后端（{chain}），最终使用 {row['solver']}。" if chain else f"求解过程中已自动切换后端，最终使用 {row['solver']}。"
    return text + ("切换原因：" + "；".join(reasons) + "。" if reasons else "")


def _summarize(report, instances, solutions):
    artifacts = report.get("artifacts", [])
    rows, business = [], {key: [] for key in ("depots", "assignments", "routes", "route_arcs")}
    warnings = list(report.get("warnings", []))
    for index, instance in enumerate(instances):
        solution = solutions[index] if index < len(solutions) and solutions[index] else {}
        artifact = artifacts[index] if index < len(artifacts) else {}
        status = solution.get("status", "not_solved")
        check = solution.get("primal_check", {})
        objective_check = solution.get("objective_check")
        acceptable = (status in {"optimal", "feasible_limit"} and check.get("valid") is True
                      and _finite(solution.get("objective"))
                      and (objective_check is None or objective_check.get("valid") is True))
        if status in {"optimal", "feasible_limit"} and not acceptable:
            status = "invalid_solution"
        row = dict(model_id=instance.get("model_id"), model_name=MODEL_NAMES.get(instance.get("model_id")),
            instance_id=instance.get("instance_id"), model_version=instance.get("model_version"),
            depot_id=instance.get("depot_id", "全部仓库"), status=status,
            status_label=_status_label(status, solution.get("mip_gap")), solver=solution.get("solver", report.get("solver", "none")),
            objective=solution.get("objective") if acceptable else None,
            mip_gap=solution.get("mip_gap") if acceptable else None,
            mip_dual_bound=solution.get("mip_dual_bound"), runtime_seconds=solution.get("runtime_seconds"),
            customer_count=len(instance.get("C", [])), demand=sum(instance.get("q", {}).values()),
            primal_check=check, check_label=_check_label(check),
            max_absolute_violation=check.get("max_absolute_violation"),
            objective_check=objective_check, objective_check_label=_check_label(objective_check, legacy=acceptable),
            fallback_used=solution.get("fallback_used", False), solver_attempts=solution.get("solver_attempts", []),
            has_verified_solution=acceptable, message=solution.get("message"), artifacts=artifact, costs={})
        row["variable_count"], row["constraint_count"] = _size(instance, artifact)
        if acceptable:
            try:
                values = _values(solution)
                decoded = _assignment(instance, values) if instance["model_id"] in {"M1", "M2"} else _routing(instance, values)
                if not math.isclose(sum(decoded["costs"].values()), solution["objective"], rel_tol=1e-6, abs_tol=1e-4):
                    raise ValueError("业务分项成本与已校验目标值不一致")
                row["costs"] = decoded["costs"]
                row["business_report_status"] = "complete"
                for key in business:
                    business[key].extend(decoded[key])
            except (KeyError, TypeError, ValueError) as exc:
                row["business_report_status"] = "decode_failed"
                row["business_report_message"] = f"业务方案未展示：{exc}"
                warnings.append(row["business_report_message"])
        rows.append(row)
    verified = [r for r in rows if r["has_verified_solution"]]
    complete = bool(rows) and len(verified) == len(rows)
    if complete:
        overall = "optimal" if all(r["status"] == "optimal" for r in rows) else "feasible_limit"
    elif verified:
        overall = "partial"
    elif rows and all(r["status"] == "not_solved" for r in rows):
        overall = "not_solved"
    elif rows:
        statuses = {r["status"] for r in rows}
        overall = next(iter(statuses)) if len(statuses) == 1 else "no_solution"
    else:
        overall = report.get("status", "not_solved")
    costs = {}
    for row in verified:
        for key, value in row["costs"].items():
            costs[key] = costs.get(key, 0) + value
    provenance = report.get("provenance", {})
    compact_provenance = {key: provenance[key] for key in
        ("scenario_description", "source", "transformations", "scenario_values", "model_version") if key in provenance}
    compact_provenance["source_cell_count"] = len(provenance.get("source_cells", {}))
    notes = ["目标值沿用原始数据成本单位；原数据未确认币种时，不标注为人民币。",
             "最优性差距为各子问题的相对 MIP gap；多仓报告不使用各仓 gap 的简单平均。",
             "求解器 optimal 状态表示达到其最优性容差；只有已记录的 gap 接近零时才标注 gap 0。限时可行方案仍可使用，但尚未达到所设最优性要求。",
             "原始约束检查包含变量上下界、整数性和全部原始线性约束；目标值复核比较重新计算的成本与求解器目标值。精度见各 solution.json。"]
    model_ids = {instance.get("model_id") for instance in instances}
    if model_ids & {"M1", "M2"}:
        notes.append("M1/M2 的 g 是客户整单往返直送成本，已含一次车辆启动费；不再乘需求量 q。分配结果不等同于多客户串点路线。")
    if "M2" in model_ids:
        notes.append("M2 仓库启用状态由场景给定，目标不含仓库启用成本；本次只优化客户归属。")
    if model_ids & {"M3", "M4"}:
        notes.append("M3/M4 的仓库启用与客户归属已由场景固定；各仓独立求解，汇总目标仅覆盖这些既定归属。每条路线按一辆车计，未建模车辆跨趟调度。")
        notes.append("路径模型仅为有客户的仓库生成子问题；已启用但无客户的仓库以场景记录为准。")
    if "M3" in model_ids:
        notes.append("M3 的 c 是行驶成本矩阵，不能直接解释为公里数；因此距离与能耗留空。")
    if "M4" in model_ids:
        notes.append("M4 实际能耗逐弧按 ell × [r0 × x + (r1 − r0) × f / Q] 计算，不使用辅助变量 e 充当实际能耗；能量单位沿用输入口径，距离缩放假设见来源。")
    runtimes = [r["runtime_seconds"] for r in rows]
    gaps = [r["mip_gap"] for r in rows]
    maximum_gap = max(gaps) if gaps and all(_finite(gap) for gap in gaps) else None
    summary = dict(schema_version="1.0", generated_at=datetime.now(timezone.utc).isoformat(),
        scenario_id=report.get("scenario_id"), build_status=report.get("build_status", report.get("status")), status=overall,
        status_label=_status_label(overall, maximum_gap), subproblem_count=len(rows),
        verified_subproblem_count=len(verified), complete_solution=complete,
        total_objective=sum(r["objective"] for r in verified) if complete else None,
        partial_objective=sum(r["objective"] for r in verified) if verified and not complete else None,
        costs=costs, cost_scope="完整方案" if complete else "已验证部分，不能作为完整总成本",
        cost_breakdown_complete=bool(verified) and all(r.get("business_report_status") == "complete" for r in verified),
        runtime_seconds=sum(runtimes) if runtimes and all(_finite(x) for x in runtimes) else None,
        models=rows, business=business, provenance=compact_provenance,
        missing_fields=report.get("missing_fields", []), errors=report.get("errors", []),
        unsupported_requirements=report.get("unsupported_requirements", []), warnings=warnings, notes=notes)
    summary["warning_messages"] = [_warning_message(item, summary) for item in warnings]
    return summary


def _markdown_cell(value, key=""):
    return _format(value, key).replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ").replace("\r", " ")


def _markdown_table(rows, fields):
    lines = ["| " + " | ".join(label for _, label in fields) + " |",
             "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(_markdown_cell(row.get(key), key) for key, _ in fields) + " |" for row in rows)
    return "\n".join(lines)


def _html_table(rows, fields):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        f'<th>{escape(label)}</th>' for _, label in fields) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(_format(row.get(key), key))}</td>' for key, _ in fields) + '</tr>'
        for row in rows) + '</tbody></table></div>'


def _csv_value(value):
    # Excel must treat arbitrary customer IDs/descriptions as data, never formulas.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _capacity_chart(rows):
    opened = [row for row in rows if row.get("opened") and _finite(row.get("utilization"))]
    if not opened:
        return ""
    charts = []
    for row in opened[:16]:
        ratio = row["utilization"]
        label = f'{row["depot_id"]}：容量利用率 {_format(ratio, "utilization")}，需求 {_format(row["demand"])} / 容量 {_format(row["capacity"])}'
        charts.append('<div class="capacity-row" role="img" aria-label="' + escape(label, quote=True) + '">'
            + f'<span>{escape(str(row["depot_id"]))}</span><div class="capacity-track"><div style="width:{min(100, max(0, ratio * 100)):.3f}%"></div></div>'
            + f'<strong>{escape(_format(ratio, "utilization"))}</strong></div>')
    suffix = "（前 16 个）" if len(opened) > 16 else ""
    return '<div class="capacity-chart"><p class="chart-title">已验证方案中的已启用仓库 · 容量利用率' + suffix + '</p>' + ''.join(charts) + '<p class="chart-note">需求量 / 仓库容量；精确数值见下表。</p></div>'


def write_run_report(output_dir, report, instances, solutions=None):
    """Write self-contained HTML, Markdown, JSON and UTF-8-BOM business CSVs.

    ``instances`` are canonical full instances; ``solutions`` correspond by order
    and may be absent/None. Return file names relative to ``output_dir``.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    instances = list(instances or [])
    solutions = list(solutions or [])
    summary = _summarize(report, instances, solutions)
    files = {"summary_json": "summary.json", "summary_markdown": "summary.md", "report_html": "report.html", "business_csv": []}
    table_rows = {"model_summary": summary["models"], **summary["business"]}
    for name, rows in table_rows.items():
        filename = f"{name}.csv"
        if not rows:
            # Reusing an output directory must not leave an old feasible plan
            # visible after a failed/unsolved run. Only our exact CSV names are removed.
            (output / filename).unlink(missing_ok=True)
            continue
        fields = TABLES[name][1]
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(label for _, label in fields)
            writer.writerows([_csv_value(row.get(key)) for key, _ in fields] for row in rows)
        files["business_csv"].append(filename)
    summary["report_files"] = files
    file_rows = [dict(name="report.html", description="首先打开：状态、成本、仓库、分配及路线的可读报告"),
        dict(name="summary.md / summary.json", description="文字摘要 / 程序可读取摘要"),
        dict(name="*.csv", description="Excel 可直接打开的业务明细；比例按小数保存"),
        dict(name="build_report.json", description="建模过程、警告、参数转换与原始数据来源"),
        dict(name="*.instance.json", description="本次实际绑定的完整模型参数"),
        dict(name="*.model.json / *.lp", description="固定数学模型、变量映射与求解器输入"),
        dict(name="*.solution.json", description="求解器状态、完整决策值、差距及独立约束检查（求解后生成）")]
    title = "ORLLM 求解与业务报告"
    metrics = [("总体状态", summary["status_label"]),
        ("已验证子问题", f'{summary["verified_subproblem_count"]} / {summary["subproblem_count"]}'),
        ("完整方案总成本", _format(summary["total_objective"])),
        ("求解耗时合计/秒", _format(summary["runtime_seconds"]))]
    markdown = [f"# {title}", "", f'场景：{_markdown_cell(summary["scenario_id"])}', "",
        *[f"- {label}：{value}" for label, value in metrics], ""]
    html_parts = [f'<header><p class="eyebrow">FIXED MODEL · VERIFIED SOLUTION</p><h1>{title}</h1><p>场景：{escape(_format(summary["scenario_id"]))}</p></header>',
        '<div class="metrics">' + ''.join(f'<div class="metric"><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>' for label, value in metrics) + '</div>']
    source = summary["provenance"]
    description = source.get("scenario_description") or source.get("scenario_values", {}).get("description")
    if description:
        is_demo = any(word in str(description).lower() for word in ("演示", "假设", "demo", "demonstration"))
        heading = "演示场景与假设" if is_demo else "场景说明"
        markdown += [f"## {heading}", "", _markdown_cell(description), ""]
        html_parts.append(f'<div class="scenario-note"><strong>{heading}</strong><p>{escape(str(description))}</p></div>')
    if summary["partial_objective"] is not None:
        message = f'目前仅覆盖 {summary["verified_subproblem_count"]} 个子问题；已验证部分成本 {_format(summary["partial_objective"])}，不能作为完整总成本。下方业务明细仅展示通过检查的子问题。'
        markdown += [message, ""]
        html_parts.append(f'<p class="notice">{escape(message)}</p>')
    if not summary["verified_subproblem_count"]:
        message = "本次没有可展示的已验证业务方案。请先查看下方状态、错误或待补充信息。"
        markdown += [message, ""]
        html_parts.append(f'<p class="notice">{message}</p>')
    if summary["costs"]:
        cost_rows = [dict(item=COST_NAMES.get(k, k), value=v) for k, v in summary["costs"].items()]
        cost_title = f'成本分项（{summary["cost_scope"]}）'
        if not summary["cost_breakdown_complete"]:
            cost_title += "：部分明细恢复失败"
        fields = [("item", "成本项"), ("value", "数值/原始成本单位")]
        markdown += [f"## {cost_title}", "", _markdown_table(cost_rows, fields), ""]
        html_parts.append(f'<section><h2>{escape(cost_title)}</h2>{_html_table(cost_rows, fields)}</section>')
    for name, rows in table_rows.items():
        if not rows:
            continue
        table_title, fields = TABLES[name]
        markdown += [f"## {table_title}", "", _markdown_table(rows, fields), ""]
        chart = _capacity_chart(rows) if name == "depots" else ""
        html_parts.append(f'<section><h2>{table_title}</h2>{chart}{_html_table(rows, fields)}</section>')
    messages = []
    for row in summary["models"]:
        if row["fallback_used"]:
            messages.append(f'{row["model_id"]} / {row["depot_id"]}：{_fallback_message(row)}')
        if row.get("message"):
            messages.append(f'{row["model_id"]} / {row["depot_id"]}：{row["message"]}')
    for key, label in (("missing_fields", "待补字段"), ("errors", "错误"),
                       ("unsupported_requirements", "未覆盖需求"), ("warning_messages", "检查说明")):
        messages.extend(f"{label}：{_format(item)}" for item in summary[key])
    if messages:
        markdown += ["## 检查信息", "", *[f"- {_markdown_cell(m)}" for m in messages], ""]
        html_parts.append('<section><h2>检查信息</h2><ul>' + ''.join(f'<li>{escape(m)}</li>' for m in messages) + '</ul></section>')
    source_text = json.dumps(source, ensure_ascii=False, indent=2)
    markdown += ["## 口径说明", "", *[f"- {note}" for note in summary["notes"]], "",
        "## 来源与场景依据", "", "```json", source_text, "```", "",
        "## 文件说明", "", _markdown_table(file_rows, [("name", "文件"), ("description", "用途")]), ""]
    html_parts.append('<section><h2>口径说明</h2><ul>' + ''.join(f'<li>{escape(note)}</li>' for note in summary["notes"]) + '</ul></section>')
    html_parts.append(f'<section><h2>来源与场景依据</h2><details><summary>展开数据来源、场景设置与参数转换</summary><pre>{escape(source_text)}</pre></details></section>')
    html_parts.append('<section><h2>文件说明</h2>' + _html_table(file_rows, [("name", "文件"), ("description", "用途")]) + '</section>')
    html_parts.append(f'<footer>生成时间（UTC）：{escape(summary["generated_at"])} · ORLLM</footer>')
    stylesheet = """
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#203148;font:15px/1.6 "Microsoft YaHei","Segoe UI",sans-serif}
    main{max-width:1440px;margin:auto;padding:36px 30px 48px}header{border-bottom:1px solid #d7e0ea;padding-bottom:18px;margin-bottom:24px}
    .eyebrow{font:12px "Segoe UI",sans-serif;letter-spacing:2px;color:#59748d;margin:0}h1{font-size:30px;margin:8px 0}h2{font-size:19px;margin:0 0 14px}
    p{margin:8px 0}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:22px 0}.metric{padding:20px;background:#fff;border:1px solid #dce5ef;border-radius:12px}
    .metric span{display:block;color:#61748a;font-size:13px}.metric strong{display:block;font-size:22px;line-height:1.45;margin-top:7px;overflow-wrap:anywhere}
    section{margin-top:22px;background:#fff;padding:24px;border:1px solid #dce5ef;border-radius:12px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}
    th{background:#eaf0f7;color:#334e6a;text-align:left;white-space:nowrap}td,th{padding:10px 12px;border-bottom:1px solid #e3eaf2;vertical-align:top}td{min-width:72px;overflow-wrap:anywhere}
    tbody tr:nth-child(even){background:#f7f9fc}tbody tr:hover{background:#eef5fa}.notice{background:#fff3d9;border-left:4px solid #d59a29;padding:14px 18px;border-radius:5px}
    .scenario-note{background:#eaf2fb;border-left:4px solid #527da6;padding:16px 20px;border-radius:5px;margin:18px 0;overflow-wrap:anywhere}.scenario-note p{margin-bottom:0}
    .capacity-chart{padding:0 2px 18px;max-width:900px}.chart-title{font-size:13px;color:#4a637b}.capacity-row{display:grid;grid-template-columns:90px 1fr 80px;align-items:center;gap:16px;margin:12px 0;font-size:13px}
    .capacity-row span{overflow-wrap:anywhere}.capacity-row strong{text-align:right;font-weight:500}.capacity-track{height:12px;background:#e5edf5;border-radius:6px;overflow:hidden}.capacity-track div{height:100%;background:#688faf;border-radius:6px}.chart-note{font-size:12px;color:#77869a}
    li{margin:8px 0;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:18px;font:13px/1.7 Consolas,monospace}
    summary{cursor:pointer;color:#31587d}footer{font-size:12px;color:#77869a;margin-top:28px}@media(max-width:800px){main{padding:20px 14px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}section{padding:16px}}
    @media print{body{background:white}main{padding:0;max-width:none}section,.metric{break-inside:avoid;border-radius:0}.table-wrap{overflow:visible}table{font-size:10px}td,th{padding:5px}details{display:block}}
    """
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>' + stylesheet + '</style></head><body><main>' + ''.join(html_parts) + '</main></body></html>'
    (output / files["summary_json"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / files["summary_markdown"]).write_text("\n".join(markdown), encoding="utf-8")
    (output / files["report_html"]).write_text(document, encoding="utf-8")
    return files
