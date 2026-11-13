"""Command-line entry point. Run `python main.py --help`."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_IDS = ("M1", "M2", "M3", "M4")


def default_config(name):
    local = ROOT / f"config/{name}.local.json"
    return str(local if local.exists() else ROOT / f"config/{name}.example.json")


def load_local_environment():
    """Load local credentials without replacing existing environment values."""
    if (ROOT / ".env").exists():
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")


def new_run_output():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    return str(ROOT / "outputs/runs" / stamp)


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"JSON 不允许非有限数值: {value}")
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, parse_constant=reject_constant)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def print_report(report, output):
    labels = {"built": "模型已生成（未求解）", "solved": "求解完成，解已通过独立校验",
              "solve_failed": "求解未完成，请查看报告", "optimal": "达到求解器最优性容差",
              "feasible_limit": "已获得可行解，尚未证明最优", "infeasible": "模型不可行",
              "unbounded": "模型无界", "no_solution": "未获得通过校验的可行解",
              "partial": "部分子模型求解成功", "error": "执行错误",
              "invalid": "参数校验失败", "needs_clarification": "需要补充信息",
              "unsupported": "模型不能覆盖业务要求", "needs_review": "选型需复核",
              "no_match": "未找到适用模型"}
    print("\nORLLM 运行结果")
    print(f"状态：{labels.get(report['status'], report['status'])}")
    if report.get("scenario_id"):
        print(f"任务：{report['scenario_id']}")
    for artifact in report.get("artifacts", []):
        model = artifact.get("model_id", artifact.get("instance", ""))
        group = f" / {artifact['depot_id']}" if artifact.get("depot_id") else ""
        if "solve_status" in artifact:
            status = labels.get(artifact["solve_status"], artifact["solve_status"])
            print(f"  {model}{group}：{status}；求解器 {artifact.get('solver', report.get('solver'))}")
            if artifact.get("objective") is not None:
                gap = artifact.get("mip_gap")
                gap_text = f"；gap {gap:.4%}" if gap is not None else ""
                print(f"    目标值 {artifact['objective']:,.4f}{gap_text}；耗时 {artifact.get('runtime_seconds', 0):.3f} 秒")
        else:
            print(f"  {model}{group}：{artifact.get('variable_count', 0)} 个变量 / {artifact.get('constraint_count', 0)} 条约束")
    for key, title in (("missing_fields", "缺少信息"), ("unsupported_requirements", "未覆盖要求"), ("errors", "错误")):
        if report.get(key):
            print(f"{title}：" + "; ".join(item.get("message", str(item)) if isinstance(item, dict) else str(item) for item in report[key]))
    out = Path(output).resolve()
    if report.get("reports", {}).get("report_html"):
        print(f"查看报告：{out / report['reports']['report_html']}")
    print(f"完整记录：{out / 'build_report.json'}")


def report_exit_code(report, solver):
    if report["status"] == "solve_failed":
        return 3
    if report["status"] not in {"built", "solved"}:
        return 2
    if solver != "none" and any(a.get("solve_status") not in ("optimal", "feasible_limit") for a in report["artifacts"]):
        return 3
    return 0


def save_run_report(output, report, instances=None, solutions=None):
    from orllm.reporting import write_run_report
    report["reports"] = write_run_report(output, report, instances or [], solutions)
    write_json(Path(output) / "build_report.json", report)
    return report


def manifest():
    path = ROOT / "data/processed/manifest.json"
    if not path.exists():
        raise ValueError("尚未准备数据，请先运行 python main.py prepare")
    return read_json(path)


def select_model(args):
    """Read a real JEV result, or an explicitly supplied model for debugging."""
    if getattr(args, "jev_result", None):
        result = read_json(args.jev_result)
        if not isinstance(result, dict):
            raise ValueError("JEV 结果必须是包含 model_id 的 JSON 对象")
        if result.get("status", "selected") != "selected":
            raise ValueError("JEV 尚未确认模型，不能使用待审核或无匹配的结果")
        model_id = result.get("model_id")
        source = "external_jev_result"
    else:
        model_id = args.model
        source = "manual_model_id"
    if model_id not in MODEL_IDS:
        raise ValueError("model_id 必须为 M1、M2、M3、M4；不支持的需求应先澄清")
    return model_id, source


def selection_for_description(args, description):
    if getattr(args, "jev_result", None):
        result = read_json(args.jev_result)
        if not isinstance(result, dict):
            raise ValueError("JEV 结果必须为 JSON 对象")
        result = {**result, "provider": "external_jev_result"}
        result.setdefault("status", "selected")
        result.setdefault("model_version", "1.0")
    elif getattr(args, "model", None):
        result = {"status": "selected", "model_id": args.model,
                  "model_version": "1.0", "provider": "manual_model_id"}
    else:
        from orllm.jev import select_with_jev
        result = select_with_jev(ROOT, description, read_json(args.jev_config))
    if not isinstance(result, dict) or not isinstance(result.get("status"), str) or result["status"] not in {"selected", "needs_review", "no_match"}:
        raise ValueError("选型状态必须为 selected、needs_review 或 no_match")
    if result.get("model_version") != "1.0":
        raise ValueError("选型结果的模型版本与当前接口不一致")
    if result["status"] == "selected":
        if result.get("model_id") not in MODEL_IDS:
            raise ValueError("选型结果必须指向 M1、M2、M3 或 M4")
        if "confidence" in result:
            confidence = result["confidence"]
            threshold = result.get("threshold", 0.7)
            for name, value in (("confidence", confidence), ("threshold", threshold)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"选型 {name} 必须为 0 到 1 的有限数值")
            if confidence < threshold:
                result.update(status="needs_review", proposed_model_id=result["model_id"], model_id=None)
    return result


def description_from_args(args):
    description = Path(args.description_file).read_text(encoding="utf-8-sig")
    if not description.strip():
        raise ValueError("业务描述不能为空")
    return description


def retrieval_options(args, default_backend="keyword"):
    config_path = getattr(args, "rag_config", None)
    if not config_path and args.command in {"run", "index-rag"}:
        local = ROOT / "config/rag.local.json"
        config_path = str(local) if local.exists() else None
    config = read_json(config_path) if config_path else {}
    if not isinstance(config, dict) or set(config) - {"backend", "embedding_model", "index_path", "top_k"}:
        raise ValueError("RAG 配置只能包含 backend、embedding_model、index_path、top_k")
    options = {"backend": default_backend, "top_k": 5, "embedding_model": None, "index_path": None, **config}
    for key in options:
        override = getattr(args, key, None)
        if override is not None:
            options[key] = override
    if not isinstance(options["backend"], str) or options["backend"] not in {"keyword", "chroma", "model_pack"}:
        raise ValueError("RAG backend 必须为 model_pack、keyword 或 chroma")
    if type(options["top_k"]) is not int or options["top_k"] <= 0:
        raise ValueError("RAG top_k 必须为正整数")
    for key in ("embedding_model", "index_path"):
        value = options[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"RAG {key} 必须为空或非空字符串")
    embedding = options["embedding_model"]
    if embedding and not Path(embedding).is_absolute() and (ROOT / embedding).exists():
        options["embedding_model"] = str((ROOT / embedding).resolve())
    return options


def mapping_for_selection(args, description, selection, context):
    from orllm.llm import extract_mapping, validate_mapping
    ids = [item["instance_id"] for item in manifest()["instances"]]
    model_id = selection["model_id"]
    trace = getattr(args, "llm_trace", None)
    options = {"trace": trace} if trace is not None else {}
    config=read_json(args.config)
    if config.get('workflow_contract') is True:
        from orllm.workflow import map_with_contract
        out=Path(args.output)
        if out.suffix=='.json':out=out.parent/(out.stem+'_audit')
        mapping=map_with_contract(ROOT,description,selection,context,config,ids,out,trace=trace,
                                  input_bundle=getattr(args,'mapping_input',None))
    else:
        mapping = extract_mapping(ROOT, description, model_id, context, config, ids, **options)
    validate_mapping(ROOT, mapping, model_id, ids)
    scenario_from_mapping(mapping)
    return mapping


def stopped_report(status, missing_fields=None, unsupported_requirements=None):
    return {"status": status, "solver": "none", "missing_fields": missing_fields or [],
            "unsupported_requirements": unsupported_requirements or [], "artifacts": []}


def scenario_from_mapping(mapping):
    """Turn a schema-validated LLM mapping into a deterministic binding request."""
    from jsonschema import Draft202012Validator
    schema = read_json(ROOT / "schemas/mapping.schema.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(mapping), key=lambda e: str(list(e.path)))
    if errors:
        raise ValueError("映射 JSON 不符合 Schema: " + "; ".join(e.message for e in errors[:8]))
    if mapping["status"] != "ready" or mapping["missing_fields"] or mapping["unsupported_requirements"]:
        return None
    assignment = mapping["assignment"]
    if assignment is not None:
        ids = [item["customer_id"] for item in assignment]
        if len(ids) != len(set(ids)):
            raise ValueError("客户归属包含重复客户，不能覆盖后继续建模")
        assignment = {item["customer_id"]: item["depot_id"] for item in assignment}
    keys = ("scenario_id", "instance_id", "model_id", "model_version", "opened_depots",
            "distance_scale_km", "distance_scale_source")
    scenario = {key: mapping[key] for key in keys}
    scenario["assignment"] = assignment
    scenario["description"] = "由 LLM 映射产生；参数仍由固定数据接口读取。"
    return scenario


def reference_route_start(instance, model, routes):
    """Reconstruct an EV start from upstream routes; original rows decide validity."""
    if model.model_id != "M4" or not routes:
        return None
    depot = instance["depot_id"]
    selected = [r for r in routes if r["depot_id"] == depot]
    values = {(v["family"], tuple(v["indices"])): 0.0 for v in model.variables}
    visited = []
    try:
        for route in selected:
            path = route["nodes"]
            if len(path) < 3 or path[0] != depot or path[-1] != depot:
                return None
            visited.extend(path[1:-1])
            load = sum(instance["q"][c] for c in path[1:-1])
            energy = 0.0
            for a, b in zip(path, path[1:]):
                values["x", (a, b)] = 1.0
                values["f", (a, b)] = load
                energy += instance["ell"][a][b] * (
                    instance["r0"] + (instance["r1"] - instance["r0"]) * load / instance["Q"])
                if b != depot:
                    values["e", (b,)] = energy
                    load -= instance["q"][b]
        if len(visited) != len(set(visited)) or set(visited) != set(instance["C"]):
            return None
        vector = [values[v["family"], tuple(v["indices"])] for v in model.variables]
    except (KeyError, TypeError, ValueError):
        return None
    if not model.check_primal(vector)["valid"]:
        return None
    return {"values": vector, "objective": sum(c * vector[i] for i, c in model.objective.items()),
            "source": "independently_checked_M3_routes"}


def packing_route_start(instance, model):
    """Try a bounded capacity packing, then validate the complete EV point.

    This is only an incumbent heuristic: failure proves nothing about feasibility.
    Customer-to-depot decisions and the exported model are unchanged.
    """
    if model.model_id != "M4" or len(instance["C"]) > 45:
        return None
    import time
    clients = sorted(instance["C"], key=lambda c: (-instance["q"][c], c))
    capacity = instance["Q"]
    count = math.ceil(math.fsum(instance["q"].values()) / capacity - 1e-9)
    bins, loads, failed = [[] for _ in range(count)], [0.0] * count, set()
    deadline = time.perf_counter() + 1.0

    def place(index):
        if time.perf_counter() > deadline or len(failed) >= 100000:
            return False
        if index == len(clients):
            return True
        state = (index, tuple(sorted(loads)))
        if state in failed:
            return False
        customer, seen = clients[index], set()
        for vehicle in sorted(range(count), key=lambda j: loads[j], reverse=True):
            before = loads[vehicle]
            if before in seen or before + instance["q"][customer] > capacity + 1e-9:
                continue
            seen.add(before)
            bins[vehicle].append(customer)
            loads[vehicle] += instance["q"][customer]
            if place(index + 1):
                return True
            bins[vehicle].pop()
            loads[vehicle] = before
            if before == 0:
                break
        failed.add(state)
        return False

    if not place(0):
        return None
    routes = []
    for customers in bins:
        remaining, path = set(customers), [instance["depot_id"]]
        while remaining:
            next_customer = min(remaining, key=lambda c: (instance["ell"][path[-1]][c], c))
            path.append(next_customer)
            remaining.remove(next_customer)
        path.append(instance["depot_id"])
        routes.append({"depot_id": instance["depot_id"], "nodes": path})
    start = reference_route_start(instance, model, routes)
    if start is not None:
        start["source"] = "capacity_packing_of_fixed_assignment"
    return start


def best_checked_incumbent(model, native, previous=None, initial=None):
    """Keep the best feasible point and compatible bounds for this exact model.

    A solver may stop at its gap tolerance with a worse point than an external
    checked start. Keep native statistics separate from the selected point.
    """
    candidates = []
    validated = []
    for source, raw in (("native_solver", native), ("previous_solution", previous)):
        if not raw or raw.get("status") not in {"optimal", "feasible_limit"}:
            continue
        try:
            vector = [raw["variable_values"][v["name"]] for v in model.variables]
            objective = sum(c * vector[i] for i, c in model.objective.items())
            if (model.check_primal(vector)["valid"] and raw.get("objective_check", {}).get("valid")
                    and math.isclose(objective, raw["objective"], rel_tol=1e-8, abs_tol=1e-6)):
                candidates.append((objective, source, vector, raw))
                validated.append(raw)
        except (KeyError, TypeError, ValueError):
            continue
    if initial is not None and model.check_primal(initial["values"])["valid"]:
        objective = sum(c * initial["values"][i] for i, c in model.objective.items())
        candidates.append((objective, initial["source"], initial["values"], None))
    if not candidates:
        return native
    objective, source, vector, origin = min(candidates, key=lambda x: x[0])
    if source == "native_solver":
        result = dict(native)
    else:
        statistics = {k: v for k, v in native.items() if k not in {
            "status", "message", "objective", "variable_values", "decision_values",
            "primal_check", "objective_check", "mip_gap", "mip_dual_bound"}}
        result = model._solution(native.get("status") if native.get("status") in {
            "optimal", "feasible_limit"} else "feasible_limit",
            "Pipeline retained the best independently checked point; native statistics are recorded separately.",
            vector, **statistics)
        result["objective_check"] = dict(valid=True, recomputed=result["objective"],
            solver_reported=None, method="fixed_model_expression_for_retained_incumbent")
        if origin is not None:
            result["original_incumbent_runtime_seconds"] = origin.get("runtime_seconds")
    bounds = [raw.get("mip_dual_bound") for raw in validated]
    bounds = [b for b in bounds if isinstance(b, (int, float)) and math.isfinite(b)
              and b <= objective + 1e-6 + 1e-8 * abs(objective)]
    if bounds:
        bound = min(objective, max(bounds))
        result.update(mip_dual_bound=bound,
                      mip_gap=max(0.0, objective - bound) / max(1e-12, abs(objective)))
    else:
        result.update(mip_dual_bound=None, mip_gap=None)
    if source != "native_solver" and (not bounds or not any(raw is native for raw in validated)):
        result["status"] = "feasible_limit"
    result.update(incumbent_source=source, native_objective=native.get("objective"),
                  native_mip_gap=native.get("mip_gap"), native_status=native.get("status"))
    return result


def reference_assignment_start(model, assignments):
    """Retain the upstream M1 allocation as a fully checked M2 candidate."""
    if model.model_id != "M2" or not assignments:
        return None
    try:
        assignment = {row["customer_id"]: row["depot_id"] for row in assignments}
        if (len(assignment) != len(assignments) or set(assignment) != set(model.metadata["C"])
                or not set(assignment.values()).issubset(model.metadata["D"])):
            return None
        values = [float(assignment[v["indices"][1]] == v["indices"][0]) for v in model.variables]
    except (KeyError, TypeError, IndexError):
        return None
    if not model.check_primal(values)["valid"]:
        return None
    return {"values": values, "objective": sum(c * values[i] for i, c in model.objective.items()),
            "source": "independently_checked_M1_assignment"}


def run_build(scenario, output_dir, solver="none", time_limit=300, mip_gap=0.08,
              reuse_verified=False, reference_routes=None, reference_assignments=None,
              progress_callback=None):
    from orllm.data import load_base
    from orllm.binding import bind_scenario
    from orllm.validation import validate_instance
    from models import build_model

    if solver not in {'none', 'copt'}:
        raise ValueError('本系统只支持 COPT 求解；none 仅导出模型。')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not isinstance(scenario, dict):
        raise ValueError("场景配置必须为 JSON 对象")
    if scenario.get('model_fingerprint'):
        from orllm.workflow import model_version
        if scenario['model_fingerprint']!=model_version(ROOT,scenario['model_id']):
            raise ValueError('模型版本已变化，必须重新确认选型与绑定。')
    if not scenario.get("instance_id"):
        result = {"status": "needs_clarification", "missing_fields": ["instance_id"],
                  "errors": [], "warnings": [], "instances": [], "provenance": {}}
    else:
        base = load_base(ROOT, scenario["instance_id"])
        result = bind_scenario(base, scenario)
    report = {k: v for k, v in result.items() if k != "instances"}
    report.update({"scenario_id": scenario.get("scenario_id"), "solver": solver, "artifacts": [],
                   "model_id": scenario.get("model_id"), "instance_id": scenario.get("instance_id"),
                   "time_limit_seconds": time_limit, "requested_mip_gap": mip_gap})
    if result["status"] != "ready":
        return save_run_report(out, report)
    # Validate every instance before writing any LP, avoiding partial builds.
    errors = []
    for instance in result["instances"]:
        errors.extend(validate_instance(instance))
    if errors:
        report["status"] = "invalid"
        report["errors"] = errors
        return save_run_report(out, report)
    solutions = []
    for instance_index, instance in enumerate(result["instances"]):
        suffix = instance.get("depot_id", "all")
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{instance['model_id']}_{suffix}")
        instance_path = out / f"{stem}.instance.json"
        lp_path = out / f"{stem}.lp"
        model = build_model(instance)
        solution_path = out / f"{stem}.solution.json"
        previous = None
        model_path = out / f"{stem}.model.json"
        if reuse_verified and model_path.is_file():
            if read_json(model_path) != model.to_dict():
                raise ValueError("已有数学模型与本次任务不一致，不能复用旧解")
            if solution_path.is_file():
                candidate = read_json(solution_path)
                try:
                    values = [candidate["variable_values"][v["name"]] for v in model.variables]
                    valid = (candidate.get("solver") == "copt"
                             and candidate.get("status") in {"optimal", "feasible_limit"}
                             and candidate.get("objective_check", {}).get("valid") is True
                             and model.check_primal(values)["valid"]
                             and math.isclose(sum(c * values[i] for i, c in model.objective.items()),
                                              candidate["objective"], rel_tol=1e-8, abs_tol=1e-6))
                    if valid:
                        previous = candidate
                        model.mip_start = values
                except (KeyError, TypeError, ValueError):
                    pass
        write_json(instance_path, instance)
        model.write_lp(lp_path)
        model.write_json(model_path)
        artifact = {"instance": instance_path.name, "lp": lp_path.name,
                    "model_with_variable_mapping": model_path.name,
                    "model_id": instance["model_id"], "depot_id": instance.get("depot_id"),
                    "variable_count": len(model.variables), "constraint_count": len(model.constraints)}
        reference = reference_route_start(instance, model, reference_routes) if solver == "copt" else None
        assignment_start = reference_assignment_start(model, reference_assignments) if solver != "none" else None
        if assignment_start is not None:
            reference = assignment_start
        packed = packing_route_start(instance, model) if solver == "copt" else None
        if packed is not None and (reference is None or packed["objective"] < reference["objective"]):
            reference = packed
        reference_better = (reference is not None and
                            (previous is None or reference["objective"] < previous["objective"] - 1e-5))
        if reference_better:
            model.mip_start = reference["values"]
            artifact["mip_start_source"] = reference["source"]
            artifact["mip_start_objective"] = reference["objective"]
        solution = None
        if solver != "none":
            if progress_callback is not None:
                progress_callback("solve", event="subproblem_started", depot_id=instance.get("depot_id"),
                                  subproblem_index=instance_index + 1,
                                  subproblem_count=len(result["instances"]))
            log_path = out / f"{stem}.solver.log"
            attempt_path = out / f"{stem}.continuation_attempt.json"
            if reuse_verified and previous is not None and attempt_path.is_file():
                archived_native = read_json(attempt_path)
                combined = best_checked_incumbent(model, archived_native, previous, reference)
                if combined.get("objective") is not None:
                    combined["continuation_attempt"] = attempt_path.name
                    combined["bound_source"] = attempt_path.name
                    previous = combined
                    model.mip_start = [previous["variable_values"][v["name"]] for v in model.variables]
                    reference_better = (reference is not None and
                                        reference["objective"] < previous["objective"] - 1e-5)
                    artifact["continuation_proof_reused"] = attempt_path.name
            gap = previous.get("mip_gap") if previous else None
            converged = (previous is not None and previous["status"] in {"optimal", "feasible_limit"}
                         and gap is not None and gap <= (mip_gap if mip_gap is not None else 1e-8) + 1e-10
                         and not reference_better)
            if reuse_verified and converged:
                solution = previous
                artifact["solution_reused"] = True
                artifact["meets_requested_gap"] = True
                print(f"REUSE {stem} objective={solution['objective']:,.3f} gap={gap:.3%}", flush=True)
            else:
                if previous is not None:
                    snapshot = out / "previous_solves" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                    snapshot.mkdir(parents=True)
                    shutil.copy2(solution_path, snapshot / solution_path.name)
                    if log_path.exists():
                        shutil.copy2(log_path, snapshot / log_path.name)
                    artifact["previous_solution"] = str((snapshot / solution_path.name).relative_to(out))
                if reuse_verified:
                    initial_objective = reference["objective"] if reference_better else previous.get("objective") if previous else None
                    print(f"SOLVE {stem} time_limit={time_limit:g}s initial_objective={initial_objective} initial_source={reference['source'] if reference_better else 'previous_solution'}", flush=True)
                native_solution = model.solve(solver=solver, time_limit=time_limit, mip_gap=mip_gap, log_path=log_path)
                solution = best_checked_incumbent(model, native_solution, previous, reference)
                if reference_better:
                    solution["mip_start_source"] = artifact["mip_start_source"]
                    solution["mip_start_objective"] = reference["objective"]
                if solution.get("incumbent_source") != "native_solver" and solution.get("objective") is not None:
                    write_json(attempt_path, native_solution)
                    solution["continuation_attempt"] = attempt_path.name
                    report.setdefault("warnings", []).append(
                        f"{stem}：采用成本更低的已校验方案，结合同一模型的有效下界计算 gap；原始求解器结果见 {attempt_path.name}。")
                if reuse_verified:
                    print(f"SUBPROBLEM DONE {stem} status={solution['status']} objective={solution.get('objective')} gap={solution.get('mip_gap')}", flush=True)
            write_json(solution_path, solution)
            artifact.update({"solution": solution_path.name, "solve_status": solution.get("status"),
                             "objective": solution.get("objective"), "solver": solution.get("solver"),
                             "mip_gap": solution.get("mip_gap"),
                             "mip_dual_bound": solution.get("mip_dual_bound"),
                             "runtime_seconds": solution.get("runtime_seconds", 0),
                             "primal_check": solution.get("primal_check")})
            if log_path.exists():
                artifact["solver_log"] = log_path.name
        solutions.append(solution)
        report["artifacts"].append(artifact)
        if solver != "none" and progress_callback is not None:
            objective = solution.get("objective")
            verified = (solution.get("status") in {"optimal", "feasible_limit"}
                        and (solution.get("primal_check") or {}).get("valid") is True
                        and (solution.get("objective_check") or {}).get("valid") is True
                        and isinstance(objective, (int, float)) and not isinstance(objective, bool)
                        and math.isfinite(objective))
            details = {"event": "subproblem_finished", "depot_id": instance.get("depot_id"),
                       "subproblem_index": instance_index + 1, "subproblem_count": len(result["instances"]),
                       "solve_status": solution.get("status"), "runtime_seconds": solution.get("runtime_seconds"),
                       "mip_gap": solution.get("mip_gap") if verified else None, "verified": verified}
            if verified:
                details["objective"] = objective
            progress_callback("solve", **details)
    report["build_status"] = "built"
    report["status"] = "built"
    if solver != "none":
        feasible = [s.get("status") in {"optimal", "feasible_limit"} and s.get("primal_check", {}).get("valid") is True
                    for s in solutions]
        if solutions and all(feasible):
            report["status"] = "solved"
            report["solve_status"] = "optimal" if all(s["status"] == "optimal" for s in solutions) else "feasible_limit"
        else:
            report["status"] = "solve_failed"
            statuses = {s["status"] for s in solutions}
            report["solve_status"] = "partial" if any(feasible) else next(iter(statuses)) if len(statuses) == 1 else "no_solution"
    if progress_callback is not None:
        progress_callback("report")
    return save_run_report(out, report, result["instances"], solutions)


def cmd_prepare(args):
    from orllm.data import prepare_data
    candidates = list((ROOT / "data/raw").glob("*.xlsx"))
    workbook = Path(args.workbook) if args.workbook else candidates[0] if len(candidates) == 1 else None
    if workbook is None:
        raise ValueError("请用 --workbook 指定 Excel")
    result = prepare_data(workbook, ROOT / "data/processed")
    print(json.dumps({"status": "prepared", "instance_count": len(result["instances"]),
                      "manifest": "data/processed/manifest.json"}, ensure_ascii=False))
    return 0


def cmd_list(args):
    for item in manifest()["instances"]:
        print(item["instance_id"])
    return 0


def cmd_validate(args):
    from orllm.validation import validate_instance
    errors = validate_instance(read_json(args.input))
    print(json.dumps({"status": "valid" if not errors else "invalid", "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def cmd_build(args):
    if args.mapping:
        mapping = read_json(args.mapping)
        scenario = scenario_from_mapping(mapping)
        if scenario is None:
            report = {"status": mapping["status"], "missing_fields": mapping["missing_fields"],
                      "unsupported_requirements": mapping["unsupported_requirements"], "artifacts": []}
            save_run_report(args.output, report)
            print_report(report, args.output)
            return 2
    else:
        scenario = read_json(args.scenario)
    report = run_build(scenario, args.output, args.solve, args.time_limit, args.mip_gap)
    print_report(report, args.output)
    return report_exit_code(report, args.solve)


def cmd_solve(args):
    """Continue from a saved, validated mapping without repeating any API call."""
    if args.run_dir:
        source = Path(args.run_dir).resolve()
    else:
        candidates = list((ROOT / "outputs/runs").glob("*/mapping.json"))
        candidates += list((ROOT / "outputs").glob("*/mapping.json"))
        if not candidates:
            candidates = list((ROOT / "outputs/experiments").glob("*/*/M[1-4]/mapping.json"))
        if not candidates:
            raise ValueError("没有已保存的映射，请使用 build --scenario，或先执行 run")
        source = max(candidates, key=lambda p: p.stat().st_mtime_ns).parent.resolve()
    mapping_path = source / "mapping.json"
    if not mapping_path.is_file():
        raise ValueError(f"结果目录缺少 mapping.json：{source}")
    out = Path(args.output).resolve()
    if out == source:
        raise ValueError("请使用新的 --output 目录，以保留原始选型和映射结果")
    mapping = read_json(mapping_path)
    scenario = scenario_from_mapping(mapping)
    if scenario is not None and (source/'binding_state.json').is_file():
        scenario=read_json(source/'binding_state.json')['scenario']
        out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'binding_state.json',out/'binding_state.json')
    out.mkdir(parents=True, exist_ok=True)
    if scenario is None:
        report = stopped_report(mapping["status"], mapping["missing_fields"], mapping["unsupported_requirements"])
        save_run_report(out, report)
        print_report(report, out)
        return 2
    # These are local trace artifacts, never credentials or configuration files.
    for filename in ("selection.json", "retrieval.json", "mapping.json"):
        if (source / filename).is_file():
            shutil.copy2(source / filename, out / filename)
    report = run_build(scenario, out, args.solve, args.time_limit, args.mip_gap)
    report["resumed_from"] = str(source)
    write_json(out / "build_report.json", report)
    write_json(out / "pipeline_report.json", {"status": report["status"], "stage": "solve",
               "solver": args.solve, "solve_status": report.get("solve_status"),
               "resumed_from": str(source), "api_calls": 0,
               "artifacts": ["mapping.json", "build_report.json", *report.get("reports", {}).get("business_csv", [])]})
    print_report(report, out)
    return report_exit_code(report, args.solve)


def cmd_retrieve(args):
    from orllm.rag import retrieve
    results = retrieve(ROOT, args.query, args.model, version="1.0", **retrieval_options(args))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def cmd_index(args):
    from orllm.rag import build_index
    options = retrieval_options(args, default_backend="chroma")
    if not options["embedding_model"]:
        raise ValueError("请通过 --embedding-model 或 RAG 配置指定嵌入模型")
    result = build_index(ROOT, options["embedding_model"], index_path=options["index_path"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_map(args):
    from orllm.rag import retrieve_for_mapping as retrieve
    description = description_from_args(args)
    selection = selection_for_description(args, description)
    write_json(str(args.output) + ".selection.json", selection)
    if selection["status"] != "selected":
        print(json.dumps({"status": selection["status"], "selection": str(args.output) + ".selection.json"}, ensure_ascii=False))
        return 2
    options = retrieval_options(args)
    context = retrieve(ROOT, description, selection["model_id"], version="1.0", **options)
    mapping = mapping_for_selection(args, description, selection, context)
    write_json(args.output, mapping)
    write_json(str(args.output) + ".trace.json", {"selection_source": selection["provider"],
        "model_id": selection["model_id"], "retrieval_backend": options["backend"],
        "context": context, "description_file": str(args.description_file)})
    print(json.dumps({"status": mapping["status"], "output": str(args.output),
                      "missing_fields": mapping["missing_fields"],
                      "unsupported_requirements": mapping["unsupported_requirements"]}, ensure_ascii=False, indent=2))
    return 0 if mapping["status"] == "ready" else 2


def cmd_select(args):
    selection = selection_for_description(args, description_from_args(args))
    write_json(args.output, selection)
    print(json.dumps({"status": selection["status"], "model_id": selection.get("model_id"),
                      "confidence": selection.get("confidence"), "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if selection["status"] == "selected" else 2


def cmd_run(args):
    """JEV -> retrieval -> LLM -> validation/build -> optional local solve."""
    from orllm.rag import retrieve_for_mapping as retrieve
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pipeline = {"status": "running", "stage": "selection", "solver": args.solve, "artifacts": []}
    try:
        description = description_from_args(args)
        selection = selection_for_description(args, description)
        write_json(out / "selection.json", selection)
        pipeline["artifacts"].append("selection.json")
        pipeline["selection_source"] = selection["provider"]
        pipeline["model_id"] = selection.get("model_id")
        if selection["status"] != "selected":
            report = stopped_report(selection["status"], ["model_selection"])
        else:
            options = retrieval_options(args, default_backend="model_pack")
            pipeline.update(stage="retrieval", retrieval_backend=options["backend"])
            context = retrieve(ROOT, description, selection["model_id"], version="1.0", **options)
            write_json(out / "retrieval.json", {"model_id": selection["model_id"], **options, "context": context})
            pipeline["artifacts"].append("retrieval.json")
            pipeline["stage"] = "mapping"
            mapping = mapping_for_selection(args, description, selection, context)
            write_json(out / "mapping.json", mapping)
            pipeline["artifacts"].append("mapping.json")
            scenario = scenario_from_mapping(mapping)
            if scenario is not None and (out/'binding_state.json').is_file():
                scenario=read_json(out/'binding_state.json')['scenario']
            if scenario is None:
                report = stopped_report(mapping["status"], mapping["missing_fields"], mapping["unsupported_requirements"])
            else:
                pipeline["stage"] = "build"
                report = run_build(scenario, out, solver=args.solve, time_limit=args.time_limit, mip_gap=args.mip_gap)
                if args.solve != "none" and report.get("build_status") == "built":
                    pipeline["stage"] = "solve"
        if "reports" not in report:
            save_run_report(out, report)
        pipeline["artifacts"].append("build_report.json")
        pipeline["status"] = report["status"]
        pipeline["solve_status"] = report.get("solve_status")
        pipeline["reports"] = report.get("reports", {})
        write_json(out / "pipeline_report.json", pipeline)
        print_report(report, out)
        return report_exit_code(report, args.solve)
    except (ImportError, ValueError, KeyError, OSError, RuntimeError) as exc:
        pipeline.update(status="error", error=str(exc))
        report = stopped_report("error")
        report["errors"] = [str(exc)]
        save_run_report(out, report)
        if "build_report.json" not in pipeline["artifacts"]:
            pipeline["artifacts"].append("build_report.json")
        write_json(out / "pipeline_report.json", pipeline)
        raise


def add_solver_args(parser, default="none"):
    parser.add_argument("--solve", choices=("none", "copt"), default=default,
                        help="copt 使用 COPT 求解；none 只建模，不启动求解器")
    parser.add_argument("--time-limit", type=float, default=300, help="每个子模型的求解时限（秒），默认 300")
    parser.add_argument("--mip-gap", type=float, default=0.08, help="相对最优性间隙，默认 0.08（8%%）")


def add_retrieval_args(parser):
    parser.add_argument("--rag-config", help="RAG 配置 JSON，命令行选项可覆盖其中字段")
    parser.add_argument("--backend", choices=("model_pack", "keyword", "chroma"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--embedding-model", help="向量索引使用的同一嵌入模型名称或本地路径")
    parser.add_argument("--index-path", help="可选 Chroma 索引目录")


def add_selection_args(parser, required=False):
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--model", choices=MODEL_IDS, help="手工指定模型，调试用")
    group.add_argument("--jev-result", help="已确认的 JEV 选型结果 JSON")
    group.add_argument("--selector", choices=("jev",), help="直接调用 JEV 自动选型")
    parser.add_argument("--jev-config", default=default_config("jev"))


def make_parser():
    parser = argparse.ArgumentParser(description="固定模型 ORLLM：选型、检索、映射、校验、求解和业务报告")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="从原始 Excel 重建容量明确的数据实例")
    p.add_argument("--workbook")
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("list", help="列出数据实例编号")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("validate", help="校验一个完整模型参数实例")
    p.add_argument("--input", required=True)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("build", help="从业务配置或 LLM 映射生成参数 JSON 和 LP")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario")
    g.add_argument("--mapping")
    p.add_argument("--output", default=str(ROOT / "outputs/build"))
    add_solver_args(p)
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("solve", help="复用已有 mapping.json，本地建模并求解，不重新调用 API")
    p.add_argument("--run-dir", help="含 mapping.json 的结果目录；默认使用最近保存的映射")
    p.add_argument("--output", default=new_run_output())
    add_solver_args(p, default="copt")
    p.set_defaults(func=cmd_solve)
    p = sub.add_parser("retrieve", help="查看选中模型的映射知识")
    p.add_argument("--model", choices=MODEL_IDS, required=True)
    p.add_argument("--query", required=True)
    add_retrieval_args(p)
    p.set_defaults(func=cmd_retrieve)
    p = sub.add_parser("index-rag", help="构建真实 Chroma 向量索引")
    add_retrieval_args(p)
    p.set_defaults(func=cmd_index)
    p = sub.add_parser("map", help="用已选模型、检索资料和 Schema 调用 LLM")
    add_selection_args(p, required=True)
    p.add_argument("--description-file", required=True)
    p.add_argument("--config", default=default_config("llm"))
    p.add_argument("--output", default=str(ROOT / "outputs/mapping.json"))
    add_retrieval_args(p)
    p.set_defaults(func=cmd_map)
    p = sub.add_parser("select", help="直接调用 JEV 选择 M1–M4；低置信度停止")
    p.add_argument("--description-file", required=True)
    p.add_argument("--jev-config", default=default_config("jev"))
    p.add_argument("--output", default=str(ROOT / "outputs/jev_result.json"))
    p.set_defaults(func=cmd_select)
    p = sub.add_parser("run", help="JEV + RAG + LLM 参数映射、建模和可选本地求解")
    add_selection_args(p)
    p.add_argument("--description-file", required=True)
    p.add_argument("--config", default=default_config("llm"))
    p.add_argument("--output", default=new_run_output())
    add_retrieval_args(p)
    add_solver_args(p)
    p.set_defaults(func=cmd_run)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if not math.isfinite(getattr(args, "time_limit", 1)) or getattr(args, "time_limit", 1) <= 0:
        parser.error("--time-limit 必须为正数")
    gap = getattr(args, "mip_gap", None)
    if gap is not None and (not math.isfinite(gap) or not 0 <= gap < 1):
        parser.error("--mip-gap 必须是 [0, 1) 内的有限数值")
    try:
        load_local_environment()
        return args.func(args)
    except ImportError as exc:
        print(f"缺少依赖：{exc}。请按 README 安装对应 requirements 文件。", file=sys.stderr)
        return 1
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
