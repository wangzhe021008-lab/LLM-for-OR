"""Isolated worker for the unchanged JEV/RAG/mapping/model pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import main as pipeline


def event(stage, **details):
    print("ORLLM_EVENT " + json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)


def check_reference(mapping, fixed):
    """A mapping must preserve externally fixed upstream decisions exactly."""
    if not fixed or mapping.get("status") != "ready":
        return
    if mapping.get("instance_id") != fixed["instance_id"]:
        raise ValueError("模型映射改变了本次引用的数据实例，已停止求解。")
    if set(mapping.get("opened_depots") or []) != set(fixed["opened_depots"]):
        raise ValueError("模型映射改变了前序确定的开放仓库，已停止求解。")
    reuse = fixed["reuse_decisions"]
    model_id = mapping.get("model_id")
    if reuse == "opened_depots" and model_id != "M2":
        raise ValueError("本次要求固定仓库后优化分配，选型或任务映射不一致，已停止求解。")
    if reuse == "assignment":
        if model_id not in {"M3", "M4"}:
            raise ValueError("本次要求沿用客户归属规划路线，选型或任务映射不一致，已停止求解。")
        expected = {item["customer_id"]: item["depot_id"] for item in fixed["assignment"]}
        actual = {item["customer_id"]: item["depot_id"] for item in mapping.get("assignment") or []}
        if actual != expected:
            raise ValueError("模型映射改变了前序确定的客户归属，已停止求解。")


def run(request):
    if request.get('parameter_update'):
        return run_update(request)
    from orllm.rag import retrieve_for_mapping as retrieve
    output = Path(request["output"])
    args = pipeline.make_parser().parse_args([
        "run", "--description-file", request["description_file"], "--output", str(output),
        "--solve", request["solver"], "--time-limit", str(request["time_limit"]), "--mip-gap", str(request["mip_gap"])])
    if request.get("llm_config"):
        args.config = request["llm_config"]
        args.llm_trace = {}
    args.mapping_input = request.get('mapping_input')
    pipeline.load_local_environment()
    record = {"status": "running", "stage": "selection", "solver": args.solve, "artifacts": [], "source": "web_chat"}
    if request.get("llm"):
        record["llm"] = request["llm"]
    started = time.monotonic()
    subproblem_starts = {}
    def progress(stage, **details):
        record["stage"] = stage
        if stage == "solve" and details.get("event") in {"subproblem_started", "subproblem_finished"}:
            index = details["subproblem_index"]
            rows = record.setdefault("subproblems", [])
            row = next((item for item in rows if item["subproblem_index"] == index), None)
            if row is None:
                row = {"subproblem_index": index, "subproblem_count": details["subproblem_count"],
                       "depot_id": details.get("depot_id")}
                rows.append(row)
            row.update({key: value for key, value in details.items() if key != "event"})
            stamp = datetime.now(timezone.utc).isoformat()
            if details["event"] == "subproblem_started":
                row.update(status="running", started_at=stamp)
                subproblem_starts[index] = time.monotonic()
            else:
                row.update(status="done", finished_at=stamp)
                if index in subproblem_starts:
                    row["elapsed_seconds"] = time.monotonic() - subproblem_starts.pop(index)
            # The event carries the same flat metadata that is persisted locally.
            details = {**row, "event": details["event"]}
        pipeline.write_json(output / "pipeline_report.json", record)
        event(stage, **details)
    try:
        description = pipeline.description_from_args(args)
        progress("selection")
        selection = pipeline.selection_for_description(args, description)
        pipeline.write_json(output / "selection.json", selection)
        record.update(selection_source=selection["provider"], model_id=selection.get("model_id"))
        record["artifacts"].append("selection.json")
        if selection["status"] != "selected":
            report = pipeline.stopped_report(selection["status"], ["model_selection"])
        else:
            record["stage"] = "retrieval"
            progress("retrieval")
            options = pipeline.retrieval_options(args, default_backend="model_pack")
            context = retrieve(pipeline.ROOT, description, selection["model_id"], version="1.0", **options)
            pipeline.write_json(output / "retrieval.json", {"model_id": selection["model_id"], **options, "context": context})
            record.update(retrieval_backend=options["backend"])
            record["artifacts"].append("retrieval.json")
            record["stage"] = "mapping"
            progress("mapping")
            try:
                mapping = pipeline.mapping_for_selection(args, description, selection, context)
            except Exception as exc:
                if hasattr(args, "llm_trace"):
                    args.llm_trace.update(status="error", error=str(exc))
                raise
            finally:
                if hasattr(args, "llm_trace"):
                    trace = {key: value for key, value in args.llm_trace.items() if not key.startswith("_")}
                    pipeline.write_json(output / "llm_mapping.json", trace)
                    record["llm_mapping"] = trace
                    record["artifacts"].append("llm_mapping.json")
            check_reference(mapping, request.get("reference_constraints"))
            pipeline.write_json(output / "mapping.json", mapping)
            record["artifacts"].append("mapping.json")
            scenario = pipeline.scenario_from_mapping(mapping)
            if scenario is not None and (output/'binding_state.json').is_file():
                scenario=pipeline.read_json(output/'binding_state.json')['scenario']
            if scenario is None:
                report = pipeline.stopped_report(mapping["status"], mapping["missing_fields"], mapping["unsupported_requirements"])
            else:
                record["stage"] = "build"
                progress("build")
                report = pipeline.run_build(scenario, output, solver=args.solve, time_limit=args.time_limit,
                                            mip_gap=args.mip_gap, progress_callback=progress)
        progress("report")
        if "reports" not in report:
            pipeline.save_run_report(output, report)
        record.update(status=report["status"], stage="done", solve_status=report.get("solve_status"),
                      reports=report.get("reports", {}), elapsed_seconds=time.monotonic() - started)
        record["artifacts"].append("build_report.json")
        pipeline.write_json(output / "pipeline_report.json", record)
        return pipeline.report_exit_code(report, args.solve)
    except Exception as exc:
        message = str(exc) if isinstance(exc, (ImportError, ValueError, KeyError, OSError, RuntimeError)) else "本地任务执行异常。"
        for row in record.get("subproblems", []):
            if row.get("status") == "running":
                row.update(status="error", solve_status="error", error=message, verified=False,
                           finished_at=datetime.now(timezone.utc).isoformat())
                index = row["subproblem_index"]
                if index in subproblem_starts:
                    row["elapsed_seconds"] = time.monotonic() - subproblem_starts.pop(index)
        record.update(status="error", error=message, elapsed_seconds=time.monotonic() - started)
        report = pipeline.stopped_report("error")
        report["errors"] = [message]
        pipeline.save_run_report(output, report)
        pipeline.write_json(output / "pipeline_report.json", record)
        return 1


def run_update(request):
    """Reuse the fixed model and validated binding; no selection/retrieval/mapping calls."""
    from orllm.workflow import apply_changes
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=True)
    update=request['parameter_update'];started=time.monotonic()
    record={'status':'running','stage':'mapping','source':'parameter_update','solver':request['solver'],
            'model_calls':0,'selection_reused':True,'artifacts':['incremental_update.json']}
    try:
        event('mapping',message='验证局部修改和数据版本')
        state=apply_changes(pipeline.ROOT,update['binding_state'],update['changes'],update['expected_revision'],update['request_id'],update['description'])
        pipeline.write_json(output/'incremental_update.json',{'request_id':update['request_id'],'expected_revision':update['expected_revision'],
            'changes':update['changes'],'source_text':update['description'],'source':'user_description',
            'new_revision':state['revision'],'model_fingerprint':state['model_fingerprint']})
        pipeline.write_json(output/'binding_state.json',state)
        pipeline.write_json(output/'mapping.json',state['mapping'])
        pipeline.write_json(output/'selection.json',{'status':'selected','model_id':state['scenario']['model_id'],'model_version':'1.0','provider':'reused_fixed_binding'})
        report=pipeline.run_build(state['scenario'],output,solver=request['solver'],time_limit=request['time_limit'],mip_gap=request['mip_gap'],progress_callback=event)
        record.update(status=report['status'],stage='done',model_id=state['scenario']['model_id'],reports=report.get('reports',{}),elapsed_seconds=time.monotonic()-started)
        pipeline.write_json(output/'pipeline_report.json',record)
        return pipeline.report_exit_code(report,request['solver'])
    except (ValueError,KeyError,RuntimeError,OSError) as exc:
        record.update(status='error',error=str(exc),elapsed_seconds=time.monotonic()-started)
        report=pipeline.stopped_report('error');report['errors']=[str(exc)];pipeline.save_run_report(output,report)
        pipeline.write_json(output/'pipeline_report.json',record)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    argument = parser.parse_args()
    raise SystemExit(run(pipeline.read_json(argument.request)))
