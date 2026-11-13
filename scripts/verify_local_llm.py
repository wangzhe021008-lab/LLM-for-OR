"""Live local Qwen check, reusing a saved JEV selection (no cloud API calls)."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as pipeline
import web_worker
from orllm.llm_profiles import load_profile, model_identity
from orllm.web_chat import plan_reply, prepare_description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev-result", type=Path, required=True, help="Previously saved, confirmed selection.json")
    parser.add_argument("--prompt", default="使用 20-5-1a__Q70 数据，选择开放仓库并将所有客户分配至仓库，在仓库容量约束下使开放成本与配送成本之和最小。请求解 M1。")
    args = parser.parse_args()
    selection = pipeline.read_json(args.jev_result)
    if selection.get("status") != "selected":
        raise ValueError("验证需要已确认的 JEV 选型结果")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "outputs/runs" / (stamp + "_qwen_validation")
    output.mkdir(parents=True)
    config = load_profile(ROOT, "qwen_local")
    trace = {}
    plan = plan_reply(ROOT, args.prompt, [], config=config, trace=trace)
    if plan["action"] != "run":
        raise RuntimeError("本地模型没有识别为计算任务：" + plan["reply"])
    (output / "problem.txt").write_text(prepare_description(ROOT, plan) + "\n", encoding="utf-8")
    pipeline.write_json(output / "chat_plan.json", plan)
    pipeline.write_json(output / "llm_chat.json", trace)
    pipeline.write_json(output / "llm_config.json", config)
    request = {"description_file": str(output / "problem.txt"), "output": str(output), "solver": "copt",
               "time_limit": 30, "mip_gap": 0.08, "llm_config": str(output / "llm_config.json"),
               "llm": model_identity(config)}
    pipeline.write_json(output / "web_request.json", request)
    original = pipeline.selection_for_description
    def saved_selection(options, description):
        options.jev_result = str(args.jev_result.resolve())
        return original(options, description)
    # Only JEV's network request is replaced with its saved result. Retrieval,
    # local Qwen mapping, schema checks, model construction and solve are real.
    with patch.object(pipeline, "selection_for_description", side_effect=saved_selection), \
            patch.object(pipeline, "load_local_environment"):
        code = web_worker.run(request)
    pipeline.write_json(output / "web_job.json", {"llm": model_identity(config), "llm_chat": trace,
                        "validation": {"selection_reused_from": str(args.jev_result), "external_api_calls": 0}})
    summary = pipeline.read_json(output / "summary.json")
    result = {"output": str(output), "exit_code": code, "status": summary.get("status"),
              "objective": summary.get("total_objective"), "verified": summary.get("complete_solution"),
              "llm_chat": trace, "llm_mapping": pipeline.read_json(output / "llm_mapping.json")}
    diagnostics = ROOT / "outputs/diagnostics/llm_switch"
    diagnostics.mkdir(parents=True, exist_ok=True)
    pipeline.write_json(diagnostics / "local_pipeline_validation.json", result)
    print(json.dumps({key: result[key] for key in ("output", "exit_code", "status", "objective", "verified")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
