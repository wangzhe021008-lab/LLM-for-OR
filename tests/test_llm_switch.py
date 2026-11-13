"""Switching and context/output failures must never mix models or launch a solve."""
import copy
import shutil
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import web_app
import web_worker
import main as pipeline
from orllm.llm import extract_mapping, instance_availability
from orllm.llm_profiles import load_profile
from orllm.llm_runtime import check_local_context, local_json
from orllm.web_chat import plan_reply


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "20-5-1a__Q70"
ANSWER = {"action": "answer", "reply": "M1 选择仓库并优化客户分配。", "description": None, "reuse_decisions": "none"}
MAPPING = {"scenario_id": "switch", "instance_id": INSTANCE, "model_id": "M1", "model_version": "1.0",
           "opened_depots": None, "assignment": None, "distance_scale_km": None, "distance_scale_source": None,
           "status": "ready", "missing_fields": [], "unsupported_requirements": [], "evidence": []}


class Response:
    def __init__(self, data): self.data = data
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.data).encode("utf-8")


def completion(value, finish="stop"):
    return {"model": "Qwen3.5-9B-Q8_0", "choices": [{"finish_reason": finish,
            "message": {"content": json.dumps(value), "reasoning_content": None}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 40, "total_tokens": 163}}


class SwitchingTests(unittest.TestCase):
    def test_prepared_conventional_instance_does_not_imply_ev_parameters(self):
        with patch("orllm.data.load_base", return_value={"ev": {}}):
            row = instance_availability(ROOT, [INSTANCE])[0]
        self.assertTrue(row["existing_core_parameters_loaded_by_python"])
        self.assertFalse(row["ev_parameters_available"])
        self.assertEqual(set(row["missing_ev_fields"]), {"B", "p", "r0", "r1"})
        with patch("orllm.data.load_base", return_value={"ev": {"B": 100, "p": 1, "r0": 0.1, "r1": 0.2}}):
            self.assertTrue(instance_availability(ROOT, [INSTANCE])[0]["ev_parameters_available"])
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copytree(ROOT / 'knowledge', self.root / 'knowledge')
        (self.root / "config").mkdir()
        for stem, source in (("llm", "deepseek"), ("qwen", "qwen_local")):
            config = load_profile(ROOT, source)
            (self.root / f"config/{stem}.local.json").write_text(json.dumps(config), encoding="utf-8")
        (self.root / "data/processed").mkdir(parents=True)
        (self.root / "data/processed/manifest.json").write_text(json.dumps({"instances": [{"instance_id": INSTANCE}]}), encoding="utf-8")

    def test_submission_freezes_config_and_excludes_private_config_from_public_job(self):
        app = web_app.Workspace(self.root)
        session = app.new_session()
        body = {"session_id": session["id"], "request_id": "frozen-source-test", "prompt": "解释 M1", "llm_source": "qwen_local"}
        with patch("web_app.threading.Thread"):
            result = app.submit(body)
        job = app.jobs[result["job_id"]]
        path = self.root / "config/qwen.local.json"
        modified = json.loads(path.read_text(encoding="utf-8"))
        modified["model"] = "changed-after-submission"
        path.write_text(json.dumps(modified), encoding="utf-8")
        self.assertEqual(job["_llm_config"]["model"], "Qwen3.5-9B-Q8_0")
        self.assertNotIn("_llm_config", app.job_snapshot(job["id"]))
        self.assertNotIn("api_key_env", app.job_snapshot(job["id"])["llm"])
        self.assertEqual(app.submit(body), result)
        with self.assertRaises(ValueError):
            app.submit({**body, "llm_source": "deepseek"})
        with self.assertRaises(ValueError):
            app.submit({**body, "llm_source": "../../arbitrary.json"})

    def test_real_execution_passes_the_frozen_local_config_and_persists_chat_identity(self):
        app = web_app.Workspace(self.root)
        session = app.new_session()
        with patch("web_app.threading.Thread"):
            result = app.submit({"session_id": session["id"], "request_id": "same-model-chat", "prompt": "解释 M1", "llm_source": "qwen_local"})
        seen = {}
        def planner(*args, config, trace):
            seen.update(config)
            trace.update(status="validated", finish_reason="stop", response_model=config["model"])
            return ANSWER.copy()
        with patch("orllm.web_chat.plan_reply", side_effect=planner), patch("web_app.subprocess.Popen") as worker:
            app.execute(result["job_id"], "解释 M1", [], None)
        worker.assert_not_called()
        self.assertEqual(seen["provider_id"], "qwen_local")
        saved = json.loads((app.sessions_path / (session["id"] + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(saved["messages"][-1]["llm"]["model"], "Qwen3.5-9B-Q8_0")

    def test_chat_and_mapping_send_the_same_local_options_and_model(self):
        config = load_profile(self.root, "qwen_local")
        config["chat_max_tokens"] = 4096  # Exercise a distinct stage budget independently of the default.
        config["max_tokens"] = 8192  # Keep this fixture independent of the production mapping budget.
        context = {"prompt_tokens": 123, "output_budget": 8192, "effective_context": 16384}
        captured = []
        def capture(request, timeout):
            body = json.loads(request.data)
            captured.append((body, timeout, request))
            return Response(completion(ANSWER if len(captured) == 1 else MAPPING))
        trace_chat, trace_mapping = {}, {}
        with patch("orllm.llm_runtime.check_local_context", return_value=context), \
                patch("orllm.llm_runtime.build_opener") as opener:
            opener.return_value.open.side_effect = capture
            plan_reply(self.root, "解释 M1", [], config=config, trace=trace_chat)
            extract_mapping(ROOT, f"求解 {INSTANCE} 的 M1", "M1", [], config, [INSTANCE], trace=trace_mapping)
        self.assertEqual([item[1] for item in captured], [600, 600])
        for index, (body, _, request) in enumerate(captured):
            self.assertEqual(body["model"], "Qwen3.5-9B-Q8_0")
            expected_thinking = config["chat_template_kwargs"]["enable_thinking"] if index == 0 else config["mapping_enable_thinking"]
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": expected_thinking})
            self.assertEqual(body["n_keep"], -1)
            self.assertEqual(body["response_format"]["type"], "json_schema")
            self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual([item[0]["max_tokens"] for item in captured], [4096, 8192])
        self.assertEqual(trace_chat["status"], trace_mapping["status"])
        self.assertEqual(trace_mapping["usage"]["total_tokens"], 163)

    def test_truncated_or_invalid_local_json_never_is_accepted(self):
        config = load_profile(self.root, "qwen_local")
        for value, finish in ((ANSWER, "length"), ({"action": "run"}, "stop")):
            trace = {}
            with self.subTest(finish=finish), patch("orllm.llm_runtime.check_local_context", return_value={}), \
                    patch("orllm.llm_runtime.build_opener") as opener:
                opener.return_value.open.return_value = Response(completion(value, finish))
                with self.assertRaises(RuntimeError):
                    plan_reply(self.root, "求解", [], config=config, trace=trace)
                self.assertEqual(trace["finish_reason"], finish)

    def test_context_checks_actual_server_size_and_blocks_before_inference(self):
        config = load_profile(self.root, "qwen_local")
        def run(prompt_tokens, actual=16384):
            responses = [{"default_generation_settings": {"n_ctx": actual}},
                         {"data": [{"id": config["model"]}]}, {"prompt": "templated prompt"},
                         {"tokens": [1] * prompt_tokens}]
            with patch("orllm.llm_runtime.local_json", side_effect=responses):
                return check_local_context(config, {"messages": [], "max_tokens": 8192})
        self.assertEqual(run(2000)["remaining_tokens"], 6064)
        for length, actual in ((9000, 16384), (2000, 8192)):
            with self.subTest(length=length, actual=actual), self.assertRaisesRegex(RuntimeError, "上下文不足"):
                run(length, actual)
        with patch("orllm.llm_runtime.local_json", return_value={"default_generation_settings": {"n_ctx": 16384}, "data": []}):
            with self.assertRaisesRegex(RuntimeError, "模型.*不一致"):
                check_local_context(config, {"messages": [], "max_tokens": 8192})
        with self.assertRaises(ValueError):
            local_json("https://example.com/v1", "/props")

    def test_worker_uses_snapshot_instead_of_the_changed_default(self):
        config = load_profile(self.root, "qwen_local")
        frozen = self.root / "frozen.json"
        frozen.write_text(json.dumps(config), encoding="utf-8")
        output = self.root / "worker_output"
        request = {"description_file": "unused.txt", "output": str(output), "solver": "none",
                   "time_limit": 5, "mip_gap": 0, "llm_config": str(frozen),
                   "llm": {"model": config["model"], "provider_id": "qwen_local"}}
        stopped = {"status": "needs_clarification", "model_id": "M1"}
        def mapping(args, *unused):
            self.assertEqual(pipeline.read_json(args.config)["model"], "Qwen3.5-9B-Q8_0")
            args.llm_trace.update(response_model="Qwen3.5-9B-Q8_0", finish_reason="stop", status="validated")
            return stopped
        with patch.object(pipeline, "load_local_environment"), \
                patch.object(pipeline, "description_from_args", return_value="business"), \
                patch.object(pipeline, "selection_for_description", return_value={"provider": "saved_test", "status": "selected", "model_id": "M1"}), \
                patch("orllm.rag.retrieve", return_value=[]), \
                patch.object(pipeline, "retrieval_options", return_value={"backend": "keyword"}), \
                patch.object(pipeline, "mapping_for_selection", side_effect=mapping), \
                patch.object(pipeline, "scenario_from_mapping", return_value=None), \
                patch.object(pipeline, "stopped_report", return_value={"status": "needs_clarification", "artifacts": []}), \
                patch.object(pipeline, "save_run_report"), patch.object(pipeline, "report_exit_code", return_value=2):
            stopped.update(missing_fields=["instance_id"], unsupported_requirements=[])
            self.assertEqual(web_worker.run(request), 2)
        report = pipeline.read_json(output / "pipeline_report.json")
        self.assertEqual(report["llm"]["provider_id"], "qwen_local")
        self.assertEqual(report["llm_mapping"]["response_model"], "Qwen3.5-9B-Q8_0")


if __name__ == "__main__":
    unittest.main()
