"""Local HTTP boundary, task deduplication, and truthful result presentation."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import main
import web_app
import web_worker


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = web_app.Workspace(self.root)
        self.output = self.root / "outputs/runs/sample"
        self.output.mkdir(parents=True)
        self.summary = {"schema_version": "1.0", "status": "partial", "status_label": "部分可行",
                        "generated_at": "2026-10-12T08:00:00Z", "complete_solution": False,
                        "partial_objective": 123, "total_objective": None,
                        "verified_subproblem_count": 1, "business": {"depots": [], "assignments": [], "routes": []},
                        "models": [{"model_id": "M3", "instance_id": "20-5-1a__Q70", "mip_gap": 0.05}]}
        web_app.save_json(self.output / "summary.json", self.summary)
        self.server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.Handler)
        self.server.workspace = self.app
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)

    def request(self, route, body=None, headers=None):
        supplied = {"Content-Type": "application/json", **(headers or {})}
        request = Request(self.base + route, headers=supplied,
                          data=json.dumps(body).encode() if body is not None else None)
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)

    def test_origin_and_static_file_boundaries(self):
        for route in ("/.env", "/config/llm.local.json", "/../main.py", "/api/result?id=../../config"):
            with self.subTest(route=route), self.assertRaises(HTTPError) as context:
                self.request(route)
            self.assertIn(context.exception.code, {400, 403, 404})
        for name in ("../main.py", "..%2F.env", "web_request.json", ".env"):
            with self.subTest(name=name), self.assertRaises(HTTPError):
                self.request("/artifact?id=runs/sample&name=" + name)
        with self.assertRaises(HTTPError) as context:
            self.request("/api/session", {}, {"Origin": "https://evil.example"})
        self.assertEqual(context.exception.code, 403)

    def test_delete_history_persists_and_clears_references(self):
        session = self.app.new_session()
        session.update(current_result_id="runs/sample", reference_result_id="runs/sample")
        session["messages"].append({"role": "assistant", "content": "saved", "result_id": "runs/sample"})
        self.app.save_session(session)
        self.app.history()  # Prime the cache before deletion.
        code, result = self.request("/api/history/delete", {"result_id": "runs/sample"})
        self.assertEqual(code, 200)
        self.assertTrue(result["files_preserved"])
        self.assertTrue((self.output / "summary.json").exists())
        self.assertEqual(self.app.history(), [])
        restarted = web_app.Workspace(self.root)
        self.assertEqual(restarted.history(), [])
        restored = restarted.session(session["id"])
        self.assertIsNone(restored["current_result_id"])
        self.assertIsNone(restored["reference_result_id"])
        self.assertNotIn("result_id", restored["messages"][0])
        with self.assertRaises(ValueError):
            restarted.result_path("runs/sample")
        with self.assertRaises(ValueError):
            restarted.result_path("runs/sample/../sample")

    def test_delete_history_rejects_busy_invalid_and_foreign_origin(self):
        for result_id in ("../config", "runs", "runs/missing", None):
            with self.subTest(result_id=result_id), self.assertRaises(HTTPError) as error:
                self.request("/api/history/delete", {"result_id": result_id})
            self.assertEqual(error.exception.code, 400)
        with self.assertRaises(HTTPError) as error:
            self.request("/api/history/delete", {"result_id": "runs/sample"}, {"Origin": "https://evil.example"})
        self.assertEqual(error.exception.code, 403)
        self.app.active_job_id = "running"
        with self.assertRaises(HTTPError) as error:
            self.request("/api/history/delete", {"result_id": "runs/sample"})
        self.assertEqual(error.exception.code, 409)
        self.assertFalse(self.app.deleted_path.exists())
        with self.assertRaises(HTTPError) as context:
            self.request("/api/health", headers={"Host": "evil.example"})
        self.assertEqual(context.exception.code, 403)

    def test_session_and_duplicate_submission(self):
        _, session = self.request("/api/session", {})
        payload = {"session_id": session["id"], "request_id": "same-request-id", "prompt": "解释 M3", "settings": {}}
        with patch.object(self.app, "execute"):
            _, first = self.request("/api/chat", payload)
            _, second = self.request("/api/chat", payload)
            self.assertEqual(first, second)
            _, restored = self.request("/api/session?id=" + session["id"])
            self.assertEqual(restored["active_job_id"], first["job_id"])
            self.assertEqual(len(restored["messages"]), 1)
            with self.assertRaises(HTTPError) as context:
                self.request("/api/chat", {**payload, "request_id": "another-request-id"})
            self.assertEqual(context.exception.code, 409)
            with self.assertRaises(HTTPError):
                self.request("/api/chat", {**payload, "prompt": "different"})

    def test_partial_solution_and_safe_history(self):
        with patch("orllm.data.load_base", return_value={"coordinates": {"d1": {"x": 1, "y": 2}, "c1": {"x": 3, "y": 4}},
                                                       "D": ["d1"], "q": {"c1": 5}, "U": {"d1": 9}}):
            _, result = self.request("/api/result?id=runs/sample")
        self.assertIsNone(result["summary"]["total_objective"])
        self.assertEqual(result["plot"]["assignments"], [])
        self.assertEqual(result["plot"]["nodes"][1]["depot_id"], None)
        self.assertIn("仅部分", web_app.result_reply(result))
        _, history = self.request("/api/history")
        self.assertEqual(history["history"][0]["model_id"], "M3")
        self.assertFalse(any("_mtime" in row for row in history["history"]))

    def test_restart_does_not_replay_requests(self):
        session = self.app.new_session()
        session["active_job_id"] = "not-in-memory"
        self.app.save_session(session)
        restarted = web_app.Workspace(self.root)
        restored = restarted.session(session["id"])
        self.assertIsNone(restored["active_job_id"])
        self.assertIn("不会自动重复提交", restored["messages"][-1]["content"])

    def test_invalid_settings_are_rejected(self):
        for settings in ({"time_limit": 0}, {"mip_gap": 1}, {"mip_gap": float("nan")},
                         {"solver": "shell"}, {"solver": "auto"}, {"solver": "gurobi"},
                         {"solver": "scipy"}, {"time_limit": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                web_app.validate_settings(settings)

    def test_workspace_uses_copt_by_default(self):
        _, bootstrap = self.request("/api/bootstrap")
        self.assertEqual(bootstrap["config"]["solver"], "copt")
        self.assertEqual(web_app.validate_settings({})["solver"], "copt")


class WorkerTests(unittest.TestCase):
    def test_upstream_decisions_cannot_be_changed_by_mapping(self):
        fixed = {"instance_id": "20-5-1a__Q70", "opened_depots": ["d1", "d2"], "reuse_decisions": "assignment",
                 "assignment": [{"customer_id": "c1", "depot_id": "d1"}, {"customer_id": "c2", "depot_id": "d2"}]}
        mapping = {"status": "ready", "model_id": "M3", "instance_id": fixed["instance_id"],
                   "opened_depots": ["d2", "d1"], "assignment": list(reversed(fixed["assignment"]))}
        web_worker.check_reference(mapping, fixed)
        with self.assertRaises(ValueError):
            web_worker.check_reference({**mapping, "assignment": [{"customer_id": "c1", "depot_id": "d2"}, {"customer_id": "c2", "depot_id": "d2"}]}, fixed)
        with self.assertRaises(ValueError):
            web_worker.check_reference({**mapping, "instance_id": "another"}, fixed)

    def test_worker_uses_fixed_pipeline_and_emits_stages_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            description = output / "problem.txt"
            description.write_text("使用 20-5-1a__Q70 实例选择仓库与分配客户。", encoding="utf-8")
            mapping = {"scenario_id": "web_offline_check", "instance_id": "20-5-1a__Q70", "model_id": "M1",
                       "model_version": "1.0", "status": "ready", "opened_depots": None, "assignment": None,
                       "distance_scale_km": None, "distance_scale_source": None,
                       "missing_fields": [], "unsupported_requirements": [], "evidence": []}
            selection = {"status": "selected", "model_id": "M1", "model_version": "1.0", "provider": "mocked-test"}
            stream = io.StringIO()
            with patch("main.load_local_environment"), \
                    patch("main.selection_for_description", return_value=selection), \
                    patch("orllm.rag.retrieve", return_value=[]), \
                    patch("main.mapping_for_selection", return_value=mapping), redirect_stdout(stream):
                code = web_worker.run({"output": temp, "description_file": str(description), "solver": "none", "time_limit": 300, "mip_gap": 0.08})
            self.assertEqual(code, 0)
            summary = main.read_json(output / "summary.json")
            self.assertFalse(summary["complete_solution"])
            self.assertIsNone(summary["total_objective"])
            self.assertEqual(summary["status"], "not_solved")
            self.assertTrue((output / "M1_all.lp").exists())
            stages = [json.loads(line[len("ORLLM_EVENT "):])["stage"] for line in stream.getvalue().splitlines() if line.startswith("ORLLM_EVENT ")]
            self.assertEqual(stages[:4], ["selection", "retrieval", "mapping", "build"])


if __name__ == "__main__":
    unittest.main()
