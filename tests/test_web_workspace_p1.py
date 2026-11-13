"""Offline checks for history provenance, explicit references, and solution quality."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import web_app


def verified_row(depot="d1", gap=0.05, runtime=12.0):
    return {"model_id": "M3", "instance_id": "sample__Q70", "depot_id": depot,
            "status": "feasible_limit", "status_label": "限时可行",
            "has_verified_solution": True, "mip_gap": gap, "runtime_seconds": runtime,
            "objective": 100.0, "primal_check": {"valid": True},
            "objective_check": {"valid": True}}


def summary_for(rows, complete=True, count=None):
    return {"schema_version": "1.0", "generated_at": "2026-10-12T03:00:00Z",
            "status": "feasible_limit" if complete else "partial", "status_label": "限时可行",
            "models": rows, "subproblem_count": len(rows) if count is None else count,
            "verified_subproblem_count": sum(r.get("has_verified_solution") is True for r in rows),
            "complete_solution": complete, "total_objective": 100.0 * len(rows) if complete else None,
            "business": {"depots": [], "assignments": [], "routes": []}}


class QualityTests(unittest.TestCase):
    def quality(self, rows, complete=True, count=None, report=None):
        return web_app.result_quality(summary_for(rows, complete, count),
                                      report if report is not None else {
                                          "requested_mip_gap": 0.08, "time_limit_seconds": 300})

    def test_complete_feasible_solution_above_target_stays_verified(self):
        result = self.quality([verified_row("d3", 0.0903, 300), verified_row("d5", 0.03, 88)])
        self.assertEqual(result["solution_state"], "verified_complete")
        self.assertEqual((result["verified_count"], result["total_count"]), (2, 2))
        self.assertEqual(result["gap_state"], "unmet")
        self.assertAlmostEqual(result["actual_max_gap"], 0.0903)
        self.assertEqual(result["requested_gap"], 0.08)
        self.assertEqual(result["time_limit_seconds"], 300)
        self.assertEqual([row["runtime_seconds"] for row in result["subproblems"]], [300, 88])
        self.assertTrue(all(row["verified"] for row in result["subproblems"]))
        self.assertTrue(result["message"])
        self.assertIn("gap", result["message"].lower())

    def test_target_zero_is_valid_and_requires_zero_gap(self):
        report = {"requested_mip_gap": 0, "time_limit_seconds": 300}
        exact = self.quality([verified_row(gap=0)], report=report)
        nonzero = self.quality([verified_row(gap=0.01)], report=report)
        self.assertEqual(exact["requested_gap"], 0)
        self.assertEqual(exact["gap_state"], "met")
        self.assertEqual(nonzero["gap_state"], "unmet")
        ordinary = self.quality([verified_row("d1", 0.03), verified_row("d2", 0.07)])
        self.assertEqual(ordinary["gap_state"], "met")
        self.assertAlmostEqual(ordinary["actual_max_gap"], 0.07)

    def test_legacy_result_does_not_invent_target_or_time_limit(self):
        result = self.quality([verified_row(gap=0.05)], report={})
        self.assertIsNone(result["requested_gap"])
        self.assertIsNone(result["time_limit_seconds"])
        self.assertEqual(result["gap_state"], "unknown")
        self.assertAlmostEqual(result["actual_max_gap"], 0.05)
        for invalid in (True, -1, float("nan"), float("inf")):
            with self.subTest(invalid_setting=invalid):
                invalid_result = self.quality([verified_row()], report={
                    "requested_mip_gap": invalid, "time_limit_seconds": invalid})
                self.assertIsNone(invalid_result["requested_gap"])
                self.assertIsNone(invalid_result["time_limit_seconds"])

    def test_invalid_or_missing_gap_is_not_treated_as_zero(self):
        for gap in (None, True, -0.01, float("nan"), float("inf")):
            with self.subTest(gap=gap):
                result = self.quality([verified_row("d1", 0), verified_row("d2", gap)])
                self.assertIsNone(result["actual_max_gap"])
                self.assertEqual(result["gap_state"], "unknown")

    def test_missing_subproblem_prevents_complete_gap_claim(self):
        result = self.quality([verified_row(gap=0.01)], complete=True, count=2)
        self.assertEqual(result["solution_state"], "verified_partial")
        self.assertEqual((result["verified_count"], result["total_count"]), (1, 2))
        self.assertIsNone(result["actual_max_gap"])
        self.assertNotEqual(result["gap_state"], "met")

    def test_partial_solution_cannot_satisfy_whole_task_gap(self):
        failed = {**verified_row("d2", None), "has_verified_solution": False,
                  "status": "no_solution", "objective": None, "primal_check": {"valid": False}}
        result = self.quality([verified_row("d1", 0.01), failed], complete=False)
        self.assertEqual(result["solution_state"], "verified_partial")
        self.assertEqual(result["verified_count"], 1)
        self.assertIsNone(result["actual_max_gap"])
        self.assertNotEqual(result["gap_state"], "met")
        none = self.quality([failed], complete=False)
        self.assertEqual(none["solution_state"], "none")
        for invalid in ({"primal_check": {"valid": False}}, {"objective": True},
                        {"objective_check": {"valid": False}}, {"has_verified_solution": False}):
            with self.subTest(invalid_verification=invalid):
                unusable = self.quality([{**verified_row(), **invalid}], complete=True)
                self.assertEqual(unusable["solution_state"], "none")
                self.assertNotEqual(unusable["gap_state"], "met")


class HistoryReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = web_app.Workspace(self.root)
        self.single = self.write_result("runs/single")
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

    def write_result(self, result_id):
        folder = self.root / "outputs" / result_id
        web_app.save_json(folder / "summary.json", summary_for([verified_row()]))
        web_app.save_json(folder / "build_report.json", {"requested_mip_gap": 0.08,
                                                       "time_limit_seconds": 300})
        return folder

    def request(self, route, body=None):
        req = Request(self.base + route, headers={"Content-Type": "application/json"},
                      data=json.dumps(body).encode() if body is not None else None)
        with urlopen(req, timeout=5) as response:
            return json.load(response)

    def test_history_classification_uses_manifest_not_folder_spelling(self):
        for batch_id, pipeline in (("looks_fixed", "live"), ("looks_live", "fixed")):
            batch = self.root / "outputs/experiments" / batch_id
            web_app.save_json(batch / "batch_manifest.json", {"settings": {"pipeline": pipeline}})
            self.write_result(f"experiments/{batch_id}/case_01/M3")
        history = {row["id"]: row for row in self.app.history()}
        self.assertEqual(history["runs/single"]["source_kind"], "single")
        self.assertEqual(history["experiments/looks_fixed/case_01/M3"]["source_kind"], "batch_live")
        self.assertEqual(history["experiments/looks_live/case_01/M3"]["source_kind"], "batch_fixed")
        for batch_id in ("looks_fixed", "looks_live"):
            row = history[f"experiments/{batch_id}/case_01/M3"]
            self.assertEqual(row["batch_id"], batch_id)
            self.assertEqual(row["case_id"], "case_01")
            self.assertTrue(row["source_label"])
            self.assertTrue(row["batch_label"])
        legacy_batch = self.root / "outputs/experiments/legacy_fixed"
        web_app.save_json(legacy_batch / "batch_manifest.json", {
            "settings": {"solver": "copt", "assignment_time_limit": 60}, "api_calls": 0})
        self.write_result("experiments/legacy_fixed/case_01/M3")
        self.app._history_cache = (0, [])
        legacy = next(row for row in self.app.history() if row["batch_id"] == "legacy_fixed")
        self.assertEqual(legacy["source_kind"], "batch_fixed")

    def test_history_and_result_reads_preserve_stored_results(self):
        before = {p: p.read_bytes() for p in self.single.glob("*.json")}
        with patch("orllm.data.load_base", return_value={}):
            history = self.request("/api/history")["history"]
            result = self.request("/api/result?id=runs/single")
        self.assertEqual(len(history), 1)
        self.assertEqual(result["quality"]["solution_state"], "verified_complete")
        self.assertFalse(result["progress"]["phases_recorded"])
        self.assertEqual(result["progress"]["phase_timings"], [])
        self.assertEqual(result["progress"]["subproblems"][0]["runtime_seconds"], 12)
        self.assertEqual(before, {p: p.read_bytes() for p in self.single.glob("*.json")})
        not_solved = {**verified_row(), "status": "not_solved", "has_verified_solution": False,
                      "objective": None, "runtime_seconds": None, "mip_gap": None}
        unsolved = self.write_result("runs/build_only")
        web_app.save_json(unsolved / "summary.json", summary_for([not_solved], complete=False))
        with patch("orllm.data.load_base", return_value={}):
            only_built = self.request("/api/result?id=runs/build_only")
        self.assertEqual(only_built["progress"]["completed_subproblem_count"], 0)
        self.assertFalse(only_built["progress"]["phases_recorded"])
        self.assertIsNone(only_built["progress"]["subproblems"][0]["runtime_seconds"])

    def test_explicit_reference_persists_and_clear_survives_restart(self):
        session = self.request("/api/session", {})
        bound = self.request("/api/session/context", {"session_id": session["id"], "result_id": "runs/single"})
        self.assertEqual(bound["reference_result_id"], "runs/single")
        restarted = web_app.Workspace(self.root)
        self.assertEqual(restarted.public_session(restarted.session(session["id"]))["reference_result_id"], "runs/single")
        cleared = self.request("/api/session/context", {"session_id": session["id"], "result_id": None})
        self.assertIsNone(cleared["reference_result_id"])
        restarted = web_app.Workspace(self.root)
        self.assertIsNone(restarted.public_session(restarted.session(session["id"]))["reference_result_id"])

    def test_browsing_historical_result_never_binds_conversation(self):
        session = self.request("/api/session", {})
        second = self.write_result("runs/another")
        self.app.set_reference(session["id"], "runs/single")
        before = deepcopy(self.app.public_session(self.app.session(session["id"])))
        with patch("orllm.data.load_base", return_value={}):
            self.request("/api/result?id=runs/another")
            self.request("/api/history")
        self.assertEqual(before, self.request("/api/session?id=" + session["id"]))
        self.assertTrue(second.is_dir())

    def test_new_verified_run_becomes_next_reference_and_clear_is_respected(self):
        session = self.app.new_session()
        with patch.object(self.app, 'execute'):
            submitted = self.app.submit({'session_id': session['id'], 'request_id': 'automatic-context-001',
                                         'prompt': '求解新任务', 'settings': {}})
        job = self.app.jobs[submitted['job_id']]
        job['action'] = 'run'
        result_id = 'runs/' + job['run_stamp']
        folder = self.write_result(result_id)
        summary = web_app.read_json(folder / 'summary.json')
        summary['cost_breakdown_complete'] = True
        web_app.save_json(folder / 'summary.json', summary)
        self.app.finish(job['id'], '完成', result_id=result_id)
        self.assertEqual(session['reference_result_id'], result_id)
        self.app.set_reference(session['id'], None)
        self.assertIsNone(self.app.session(session['id'])['reference_result_id'])

    def test_chat_only_reply_does_not_replace_explicit_reference(self):
        session = self.app.new_session()
        self.app.set_reference(session['id'], 'runs/single')
        with patch.object(self.app, 'execute'):
            submitted = self.app.submit({'session_id': session['id'], 'request_id': 'keep-context-001',
                                         'prompt': '解释这个结果', 'settings': {}})
        job = self.app.jobs[submitted['job_id']];job['action'] = 'answer'
        self.app.finish(job['id'], '解释', result_id='runs/single')
        self.assertEqual(session['reference_result_id'], 'runs/single')

    def test_submission_uses_reference_not_last_viewed_result(self):
        called = threading.Event()
        arguments = []
        session = self.app.new_session()
        session["current_result_id"] = "runs/single"
        self.app.save_session(session)
        def capture(*args):
            arguments.append(args)
            called.set()
        with patch.object(self.app, "execute", side_effect=capture):
            self.app.submit({"session_id": session["id"], "request_id": "offline-request-01",
                             "prompt": "解释一下模型", "settings": {}})
            self.assertTrue(called.wait(3))
        self.assertIsNone(arguments[0][-1])

    def test_subproblem_progress_counts_finishes_and_deduplicates_events(self):
        session = self.app.new_session()
        with patch.object(self.app, "execute"):
            submitted = self.app.submit({"session_id": session["id"], "request_id": "offline-progress-01",
                                         "prompt": "规划路线", "settings": {}})
        job_id = submitted["job_id"]
        with self.assertRaises(HTTPError) as busy:
            self.request("/api/session/context", {"session_id": session["id"], "result_id": "runs/single"})
        self.assertEqual(busy.exception.code, 409)
        self.app.jobs[job_id]["result_id"] = "runs/single"
        first = {"depot_id": "d1", "subproblem_index": 1, "subproblem_count": 2}
        self.app.phase(job_id, "solve", event="subproblem_started", **first)
        self.app.phase(job_id, "solve", event="subproblem_started", **first)
        started = self.app.job_snapshot(job_id)
        self.assertEqual(started["completed_subproblem_count"], 0)
        self.assertEqual(len(started["subproblems"]), 1)
        self.assertEqual(started["depot_id"], "d1")
        self.app.phase(job_id, "solve", event="subproblem_finished", **first,
                       solve_status="feasible_limit", runtime_seconds=12, mip_gap=0.06, verified=True)
        self.app.phase(job_id, "solve", event="subproblem_finished", **first,
                       solve_status="feasible_limit", runtime_seconds=12, mip_gap=0.06, verified=True)
        self.app.phase(job_id, "solve", event="subproblem_started", depot_id="d2",
                       subproblem_index=2, subproblem_count=2)
        underway = self.app.job_snapshot(job_id)
        self.assertEqual(underway["completed_subproblem_count"], 1)
        self.assertEqual(underway["depot_id"], "d2")
        self.assertEqual(len(underway["subproblems"]), 2)
        self.assertEqual(underway["subproblems"][0]["runtime_seconds"], 12)
        self.app.phase(job_id, "solve", event="subproblem_finished", depot_id="d2",
                       subproblem_index=2, subproblem_count=2,
                       solve_status="optimal", runtime_seconds=4, mip_gap=0, verified=True)
        self.app.phase(job_id, "report")
        finished = self.app.job_snapshot(job_id)
        self.assertEqual(finished["completed_subproblem_count"], 2)
        self.assertEqual([row["runtime_seconds"] for row in finished["subproblems"]], [12, 4])
        self.assertTrue(any(row["stage"] == "solve" for row in finished["phase_timings"]))
        persisted = web_app.read_json(self.single / "web_job.json")
        self.assertEqual(persisted["completed_subproblem_count"], 2)
        self.assertEqual(len(persisted["subproblems"]), 2)
        self.assertTrue(all("_started_tick" not in row for row in persisted["subproblems"]))
        with patch("orllm.data.load_base", return_value={}):
            reloaded = self.app.result("runs/single")
        self.assertEqual(reloaded["progress"]["completed_subproblem_count"], 2)
        self.assertTrue(reloaded["progress"]["phases_recorded"])
        self.assertEqual([row["runtime_seconds"] for row in reloaded["progress"]["subproblems"]], [12, 4])


if __name__ == "__main__":
    unittest.main()
