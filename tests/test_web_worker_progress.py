"""Per-depot progress stays truthful without calling providers or solver backends."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
import web_worker
from orllm.data import load_base
from orllm.linear_model import LinearModel


def routing(depot="d1", clients=("a", "b")):
    nodes = [depot, *clients]
    return {"model_id": "M3", "model_version": "1.0", "instance_id": "hand",
            "depot_id": depot, "C": list(clients), "q": dict.fromkeys(clients, 1),
            "U": 10, "Q": 2, "h": 5,
            "c": {i: {j: 1 for j in nodes if i != j} for i in nodes}}


def checked_solve(model, **options):
    """Each customer gets a direct route; original fixed rows verify the point."""
    depot = model.metadata["depot_id"]
    vector = []
    for variable in model.variables:
        first, second = variable["indices"]
        direct = first == depot or second == depot
        vector.append(float(direct) if variable["family"] == "x" else float(first == depot))
    result = model._solution("feasible_limit", "mocked checked incumbent", vector,
                             solver="mocked", runtime_seconds=0.2)
    assert result["primal_check"]["valid"]
    result.update(mip_gap=0.05, mip_dual_bound=result["objective"] * 0.95,
                  objective_check={"valid": True, "recomputed": result["objective"]})
    return result


class WorkerProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.scenario = {"scenario_id": "progress", "instance_id": "hand", "model_id": "M3"}
        self.instances = [routing(), routing("d2", ("c", "e"))]

    def local_build_mocks(self, solver=checked_solve):
        stack = ExitStack()
        stack.enter_context(patch("orllm.data.load_base", return_value={}))
        stack.enter_context(patch("orllm.binding.bind_scenario", return_value={
            "status": "ready", "errors": [], "warnings": [], "missing_fields": [],
            "provenance": {}, "instances": copy.deepcopy(self.instances)}))
        stack.enter_context(patch.object(LinearModel, "solve", autospec=True, side_effect=solver))
        return stack

    def test_build_started_finished_events_follow_saved_verified_artifacts(self):
        events = []
        def callback(stage, **details):
            events.append((stage, details))
            if details.get("event") == "subproblem_finished":
                solution = main.read_json(self.folder / f"M3_{details['depot_id']}.solution.json")
                self.assertEqual(details["objective"], solution["objective"])
                self.assertTrue(details["verified"])
                self.assertEqual(details["runtime_seconds"], 0.2)
                self.assertAlmostEqual(details["mip_gap"], 0.05)
        with self.local_build_mocks():
            report = main.run_build(self.scenario, self.folder, solver="copt", progress_callback=callback)
        solve_events = [details for stage, details in events if stage == "solve"]
        self.assertEqual([item["event"] for item in solve_events],
                         ["subproblem_started", "subproblem_finished"] * 2)
        self.assertEqual([item["subproblem_index"] for item in solve_events], [1, 1, 2, 2])
        self.assertTrue(all(item["subproblem_count"] == 2 for item in solve_events))
        self.assertEqual(events[-1], ("report", {}))
        self.assertEqual(report["status"], "solved")

    def test_unverified_outcome_does_not_emit_objective_or_gap(self):
        events = []
        def unverified(model, **options):
            return {"status": "feasible_limit", "objective": 99, "mip_gap": 0.01,
                    "primal_check": {"valid": True}, "objective_check": {"valid": False}}
        with self.local_build_mocks(unverified):
            main.run_build(self.scenario, self.folder, solver="copt",
                           progress_callback=lambda stage, **details: events.append((stage, details)))
        for stage, details in events:
            if details.get("event") == "subproblem_finished":
                self.assertFalse(details["verified"])
                self.assertNotIn("objective", details)
                self.assertIsNone(details["mip_gap"])

    def test_build_only_has_no_solve_events_and_no_backend_calls(self):
        events = []
        with self.local_build_mocks(), patch.object(LinearModel, "solve") as solve:
            report = main.run_build(self.scenario, self.folder, solver="none",
                                   progress_callback=lambda stage, **details: events.append(stage))
        solve.assert_not_called()
        self.assertEqual(report["status"], "built")
        self.assertNotIn("solve", events)

    def test_capacity_rejection_has_no_solve_events(self):
        scenario = main.read_json(main.ROOT / "tests/fixtures/m3_routes.scenario.json")
        base = load_base(main.ROOT, scenario["instance_id"])
        base["U"] = dict.fromkeys(base["D"], 1)
        events = []
        with patch("orllm.data.load_base", return_value=base), patch.object(LinearModel, "solve") as solve:
            report = main.run_build(scenario, self.folder, solver="copt",
                                   progress_callback=lambda stage, **details: events.append(stage))
        solve.assert_not_called()
        self.assertEqual(report["status"], "invalid")
        self.assertNotIn("solve", events)
        self.assertFalse(list(self.folder.glob("*.lp")))

    def test_progress_preserves_every_fixed_mathematical_model(self):
        with self.local_build_mocks():
            main.run_build(self.scenario, self.folder / "before", solver="copt")
            main.run_build(self.scenario, self.folder / "after", solver="copt", progress_callback=lambda *a, **kw: None)
        for depot in ("d1", "d2"):
            for suffix in ("model.json", "lp", "instance.json", "solution.json"):
                name = f"M3_{depot}.{suffix}"
                self.assertEqual((self.folder / "before" / name).read_bytes(),
                                 (self.folder / "after" / name).read_bytes())

    def run_worker(self, solver=checked_solve):
        description = self.folder / "problem.txt"
        description.write_text("固定仓库和客户归属后安排常规车辆路线。", encoding="utf-8")
        mapping = {"scenario_id": "worker_progress", "instance_id": "hand", "model_id": "M3",
                   "model_version": "1.0", "status": "ready", "opened_depots": ["d1", "d2"],
                   "assignment": [{"customer_id": c, "depot_id": instance["depot_id"]}
                                  for instance in self.instances for c in instance["C"]],
                   "distance_scale_km": None, "distance_scale_source": None,
                   "missing_fields": [], "unsupported_requirements": [], "evidence": []}
        selection = {"status": "selected", "model_id": "M3", "model_version": "1.0", "provider": "mocked"}
        stream = io.StringIO()
        with self.local_build_mocks(solver), patch("main.load_local_environment"), \
                patch("main.selection_for_description", return_value=selection), \
                patch("main.retrieval_options", return_value={"backend": "keyword", "top_k": 5}), \
                patch("orllm.rag.retrieve", return_value=[]), \
                patch("main.mapping_for_selection", return_value=mapping), redirect_stdout(stream):
            code = web_worker.run({"output": str(self.folder), "description_file": str(description),
                                   "solver": "copt", "time_limit": 300, "mip_gap": 0.08})
        emitted = [json.loads(line[len("ORLLM_EVENT "):]) for line in stream.getvalue().splitlines()
                   if line.startswith("ORLLM_EVENT ")]
        return code, main.read_json(self.folder / "pipeline_report.json"), emitted

    def test_worker_persists_completed_rows_after_report_stage(self):
        code, record, emitted = self.run_worker()
        self.assertEqual(code, 0)
        self.assertEqual(record["stage"], "done")
        self.assertEqual(record["status"], "solved")
        self.assertEqual(len(record["subproblems"]), 2)
        for row in record["subproblems"]:
            self.assertEqual(row["status"], "done")
            self.assertTrue(row["verified"])
            self.assertIn("started_at", row)
            self.assertIn("finished_at", row)
            self.assertGreaterEqual(row["elapsed_seconds"], 0)
            self.assertEqual(row["runtime_seconds"], 0.2)
        finished = [item for item in emitted if item.get("event") == "subproblem_finished"]
        self.assertEqual(len(finished), 2)
        self.assertEqual(finished[0]["objective"], record["subproblems"][0]["objective"])

    def test_worker_exception_marks_only_current_row_error(self):
        calls = []
        def fail_second(model, **options):
            calls.append(model.metadata["depot_id"])
            if len(calls) == 2:
                raise RuntimeError("mocked solver failure")
            return checked_solve(model, **options)
        code, record, emitted = self.run_worker(fail_second)
        self.assertEqual(code, 1)
        self.assertEqual(record["status"], "error")
        first, second = record["subproblems"]
        self.assertEqual(first["status"], "done")
        self.assertTrue(first["verified"])
        self.assertEqual(second["status"], "error")
        self.assertFalse(second["verified"])
        self.assertEqual(second["error"], "mocked solver failure")
        self.assertNotIn("objective", second)
        self.assertEqual(len([item for item in emitted if item.get("event") == "subproblem_finished"]), 1)


if __name__ == "__main__":
    unittest.main()
