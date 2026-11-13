"""Local solve integration, including truthful outcomes and offline run reuse."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
from orllm.linear_model import LinearModel


SELECTION = {
    "status": "selected", "model_id": "M1", "model_version": "1.0",
    "provider": "typesafe", "confidence": 0.98, "threshold": 0.7,
}
MAPPING = {
    "scenario_id": "offline_solve_pipeline", "instance_id": "20-5-1a__Q70",
    "model_id": "M1", "model_version": "1.0", "opened_depots": None,
    "assignment": None, "distance_scale_km": None, "distance_scale_source": None,
    "status": "ready", "missing_fields": [], "unsupported_requirements": [],
    "evidence": [{"field": "instance_id", "source": "user_description",
                  "excerpt": "使用 20-5-1a__Q70 实例"}],
}
CONTEXT = [{"id": "m1-rule", "model_id": "M1", "model_version": "1.0",
            "backend": "keyword", "text": "M1 决定仓库开放和客户归属。"}]


class SolvePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.scenario = main.read_json(main.ROOT / "tests/fixtures/m1_assignment.scenario.json")
        self.description = self.folder / "description.txt"
        self.description.write_text("使用 20-5-1a__Q70 实例选择仓库并分配客户。", encoding="utf-8")
        self.config = self.folder / "empty-config.json"
        self.config.write_text("{}", encoding="utf-8")

    def run_args(self, output="result", *extra):
        return main.make_parser().parse_args([
            "run", "--description-file", str(self.description),
            "--jev-config", str(self.config), "--config", str(self.config),
            "--rag-config", str(self.config), "--backend", "keyword",
            "--output", str(self.folder / output), *extra,
        ])

    def execute(self, args):
        with redirect_stdout(io.StringIO()):
            return args.func(args)

    def assert_readable_reports(self, report, output):
        self.assertIsInstance(report.get("reports"), dict)
        paths = {Path(value).name: output / value for value in report["reports"].values()
                 if isinstance(value, str)}
        for filename in ("summary.json", "summary.md", "report.html"):
            self.assertIn(filename, paths)
            self.assertGreater(paths[filename].stat().st_size, 0)
        self.assertIsInstance(main.read_json(paths["summary.json"]), dict)
        self.assertIn("<html", paths["report.html"].read_text(encoding="utf-8").lower())

    def provider_mocks(self, selection=None, mapping=None):
        stack = ExitStack()
        stack.enter_context(patch("orllm.jev.select_with_jev", return_value=copy.deepcopy(selection or SELECTION)))
        stack.enter_context(patch("orllm.rag.retrieve", return_value=CONTEXT))
        stack.enter_context(patch("orllm.llm.extract_mapping", return_value=copy.deepcopy(mapping or MAPPING)))
        return stack

    def test_build_without_solve_still_exports_human_reports(self):
        output = self.folder / "build-only"
        with patch.object(LinearModel, "solve") as solve:
            report = main.run_build(self.scenario, output)
        solve.assert_not_called()
        self.assertEqual(report["status"], "built")
        self.assertEqual(main.report_exit_code(report, "none"), 0)
        self.assertEqual(len(report["artifacts"]), 1)
        artifact = report["artifacts"][0]
        self.assertEqual(artifact["variable_count"], 105)
        self.assertEqual(artifact["constraint_count"], 125)
        self.assertNotIn("solution", artifact)
        self.assert_readable_reports(report, output)

    @unittest.skipUnless(importlib.util.find_spec("coptpy"), "optional copt unavailable")
    def test_build_solves_real_m1_and_exports_verified_solution(self):
        output = self.folder / "copt"
        args = main.make_parser().parse_args([
            "build", "--scenario", str(main.ROOT / "tests/fixtures/m1_assignment.scenario.json"),
            "--solve", "copt", "--time-limit", "15",
            "--mip-gap", "0", "--output", str(output),
        ])
        self.assertEqual(self.execute(args), 0)
        report = main.read_json(output / "build_report.json")
        self.assertEqual(report["status"], "solved")
        self.assertEqual(report["build_status"], "built")
        self.assertEqual(report["solve_status"], "optimal")
        artifact = report["artifacts"][0]
        self.assertIn(artifact["solver"], {"copt"})
        self.assertAlmostEqual(artifact["objective"], 92535, places=4)
        solution = main.read_json(output / artifact["solution"])
        self.assertTrue(solution["primal_check"]["valid"])
        self.assertEqual(len(solution["variable_values"]), 105)
        self.assertAlmostEqual(solution["objective"], artifact["objective"])
        self.assert_readable_reports(report, output)

    def test_limited_feasible_outcome_and_options_are_preserved(self):
        output = self.folder / "limited"
        solution = {
            "status": "feasible_limit", "solver": "copt", "message": "Time limit",
            "objective": 92535.0, "best_bound": 90000.0, "mip_gap": 0.0274,
            "primal_check": {"valid": True}, "decision_values": [], "variable_values": {},
        }
        with patch.object(LinearModel, "solve", return_value=solution) as solve:
            report = main.run_build(self.scenario, output, solver="copt", time_limit=7.5, mip_gap=0.02)
        solve.assert_called_once()
        self.assertEqual(solve.call_args.kwargs["solver"], "copt")
        self.assertEqual(solve.call_args.kwargs["time_limit"], 7.5)
        self.assertEqual(solve.call_args.kwargs["mip_gap"], 0.02)
        self.assertEqual(Path(solve.call_args.kwargs["log_path"]).parent, output)
        self.assertEqual(report["status"], "solved")
        self.assertEqual(report["solve_status"], "feasible_limit")
        self.assertEqual(report["artifacts"][0]["solve_status"], "feasible_limit")
        self.assertEqual(main.report_exit_code(report, "copt"), 0)
        self.assert_readable_reports(report, output)

    def test_solver_outcomes_without_incumbent_are_not_success(self):
        for status in ("infeasible", "no_solution", "error"):
            output = self.folder / status
            with self.subTest(status=status), patch.object(LinearModel, "solve", return_value={
                    "status": status, "solver": "copt", "message": status}):
                report = main.run_build(self.scenario, output, solver="copt")
            self.assertEqual(report["status"], "solve_failed")
            self.assertEqual(report["build_status"], "built")
            self.assertEqual(report["solve_status"], status)
            self.assertEqual(main.report_exit_code(report, "copt"), 3)
            artifact = report["artifacts"][0]
            solution = main.read_json(output / artifact["solution"])
            self.assertNotIn("objective", solution)
            self.assertIsNone(artifact.get("objective"))
            self.assert_readable_reports(report, output)

    def test_one_failed_depot_prevents_claiming_whole_scenario_solved(self):
        output = self.folder / "partial"
        scenario = main.read_json(main.ROOT / "tests/fixtures/m3_routes.scenario.json")
        feasible = {"status": "optimal", "solver": "copt", "message": "Optimal",
                    "objective": 100.0, "primal_check": {"valid": True},
                    "decision_values": [], "variable_values": {}}
        failed = {"status": "no_solution", "solver": "copt", "message": "No incumbent"}
        with patch.object(LinearModel, "solve", side_effect=[feasible, failed, feasible, feasible, feasible]) as solve:
            report = main.run_build(scenario, output, solver="copt")
        self.assertEqual(solve.call_count, 5)
        self.assertEqual(report["status"], "solve_failed")
        self.assertEqual(report["solve_status"], "partial")
        self.assertEqual(main.report_exit_code(report, "copt"), 3)
        self.assert_readable_reports(report, output)

    def test_run_can_solve_and_reports_solver_failure_as_exit_three(self):
        with self.provider_mocks(), patch.object(LinearModel, "solve", return_value={
                "status": "infeasible", "solver": "copt", "message": "Infeasible"}) as solve:
            args = self.run_args("run-failed", "--solve", "copt", "--time-limit", "12", "--mip-gap", "0.01")
            self.assertEqual(self.execute(args), 3)
        solve.assert_called_once()
        self.assertEqual(solve.call_args.kwargs["time_limit"], 12)
        self.assertEqual(solve.call_args.kwargs["mip_gap"], 0.01)
        report = main.read_json(self.folder / "run-failed/build_report.json")
        self.assertEqual(report["status"], "solve_failed")
        pipeline = main.read_json(self.folder / "run-failed/pipeline_report.json")
        self.assertEqual(pipeline["status"], "solve_failed")

    def test_run_selection_and_mapping_stop_states_never_solve(self):
        cases = [
            ("needs_review", {**SELECTION, "status": "needs_review", "model_id": None}, MAPPING),
            ("unsupported", SELECTION, {**MAPPING, "status": "unsupported", "unsupported_requirements": ["时间窗"]}),
            ("needs_clarification", SELECTION, {**MAPPING, "status": "needs_clarification",
                                              "instance_id": None, "missing_fields": ["instance_id"]}),
        ]
        for status, selection, mapping in cases:
            with self.subTest(status=status), self.provider_mocks(selection, mapping), \
                    patch.object(LinearModel, "solve") as solve:
                self.assertEqual(self.execute(self.run_args(status, "--solve", "copt")), 2)
            solve.assert_not_called()
            output = self.folder / status
            report = main.read_json(output / "build_report.json")
            self.assertEqual(report["status"], status)
            self.assertEqual(report["artifacts"], [])
            self.assert_readable_reports(report, output)

    def test_solve_reuses_saved_mapping_without_provider_calls_or_source_changes(self):
        source = self.folder / "previous-run"
        main.write_json(source / "mapping.json", MAPPING)
        main.write_json(source / "selection.json", SELECTION)
        before = {path.name: path.read_bytes() for path in source.iterdir()}
        output = self.folder / "reuse"
        args = main.make_parser().parse_args([
            "solve", "--run-dir", str(source), "--output", str(output),
            "--solve", "copt", "--time-limit", "11", "--mip-gap", "0.03",
        ])
        with patch("orllm.jev.select_with_jev") as select, \
                patch("orllm.rag.retrieve") as retrieve, \
                patch("orllm.llm.extract_mapping") as extract, \
                patch.object(LinearModel, "solve", return_value={
                    "status": "no_solution", "solver": "copt", "message": "No incumbent"}) as solve:
            self.assertEqual(self.execute(args), 3)
        select.assert_not_called()
        retrieve.assert_not_called()
        extract.assert_not_called()
        solve.assert_called_once()
        self.assertEqual(solve.call_args.kwargs["solver"], "copt")
        self.assertEqual(solve.call_args.kwargs["time_limit"], 11)
        self.assertEqual(solve.call_args.kwargs["mip_gap"], 0.03)
        self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})
        report = main.read_json(output / "build_report.json")
        self.assertEqual(report["scenario_id"], MAPPING["scenario_id"])
        self.assert_readable_reports(report, output)

    def test_solve_after_demo_cleanup_falls_back_to_batch_and_prefers_single_runs(self):
        for has_single_run in (False, True):
            with self.subTest(has_single_run=has_single_run):
                workspace = self.folder / str(has_single_run)
                main.write_json(workspace / "schemas/mapping.schema.json",
                                main.read_json(main.ROOT / "schemas/mapping.schema.json"))
                single = workspace / "outputs/runs/new_task"
                batch = workspace / "outputs/experiments/live_chain_saved/01_data/M1"
                if has_single_run:
                    main.write_json(single / "mapping.json", {**MAPPING, "scenario_id": "single_task"})
                main.write_json(batch / "mapping.json", {**MAPPING, "scenario_id": "batch_task"})
                source = single if has_single_run else batch
                before = (source / "mapping.json").read_bytes()
                output = workspace / "outputs/runs/reused"
                args = main.make_parser().parse_args(["solve", "--output", str(output)])
                with patch.object(main, "ROOT", workspace), \
                        patch.object(main, "run_build", return_value={
                            "status": "solved", "solve_status": "optimal", "reports": {}, "artifacts": []}) as build, \
                        patch("orllm.jev.select_with_jev") as select, \
                        patch("orllm.rag.retrieve") as retrieve, \
                        patch("orllm.llm.extract_mapping") as extract:
                    self.assertEqual(self.execute(args), 0)
                select.assert_not_called()
                retrieve.assert_not_called()
                extract.assert_not_called()
                self.assertEqual(build.call_args.args[0]["scenario_id"],
                                 "single_task" if has_single_run else "batch_task")
                self.assertEqual(main.read_json(output / "build_report.json")["resumed_from"], str(source.resolve()))
                self.assertEqual((source / "mapping.json").read_bytes(), before)

    def test_solve_does_not_bypass_incomplete_mapping(self):
        source = self.folder / "incomplete-run"
        main.write_json(source / "mapping.json", {
            **MAPPING, "status": "needs_clarification", "instance_id": None,
            "missing_fields": ["instance_id"],
        })
        output = self.folder / "stopped-reuse"
        args = main.make_parser().parse_args(["solve", "--run-dir", str(source), "--output", str(output)])
        self.assertEqual(args.solve, "copt")
        with patch.object(LinearModel, "solve") as solve:
            self.assertEqual(self.execute(args), 2)
        solve.assert_not_called()
        report = main.read_json(output / "build_report.json")
        self.assertEqual(report["status"], "needs_clarification")
        self.assertFalse(list(output.glob("*.lp")))
        self.assert_readable_reports(report, output)

    def test_main_rejects_invalid_solver_limits_before_dispatch(self):
        for option, value in (("--time-limit", "0"), ("--time-limit", "nan"),
                              ("--mip-gap", "-0.1"), ("--mip-gap", "inf")):
            with self.subTest(option=option, value=value), \
                    patch("sys.argv", ["main.py", "build", "--scenario",
                                      str(main.ROOT / "tests/fixtures/m1_assignment.scenario.json"), option, value]), \
                    patch("main.cmd_build") as build, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main.main()
            self.assertEqual(error.exception.code, 2)
            build.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("coptpy"), "optional copt unavailable")
    def test_relaxed_gap_reuses_checked_limited_solution_and_tighter_gap_resolves(self):
        output = self.folder / "resume-gap"
        main.run_build(self.scenario, output, solver="copt", time_limit=15, mip_gap=0)
        path = output / "M1_all.solution.json"
        previous = main.read_json(path)
        previous.update(status="feasible_limit", mip_gap=0.03,
                        mip_dual_bound=previous["objective"] * 0.97)
        main.write_json(path, previous)
        with patch.object(LinearModel, "solve") as solve:
            relaxed = main.run_build(self.scenario, output, solver="copt",
                                     mip_gap=0.05, reuse_verified=True)
        solve.assert_not_called()
        self.assertTrue(relaxed["artifacts"][0]["solution_reused"])
        self.assertTrue(relaxed["artifacts"][0]["meets_requested_gap"])
        self.assertEqual(main.read_json(path)["status"], "feasible_limit")
        with patch.object(LinearModel, "solve", return_value=copy.deepcopy(previous)) as solve:
            main.run_build(self.scenario, output, solver="copt", mip_gap=0.01,
                           reuse_verified=True)
        solve.assert_called_once()
        self.assertEqual(solve.call_args.kwargs["mip_gap"], 0.01)


if __name__ == "__main__":
    unittest.main()
