"""Business interpretation, honest partial results, and report escaping."""
import json
from pathlib import Path
import tempfile
import unittest

from models import build_model
from orllm.reporting import write_run_report
from test_models import assignment, route, electric


def solution(instance, decisions):
    model = build_model(instance)
    values = [decisions.get((v["family"], tuple(v["indices"])), 0.0) for v in model.variables]
    result = model._solution("optimal", "hand-verified incumbent", values,
        solver="test", mip_gap=0.0, runtime_seconds=0.2)
    assert result["primal_check"]["valid"], result
    return result


class ReportingTests(unittest.TestCase):
    def write(self, instances, solutions, report=None):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        output = Path(folder.name)
        report = report or {"status": "built", "scenario_id": "test", "artifacts": [], "provenance": {}}
        files = write_run_report(output, report, instances, solutions)
        return json.loads((output / files["summary_json"]).read_text(encoding="utf-8")), output, files

    def test_assignment_cost_not_multiplied_by_demand(self):
        instance = assignment()
        sol = solution(instance, {("y", ("d2",)): 1, ("z", ("d2", "a")): 1, ("z", ("d2", "b")): 1})
        summary, output, files = self.write([instance], [sol])
        self.assertEqual(summary["total_objective"], 26)
        self.assertEqual(summary["costs"], {"opening_cost": 20, "delivery_cost": 6})
        self.assertEqual(summary["business"]["depots"][1]["utilization"], 5 / 6)
        self.assertTrue((output / "assignments.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertIn("report.html", files.values())

    def test_route_recovered_and_costed(self):
        instance = route()
        sol = solution(instance, {("x", ("d1", "a")): 1, ("x", ("a", "b")): 1,
            ("x", ("b", "d1")): 1, ("f", ("d1", "a")): 2, ("f", ("a", "b")): 1})
        summary, _, _ = self.write([instance], [sol])
        route_row = summary["business"]["routes"][0]
        self.assertEqual(route_row["nodes"], ["d1", "a", "b", "d1"])
        self.assertEqual(route_row["total_cost"], 11)
        self.assertEqual(route_row["demand"], 2)
        self.assertIsNone(route_row["distance_km"])
        self.assertEqual(summary["business"]["depots"][0]["vehicle_count"], 1)

    def test_optimal_with_nonzero_gap_does_not_claim_zero_gap(self):
        instance = assignment()
        sol = solution(instance, {("y", ("d2",)): 1, ("z", ("d2", "a")): 1, ("z", ("d2", "b")): 1})
        tolerance_sol = dict(sol, mip_gap=0.005)
        summary, output, _ = self.write([instance, instance], [sol, tolerance_sol])
        self.assertEqual(summary["status"], "optimal")
        self.assertEqual(summary["total_objective"], 52)
        self.assertEqual(summary["models"][0]["status_label"], "最优（gap 0）")
        self.assertIn("容差", summary["models"][1]["status_label"])
        self.assertIn("容差", summary["status_label"])
        self.assertNotIn("已证明最优", (output / "report.html").read_text(encoding="utf-8"))
        # Older solutions without a recorded gap must not imply zero gap either.
        summary, _, _ = self.write([instance], [dict(sol, mip_gap=None)])
        self.assertIn("容差", summary["status_label"])

    def test_energy_uses_arcs_and_load_not_auxiliary_e(self):
        instance = electric(10)
        sol = solution(instance, {("x", ("d1", "a")): 1, ("x", ("a", "d1")): 1,
            ("f", ("d1", "a")): 2, ("e", ("a",)): 8})
        summary, _, _ = self.write([instance], [sol])
        route_row = summary["business"]["routes"][0]
        self.assertEqual(route_row["energy"], 6)
        self.assertEqual(route_row["energy_cost"], 18)
        self.assertEqual(summary["total_objective"], 23)

    def test_partial_never_reports_complete_total(self):
        instance = electric(10)
        sol = solution(instance, {("x", ("d1", "a")): 1, ("x", ("a", "d1")): 1,
            ("f", ("d1", "a")): 2, ("e", ("a",)): 4})
        summary, _, _ = self.write([instance, instance], [sol, {"status": "infeasible", "solver": "test"}])
        self.assertEqual(summary["status"], "partial")
        self.assertIsNone(summary["total_objective"])
        self.assertEqual(summary["partial_objective"], 23)
        self.assertEqual(len(summary["business"]["routes"]), 1)

    def test_invalid_and_unsolved_never_show_business_plan(self):
        for sol in (None, {"status": "optimal", "objective": 99, "primal_check": {"valid": False}},
                    {"status": "infeasible", "objective": 99}):
            with self.subTest(solution=sol):
                summary, _, files = self.write([assignment()], [sol])
                self.assertIsNone(summary["total_objective"])
                self.assertFalse(summary["business"]["assignments"])
                self.assertNotIn("assignments.csv", files["business_csv"])

    def test_dynamic_html_escaped(self):
        payload = '<script>alert("x")</script>'
        report = {"status": "invalid", "scenario_id": payload, "errors": [payload],
                  "provenance": {"scenario_description": payload}, "artifacts": []}
        _, output, _ = self.write([], [], report)
        html = (output / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("href=", html)

    def test_missing_decision_values_does_not_invent_routes(self):
        sol = {"status": "optimal", "objective": 23, "primal_check": {"valid": True}}
        summary, _, _ = self.write([electric(10)], [sol])
        self.assertFalse(summary["business"]["routes"])
        self.assertEqual(summary["models"][0]["business_report_status"], "decode_failed")

    def test_reused_directory_drops_stale_business_csv(self):
        instance = assignment()
        sol = solution(instance, {("y", ("d2",)): 1, ("z", ("d2", "a")): 1, ("z", ("d2", "b")): 1})
        _, output, _ = self.write([instance], [sol])
        self.assertTrue((output / "assignments.csv").exists())
        write_run_report(output, {"status": "built", "artifacts": []}, [instance], [None])
        self.assertFalse((output / "assignments.csv").exists())
        self.assertFalse((output / "depots.csv").exists())

    def test_objective_check_failure_blocks_plan_and_old_results_still_work(self):
        instance = assignment()
        sol = solution(instance, {("y", ("d2",)): 1, ("z", ("d2", "a")): 1, ("z", ("d2", "b")): 1})
        summary, _, _ = self.write([instance], [sol])
        self.assertTrue(summary["complete_solution"])
        self.assertEqual(summary["models"][0]["objective_check_label"], "旧结果未记录")
        sol["objective_check"] = {"valid": False, "absolute_difference": 4}
        summary, _, _ = self.write([instance], [sol])
        self.assertEqual(summary["status"], "invalid_solution")
        self.assertFalse(summary["business"]["assignments"])
        self.assertIsNone(summary["total_objective"])
        self.assertEqual(summary["models"][0]["objective_check_label"], "未通过")

    def test_preflight_warning_is_contextual_and_original_preserved(self):
        instance = assignment()
        sol = solution(instance, {("y", ("d2",)): 1, ("z", ("d2", "a")): 1, ("z", ("d2", "b")): 1})
        warning = {"code": "FEASIBILITY_NOT_CERTIFIED", "field": "assignment", "message": "Original preflight warning"}
        report = {"status": "optimal", "build_status": "built", "warnings": [warning], "artifacts": []}
        summary, output, _ = self.write([instance], [sol], report)
        self.assertEqual(summary["build_status"], "built")
        self.assertEqual(summary["warnings"], [warning])
        self.assertEqual(report["warnings"], [warning])
        self.assertIn("方案可行性已验证", summary["warning_messages"][0])
        html = (output / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("Original preflight warning", html)
        self.assertIn("容量利用率", html)
        summary, _, _ = self.write([instance], [None], report)
        self.assertIn("仍需获得", summary["warning_messages"][0])

    def test_scenario_assumptions_and_copt_solver_are_prominent(self):
        instance = assignment()
        sol = solution(instance, {("y", ("d2",)): 1, ("z", ("d2", "a")): 1, ("z", ("d2", "b")): 1})
        sol.update(solver="copt", fallback_used=False, solver_attempts=[
            {"solver": "copt", "status": "optimal"}])
        report = {"status": "built", "artifacts": [], "provenance": {"scenario_description": "仅供演示：客户归属为实验假设。"}}
        summary, output, _ = self.write([instance], [sol], report)
        html = (output / "report.html").read_text(encoding="utf-8")
        self.assertLess(html.index("演示场景与假设"), html.index("子问题求解情况"))
        self.assertIn("<td>copt</td>", html)
        self.assertEqual(summary["models"][0]["solver"], "copt")
        self.assertFalse(summary["models"][0]["fallback_used"])

    def test_battery_baseline_warning_refers_to_correct_subproblem(self):
        instance = electric(10)
        sol = solution(instance, {("x", ("d1", "a")): 1, ("x", ("a", "d1")): 1,
            ("f", ("d1", "a")): 2, ("e", ("a",)): 4})
        report = {"status": "partial", "artifacts": [], "warnings": [
            {"code": "DIRECT_BASELINE_NOT_CERTIFIED", "field": "instances[0]"},
            {"code": "DIRECT_BASELINE_NOT_CERTIFIED", "field": "instances[1]"}]}
        summary, _, _ = self.write([instance, instance], [sol, {"status": "no_solution"}], report)
        self.assertIn("优化方案的可行性已验证", summary["warning_messages"][0])
        self.assertIn("不能据此断言", summary["warning_messages"][1])

    def test_extended_states_are_chinese(self):
        for status, expected in (("infeasible_or_unbounded", "不可行或无界"), ("unsupported", "不支持")):
            summary, _, _ = self.write([assignment()], [{"status": status}])
            self.assertIn(expected, summary["status_label"])


if __name__ == "__main__":
    unittest.main()
