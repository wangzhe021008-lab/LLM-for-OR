"""End-to-end checks for provided data and preflight boundaries (no API calls)."""
import json
import tempfile
import unittest
from pathlib import Path

from main import ROOT, read_json, run_build, report_exit_code
from orllm.data import load_base
from orllm.binding import bind_scenario
from orllm.validation import validate_instance


class PipelineTests(unittest.TestCase):
    def test_all_prepared_instances_bind_to_m1(self):
        manifest = read_json(ROOT / "data/processed/manifest.json")
        self.assertEqual(len(manifest["instances"]), 41)
        scenario = read_json(ROOT / "tests/fixtures/m1_assignment.scenario.json")
        for item in manifest["instances"]:
            with self.subTest(instance_id=item["instance_id"]):
                scenario["instance_id"] = item["instance_id"]
                result = bind_scenario(load_base(ROOT, item["instance_id"]), scenario)
                self.assertEqual(result["status"], "ready", result["errors"])
                self.assertEqual(validate_instance(result["instances"][0]), [])

    def test_direct_delivery_cost_is_whole_order(self):
        scenario = read_json(ROOT / "tests/fixtures/m1_assignment.scenario.json")
        result = bind_scenario(load_base(ROOT, scenario["instance_id"]), scenario)
        instance = result["instances"][0]
        # Source d1=(6,7), c1=(20,35): floor(100*sqrt(980))=3130.
        self.assertEqual(instance["q"]["c1"], 17)
        self.assertEqual(instance["g"]["d1"]["c1"], 1000 + 3130 + 3130)

    def test_missing_m4_does_not_create_model(self):
        scenario = read_json(ROOT / "tests/fixtures/m4_missing_parameters.scenario.json")
        with tempfile.TemporaryDirectory() as folder:
            report = run_build(scenario, folder)
            self.assertEqual(report["status"], "needs_clarification")
            self.assertIn("assignment", report["missing_fields"])
            self.assertIn("distance_scale_km", report["missing_fields"])
            self.assertFalse(list(Path(folder).glob("*.lp")))

    def test_m4_scaling_is_explicit_and_baseline_verified(self):
        scenario = read_json(ROOT / "tests/fixtures/m4_routes.scenario.json")
        base = load_base(ROOT, scenario["instance_id"])
        result = bind_scenario(base, scenario)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(result["instances"]), 5)
        self.assertEqual(scenario["distance_scale_km"], 0.1)
        self.assertTrue(scenario["distance_scale_source"].strip())
        self.assertEqual(len(result["provenance"]["baseline_certificates"]), 5)

    def test_bad_indices_and_extra_requirements_rejected(self):
        scenario = read_json(ROOT / "tests/fixtures/m3_routes.scenario.json")
        base = load_base(ROOT, scenario["instance_id"])
        scenario["assignment"]["c1"] = "d999"
        result = bind_scenario(base, scenario)
        self.assertEqual(result["status"], "invalid")
        scenario["time_windows"] = {"c1": [8, 10]}
        result = bind_scenario(base, scenario)
        self.assertIn("UNKNOWN_FIELD", [error["code"] for error in result["errors"]])

    def test_failed_solver_not_success_exit(self):
        report = {"status": "built", "artifacts": [{"solve_status": "error"}]}
        self.assertEqual(report_exit_code(report, "copt"), 3)

    def test_nonstring_model_id_is_validation_error(self):
        self.assertTrue(validate_instance({"model_id": []}))
        self.assertTrue(validate_instance({"model_id": {}}))


if __name__ == "__main__":
    unittest.main()
