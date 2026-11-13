"""Live batch requests must use providers and preserve upstream decisions."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import run_chain_batch as batch
from orllm.linear_model import build_assignment
from test_live_pipeline import MAPPING, SELECTION, CONTEXT


class LiveChainTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        self.folder = self.output / "01" / "M1"
        self.folder.mkdir(parents=True)
        config = self.output / "config.json"
        config.write_text("{}", encoding="utf-8")
        self.args = argparse.Namespace(jev_config=str(config), llm_config=str(config),
            rag_config=str(config), solver="copt")
        self.manifest = dict(status="running", instances=["G20-5-1a__Q70"],
            results=[], chains=[], api_calls=0, _started_perf=time.perf_counter(),
            settings=dict(pipeline="live", solver="copt", llm_model="test-deepseek",
                assignment_time_limit=300, assignment_mip_gap=0.08,
                route_time_limit=300, route_mip_gap=0.08,
                distance_scale_source="explicit test assumption"))
        self.mapping = {**copy.deepcopy(MAPPING), "instance_id": "G20-5-1a__Q70",
                        "scenario_id": "actual_provider_mapping"}
        self.expected = {key: self.mapping[key] for key in (
            "scenario_id", "instance_id", "model_id", "model_version", "opened_depots",
            "assignment", "distance_scale_km", "distance_scale_source")}
        (self.folder / "problem.txt").write_text(batch.problem_text(self.expected), encoding="utf-8")

    def execute(self):
        return batch.map_live_request(self.args, self.expected, self.folder, self.manifest, self.output)

    def test_calls_jev_rag_and_llm_and_uses_llm_mapping(self):
        self.expected["scenario_id"] = "requested_task"
        with patch("orllm.jev.select_with_jev", return_value=copy.deepcopy(SELECTION)) as jev, \
                patch("orllm.rag.retrieve", return_value=copy.deepcopy(CONTEXT)) as rag, \
                patch("orllm.llm.extract_mapping", return_value=self.mapping) as llm:
            scenario, pipeline = self.execute()
        jev.assert_called_once()
        rag.assert_called_once()
        llm.assert_called_once()
        self.assertEqual(scenario["scenario_id"], "actual_provider_mapping")
        self.assertTrue(pipeline["mapping_matches_request"])
        self.assertEqual(pipeline["api_calls"], 2)
        self.assertEqual(self.manifest["api_calls"], 2)
        for name in ("selection.json", "retrieval.json", "mapping.json", "scenario.json"):
            self.assertTrue((self.folder / name).is_file())

    def test_low_confidence_does_not_map_or_force_expected_model(self):
        with patch("orllm.jev.select_with_jev", return_value={**SELECTION, "confidence": 0.1}) as jev, \
                patch("orllm.rag.retrieve") as rag, patch("orllm.llm.extract_mapping") as llm:
            with self.assertRaisesRegex(ValueError, "JEV"):
                self.execute()
        jev.assert_called_once()
        rag.assert_not_called()
        llm.assert_not_called()
        selected = json.loads((self.folder / "selection.json").read_text(encoding="utf-8"))
        self.assertEqual(selected["status"], "needs_review")
        self.assertIsNone(selected["model_id"])

    def test_resume_reuses_real_provider_records_without_new_calls(self):
        with patch("orllm.jev.select_with_jev", return_value=copy.deepcopy(SELECTION)), \
                patch("orllm.rag.retrieve", return_value=copy.deepcopy(CONTEXT)), \
                patch("orllm.llm.extract_mapping", return_value=self.mapping):
            self.execute()
        self.args.resume = True
        with patch("orllm.jev.select_with_jev") as jev, patch("orllm.rag.retrieve") as rag, \
                patch("orllm.llm.extract_mapping") as llm:
            scenario, pipeline = self.execute()
        jev.assert_not_called()
        rag.assert_not_called()
        llm.assert_not_called()
        self.assertEqual(self.manifest["api_calls"], 2)
        self.assertTrue(pipeline["provider_records_reused"])
        self.assertEqual(pipeline["api_calls_this_resume"], 0)
        self.assertEqual(scenario["scenario_id"], self.mapping["scenario_id"])

    def test_changed_upstream_assignment_is_rejected_before_solve(self):
        self.expected.update(model_id="M3", opened_depots=["d1", "d2"], assignment={"c1": "d1"})
        mapping = {**self.mapping, "model_id": "M3", "opened_depots": ["d1", "d2"],
                   "assignment": [{"customer_id": "c1", "depot_id": "d2"}]}
        (self.folder / "problem.txt").write_text(batch.problem_text(self.expected), encoding="utf-8")
        with patch("orllm.jev.select_with_jev", return_value={**SELECTION, "model_id": "M3"}), \
                patch("orllm.rag.retrieve", return_value=CONTEXT), \
                patch("orllm.llm.extract_mapping", return_value=mapping), \
                patch("main.run_build") as solve:
            with self.assertRaisesRegex(ValueError, "assignment"):
                self.execute()
        solve.assert_not_called()
        self.assertFalse((self.folder / "scenario.json").exists())
        pipeline = json.loads((self.folder / "pipeline_report.json").read_text(encoding="utf-8"))
        self.assertEqual(pipeline["status"], "error")

    def test_resume_can_relax_gap_and_records_previous_settings(self):
        previous = {**self.manifest, "instances": [],
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "settings": dict(pipeline="fixed", solver="copt", distance_scale_km=0.1,
                                     route_mip_gap=0.01, assignment_time_limit=600,
                                     route_time_limit=1800, distance_scale_source="test assumption")}
        batch.write_json(self.output / "batch_manifest.json", previous)
        args = argparse.Namespace(output=str(self.output), pipeline="fixed", resume=True,
                                  instances=[], solver="copt", distance_scale_km=0.1,
                                  route_mip_gap=0.05, assignment_mip_gap=0.08, assignment_time_limit=600,
                                  route_time_limit=1800)
        self.assertEqual(batch.run(args), 0)
        saved = batch.read_json(self.output / "batch_manifest.json")
        self.assertEqual(saved["settings"]["route_mip_gap"], 0.05)
        self.assertEqual(saved["settings"]["assignment_mip_gap"], 0.08)
        self.assertEqual(saved["settings_history"][-1]["previous_settings"]["route_mip_gap"], 0.01)

    def test_loose_gap_keeps_M1_allocation_when_M2_native_incumbent_is_worse(self):
        instance = dict(model_id="M2", model_version="1.0", instance_id="checked_assignment",
                        D=["d1", "d2"], C=["c1", "c2"], q={"c1": 2, "c2": 3},
                        U={"d1": 5, "d2": 5}, g={"d1": {"c1": 10, "c2": 10},
                                               "d2": {"c1": 10.5, "c2": 10.5}})
        model = build_assignment(instance, False)
        initial = batch.pipeline_app.reference_assignment_start(model,
            [{"customer_id": c, "depot_id": "d1"} for c in instance["C"]])
        worse = batch.pipeline_app.reference_assignment_start(model,
            [{"customer_id": c, "depot_id": "d2"} for c in instance["C"]])
        native = model._solution("optimal", "Gap tolerance reached", worse["values"],
                                 solver="copt", mip_dual_bound=19.9, mip_gap=(21-19.9)/21)
        native["objective_check"] = {"valid": True}
        result = batch.pipeline_app.best_checked_incumbent(model, native, initial=initial)
        self.assertEqual(result["objective"], 20)
        self.assertLessEqual(result["mip_gap"], 0.08)
        self.assertEqual(result["incumbent_source"], "independently_checked_M1_assignment")
        self.assertTrue(result["primal_check"]["valid"])

    def test_invalid_M1_allocation_is_not_used_as_M2_start(self):
        instance = dict(model_id="M2", model_version="1.0", instance_id="invalid_assignment",
                        D=["d1"], C=["c1", "c2"], q={"c1": 2, "c2": 3},
                        U={"d1": 4}, g={"d1": {"c1": 10, "c2": 10}})
        model = build_assignment(instance, False)
        full = [{"customer_id": c, "depot_id": "d1"} for c in instance["C"]]
        for rows in (full, full[:1], full + full[:1],
                     [{"customer_id": c, "depot_id": "d2"} for c in instance["C"]]):
            with self.subTest(rows=rows):
                self.assertIsNone(batch.pipeline_app.reference_assignment_start(model, rows))


if __name__ == "__main__":
    unittest.main()
