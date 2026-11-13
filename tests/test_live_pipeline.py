"""Exercise the default build-only live pipeline offline with mocked providers."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
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
    "scenario_id": "offline_live_pipeline", "instance_id": "20-5-1a__Q70",
    "model_id": "M1", "model_version": "1.0", "opened_depots": None,
    "assignment": None, "distance_scale_km": None, "distance_scale_source": None,
    "status": "ready", "missing_fields": [], "unsupported_requirements": [],
    "evidence": [{"field": "instance_id", "source": "user_description",
                  "excerpt": "使用 20-5-1a__Q70 实例"}],
}
CONTEXT = [{"id": "offline-m1-rule", "model_id": "M1", "model_version": "1.0",
            "backend": "chroma", "text": "仓库开放与客户分配是 M1 的决策。"}]


class LivePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.description = self.folder / "description.txt"
        self.description.write_text("使用 20-5-1a__Q70 实例，决定仓库开放和客户归属。", encoding="utf-8")
        self.config = self.folder / "offline-config.json"
        self.config.write_text("{}", encoding="utf-8")

    def args(self, output="result", *extra):
        # Explicit empty configs keep tests independent of private local settings.
        return main.make_parser().parse_args([
            "run", "--description-file", str(self.description),
            "--jev-config", str(self.config), "--config", str(self.config),
            "--rag-config", str(self.config), "--output", str(self.folder / output),
            *extra,
        ])

    def execute(self, args):
        with redirect_stdout(io.StringIO()):
            return args.func(args)

    def read(self, output, filename):
        return main.read_json(self.folder / output / filename)

    def test_selection_stop_states_never_retrieve_map_or_build(self):
        for status in ("needs_review", "no_match"):
            with self.subTest(status=status):
                selection = {**SELECTION, "status": status, "model_id": None}
                with patch("orllm.jev.select_with_jev", return_value=selection) as select, \
                        patch("orllm.rag.retrieve") as retrieve, \
                        patch("orllm.llm.extract_mapping") as extract, \
                        patch("main.run_build") as build:
                    self.assertEqual(self.execute(self.args(status)), 2)
                select.assert_called_once()
                retrieve.assert_not_called()
                extract.assert_not_called()
                build.assert_not_called()
                self.assertEqual(self.read(status, "selection.json")["status"], status)
                report = self.read(status, "build_report.json")
                self.assertEqual(report["status"], status)
                self.assertEqual(report["solver"], "none")
                self.assertEqual(report["artifacts"], [])
                self.assertEqual(self.read(status, "pipeline_report.json")["stage"], "selection")
                self.assertFalse((self.folder / status / "retrieval.json").exists())

    def test_low_confidence_selected_result_is_stopped(self):
        selection = {**SELECTION, "confidence": 0.3}
        with patch("orllm.jev.select_with_jev", return_value=selection), \
                patch("orllm.rag.retrieve") as retrieve, \
                patch("orllm.llm.extract_mapping") as extract, \
                patch("main.run_build") as build:
            self.assertEqual(self.execute(self.args()), 2)
        retrieve.assert_not_called()
        extract.assert_not_called()
        build.assert_not_called()
        selected = self.read("result", "selection.json")
        self.assertEqual(selected["status"], "needs_review")
        self.assertIsNone(selected["model_id"])
        self.assertEqual(selected["proposed_model_id"], "M1")

    def test_invalid_selection_never_reaches_downstream(self):
        invalid = [
            {"model_id": "M999"}, {"model_version": "2.0"},
            {"confidence": float("nan")}, {"confidence": float("inf")},
            {"confidence": True}, {"threshold": -0.1}, {"status": "ready"}, {"status": []},
        ]
        for number, change in enumerate(invalid):
            output = f"bad-selection-{number}"
            with self.subTest(change=change), \
                    patch("orllm.jev.select_with_jev", return_value={**SELECTION, **change}), \
                    patch("orllm.rag.retrieve") as retrieve, \
                    patch("orllm.llm.extract_mapping") as extract, \
                    patch("main.run_build") as build:
                with self.assertRaises(ValueError):
                    self.execute(self.args(output))
            retrieve.assert_not_called()
            extract.assert_not_called()
            build.assert_not_called()
            report = self.read(output, "pipeline_report.json")
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["stage"], "selection")
            self.assertEqual(report["artifacts"], ["build_report.json"])
            self.assertEqual(self.read(output, "build_report.json")["artifacts"], [])
            self.assertFalse(list((self.folder / output).glob("*.lp")))

    def test_mapping_stop_states_never_build(self):
        cases = [
            ("needs_clarification", {"instance_id": None, "missing_fields": ["instance_id"]}),
            ("unsupported", {"unsupported_requirements": ["客户时间窗"]}),
        ]
        for status, change in cases:
            mapping = {**MAPPING, **change, "status": status}
            with self.subTest(status=status), \
                    patch("orllm.jev.select_with_jev", return_value=copy.deepcopy(SELECTION)), \
                    patch("orllm.rag.retrieve", return_value=CONTEXT) as retrieve, \
                    patch("orllm.llm.extract_mapping", return_value=mapping) as extract, \
                    patch("main.run_build") as build:
                self.assertEqual(self.execute(self.args(status)), 2)
            retrieve.assert_called_once()
            extract.assert_called_once()
            build.assert_not_called()
            report = self.read(status, "build_report.json")
            self.assertEqual(report["status"], status)
            self.assertEqual(report["missing_fields"], mapping["missing_fields"])
            self.assertEqual(report["unsupported_requirements"], mapping["unsupported_requirements"])
            self.assertEqual(report["artifacts"], [])
            self.assertEqual(self.read(status, "mapping.json"), mapping)
            self.assertFalse(list((self.folder / status).glob("*.lp")))

    def test_llm_model_version_and_instance_are_revalidated(self):
        invalid = [
            {"model_id": "M2"}, {"model_version": "2.0"},
            {"instance_id": "invented-instance"}, {"missing_fields": ["capacity"]},
            {"unexpected": "field"},
        ]
        for number, change in enumerate(invalid):
            output = f"bad-mapping-{number}"
            with self.subTest(change=change), \
                    patch("orllm.jev.select_with_jev", return_value=copy.deepcopy(SELECTION)), \
                    patch("orllm.rag.retrieve", return_value=CONTEXT), \
                    patch("orllm.llm.extract_mapping", return_value={**MAPPING, **change}), \
                    patch("main.run_build") as build:
                with self.assertRaises(ValueError):
                    self.execute(self.args(output))
            build.assert_not_called()
            report = self.read(output, "pipeline_report.json")
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["stage"], "mapping")
            self.assertFalse((self.folder / output / "mapping.json").exists())
            self.assertFalse(list((self.folder / output).glob("*.lp")))

    def test_retrieval_failure_does_not_call_llm_or_build(self):
        with patch("orllm.jev.select_with_jev", return_value=copy.deepcopy(SELECTION)), \
                patch("orllm.rag.retrieve", side_effect=ValueError("Index needs rebuilding")), \
                patch("orllm.llm.extract_mapping") as extract, \
                patch("main.run_build") as build:
            with self.assertRaisesRegex(ValueError, "Index needs rebuilding"):
                self.execute(self.args())
        extract.assert_not_called()
        build.assert_not_called()
        report = self.read("result", "pipeline_report.json")
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["stage"], "retrieval")

    def test_happy_path_exports_real_model_without_solving(self):
        real_build = main.run_build
        with patch("orllm.jev.select_with_jev", return_value=copy.deepcopy(SELECTION)) as select, \
                patch("orllm.rag.retrieve", return_value=CONTEXT) as retrieve, \
                patch("orllm.llm.extract_mapping", return_value=copy.deepcopy(MAPPING)) as extract, \
                patch("main.run_build", wraps=real_build) as build, \
                patch.object(LinearModel, "solve_copt", side_effect=AssertionError("Must not solve")) as copt:
            self.assertEqual(self.execute(self.args()), 0)
        select.assert_called_once()
        retrieve.assert_called_once()
        self.assertEqual(retrieve.call_args.kwargs["backend"], "model_pack")
        self.assertEqual(retrieve.call_args.kwargs["version"], "1.0")
        self.assertEqual(retrieve.call_args.args[2], "M1")
        extract.assert_called_once()
        self.assertEqual(extract.call_args.args[2], "M1")
        self.assertEqual(extract.call_args.args[3], [{**r, 'retrieval_role': 'required'} for r in CONTEXT])
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs["solver"], "none")
        copt.assert_not_called()
        folder = self.folder / "result"
        report = self.read("result", "build_report.json")
        self.assertEqual(report["status"], "built")
        self.assertEqual(report["solver"], "none")
        self.assertEqual(len(report["artifacts"]), 1)
        for key in ("instance", "lp", "model_with_variable_mapping"):
            self.assertTrue((folder / report["artifacts"][0][key]).is_file())
        self.assertNotIn("solution", report["artifacts"][0])
        self.assertFalse(list(folder.glob("*.solution.json")))
        self.assertEqual(self.read("result", "mapping.json"), MAPPING)
        self.assertEqual(self.read("result", "retrieval.json")["context"], [{**r, 'retrieval_role': 'required'} for r in CONTEXT])
        pipeline = self.read("result", "pipeline_report.json")
        self.assertEqual(pipeline["status"], "built")
        self.assertEqual(pipeline["solver"], "none")
        self.assertEqual(pipeline["artifacts"], ["selection.json", "retrieval.json", "mapping.json", "build_report.json"])

    def test_run_cli_accepts_explicit_solver_and_defaults_to_build_only(self):
        self.assertEqual(self.args().solve, "none")
        self.assertEqual(self.args("result", "--solve", "copt").solve, "copt")


if __name__ == "__main__":
    unittest.main()
