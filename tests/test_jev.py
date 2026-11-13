"""Offline TypeSafe adapter checks: SDK mocked, no paid requests or solving."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from orllm.jev import select_with_jev


ROOT = Path(__file__).resolve().parents[1]
KEY = "test-secret-never-print"


def response(choice="M1", confidence=0.9):
    probabilities = {option: (0.9 if option == choice else 0.025)
                     for option in ("M1", "M2", "M3", "M4", "no_match")}
    return SimpleNamespace(model="jev-1.13.0", request_id="req-offline",
                           usage={"input_tokens": 20, "output_tokens": 3},
                           choices={"expert_model": SimpleNamespace(
                               choice=choice, confidence=confidence, probabilities=probabilities)})


class JevTests(unittest.TestCase):
    def setUp(self):
        self.client = SimpleNamespace(system_one=MagicMock(return_value=response()), close=MagicMock())
        self.factory = MagicMock(return_value=self.client)
        self.choice = MagicMock(side_effect=lambda **kwargs: kwargs)
        self.retry = MagicMock(side_effect=lambda **kwargs: kwargs)
        self.sdk = SimpleNamespace(TypeSafeClient=self.factory, Choice=self.choice, RetryPolicy=self.retry)
        self.env = patch.dict(os.environ, {"TYPESAFE_API_KEY": KEY})
        self.modules = patch.dict(sys.modules, {"typesafe_sdk": self.sdk})
        self.env.start()
        self.modules.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.modules.stop)

    def test_selected_and_catalog_driven_request(self):
        result = select_with_jev(ROOT, "需要选择开放仓库并分配客户，不安排路线。", {})
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["model_id"], "M1")
        self.assertEqual(result["proposed_model_id"], "M1")
        self.assertEqual(result["model_version"], "1.0")
        self.assertEqual(result["provider"], "typesafe")
        self.assertEqual(result["provider_model"], "jev-1.13.0")
        self.assertNotIn(KEY, json.dumps(result))
        request = self.client.system_one.call_args.kwargs
        catalog = json.loads((ROOT / "models/model_catalog.json").read_text(encoding="utf-8"))
        criteria = request["questions"]["expert_model"]["criteria"]
        self.assertEqual(set(criteria), {"M1", "M2", "M3", "M4", "no_match"})
        for model in catalog["models"]:
            self.assertEqual(criteria[model["model_id"]]["match_when"], model["match_when"])
            self.assertEqual(criteria[model["model_id"]]["assumptions"], model["assumptions"])
        self.assertEqual(request["state"]["shared_conventions"], catalog["shared_conventions"])
        self.factory.assert_called_once_with(api_key=KEY, base_url="https://api.typesafe.ai",
                                             model="jev-latest", timeout=30.0, retry={"max_retries": 0})
        self.client.close.assert_called_once()

    def test_gate_low_confidence_and_no_match_without_fallback(self):
        for choice, confidence, expected in [("M2", 0.69, "needs_review"),
                                             ("M3", 0.7, "selected"),
                                             ("no_match", 0.3, "no_match")]:
            with self.subTest(choice=choice):
                self.client.system_one.return_value = response(choice, confidence)
                result = select_with_jev(ROOT, "业务描述", {})
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["model_id"], choice if expected == "selected" else None)
                self.assertEqual(result["proposed_model_id"], None if choice == "no_match" else choice)
        self.assertEqual(self.client.system_one.call_count, 3)

    def test_answers_sdk_compatibility(self):
        value = response()
        value.answers = value.choices
        del value.choices
        self.client.system_one.return_value = value
        self.assertEqual(select_with_jev(ROOT, "业务描述", {})["model_id"], "M1")

    def test_bad_config_rejected_before_sdk_call(self):
        configs = [{"base_url": url} for url in ["http://api.typesafe.ai", "https://user:secret@api.typesafe.ai",
                   "https://api.typesafe.ai?token=secret", "https://api.typesafe.ai/#x", "https://api.typesafe.ai:bad",
                   "file:///tmp/service", "https://api.type safe.ai", "https://api.typesafe.ai\\@evil.example"]]
        configs += [{"confidence_threshold": value} for value in [True, "0.7", -0.1, 1.1, float("nan"), float("inf")]]
        configs += [{"timeout_seconds": value} for value in [0, -1, True, float("inf")]]
        configs += [{"model": ""}, {"api_key_env": "BAD KEY"}, {"api_key": KEY}]
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                select_with_jev(ROOT, "业务描述", config)
        self.factory.assert_not_called()

    def test_empty_key_or_description_rejected(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}), self.assertRaises(ValueError):
            select_with_jev(ROOT, "业务描述", {})
        for description in ["", "  ", None]:
            with self.subTest(description=description), self.assertRaises(ValueError):
                select_with_jev(ROOT, description, {})
        self.factory.assert_not_called()

    def test_invalid_provider_response_never_selects(self):
        invalid = []
        for field, value in [("choice", "invented"), ("choice", []), ("confidence", float("nan")),
                             ("confidence", 1.1), ("confidence", "0.99"), ("confidence", True),
                             ("probabilities", {}), ("probabilities", {"M1": 0.9}),
                             ("probabilities", {"M1": 0.9, "invented": 0.1}),
                             ("probabilities", {"M1": float("inf")}),
                             ("probabilities", {"M1": 1.1, "M2": -0.1})]:
            value_response = response()
            setattr(value_response.choices["expert_model"], field, value)
            invalid.append(value_response)
        invalid.append(SimpleNamespace(choices={}))
        wrong_choice = response("M2")
        wrong_choice.choices["expert_model"].choice = "M1"
        invalid.append(wrong_choice)
        for value in invalid:
            with self.subTest(value=value):
                self.client.system_one.return_value = value
                with self.assertRaisesRegex(RuntimeError, "invalid selection response"):
                    select_with_jev(ROOT, "业务描述", {})

    def test_sdk_failures_hide_secrets_and_close_client(self):
        self.client.system_one.side_effect = RuntimeError("Authorization: Bearer " + KEY)
        with self.assertRaises(RuntimeError) as error:
            select_with_jev(ROOT, "业务描述", {})
        self.assertNotIn(KEY, str(error.exception))
        self.assertTrue(error.exception.__suppress_context__)
        self.client.close.assert_called_once()
        self.assertEqual(self.client.system_one.call_count, 1)

    def test_catalog_changes_are_used(self):
        catalog = json.loads((ROOT / "models/model_catalog.json").read_text(encoding="utf-8"))
        catalog = copy.deepcopy(catalog)
        catalog["models"][0]["match_when"] = "changed catalog condition"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            (root / "models/model_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            select_with_jev(root, "业务描述", {})
        criteria = self.choice.call_args.kwargs["criteria"]
        self.assertEqual(criteria["M1"]["match_when"], "changed catalog condition")

    def test_request_errors_keep_safe_status_without_secrets(self):
        for status in (401, 403, 429, 500, 503):
            failure = RuntimeError("Authorization: Bearer " + KEY)
            failure.status = status
            self.client.system_one.side_effect = failure
            with self.subTest(status=status), self.assertRaises(RuntimeError) as error:
                select_with_jev(ROOT, "业务描述", {})
            self.assertIn(f"HTTP {status}", str(error.exception))
            self.assertNotIn(KEY, str(error.exception))
            self.assertTrue(error.exception.__suppress_context__)
        for failure, expected in ((TimeoutError(KEY), "超时"), (ConnectionError(KEY), "连接失败")):
            self.client.system_one.side_effect = failure
            with self.assertRaises(RuntimeError) as error:
                select_with_jev(ROOT, "业务描述", {})
            self.assertIn(expected, str(error.exception))
            self.assertNotIn(KEY, str(error.exception))


class JevSdkTransportTests(unittest.TestCase):
    def test_real_sdk_with_offline_http_transport(self):
        try:
            import httpx2
            import typesafe_sdk
        except ImportError:
            self.skipTest("Optional TypeSafe SDK is not installed")
        real_client = typesafe_sdk.TypeSafeClient
        requests = []

        def respond(request):
            content = request.content
            # The SDK uses stdlib zstd automatically on Python 3.14; emulate
            # the provider's HTTP decoding before inspecting the JSON body.
            if request.headers.get("Content-Encoding") == "zstd":
                try:
                    from compression.zstd import decompress
                except ImportError:
                    from backports.zstd import decompress
                content = decompress(content)
            requests.append(json.loads(content))
            return httpx2.Response(200, json={
                "model": "jev-offline",
                "answers": {"expert_model": {"type": "choice", "choice": "M1", "confidence": 0.9,
                            "probabilities": {"M1": 0.9, "M2": 0.025, "M3": 0.025, "M4": 0.025, "no_match": 0.025}}},
                "usage": {"input_tokens": 10, "output_tokens": 1}})

        def client(**kwargs):
            return real_client(**kwargs, transport=httpx2.MockTransport(respond))

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": KEY}), patch.object(typesafe_sdk, "TypeSafeClient", client):
            result = select_with_jev(ROOT, "Select warehouses and customer assignments", {})
        self.assertEqual(result["model_id"], "M1")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["questions"]["expert_model"]["type"], "choice")


if __name__ == "__main__":
    unittest.main()
