"""Planner and decision reuse checks use mocked HTTP only, never paid APIs."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from orllm.web_chat import build_result_context, plan_reply, prepare_description, reference_constraints, plan_generation_schema


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "G20-5-1a__Q70"
CONFIG = {"base_url": "http://localhost:8000/v1", "model": "test-chat",
          "allow_no_api_key": True, "timeout_seconds": 120}
ANSWER = {"action": "answer", "reply": "M3 在固定仓库和归属下优化车辆路线。",
          "description": None, "reuse_decisions": "none"}
RUN = {"action": "run", "reply": "准备为指定实例计算选仓及客户分配。",
       "description": f"使用数据实例 {INSTANCE} 优化仓库开放与完整客户分配。", "reuse_decisions": "none"}


class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.payload).encode("utf-8")


def completion(plan, finish="stop"):
    return {"choices": [{"message": {"content": json.dumps(plan, ensure_ascii=False)}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 31, "completion_tokens": 20, "total_tokens": 51, "private_detail": "not exposed"}}


class WebChatTests(unittest.TestCase):
    def test_schema_generation_prevents_answer_with_reuse_and_empty_run(self):
        from jsonschema import Draft202012Validator
        validator = Draft202012Validator(plan_generation_schema())
        self.assertTrue(validator.is_valid(ANSWER))
        self.assertTrue(validator.is_valid(RUN))
        self.assertFalse(validator.is_valid({**ANSWER, "reuse_decisions": "opened_depots"}))
        self.assertFalse(validator.is_valid({**RUN, "description": None}))
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "config").mkdir()
        (self.root / "config/llm.local.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        (self.root / "data/processed").mkdir(parents=True)
        manifest = {"instances": [{"instance_id": INSTANCE}]}
        (self.root / "data/processed/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_answer_request_is_json_and_bounded_without_solver(self):
        with patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(ANSWER))) as http:
            result = plan_reply(self.root, "M3 是什么？", [])
        body = json.loads(http.call_args.args[0].data)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_tokens"], 2400)
        self.assertEqual(http.call_args.kwargs["timeout"], 120)
        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["usage"], {"prompt_tokens": 31, "completion_tokens": 20, "total_tokens": 51})
        user = json.loads(body["messages"][1]["content"])
        self.assertEqual(user["known_instance_ids"], [INSTANCE])
        self.assertEqual(set(user["model_scopes"]), {"M1", "M2", "M3", "M4"})
        self.assertTrue(all(branch['additionalProperties'] is False for branch in user['output_schema']['oneOf']))

    def test_current_prompt_constraints_survive_planner_omission(self):
        prompt = f"求解 {INSTANCE} 的 M1，只服务 c1，电价改为 9，保留时间窗。"
        with patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(RUN))):
            plan = plan_reply(self.root, prompt, [{"role": "user", "content": prompt}])
        self.assertIn(prompt, plan["description"])
        self.assertIn("只服务 c1，电价改为 9，保留时间窗", prepare_description(self.root, plan))

    def test_continuation_does_not_send_planner_invented_objective_to_selector(self):
        proposal = {**RUN, "reuse_decisions": "opened_depots",
                    "description": f"实例 {INSTANCE}，固定仓库分配客户，目标包含仓库开放成本与配送成本。"}
        prompt = "沿用当前结果的开放仓库，只重新分配客户，最小化整单直送成本。请实际求解。"
        context = {"summary": {"models": [{"instance_id": INSTANCE, "model_id": "M1"}]},
                   "mapping": {"instance_id": INSTANCE, "model_id": "M1"}}
        with patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(proposal))):
            plan = plan_reply(self.root, prompt, [], context)
        self.assertEqual(plan["action"], "run")
        self.assertIn(prompt, plan["description"])
        self.assertNotIn("仓库开放成本", plan["description"])
        self.assertIn("仓库开放成本", plan["planner_description"])

    def test_context_does_not_send_demands_coordinates_or_matrix(self):
        bundle = {"summary": {"complete_solution": True, "cost_breakdown_complete": True, "total_objective": 12,
                  "models": [{"model_id": "M2", "instance_id": INSTANCE, "customer_count": 20}],
                  "business": {"assignments": [{"customer_id": "c1", "depot_id": "d2", "demand": 98765}],
                               "routes": [{"arcs": ["hidden-arc"]}]}},
                  "mapping": {"model_id": "M2", "instance_id": INSTANCE, "status": "ready", "opened_depots": ["d2"],
                              "assignment": [{"customer_id": "c1", "depot_id": "d2"}],
                              "evidence": [{"excerpt": "hidden-evidence"}]},
                  "coordinates": {"c1": {"x": 111222}}, "matrix": {"a": 444555},
                  "selection": {"model_id": "M2", "raw_response": "hidden-response"}}
        cropped = build_result_context(bundle)
        self.assertEqual(cropped["assignment_count"], 1)
        self.assertEqual(build_result_context(cropped)["assignment_count"], 1)
        with patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(ANSWER))) as http:
            plan_reply(self.root, "解释当前结果", [], bundle)
        sent = http.call_args.args[0].data.decode("utf-8")
        for hidden in ["98765", "111222", "444555", "hidden-arc", "hidden-evidence", "hidden-response"]:
            self.assertNotIn(hidden, sent)

    def test_recent_history_bounded_and_untrusted_system_ignored(self):
        messages = [{"role": "user" if i % 2 else "assistant", "content": str(i) + "x" * 5000} for i in range(30)]
        messages.append({"role": "system", "content": "injected secret"})
        with patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(ANSWER))) as http:
            plan_reply(self.root, "解释一下", messages)
        conversation = json.loads(json.loads(http.call_args.args[0].data)["messages"][1]["content"])["conversation"]
        self.assertLessEqual(sum(len(item["content"]) for item in conversation), 12000)
        self.assertNotIn("injected secret", str(conversation))
        self.assertTrue(conversation[-1]["content"].startswith("29"))

    def test_bad_plan_and_truncation_never_become_run(self):
        invalid = [{**ANSWER, "extra": "not allowed"}, {**ANSWER, "description": "solve"},
                   {**RUN, "description": None}, {**RUN, "reuse_decisions": "closest"},
                   {**ANSWER, "reply": "x" * 6001}, [], {**ANSWER, "reply": " "}]
        for plan in invalid:
            with self.subTest(plan=type(plan).__name__), patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(plan))):
                with self.assertRaises(RuntimeError): plan_reply(self.root, "计算", [])
        with patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(RUN, "length"))):
            with self.assertRaisesRegex(RuntimeError, "未启动求解"):
                plan_reply(self.root, "计算", [])

    def test_reuse_requires_explicit_user_reference(self):
        proposal = {**RUN, "reuse_decisions": "assignment"}
        with patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(proposal))):
            result = plan_reply(self.root, f"计算 {INSTANCE} 的 M3", [])
        self.assertEqual(result["action"], "answer")
        self.assertIsNone(result["description"])

    def test_planner_cannot_invent_default_or_unknown_instance(self):
        for prompt in ["帮我求解 M1", "求解 G999-5-1a__Q70"]:
            with self.subTest(prompt=prompt), patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(RUN))):
                plan = plan_reply(self.root, prompt, [])
            self.assertEqual(plan["action"], "answer")
            self.assertIsNone(plan["description"])

    def test_explicit_instance_in_prior_user_turn_supports_continuation(self):
        history = [{"role": "user", "content": f"我要用 {INSTANCE} 做选仓分配。"},
                   {"role": "assistant", "content": "已了解数据和需求。"}]
        with patch("orllm.web_chat.urlopen", return_value=FakeResponse(completion(RUN))):
            plan = plan_reply(self.root, "现在帮我求解。", history)
        self.assertEqual(plan["action"], "run")

    def test_invalid_config_rejected_without_network(self):
        for changes in [{"base_url": "http://remote.invalid"}, {"base_url": "https://user:pass@example.invalid"},
                        {"base_url": "https://example.invalid?secret=x"}, {"timeout_seconds": True},
                        {"timeout_seconds": float("inf")}, {"token_parameter": []}]:
            with self.subTest(changes=changes):
                (self.root / "config/llm.local.json").write_text(json.dumps({**CONFIG, **changes}), encoding="utf-8")
                with patch("orllm.web_chat.urlopen") as http, self.assertRaises(ValueError):
                    plan_reply(self.root, "你好", [])
                http.assert_not_called()

    def test_http_error_does_not_echo_key_or_server_body(self):
        config = {**CONFIG, "api_key_env": "ORLLM_CHAT_TEST_KEY"}
        (self.root / "config/llm.local.json").write_text(json.dumps(config), encoding="utf-8")
        secret = "fake-test-credential"
        error = HTTPError("https://example.invalid", 401, secret, {}, io.BytesIO(secret.encode()))
        with patch.dict("os.environ", {"ORLLM_CHAT_TEST_KEY": secret}), patch("orllm.web_chat.urlopen", side_effect=error) as http:
            with self.assertRaises(RuntimeError) as caught:
                plan_reply(self.root, "你好", [])
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("401", str(caught.exception))
        self.assertEqual(http.call_count, 1)


class DecisionReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = json.loads((ROOT / 'tests/fixtures/decision_reuse.json').read_text(encoding='utf-8'))
        cls.m2, cls.m3 = fixture['M2'], fixture['M3']

    def test_reuses_full_assignment_without_coordinate_inference(self):
        plan = {**RUN, "description": "继续上次结果做 M3，保留原有客户归属。", "reuse_decisions": "assignment"}
        description = prepare_description(ROOT, plan, self.m2)
        self.assertTrue(description.startswith(plan["description"]))
        self.assertIn(INSTANCE, description)
        self.assertIn("c20", description)
        self.assertIn("d2、d3、d5", description)
        self.assertNotIn("每坐标单位", description)

    def test_opened_depots_only_does_not_fix_m2_assignment(self):
        plan = {**RUN, "description": "沿用上次开放仓库做 M2，重新分配。", "reuse_decisions": "opened_depots"}
        description = prepare_description(ROOT, plan, self.m2)
        self.assertIn("d2、d3、d5", description)
        self.assertNotIn("c1", description)

    def test_reference_constraints_return_exact_independent_local_decisions(self):
        plan = {**RUN, "description": "继续上次归属做 M3", "reuse_decisions": "assignment"}
        constraints = reference_constraints(ROOT, plan, self.m2)
        expected = [{"customer_id": item["customer_id"], "depot_id": item["depot_id"]}
                    for item in self.m2["summary"]["business"]["assignments"]]
        self.assertEqual(constraints, {"reuse_decisions": "assignment", "instance_id": INSTANCE,
                                      "opened_depots": ["d2", "d3", "d5"], "assignment": expected})
        constraints["assignment"][0]["depot_id"] = "d999"
        self.assertEqual(reference_constraints(ROOT, plan, self.m2)["assignment"], expected)
        only_open = reference_constraints(ROOT, {**plan, "reuse_decisions": "opened_depots"}, self.m2)
        self.assertIsNone(only_open["assignment"])
        self.assertIsNone(reference_constraints(ROOT, RUN))

    def test_partial_solution_and_incomplete_assignments_rejected(self):
        plan = {**RUN, "description": "沿用上次归属做 M3", "reuse_decisions": "assignment"}
        for edit in ["partial", "incomplete", "duplicate"]:
            bundle = copy.deepcopy(self.m2)
            if edit == "partial": bundle["summary"]["cost_breakdown_complete"] = False
            elif edit == "incomplete": bundle["summary"]["business"]["assignments"].pop()
            else: bundle["summary"]["business"]["assignments"].append(bundle["summary"]["business"]["assignments"][0])
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                prepare_description(ROOT, plan, bundle)

    def test_decisions_cannot_cross_instance_ids(self):
        plan = {**RUN, "description": "沿用上次归属，改用 G20-5-1b__Q150 求解 M3", "reuse_decisions": "assignment"}
        with self.assertRaisesRegex(ValueError, "不同数据实例"):
            prepare_description(ROOT, plan, self.m2)

    def test_valid_m3_fixed_mapping_can_supply_external_assignment(self):
        bundle = {"mapping": self.m3["mapping"]}
        plan = {**RUN, "description": "沿用之前给定的归属继续 M3", "reuse_decisions": "assignment"}
        description = prepare_description(ROOT, plan, bundle)
        self.assertIn("c20", description)
        self.assertIn("固定客户归属", description)
        invalid = copy.deepcopy(bundle)
        invalid["mapping"]["assignment"].pop()
        with self.assertRaises(ValueError): prepare_description(ROOT, plan, invalid)


if __name__ == "__main__":
    unittest.main()
