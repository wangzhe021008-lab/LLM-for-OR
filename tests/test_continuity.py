import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from orllm.data import load_base
from orllm.rag import load_runtime_knowledge, retrieve, retrieve_for_mapping
from orllm.web_chat import build_planning_context, build_result_context, plan_reply, reference_constraints, inheritance_contract, plan_generation_schema, request_precheck
from orllm.workflow import fingerprint, model_version
from test_web_chat import CONFIG, ANSWER, RUN, FakeResponse, completion

ROOT = Path(__file__).resolve().parents[1]


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.bundle = json.loads((ROOT/'tests/fixtures/decision_reuse.json').read_text(encoding='utf-8'))['M2']
        self.bundle['id'] = 'runs/checked-example'
        self.iid = self.bundle['mapping']['instance_id']

    def test_full_decisions_are_verified_before_planning_and_survive_cropping(self):
        context = build_planning_context(ROOT, self.bundle)
        proof = context['reference_state']
        self.assertEqual(proof['status'], 'verified')
        self.assertEqual(proof['assignment_count'], 20)
        self.assertEqual(sum(map(len, proof['assignment_by_depot'].values())), 20)
        self.assertIn('assignment', proof['reusable_decisions'])
        self.assertEqual(build_result_context(context)['reference_state'], proof)
        self.assertIn('prepared_instance', context['workflow_knowledge']['document']['data_locations'])

    def test_partial_duplicate_and_missing_assignments_cannot_claim_availability(self):
        for defect in ['missing', 'duplicate', 'partial']:
            b = copy.deepcopy(self.bundle)
            if defect == 'missing': b['summary']['business']['assignments'].pop()
            elif defect == 'duplicate': b['summary']['business']['assignments'].append(b['summary']['business']['assignments'][0])
            else: b['summary']['complete_solution'] = False
            with self.subTest(defect=defect):
                proof = build_planning_context(ROOT, b)['reference_state']
                self.assertEqual(proof['status'], 'blocked')
                self.assertEqual(proof['reusable_decisions'], [])

    def test_stale_data_model_tampered_revision_and_parameter_changes_block_reuse(self):
        base = load_base(ROOT, self.iid)
        state = {'scenario': {'model_id': 'M2', 'instance_id': self.iid,
                            'data_version': fingerprint(base), 'parameter_values': {}},
                 'model_fingerprint': model_version(ROOT, 'M2')}
        for defect in ['none', 'data', 'model', 'revision', 'parameters']:
            st = copy.deepcopy(state)
            if defect == 'data': st['scenario']['data_version'] = 'old'
            if defect == 'model': st['model_fingerprint'] = 'old'
            if defect == 'parameters': st['scenario']['parameter_values'] = {'h': 123}
            st['revision'] = fingerprint(st)
            if defect == 'revision': st['revision'] = 'tampered'
            b = {**self.bundle, 'binding_state': st}
            with self.subTest(defect=defect):
                proof = build_planning_context(ROOT, b)['reference_state']
                self.assertEqual(proof['status'], 'verified' if defect == 'none' else 'blocked')

    def test_new_instance_never_inherits_prior_assignments(self):
        plan = {**RUN, 'description': '沿用归属，改用 G20-5-1b__Q150 规划路线', 'reuse_decisions': 'assignment'}
        with self.assertRaisesRegex(ValueError, '不同数据实例'):
            reference_constraints(ROOT, plan, self.bundle)

    def test_planner_gets_actual_groups_and_authoritative_readiness(self):
        context = build_planning_context(ROOT, self.bundle)
        prompt = '沿用刚才的客户归属，继续优化配送路线。'
        planned = {**RUN, 'reuse_decisions': 'assignment', 'description': prompt}
        with patch('orllm.web_chat.urlopen', return_value=FakeResponse(completion(planned))) as http:
            plan = plan_reply(ROOT, prompt, [], context, config=CONFIG)
        payload = json.loads(json.loads(http.call_args.args[0].data)['messages'][1]['content'])
        self.assertEqual(payload['context']['reference_state']['assignment_count'], 20)
        self.assertTrue(payload['context']['reference_state']['assignment_by_depot'])
        self.assertEqual(plan['action'], 'run')
        self.assertEqual(plan['description'], prompt)

    def test_available_data_does_not_force_unsupported_or_question_into_execution(self):
        context = build_planning_context(ROOT, self.bundle)
        for prompt in ['解释当前结果，不要运行', '沿用结果，新增途中充电站和时间窗后求解']:
            with self.subTest(prompt=prompt), patch('orllm.web_chat.urlopen', return_value=FakeResponse(completion(ANSWER))):
                plan = plan_reply(ROOT, prompt, [], context, config=CONFIG)
                self.assertEqual(plan['action'], 'answer')

    def test_explicit_assignment_reuse_cannot_be_downgraded_by_planner(self):
        context = build_planning_context(ROOT, self.bundle)
        prompt = '沿用当前开放仓库和客户归属，继续优化路线。'
        wrong = {**RUN, 'description': prompt, 'reuse_decisions': 'opened_depots'}
        with patch('orllm.web_chat.urlopen', return_value=FakeResponse(completion(wrong))):
            plan = plan_reply(ROOT, prompt, [], context, config=CONFIG)
        self.assertEqual(plan['reuse_decisions'], 'assignment')
        self.assertEqual(plan['planner_reuse_decisions'], 'opened_depots')
        self.assertEqual(len(reference_constraints(ROOT, plan, self.bundle)['assignment']), 20)

    def test_negation_reassignment_and_conflicting_retention_are_not_frozen(self):
        proof = build_planning_context(ROOT, self.bundle)['reference_state']
        for prompt in ['不要沿用当前客户归属，重新分配客户', '不再保留客户分配，继续重新优化',
                       '只沿用当前开放仓库，重新分配客户', '继续重新确定客户归属']:
            with self.subTest(prompt=prompt):
                self.assertIsNone(inheritance_contract(prompt, proof)['reuse_decisions'])
        self.assertEqual(inheritance_contract('保留当前客户归属不变，不调整客户归属，继续优化路线', proof)['reuse_decisions'], 'assignment')
        self.assertEqual(inheritance_contract('沿用客户归属，但重新分配客户', proof)['status'], 'conflict')

    def test_dynamic_schema_locks_reuse_without_forcing_run_or_model(self):
        from jsonschema import Draft202012Validator
        schema = plan_generation_schema(inherited='assignment')
        validator = Draft202012Validator(schema)
        self.assertTrue(validator.is_valid(ANSWER))
        self.assertTrue(validator.is_valid({**RUN, 'reuse_decisions': 'assignment'}))
        self.assertFalse(validator.is_valid({**RUN, 'reuse_decisions': 'opened_depots'}))
        self.assertNotIn('model_id', str(schema))

    def test_clear_missing_ev_has_correct_grounded_explanation_without_http(self):
        context = build_planning_context(ROOT, self.bundle)
        from orllm.llm import instance_availability
        data = instance_availability(ROOT, ['50-5-2ab__Q150'])
        check = request_precheck('继续优化电动车路线，要考虑电池续航。请直接求解。', None, data, {})
        self.assertEqual(check['code'], 'missing_ev_parameters')
        self.assertIn('单位电量价格 p', check['reply'])
        self.assertIn('不能直接迁移当前归属', check['reply'])
        with patch('orllm.web_chat.urlopen') as http:
            trace = {}
            result = plan_reply(ROOT, '使用 50-5-2ab__Q150，规划电动车路线并考虑电池续航。请直接求解。', [], config=CONFIG, trace=trace)
        http.assert_not_called()
        self.assertEqual(result['action'], 'answer')
        self.assertEqual(trace['status'], 'not_called')

    def test_prechecks_do_not_block_questions_negated_requirements_or_normal_routes(self):
        from orllm.llm import instance_availability
        data = instance_availability(ROOT, ['50-5-2ab__Q150'])
        for prompt in ['为什么不能求解电动车路线？', '继续规划普通车辆路线，不考虑电耗和电池。',
                       '沿用仓库和客户归属，继续优化路线，满足载重限制。请直接求解。',
                       '继续规划路线，不考虑时间窗。', '请规划普通车辆路线，不要考虑电耗。']:
            with self.subTest(prompt=prompt): self.assertIsNone(request_precheck(prompt, None, data, {}))

    def test_cross_instance_and_time_windows_are_factual_stops(self):
        proof = build_planning_context(ROOT, self.bundle)['reference_state']
        prompt = '沿用当前客户归属，换用 20-5-2a__Q70，请继续优化路线。'
        check = request_precheck(prompt, proof, [{'instance_id':'20-5-2a__Q70'}], inheritance_contract(prompt, proof))
        self.assertEqual(check['code'], 'cross_instance_reuse')
        self.assertIn('选择本身有效', check['reply'])
        check = request_precheck('继续规划路线，并新增每个客户必须在指定时间窗内到达的硬约束。请直接求解。', proof, [], {})
        self.assertEqual(check['code'], 'unsupported_time_windows')
        self.assertIn('仅补充时间窗数据也不能', check['reply'])


class KnowledgeCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT/'knowledge', self.root/'knowledge')
        (self.root/'models').mkdir()
        shutil.copy2(ROOT/'models/model_catalog.json', self.root/'models/model_catalog.json')

    def test_deleting_rule_from_corpus_still_fails_manifest_completeness(self):
        p = self.root/'knowledge/mapping_rules.jsonl'
        lines = p.read_text(encoding='utf-8').splitlines()
        p.write_text('\n'.join(lines[1:]), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '残缺知识包'):
            retrieve(self.root, '', 'M1', backend='model_pack')

    def test_runtime_document_and_manifest_changes_need_review(self):
        for name in ['runtime_contract.json', 'pack_manifest.json']:
            path = self.root/'knowledge'/name
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.subTest(name=name), self.assertRaises(ValueError):
                if name == 'runtime_contract.json': load_runtime_knowledge(self.root)
                else: retrieve(self.root, '', 'M1', backend='model_pack')
            path.write_bytes(original)

    def test_catalog_tampering_is_not_hidden_by_mapping_rules(self):
        p = self.root/'models/model_catalog.json'
        c = json.loads(p.read_text(encoding='utf-8'));c['models'][0]['match_when'] = 'tampered'
        p.write_text(json.dumps(c), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '模型目录'):
            retrieve(self.root, '', 'M1', backend='model_pack')

    def test_top_one_keyword_search_cannot_remove_mandatory_rules(self):
        with patch('orllm.rag._vector_dependencies', side_effect=AssertionError('no vector query')):
            result = retrieve_for_mapping(self.root, '车辆容量', 'M3', backend='keyword', top_k=1)
        required = [r for r in result if r['retrieval_role'] == 'required']
        self.assertEqual(len(required), 8)
        self.assertEqual({r['model_id'] for r in result}, {'M3'})
        self.assertTrue({'mapping','data_sources','continuation'} <= {r['category'] for r in required})


if __name__ == '__main__':
    unittest.main()
