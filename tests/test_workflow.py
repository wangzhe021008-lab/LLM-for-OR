"""Contracts reject lost requirements, invented evidence and unsafe replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
from orllm.binding import bind_scenario
from orllm.data import load_base
from orllm.llm import mapping_schema
from orllm.workflow import (apply_changes, contract, fingerprint, map_with_contract,
                            new_state, update_contract, validate_contract, expand_mapping)
from orllm.web_chat import plan_generation_schema

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = '20-5-1a__Q70'
DESCRIPTION = f'使用 {INSTANCE} 选择开放仓库并分配全部客户。'


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.base = load_base(ROOT, INSTANCE)
        self.spec = contract(ROOT, DESCRIPTION, 'M1', [INSTANCE], mapping_schema(ROOT, 'M1', [INSTANCE]))
        self.mapping = {
            'scenario_id': 'contract_test', 'instance_id': INSTANCE, 'model_id': 'M1', 'model_version': '1.0',
            'opened_depots': None, 'assignment': None, 'distance_scale_km': None, 'distance_scale_source': None,
            'status': 'ready', 'missing_fields': [], 'unsupported_requirements': [],
            'evidence': [{'field': 'instance_id', 'source': 'user_description', 'excerpt': INSTANCE}],
            'requirement_coverage': [{'requirement_id': 'R001', 'status': 'mapped',
                                      'targets': ['instance_id', 'model_scope'], 'explanation': '实例明确，模型支持选址分配。'}]}
        self.state = new_state(ROOT, self.mapping, DESCRIPTION, self.spec)

    def update(self, changes, state=None, request='request-1', revision=None):
        state = state or self.state
        return apply_changes(ROOT, state, changes, revision or state['revision'], request, '修改当前参数')

    def test_dynamic_entities_and_coverage(self):
        validate_contract(self.mapping, self.spec, DESCRIPTION, [])
        self.assertEqual(set(self.spec['schema']['properties']['opened_depots']['items']['enum']), set(self.base['D']))
        for coverage in [[], self.mapping['requirement_coverage'] * 2,
                         [{**self.mapping['requirement_coverage'][0], 'status': 'unsupported'}]]:
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                validate_contract({**self.mapping, 'requirement_coverage': coverage}, self.spec, DESCRIPTION, [])
        bad = {**self.mapping, 'opened_depots': ['invented-depot']}
        with self.assertRaises(ValueError): validate_contract(bad, self.spec, DESCRIPTION, [])

    def raw_mapping(self):
        raw={k:copy.deepcopy(v) for k,v in self.mapping.items() if k in self.spec['generation_schema']['properties']}
        raw['evidence']=[{'field':'instance_id','span_id':'U001'}]
        return raw

    def test_system_fields_cannot_be_generated_or_attributed_to_user(self):
        raw=self.raw_mapping()
        mapped=expand_mapping(raw,self.spec)
        validate_contract(mapped,self.spec,DESCRIPTION,[])
        self.assertEqual(mapped['model_id'],'M1')
        self.assertEqual(mapped['evidence'],[{'field':'instance_id','source':'user_description','excerpt':DESCRIPTION}])
        for field in ['model_id','model_version','scenario_id','opened_depots','assignment']:
            self.assertNotIn(field,self.spec['generation_schema']['properties'])
        for edit in [{'model_id':'M4'}, {'evidence':[{'field':'model_id','span_id':'U001'}]},
                     {'evidence':[{'field':'instance_id','span_id':'selected_model_id'}]},
                     {'evidence':[{'field':'instance_id','source':'user_description','excerpt':'M1'}]}]:
            with self.subTest(edit=edit),self.assertRaises(ValueError):expand_mapping({**raw,**edit},self.spec)

    def test_current_requirements_do_not_include_planner_or_reference_headers(self):
        bundle={'user_description':DESCRIPTION,'user_history':['请使用原始成本数据。'],'confirmed_reference':None}
        spec=contract(ROOT,'系统包装说明；改写后的任务。','M1',[INSTANCE],mapping_schema(ROOT,'M1',[INSTANCE]),input_bundle=bundle)
        self.assertEqual([r['text'] for r in spec['requirements']],[DESCRIPTION])
        self.assertEqual(spec['source_spans']['U001']['text'],DESCRIPTION)
        self.assertEqual(spec['source_spans']['H1_001']['source'],'user_history:1')
        self.assertNotIn('系统包装说明',str(spec['source_spans']))

    def test_confirmed_routing_decisions_are_bound_without_llm_recopy(self):
        previous=json.loads((ROOT/'tests/fixtures/decision_reuse.json').read_text(encoding='utf-8'))['M3']['mapping']
        reference={k:previous[k] for k in ['instance_id','opened_depots','assignment']}
        reference['reuse_decisions']='assignment'
        text='沿用当前结果的全部客户归属，规划常规车辆路线。'
        spec=contract(ROOT,text,'M3',[previous['instance_id']],mapping_schema(ROOT,'M3',[previous['instance_id']]),
                      input_bundle={'user_description':text,'user_history':[],'confirmed_reference':reference})
        raw={'status':'ready','missing_fields':[],'unsupported_requirements':[],'evidence':[],
             'requirement_coverage':[{'requirement_id':'R001','status':'model_capability','targets':['model_scope'],'explanation':'固定归属的常规路径'}]}
        mapped=expand_mapping(raw,spec)
        validate_contract(mapped,spec,text,[])
        self.assertEqual(mapped['assignment'],previous['assignment'])
        self.assertNotIn('assignment',spec['generation_schema']['properties'])
        self.assertTrue(all(e['source']=='confirmed_reference' for e in mapped['evidence']))
        state=new_state(ROOT,mapped,text,spec)
        self.assertEqual(bind_scenario(load_base(ROOT,previous['instance_id']),state['scenario'])['status'],'ready')

    def test_request_separates_system_knowledge_and_business_and_expands_response(self):
        from orllm.llm import extract_mapping
        from unittest.mock import MagicMock
        response=MagicMock();response.__enter__.return_value.read.return_value=json.dumps(
            {'choices':[{'message':{'content':json.dumps(self.raw_mapping())},'finish_reason':'stop'}]}).encode()
        config={'base_url':'http://localhost:8000/v1','model':'test','allow_no_api_key':True,
                'response_format':'json_schema','json_schema_supported':True}
        with patch('orllm.llm.urlopen',return_value=response) as http:
            mapped=extract_mapping(ROOT,DESCRIPTION,'M1',[],config,[INSTANCE],contract=self.spec)
        payload=json.loads(json.loads(http.call_args.args[0].data)['messages'][1]['content'])
        self.assertEqual(payload['business_input']['user_description'],DESCRIPTION)
        self.assertEqual(payload['system_context']['selection']['model_id'],'M1')
        self.assertNotIn('model_id',payload['output_schema']['properties'])
        self.assertNotIn('selected_model_id',str(payload['citable_business_spans']))
        validate_contract(mapped,self.spec,DESCRIPTION,[])

    def test_cross_model_knowledge_rejected_before_llm_call(self):
        with tempfile.TemporaryDirectory() as tmp,patch('orllm.llm.extract_mapping') as llm:
            with self.assertRaisesRegex(ValueError,'其他模型'):
                map_with_contract(ROOT,DESCRIPTION,{'model_id':'M1'},[{'model_id':'M4','model_version':'1.0'}],{},[INSTANCE],tmp)
            llm.assert_not_called()

    def test_quotes_cannot_be_invented_or_attributed_to_unknown_source(self):
        for evidence in [[], [{'field': 'instance_id', 'source': 'user_description', 'excerpt': 'invented'}],
                         [{'field': 'instance_id', 'source': 'unknown', 'excerpt': INSTANCE}]]:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                validate_contract({**self.mapping, 'evidence': evidence}, self.spec, DESCRIPTION, [])

    def test_direct_model_document_is_valid_evidence_without_retrieval(self):
        evidence = self.mapping['evidence'] + [{'field': 'model_scope',
            'source': 'models/model_catalog.json#/models/0',
            'excerpt': '分配给仓库的总需求不超过开放后的容量'}]
        validate_contract({**self.mapping, 'evidence': evidence}, self.spec, DESCRIPTION, [])
        targets = self.spec['schema']['properties']['requirement_coverage']['items']['properties']['targets']['items']['enum']
        self.assertNotIn('opened_depots', targets)
        self.assertNotIn('assignment', targets)

    def test_payload_aliases_are_grounded_and_wrong_attribution_still_rejected(self):
        evidence = self.mapping['evidence'] + [{'field':'depot_capacity','source':'fixed_model',
            'excerpt':'Σ_{i∈C} q[i]*z[d,i] ≤ U[d]*y[d]'}]
        validate_contract({**self.mapping,'evidence':evidence},self.spec,DESCRIPTION,[])
        for source, excerpt in [('user_description','selected_model_id": "M1"'),
                                ('fixed_model','arbitrary fabricated requirement')]:
            wrong = evidence + [{'field':'model_id','source':source,'excerpt':excerpt}]
            with self.subTest(source=source),self.assertRaisesRegex(ValueError,'evidence\\[2\\]'):
                validate_contract({**self.mapping,'evidence':wrong},self.spec,DESCRIPTION,[])
        sources=self.spec['schema']['properties']['evidence']['items']['properties']['source']['enum']
        self.assertIn('fixed_model',sources)
        self.assertNotIn('invented_source',sources)

    def test_retrieved_sources_are_included_in_generation_enum(self):
        context=[{'id':'record-1','source':'knowledge/test','text':'已提供的解释'}]
        spec=contract(ROOT,DESCRIPTION,'M1',[INSTANCE],mapping_schema(ROOT,'M1',[INSTANCE]),context=context)
        evidence=self.mapping['evidence']+[{'field':'model_scope','source':'record-1','excerpt':'已提供的解释'}]
        validate_contract({**self.mapping,'evidence':evidence},spec,DESCRIPTION,context)

    def test_bounded_repair_and_evidence_only_parameter_freeze(self):
        bad = {**self.mapping, 'evidence': []}
        with tempfile.TemporaryDirectory() as tmp, patch('orllm.llm.extract_mapping', side_effect=[bad, self.mapping]) as llm:
            result = map_with_contract(ROOT, DESCRIPTION, {'model_id': 'M1'}, [], {}, [INSTANCE], tmp)
            self.assertEqual(result, self.mapping)
            self.assertEqual(llm.call_count, 2)
            self.assertEqual(llm.call_args.kwargs['feedback']['scope'], 'evidence_only')
            self.assertTrue((Path(tmp) / 'binding_state.json').is_file())
        changed = {**self.mapping, 'distance_scale_km': 1}
        # Isolate the freeze check from other independent contract checks.
        with tempfile.TemporaryDirectory() as tmp, patch('orllm.llm.extract_mapping', side_effect=[bad, changed]) as llm:
            with patch('orllm.workflow.validate_contract', side_effect=[ValueError('证据校验：缺失'), None]):
                with self.assertRaisesRegex(ValueError, '不允许改变参数'):
                    map_with_contract(ROOT, DESCRIPTION, {'model_id': 'M1'}, [], {}, [INSTANCE], tmp)
            self.assertEqual(llm.call_count, 2)
            self.assertFalse((Path(tmp) / 'binding_state.json').exists())

    def test_set_increase_decrease_and_idempotence(self):
        change = [{'field': 'h', 'operation': 'increase', 'value': 20}]
        updated = self.update(change)
        self.assertEqual(updated['scenario']['parameter_values']['h'], self.base['h'] + 20)
        self.assertEqual(self.update(change, updated, revision=self.state['revision']), updated)
        with self.assertRaisesRegex(ValueError, '同一请求'):
            self.update([{'field': 'h', 'operation': 'increase', 'value': 21}], updated, revision=self.state['revision'])
        decreased = self.update([{'field': 'h', 'operation': 'decrease', 'value': 10}], updated, 'request-2')
        self.assertEqual(decreased['scenario']['parameter_values']['h'], self.base['h'] + 10)
        final = self.update([{'field': 'h', 'operation': 'set', 'value': 0}], decreased, 'request-3')
        self.assertEqual(final['scenario']['parameter_values']['h'], 0)
        self.assertEqual(load_base(ROOT, INSTANCE), self.base)

    def test_stale_revision_data_model_and_tamper_rejected(self):
        changes = [{'field': 'h', 'operation': 'increase', 'value': 20}]
        with self.assertRaisesRegex(ValueError, '绑定版本'): self.update(changes, revision='old')
        with patch('orllm.workflow.load_base', return_value={**self.base, 'h': self.base['h'] + 1}):
            with self.assertRaisesRegex(ValueError, '数据版本'): self.update(changes)
        with patch('orllm.workflow.model_version', return_value='changed'):
            with self.assertRaisesRegex(ValueError, '模型程序版本'): self.update(changes)
        tampered = copy.deepcopy(self.state); tampered['scenario']['parameter_values']['h'] = 123
        with self.assertRaisesRegex(ValueError, '快照校验'): self.update(changes, tampered)

    def test_unknown_fields_duplicate_operations_and_infeasible_values_rejected(self):
        for changes in [
            [{'field': 'objective', 'operation': 'set', 'value': 1}],
            [{'field': 'h', 'operation': 'increase', 'value': -20}],
            [{'field': 'Q', 'operation': 'set', 'value': 0}],
            [{'field': 'Q', 'operation': 'set', 'value': 0.001}],
            [{'field': 'h', 'operation': 'set', 'value': True}],
            [{'field': 'h', 'operation': 'set', 'value': 1}] * 2,
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.update(changes)

    def test_binding_recomputes_cost_and_preserves_inputs(self):
        updated = self.update([{'field': 'h', 'operation': 'increase', 'value': 20}])
        before = bind_scenario(self.base, self.state['scenario'])
        after = bind_scenario(self.base, updated['scenario'])
        self.assertEqual(after['status'], 'ready')
        a, b = before['instances'][0], after['instances'][0]
        for d in a['D']:
            for c in a['C']: self.assertAlmostEqual(b['g'][d][c] - a['g'][d][c], 20)
        self.assertEqual(a['q'], b['q']); self.assertEqual(a['F'], b['F'])
        self.assertEqual(fingerprint(self.base), self.state['scenario']['data_version'])

    def test_update_schema_locks_fields_and_revision(self):
        spec = update_contract(ROOT, self.state)
        plan = {'action': 'update', 'reply': '准备修改', 'description': None, 'reuse_decisions': 'none',
                'expected_revision': spec['expected_revision'],
                'changes': [{'field': 'h', 'operation': 'increase', 'value': 20}]}
        validator = Draft202012Validator(plan_generation_schema(spec))
        self.assertTrue(validator.is_valid(plan))
        self.assertFalse(Draft202012Validator(plan_generation_schema()).is_valid(plan))
        self.assertFalse(validator.is_valid({**plan, 'expected_revision': 'old'}))
        self.assertFalse(validator.is_valid({**plan, 'changes': [{'field': 'constraint', 'operation': 'set', 'value': 1}]}))

    def test_update_worker_does_not_reselect_retrieve_or_map(self):
        import web_worker
        report = {'status': 'ready', 'reports': {}}
        with tempfile.TemporaryDirectory() as tmp, patch('web_worker.pipeline.run_build', return_value=report), \
                patch('web_worker.pipeline.report_exit_code', return_value=0), \
                patch('web_worker.pipeline.selection_for_description', side_effect=AssertionError('selection called')), \
                patch('web_worker.pipeline.mapping_for_selection', side_effect=AssertionError('mapping called')), \
                patch('orllm.rag.retrieve', side_effect=AssertionError('retrieval called')):
            status = web_worker.run({'output': tmp, 'solver': 'copt', 'time_limit': 10, 'mip_gap': 0,
                'parameter_update': {'binding_state': self.state, 'expected_revision': self.state['revision'],
                                     'request_id': 'patch', 'description': 'h 增加 20',
                                     'changes': [{'field': 'h', 'operation': 'increase', 'value': 20}]}})
            self.assertEqual(status, 0)
            saved = json.loads((Path(tmp) / 'binding_state.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['scenario']['parameter_values']['h'], self.base['h'] + 20)
            audit = json.loads((Path(tmp) / 'incremental_update.json').read_text(encoding='utf-8'))
            self.assertEqual(audit['source_text'], 'h 增加 20')
            self.assertEqual(audit['source'], 'user_description')

    def test_modified_knowledge_cannot_reuse_review_approval(self):
        from orllm.rag import build_knowledge
        original = json.loads((ROOT / 'knowledge/review_registry.json').read_text(encoding='utf-8'))
        original['records'][next(iter(original['records']))]['fingerprint'] = 'changed'
        real_read = Path.read_text
        def read(path, *args, **kwargs):
            if path.name == 'review_registry.json': return json.dumps(original)
            return real_read(path, *args, **kwargs)
        before = len(build_knowledge(ROOT))
        with patch.object(Path, 'read_text', read): self.assertEqual(len(build_knowledge(ROOT)), before - 1)
