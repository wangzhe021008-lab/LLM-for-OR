"""Offline regression checks. HTTP is mocked; no live LLM or vector model test."""
import copy
import io
import json
import shutil
from pathlib import Path
import unittest
import tempfile
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from orllm.rag import build_knowledge, build_index, retrieve
from orllm.llm import extract_mapping, validate_mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = {"scenario_id": "demo", "instance_id": "demo", "model_id": "M1",
        "model_version": "1.0", "opened_depots": None, "assignment": None,
        "distance_scale_km": None, "distance_scale_source": None, "status": "ready",
        "missing_fields": [], "unsupported_requirements": [], "evidence": []}
CONFIG = {"base_url": "http://localhost:8000/v1", "model": "test-model",
          "allow_no_api_key": True, "json_schema_supported": True}
JSON_CONFIG = {**CONFIG, "response_format": "json_object", "json_schema_supported": False}


class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.payload).encode()


class RetrievalAndMappingTests(unittest.TestCase):
    def test_selected_model_pack_is_complete_even_with_top_one_and_unrelated_query(self):
        records=build_knowledge(ROOT)
        with patch('orllm.rag._vector_dependencies',side_effect=AssertionError('No embedding query for a fixed pack')):
            for mid in ['M1','M2','M3','M4']:
                hits=retrieve(ROOT,'完全无关的词语','M1' if mid=='M1' else mid,backend='model_pack',top_k=1)
                expected={r['id'] for r in records if r['model_id']==mid and '-catalog-' not in r['id']}
                self.assertEqual({r['id'] for r in hits},expected)
                self.assertTrue(all(r['model_id']==mid and r['review_status']=='engineering_reviewed' for r in hits))

    def test_model_pack_fails_closed_if_reviewed_knowledge_is_missing(self):
        records=build_knowledge(ROOT)
        omitted=next(r for r in records if r['model_id']=='M1' and r['category']=='mapping')
        with patch('orllm.rag.build_knowledge',return_value=[r for r in records if r['id']!=omitted['id']]):
            with self.assertRaisesRegex(ValueError,'残缺知识包'):
                retrieve(ROOT,'需求','M1',backend='model_pack')
    def test_bundled_embedding_identity_survives_workspace_move(self):
        with tempfile.TemporaryDirectory() as folder:
            first, moved = Path(folder) / "first", Path(folder) / "moved"
            model = first / "data/embedding_models/local-model"
            model.mkdir(parents=True)
            records = build_knowledge(ROOT)
            backend, settings, encoder = MagicMock(), MagicMock(), MagicMock()
            encoder.encode.return_value.tolist.return_value = [[1.0]] * len(records)
            with patch("orllm.rag.build_knowledge", return_value=records), \
                    patch("orllm.rag._vector_dependencies", return_value=(backend, settings, encoder)) as dependencies:
                manifest = build_index(first, str(model))
                self.assertEqual(manifest["embedding_model"], "data/embedding_models/local-model")
                shutil.move(str(first), str(moved))
                record = next(r for r in records if r["model_id"] == "M1")
                backend.PersistentClient.return_value.get_collection.return_value.query.return_value = {
                    "ids": [[record["id"]]], "documents": [[record["text"]]],
                    "metadatas": [[{"model_id": "M1", "model_version": "1.0"}]], "distances": [[0.0]]}
                hits = retrieve(moved, "仓库容量", "M1", backend="chroma")
                self.assertEqual(hits[0]["id"], record["id"])
                dependencies.assert_called_with(str((moved / manifest["embedding_model"]).resolve()))

    def test_knowledge_and_filters(self):
        records = build_knowledge(ROOT)
        self.assertEqual(len(records), len({r['id'] for r in records}))
        for model in ['M1', 'M2', 'M3', 'M4']:
            hits = retrieve(ROOT, '配送需求容量电池碳排放参数', model)
            self.assertTrue(hits)
            self.assertTrue(all(r['model_id'] == model and r['model_version'] == '1.0'
                                and r['backend'] == 'keyword' for r in hits))
        self.assertIn('模板边界', [r['title'] for r in retrieve(ROOT, '碳排放', 'M4')])
        with self.assertRaises(ValueError): retrieve(ROOT, '电耗', 'M4', version='unknown')

    def test_mapping_semantics_and_schema(self):
        self.assertEqual(validate_mapping(ROOT, BASE, 'M1', ['demo']), BASE)
        for change, model in [({'instance_id': 'invented'}, 'M1'),
                              ({'model_id': 'M2'}, 'M2'),
                              ({'missing_fields': ['Q']}, 'M1'),
                              ({'extra': 1}, 'M1'),
                              ({'distance_scale_km': float('nan')}, 'M1')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_mapping(ROOT, {**BASE, **change}, model, ['demo'])

    def test_strict_request_and_model_binding(self):
        payload = {'choices': [{'message': {'content': json.dumps(BASE)}, 'finish_reason': 'stop'}]}
        with patch('orllm.llm.urlopen', return_value=FakeResponse(payload)) as http:
            result = extract_mapping(ROOT, '使用 demo 实例做选址', 'M1', [], CONFIG, ['demo'])
            body = json.loads(http.call_args.args[0].data)
            response_format = body['response_format']['json_schema']
            self.assertTrue(response_format['strict'])
            self.assertEqual(response_format['schema']['properties']['model_id']['enum'], ['M1'])
            self.assertEqual(response_format['schema']['properties']['instance_id']['enum'], [None, 'demo'])
            self.assertTrue(json.loads(body['messages'][1]['content'])['complete_input_schema']['properties'])
            self.assertEqual(json.loads(body['messages'][1]['content'])['output_schema'], response_format['schema'])
            self.assertEqual(result, BASE)

    def test_json_object_request_supplies_full_schema(self):
        payload = {'choices': [{'message': {'content': json.dumps(BASE)}, 'finish_reason': 'stop'}]}
        with patch('orllm.llm.urlopen', return_value=FakeResponse(payload)) as http:
            result = extract_mapping(ROOT, '使用 demo 实例做选址', 'M1', [], JSON_CONFIG, ['demo'])
            body = json.loads(http.call_args.args[0].data)
            self.assertEqual(body['response_format'], {'type': 'json_object'})
            self.assertFalse(body['stream'])
            output_schema = json.loads(body['messages'][1]['content'])['output_schema']
            self.assertEqual(output_schema['properties']['model_id']['enum'], ['M1'])
            self.assertEqual(output_schema['properties']['model_version']['enum'], ['1.0'])
            self.assertEqual(output_schema['properties']['instance_id']['enum'], [None, 'demo'])
            self.assertEqual(set(output_schema['required']), set(BASE))
            self.assertFalse(output_schema['additionalProperties'])
            self.assertIn('JSON', body['messages'][0]['content'])
            self.assertEqual(result, BASE)

    def test_both_modes_preserve_selected_model_and_known_instances(self):
        invalid = [{'model_id': 'M2'}, {'model_version': 'other'}, {'instance_id': 'invented'},
                   {'missing_fields': ['capacity']}, {'extra': 1}]
        for config in [CONFIG, JSON_CONFIG]:
            for change in invalid:
                payload = {'choices': [{'message': {'content': json.dumps({**BASE, **change})}, 'finish_reason': 'stop'}]}
                with self.subTest(mode=config.get('response_format', 'json_schema'), change=change):
                    with patch('orllm.llm.urlopen', return_value=FakeResponse(payload)) as http:
                        with self.assertRaises(ValueError):
                            extract_mapping(ROOT, 'description', 'M1', [], config, ['demo'])
                        self.assertEqual(http.call_count, 1)

    def test_bad_api_results_rejected(self):
        invalid = [({'content': json.dumps(BASE)}, 'length'),
                   ({'refusal': 'no'}, 'stop'), ({'content': '```json\n{}\n```'}, 'stop'),
                   ({'content': json.dumps({**BASE, 'distance_scale_km': float('nan')})}, 'stop')]
        for config in [CONFIG, JSON_CONFIG]:
            for message, finish in invalid:
                payload = {'choices': [{'message': message, 'finish_reason': finish}]}
                with self.subTest(mode=config.get('response_format', 'json_schema'), message=message):
                    with patch('orllm.llm.urlopen', return_value=FakeResponse(payload)):
                        with self.assertRaises(RuntimeError):
                            extract_mapping(ROOT, 'description', 'M1', [], config, ['demo'])

    def test_malformed_response_envelopes_rejected(self):
        invalid = [[], {'error': {'message': 'bad'}}, {'choices': []}, {'choices': 'bad'},
                   {'choices': [None]}, {'choices': [{'message': 'bad', 'finish_reason': 'stop'}]},
                   {'choices': [{'message': {'content': {}}, 'finish_reason': 'stop'}]}]
        for payload in invalid:
            with self.subTest(payload=payload), patch('orllm.llm.urlopen', return_value=FakeResponse(payload)):
                with self.assertRaises(RuntimeError):
                    extract_mapping(ROOT, 'description', 'M1', [], JSON_CONFIG, ['demo'])

    def test_invalid_configuration_rejected_before_http(self):
        invalid = [{'response_format': 'auto'}, {'response_format': []},
                   {'json_schema_supported': False}, {'base_url': None}, {'model': None},
                   {'max_tokens': 0}, {'max_tokens': -1}, {'max_tokens': 2.5}, {'max_tokens': True},
                   {'timeout_seconds': 0}, {'timeout_seconds': float('inf')},
                   {'timeout_seconds': float('nan')}, {'timeout_seconds': True},
                   {'token_parameter': 'tokens'}, {'api_key_env': ''}]
        for change in invalid:
            with self.subTest(change=change), patch('orllm.llm.urlopen') as http:
                with self.assertRaises(ValueError):
                    extract_mapping(ROOT, 'description', 'M1', [], {**CONFIG, **change}, ['demo'])
                http.assert_not_called()

    def test_http_error_does_not_echo_response_or_key_and_has_no_fallback(self):
        secret = 'test-secret-key'
        error = HTTPError('https://example.invalid', 401, secret, {}, io.BytesIO(secret.encode()))
        config = {**JSON_CONFIG, 'api_key_env': 'ORLLM_TEST_SECRET'}
        with patch.dict('os.environ', {'ORLLM_TEST_SECRET': secret}):
            with patch('orllm.llm.urlopen', side_effect=error) as http:
                with self.assertRaises(RuntimeError) as caught:
                    extract_mapping(ROOT, 'description', 'M1', [], config, ['demo'])
                self.assertIn('HTTP 401', str(caught.exception))
                self.assertNotIn(secret, str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)
                self.assertEqual(http.call_count, 1)

    def test_unconfigured_api_never_called(self):
        with patch('orllm.llm.urlopen') as http:
            with self.assertRaises(ValueError):
                extract_mapping(ROOT, 'description', 'M1', [], {}, ['demo'])
            http.assert_not_called()


if __name__ == '__main__': unittest.main()
