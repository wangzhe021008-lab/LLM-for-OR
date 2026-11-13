"""Tiny hand-computable checks of model meaning, not text snapshots."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from models import build_model


def assignment(model_id='M1'):
    data = {'model_id': model_id, 'model_version':'1.0', 'instance_id':'hand',
        'D':['d1','d2'], 'C':['a','b'], 'q':{'a':2,'b':3}, 'U':{'d1':4,'d2':6},
        'g':{'d1':{'a':1,'b':2}, 'd2':{'a':5,'b':1}}}
    if model_id == 'M1':
        data['F'] = {'d1':10,'d2':20}
    return data


def route():
    # Depot and customers lie at coordinates 0, 1, 3.
    return {'model_id':'M3', 'model_version':'1.0', 'instance_id':'line', 'depot_id':'d1',
        'C':['a','b'], 'q':{'a':1,'b':1}, 'U':10, 'Q':2, 'h':5,
        'c':{'d1':{'a':1,'b':3}, 'a':{'d1':1,'b':2}, 'b':{'d1':3,'a':2}}}


def electric(B):
    # Full-load outbound: 2 km * 2 kWh/km; empty return: 2 km * 1 kWh/km.
    return {'model_id':'M4', 'model_version':'1.0', 'instance_id':'battery', 'depot_id':'d1',
        'C':['a'], 'q':{'a':2}, 'U':10, 'Q':2, 'h':5, 'ell':{'d1':{'a':2}, 'a':{'d1':2}},
        'B':B, 'p':3, 'r0':1, 'r1':2, 'distance_unit':'km'}


class FixedModels(unittest.TestCase):
    def test_lp_names_and_mapping(self):
        model = build_model(assignment())
        with tempfile.TemporaryDirectory() as directory:
            lp = model.write_lp(Path(directory)/'test.lp').read_text()
            self.assertIn('Binary', lp)
            self.assertIn('Minimize', lp)
            self.assertEqual(model.variable_mapping['v0']['indices'], ['d1'])
            self.assertEqual(model.to_dict()['model_version'], '1.0')

    def test_rejects_empty_customer_group(self):
        data = route(); data['C'] = []
        with self.assertRaises(ValueError):
            build_model(data)

    def test_primal_check_rejects_false_solution(self):
        model = build_model(assignment())
        self.assertFalse(model.check_primal([0]*len(model.variables))['valid'])
        result = model._solution('optimal', 'bad incumbent', [0]*len(model.variables))
        self.assertEqual(result['status'], 'no_solution')
        self.assertNotIn('objective', result)

    @unittest.skipUnless(importlib.util.find_spec('coptpy'), 'COPT unavailable')
    def test_assignment_cost_and_capacity(self):
        # M1: d1 alone violates U=4; opening d2 alone costs 20+5+1=26;
        # opening both costs >=30+1+1=32.
        result = build_model(assignment()).solve_copt()
        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['objective'], 26)
        # M2: costs are whole-order costs, not multiplied by demand.
        result = build_model(assignment('M2')).solve_copt()
        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['objective'], 2)

    @unittest.skipUnless(importlib.util.find_spec('coptpy'), 'COPT unavailable')
    def test_route_capacity_and_cost(self):
        data = route()
        result = build_model(data).solve_copt()
        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['objective'], 11)
        data['Q'] = 1
        result = build_model(data).solve_copt()
        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['objective'], 18)

    @unittest.skipUnless(importlib.util.find_spec('coptpy'), 'COPT unavailable')
    def test_complete_roundtrip_battery(self):
        result = build_model(electric(6)).solve_copt()
        self.assertEqual(result['status'], 'optimal')
        self.assertAlmostEqual(result['objective'], 23)
        result = build_model(electric(5)).solve_copt()
        self.assertEqual(result['status'], 'infeasible')
        self.assertNotIn('objective', result)


if __name__ == '__main__':
    unittest.main()
