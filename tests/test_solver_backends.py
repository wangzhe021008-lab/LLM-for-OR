"""COPT-only solver checks, failure handling and backend rejection."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from orllm.linear_model import LinearModel
from orllm.solvers import _error, _result


def tiny(binary=False, cost=2.0):
    model = LinearModel('solver_check', 'hand_computable')
    x = model.add_var('quantity', ['a'], binary=binary, cost=cost)
    model.add_row('requirement', ['a'], {x: 1}, lower=0.4)
    return model


class CheckedResultTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('coptpy'), 'optional copt unavailable')
    def test_copt_uses_checked_initial_solution_and_rejects_invalid_start(self):
        model = tiny(binary=True)
        model.mip_start = [1.0]
        result = model.solve_copt(time_limit=5, mip_gap=0)
        self.assertEqual(result['status'], 'optimal')
        self.assertEqual(result['objective'], 2)
        self.assertTrue(result['mip_start_used'])
        self.assertTrue(result['objective_check']['valid'])
        model.mip_start = [0.4]
        with self.assertRaisesRegex(ValueError, 'initial solution'):
            model.solve_copt(time_limit=5)

    def test_rejects_feasible_values_with_incorrect_reported_objective(self):
        result = _result(tiny(), 'test', 'optimal', 0, 'test incumbent',
                         time.perf_counter(), values=[0.4], solver_objective=200)
        self.assertTrue(result['primal_check']['valid'])
        self.assertFalse(result['objective_check']['valid'])
        self.assertEqual(result['status'], 'no_solution')
        self.assertNotIn('decision_values', result)

    def test_rejects_infeasible_incumbent_even_with_matching_objective(self):
        result = _result(tiny(binary=True), 'test', 'optimal', 0, 'test incumbent',
                         time.perf_counter(), values=[0.4], solver_objective=0.8)
        self.assertEqual(result['status'], 'no_solution')
        self.assertFalse(result['primal_check']['valid'])
        self.assertNotIn('objective', result)

    def test_vendor_license_exception_is_not_leaked(self):
        result = _error(tiny(), 'test', time.perf_counter(),
                        RuntimeError('license identifier private-secret-123'))
        self.assertEqual(result['error_type'], 'license_error')
        self.assertFalse(result['can_fallback'])
        self.assertNotIn('private-secret-123', json.dumps(result))

    def test_default_is_copt_and_errors_never_fall_back(self):
        model = tiny()
        for status in ('error', 'infeasible', 'unbounded', 'infeasible_or_unbounded',
                       'no_solution', 'feasible_limit', 'optimal'):
            failure = {'solver': 'copt', 'status': status, 'can_fallback': True}
            with self.subTest(status=status), patch.object(model, 'solve_copt', return_value=failure) as copt:
                result = model.solve()
                copt.assert_called_once()
                self.assertEqual(result['status'], status)
                self.assertEqual(result['solver_requested'], 'copt')
                self.assertFalse(result['fallback_used'])
                self.assertFalse(result['can_fallback'])
                self.assertEqual(len(result['solver_attempts']), 1)

    def test_other_backends_rejected_before_any_solve(self):
        import main
        import contextlib
        import io
        model = tiny()
        for name in ('gurobi', 'scipy', 'auto', 'unknown'):
            with self.subTest(solver=name), patch.object(model, 'solve_copt') as copt:
                with self.assertRaisesRegex(ValueError, '只支持 COPT'): model.solve(name)
                with self.assertRaisesRegex(ValueError, '只支持 COPT'):
                    main.run_build({}, '.', solver=name)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main.make_parser().parse_args(['build', '--scenario', 'tests/fixtures/m1_assignment.scenario.json', '--solve', name])
                from scripts import run_chain_batch
                with patch('sys.argv', ['run_chain_batch.py', '--solver', name]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    run_chain_batch.main()
                copt.assert_not_called()
        self.assertFalse(hasattr(model, 'solve_gurobi'))
        self.assertFalse(hasattr(model, 'solve_scipy'))

    def test_invalid_options_are_not_backend_failures(self):
        with patch.object(LinearModel, 'solve_copt') as copt:
            with self.assertRaises(ValueError):
                tiny().solve(time_limit=-1)
            with self.assertRaises(ValueError):
                tiny().solve(mip_gap=float('nan'))
            copt.assert_not_called()

    def test_limit_preserves_checked_incumbent_without_claiming_optimality(self):
        result = _result(tiny(binary=True), 'copt', 'feasible_limit', 8, 'Time limit',
                         time.perf_counter(), values=[1.0], solver_objective=2.0, gap=0.6, bound=0.8)
        self.assertEqual(result['status'], 'feasible_limit')
        self.assertEqual(result['objective'], 2)
        self.assertEqual(result['mip_dual_bound'], 0.8)
        self.assertTrue(result['objective_check']['valid'])

    def test_limit_without_incumbent_does_not_publish_decisions(self):
        result = _result(tiny(binary=True), 'copt', 'no_solution', 8, 'Time limit',
                         time.perf_counter(), bound=0.8)
        self.assertEqual(result['status'], 'no_solution')
        self.assertNotIn('decision_values', result)



class LocalBackendTests(unittest.TestCase):
    def check_backend(self, solver):
        package = 'coptpy'
        if not importlib.util.find_spec(package):
            self.skipTest(f'optional {package} unavailable')
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / '求解日志' / 'solver.log'
            result = tiny(binary=True).solve(solver, time_limit=5, mip_gap=0.0, log_path=log_path)
            if result.get('error_type') == 'license_error':
                self.skipTest(f'{solver} license unavailable')
            self.assertEqual(result['status'], 'optimal', result)
            self.assertAlmostEqual(result['objective'], 2.0)
            self.assertTrue(result['primal_check']['valid'])
            self.assertTrue(result['objective_check']['valid'])
            self.assertAlmostEqual(result['mip_dual_bound'], 2.0)
            self.assertAlmostEqual(result['mip_gap'], 0.0)
            self.assertGreater(log_path.stat().st_size, 0)
            self.assertNotIn('log_error', result)
            # LP must use LP solution attributes; HasMipSol would miss this.
            continuous = tiny().solve(solver, time_limit=5)
            self.assertEqual(continuous['status'], 'optimal', continuous)
            self.assertAlmostEqual(continuous['objective'], 0.8)
            # Ranged rows must enforce both bounds in each direct API.
            ranged = tiny(cost=-2.0)
            ranged.constraints[0]['upper'] = 3
            result = ranged.solve(solver, time_limit=5)
            self.assertEqual(result['status'], 'optimal', result)
            self.assertAlmostEqual(result['objective'], -6.0)
            infeasible = tiny(binary=True)
            infeasible.constraints[0]['lower'] = 2.0
            result = infeasible.solve(solver, time_limit=5)
            self.assertEqual(result['status'], 'infeasible', result)
            self.assertNotIn('objective', result)
            self.assertNotIn('decision_values', result)
            self.assertIsNone(result['mip_gap'])
            self.assertIsNone(result['mip_dual_bound'])
            result = tiny(cost=-2.0).solve(solver, time_limit=5)
            self.assertIn(result['status'], ('unbounded', 'infeasible_or_unbounded'), result)
            self.assertNotIn('objective', result)

    def test_copt(self):
        self.check_backend('copt')



if __name__ == '__main__':
    unittest.main()
