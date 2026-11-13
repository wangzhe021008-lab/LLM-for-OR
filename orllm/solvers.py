"""COPT LP/MILP solving with independently checked results.

Import COPT only when solving. Never include vendor exceptions in
public results: they can contain license identifiers, server URLs, or file paths.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import tempfile
import time

from .linear_model import number


def _options(time_limit, mip_gap, log_path):
    time_limit = number(time_limit, 'time_limit', positive=True)
    if mip_gap is not None:
        mip_gap = number(mip_gap, 'mip_gap')
    path = Path(log_path).resolve() if log_path is not None else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
    return time_limit, mip_gap, path


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _get(obj, name):
    """Optional metadata must not discard an otherwise valid solution."""
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _publish_log(temporary, destination, result):
    """Vendor C APIs may reject Unicode paths; Python copies the closed log."""
    if temporary is None:
        return
    try:
        source = Path(temporary.name) / 'solver.log'
        if source.is_file():
            shutil.copyfile(source, destination)
        elif result is not None:
            result['log_error'] = 'The backend did not produce a native log.'
    except OSError:
        if result is not None:
            result['log_error'] = 'Unable to copy the solver log to the output directory.'
    finally:
        try:
            temporary.cleanup()
        except OSError:
            pass


def _result(model, solver, status, solver_status, message, started, *, values=None,
            solver_objective=None, gap=None, bound=None, log_path=None, **details):
    # Model/status-specific calls decide whether an incumbent is meaningful.
    if status not in ('optimal', 'feasible_limit'):
        gap = None
    if status in ('infeasible', 'unbounded', 'infeasible_or_unbounded', 'error'):
        bound = None
    result = model._solution(status, message, values, solver=solver,
        solver_status=solver_status, runtime_seconds=max(0.0, time.perf_counter() - started),
        mip_gap=_finite(gap), mip_dual_bound=_finite(bound), **details)
    result.setdefault('primal_check', {'valid': False, 'reason': 'No complete incumbent'})
    if log_path:
        result['log_path'] = str(log_path)
    reported_objective = _finite(solver_objective)
    if 'objective' in result:
        computed = result['objective']
        difference = abs(computed - reported_objective) if reported_objective is not None else None
        tolerance = 1e-6 + 1e-8 * max(1.0, abs(computed), abs(reported_objective or 0.0))
        result['objective_check'] = {
            'valid': difference is not None and difference <= tolerance,
            'recomputed': computed, 'solver_reported': reported_objective,
            'absolute_difference': difference, 'tolerance': tolerance,
        }
        if not result['objective_check']['valid']:
            result['status'] = 'no_solution'
            result['message'] += '; incumbent failed independent objective check'
            for name in ('objective', 'variable_values', 'decision_values'):
                result.pop(name, None)
    else:
        result['objective_check'] = {'valid': False, 'reason': 'No validated incumbent'}
    if 'objective' not in result:
        result['mip_gap'] = None
    return result


def _error(model, solver, started, exc=None, missing=False, log_path=None):
    license_error = exc is not None and any(word in str(exc).lower()
        for word in ('licen', 'size-limited', 'size limited'))
    kind = 'dependency_unavailable' if missing else 'license_error' if license_error else 'backend_error'
    messages = {
        'dependency_unavailable': f'Required {solver} Python package is unavailable.',
        'license_error': f'{solver} license is unavailable or unsuitable for this model.',
        'backend_error': f'{solver} backend could not complete optimization.',
    }
    details = {'error_type': kind, 'can_fallback': False}
    if exc is not None:
        details['exception_type'] = type(exc).__name__
        error_code = _get(exc, 'errno')
        if isinstance(error_code, int):
            details['error_code'] = error_code
    return _result(model, solver, 'error', None, messages[kind], started,
                   log_path=log_path, **details)


def solve(model, solver='copt', time_limit=30, mip_gap=None, log_path=None):
    """COPT is the only backend; availability errors never trigger a fallback."""
    if solver != 'copt':
        raise ValueError('本系统只支持 COPT 求解，请指定 solver=copt。')
    time_limit, mip_gap, log_path = _options(time_limit, mip_gap, log_path)
    result = model.solve_copt(time_limit=time_limit, mip_gap=mip_gap, log_path=log_path)
    result['solver_requested'] = 'copt'
    result['solver_attempts'] = [{key: result[key] for key in
        ('solver', 'status', 'solver_status', 'error_type', 'message', 'runtime_seconds') if key in result}]
    result['fallback_used'] = False
    result['can_fallback'] = False
    return result


def _copt_status(code, has_solution, constants):
    if code == constants.OPTIMAL:
        return 'optimal' if has_solution else 'no_solution'
    if code == constants.INFEASIBLE:
        return 'infeasible'
    if code == constants.UNBOUNDED:
        return 'unbounded'
    if code == getattr(constants, 'INF_OR_UNB', None):
        return 'infeasible_or_unbounded'
    if code in tuple(getattr(constants, name, None) for name in ('NUMERICAL', 'UNSTARTED')):
        return 'error'
    return 'feasible_limit' if has_solution else 'no_solution'


def _capacity_vehicle_bound(model):
    """An implied integer capacity cut for the fixed routing templates.

    Summed delivery flow and empty returns give total q <= Q * departures.
    Binary departures give the rounded lower bound, without changing any
    integer feasible solution of the exported mathematical model.
    """
    if (model.model_id not in {"M3", "M4"}
            or model.metadata.get("template_source") != "models/model_catalog.json"):
        return None
    depot, customers = model.metadata["depot_id"], set(model.metadata["C"])
    demand = {r["indices"][0]: r["lower"] for r in model.constraints
              if r["family"] == "delivery_flow" and r["lower"] == r["upper"]}
    departures = [i for i, v in enumerate(model.variables)
                  if v["family"] == "x" and v["indices"][0] == depot]
    capacities = [v["ub"] for v in model.variables
                  if v["family"] == "f" and v["indices"][0] == depot]
    if set(demand) != customers or not capacities or len(set(capacities)) != 1:
        return None
    capacity = capacities[0]
    total = math.fsum(demand.values())
    ratio = total / capacity
    minimum = math.ceil(ratio - 1e-9 * max(1.0, abs(ratio)))
    names = [model.variables[i]["name"] for i in departures]
    return {"indices": departures, "certificate": {
        "family": "capacity_vehicle_count_bound", "minimum_vehicles": minimum,
        "total_demand": total, "vehicle_capacity": capacity,
        "coefficients": {name: 1 for name in names}, "lower": minimum,
        "derivation": "Sum delivery-flow equations, use empty returns and f <= Q*x, then round integer depot departures upward."}}


def solve_copt(model, time_limit=30, mip_gap=None, log_path=None):
    time_limit, mip_gap, log_path = _options(time_limit, mip_gap, log_path)
    mip_start = model.mip_start
    if mip_start is not None and not model.check_primal(mip_start)['valid']:
        raise ValueError('The COPT initial solution violates the original model')
    started = time.perf_counter()
    try:
        import coptpy as cp
    except ImportError:
        return _error(model, 'copt', started, missing=True, log_path=log_path)
    except Exception as exc:
        return _error(model, 'copt', started, exc, log_path=log_path)
    env = native = temporary = result = None
    try:
        config = cp.EnvrConfig()
        config.set('nobanner', '1')
        env = cp.Envr(config)
        native = env.createModel('fixed_orllm')
        native.setParam('Logging', 0)
        # coptpy's Python log callback also prints output independently of the
        # native LogToConsole setting; keep the native local logfile only.
        class QuietLog(cp.LogCallback):
            def Display(self, message):
                pass
        quiet_log = QuietLog()
        native.setLogCallback(quiet_log)
        native.setParam('LogToConsole', 0)
        native.setParam('Logging', 1 if log_path else 0)
        if log_path:
            temporary = tempfile.TemporaryDirectory(prefix='orllm_copt_')
            native.setLogFile(str(Path(temporary.name) / 'solver.log'))
        native.setParam('TimeLimit', time_limit)
        if mip_gap is not None:
            native.setParam('RelGap', mip_gap)
        kinds = {'binary': cp.COPT.BINARY, 'integer': cp.COPT.INTEGER, 'continuous': cp.COPT.CONTINUOUS}
        variables = [native.addVar(lb=v['lb'], ub=v['ub'] if v['ub'] is not None else cp.COPT.INFINITY,
            vtype=kinds[v['kind']], obj=model.objective.get(i, 0.0), name=v['name'])
            for i, v in enumerate(model.variables)]
        native.setObjSense(cp.COPT.MINIMIZE)
        for row in model.constraints:
            expression = cp.quicksum(coefficient * variables[i]
                                     for i, coefficient in row['coefficients'].items())
            lower, upper = row['lower'], row['upper']
            if lower is not None and lower == upper:
                native.addConstr(expression == lower, name=row['name'])
            else:
                if lower is not None:
                    native.addConstr(expression >= lower, name=row['name'] + '_lo')
                if upper is not None:
                    native.addConstr(expression <= upper, name=row['name'] + '_hi')
        vehicle_bound = _capacity_vehicle_bound(model)
        if vehicle_bound is not None:
            native.addConstr(cp.quicksum(variables[i] for i in vehicle_bound["indices"])
                             >= vehicle_bound["certificate"]["minimum_vehicles"],
                             name='capacity_vehicle_count_bound')
        if mip_start is not None:
            native.setMipStart(variables, mip_start)
            native.loadMipStart()
        native.solve()
        is_mip = bool(native.getAttr('IsMIP'))
        code = int(native.getAttr('MipStatus' if is_mip else 'LpStatus'))
        has_solution = bool(native.getAttr('HasMipSol' if is_mip else 'HasLpSol'))
        status = _copt_status(code, has_solution, cp.COPT)
        has_incumbent = status in ('optimal', 'feasible_limit')
        def attr(name):
            try:
                return native.getAttr(name)
            except Exception:
                return None
        bound = attr('BestBnd') if is_mip else (attr('LpObjVal') if status == 'optimal' else None)
        if _finite(bound) is not None and abs(bound) >= cp.COPT.INFINITY:
            bound = None
        result = _result(model, 'copt', status, code, f'COPT status {code}', started,
            values=[float(v.x) for v in variables] if has_incumbent else None,
            solver_objective=attr('BestObj' if is_mip else 'LpObjVal') if has_incumbent else None,
            gap=attr('BestGap') if is_mip else None,
            bound=bound,
            log_path=log_path, solver_runtime_seconds=_finite(attr('SolvingTime')),
            mip_start_used=mip_start is not None,
            **({'capacity_vehicle_bound': vehicle_bound['certificate']} if vehicle_bound is not None else {}),
            **({'error_type': 'backend_error', 'can_fallback': False} if status == 'error' else {}))
        return result
    except Exception as exc:
        result = _error(model, 'copt', started, exc, log_path=log_path)
        return result
    finally:
        for resource in (native, env):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
        _publish_log(temporary, log_path, result)
