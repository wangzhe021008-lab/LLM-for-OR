"""Fixed sparse MILP representation, readable LP export, and optional solvers.

No formula strings are evaluated. Solver incumbents are checked against bounds,
integer domains, and every original row before they are reported as feasible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Real
from pathlib import Path


def number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{label}: must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def identifiers(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label}: expected a nonempty list")
    if any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"{label}: every ID must be a nonempty string")
    if len(set(values)) != len(values):
        raise ValueError(f"{label}: duplicate IDs")
    return values


def matrix(data, nodes, label):
    for i in nodes:
        if i not in data:
            raise ValueError(f"{label}: missing row {i}")
        for j in nodes:
            if i != j:
                number(data[i][j], f"{label}[{i},{j}]")


@dataclass
class LinearModel:
    model_id: str
    instance_id: str
    model_version: str = "1.0"
    variables: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    objective: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    mip_start: list | None = field(default=None, repr=False)

    @property
    def variable_mapping(self):
        return {v['name']: {k: val for k, val in v.items() if k != 'name'} for v in self.variables}

    def add_var(self, family, indices, *, binary=False, upper=None, cost=0.0):
        idx = len(self.variables)
        cost = float(cost)
        if not math.isfinite(cost):
            raise ValueError("Nonfinite objective coefficient")
        self.variables.append({
            'name': f'v{idx}', 'family': family, 'indices': list(indices),
            'kind': 'binary' if binary else 'continuous', 'lb': 0.0,
            'ub': 1.0 if binary else upper,
        })
        if cost:
            self.objective[idx] = cost
        return idx

    def add_row(self, family, indices, terms, *, lower=None, upper=None):
        coeff = {}
        items = terms.items() if isinstance(terms, dict) else terms
        for idx, value in items:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Nonfinite constraint coefficient")
            coeff[idx] = coeff.get(idx, 0.0) + value
        coeff = {idx: value for idx, value in coeff.items() if value != 0}
        for bound in (lower, upper):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("Nonfinite row bound")
        self.constraints.append({
            'name': f'r{len(self.constraints)}', 'family': family,
            'indices': list(indices), 'coefficients': coeff,
            'lower': lower, 'upper': upper,
        })

    def to_dict(self):
        return {
            'model_id': self.model_id, 'model_version': self.model_version,
            'instance_id': self.instance_id, 'sense': 'minimize',
            'variables': self.variables, 'variable_mapping': self.variable_mapping,
            'objective': {self.variables[i]['name']: c for i, c in self.objective.items()},
            'constraints': [dict(row, coefficients={self.variables[i]['name']: c
                for i, c in row['coefficients'].items()}) for row in self.constraints],
            'metadata': self.metadata,
        }

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        return path

    def write_lp(self, path):
        """Export without a solver. Internal numeric names are safe for arbitrary IDs."""
        if not self.variables:
            raise ValueError("Cannot export an empty model")
        def expr(coeff):
            return ' '.join(f"{'+' if val >= 0 else '-'} {abs(val):.17g} {self.variables[i]['name']}"
                            for i, val in coeff.items()) or '0 v0'
        # Long rows continue on subsequent lines, avoiding parser line-length limits.
        def append_expr(lines, prefix, coefficients, suffix=''):
            parts = expr(coefficients).split(' ')
            current = prefix
            for j in range(0, len(parts), 3):
                term = ' '.join(parts[j:j+3])
                if len(current) + len(term) > 200:
                    lines.append(current)
                    current = '  '
                current += ' ' + term
            lines.append(current + suffix)
        lines = ['Minimize']
        append_expr(lines, ' obj:', self.objective)
        lines.append('Subject To')
        for row in self.constraints:
            lo, hi = row['lower'], row['upper']
            if lo is not None and lo == hi:
                append_expr(lines, f" {row['name']}:", row['coefficients'], f' = {lo:.17g}')
            else:
                if lo is not None:
                    append_expr(lines, f" {row['name']}_lo:", row['coefficients'], f' >= {lo:.17g}')
                if hi is not None:
                    append_expr(lines, f" {row['name']}_hi:", row['coefficients'], f' <= {hi:.17g}')
        lines.append('Bounds')
        for v in self.variables:
            if v['kind'] != 'binary':
                lines.append(f" {v['lb']:.17g} <= {v['name']}" + (f" <= {v['ub']:.17g}" if v['ub'] is not None else ''))
        binaries = [v['name'] for v in self.variables if v['kind'] == 'binary']
        if binaries:
            lines.append('Binary')
            for i in range(0, len(binaries), 20):
                lines.append(' ' + ' '.join(binaries[i:i+20]))
        lines.append('End')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(lines) + '\n', encoding='ascii')
        return path

    def check_primal(self, values, atol=1e-6, rtol=1e-8):
        if values is None or len(values) != len(self.variables):
            return {'valid': False, 'reason': 'No complete incumbent'}
        if not all(math.isfinite(float(x)) for x in values):
            return {'valid': False, 'reason': 'Nonfinite incumbent'}
        violations = []
        worst = 0.0
        for var, value in zip(self.variables, values):
            violation = max(0.0, var['lb'] - value,
                value - var['ub'] if var['ub'] is not None else 0.0)
            if var['kind'] == 'binary':
                violation = max(violation, abs(value - round(value)))
            worst = max(worst, violation)
            if violation > atol + rtol * max(1.0, abs(value)):
                violations.append(var['name'])
        for row in self.constraints:
            activity = sum(c * values[i] for i, c in row['coefficients'].items())
            lo, hi = row['lower'], row['upper']
            violation = max(0.0, lo - activity if lo is not None else 0.0,
                            activity - hi if hi is not None else 0.0)
            scale = max(1.0, abs(activity), abs(lo or 0.0), abs(hi or 0.0))
            worst = max(worst, violation)
            if violation > atol + rtol * scale:
                violations.append(row['name'])
        return {'valid': not violations, 'max_absolute_violation': float(worst),
                'violations': violations[:20], 'absolute_tolerance': atol, 'relative_tolerance': rtol}

    def _solution(self, status, message, values=None, **details):
        result = {'status': status, 'message': message, **details}
        if values is None:
            return result
        check = self.check_primal(values)
        result['primal_check'] = check
        if not check['valid']:
            result['status'] = 'no_solution'
            result['message'] += '; incumbent failed independent primal checks'
            return result
        result['objective'] = float(sum(c * values[i] for i, c in self.objective.items()))
        result['variable_values'] = {v['name']: float(values[i]) for i, v in enumerate(self.variables)}
        result['decision_values'] = [dict(family=v['family'], indices=v['indices'], value=float(values[i]))
                                     for i, v in enumerate(self.variables)]
        return result

    def solve(self, solver='copt', time_limit=30, mip_gap=None, log_path=None):
        """Solve only with COPT and independently verify its incumbent."""
        from .solvers import solve
        return solve(self, solver, time_limit, mip_gap, log_path)

    def solve_copt(self, time_limit=30, mip_gap=None, log_path=None):
        from .solvers import solve_copt
        return solve_copt(self, time_limit, mip_gap, log_path)



def new_model(instance, expected):
    if instance.get('model_id') != expected:
        raise ValueError(f'Expected {expected}')
    if instance.get('model_version') != '1.0':
        raise ValueError('Unsupported model_version; expected 1.0')
    if not isinstance(instance.get('instance_id'), str) or not instance['instance_id']:
        raise ValueError('instance_id must be nonempty')
    return LinearModel(expected, instance['instance_id'])


def build_assignment(instance, with_opening):
    model = new_model(instance, 'M1' if with_opening else 'M2')
    D, C = identifiers(instance['D'], 'D'), identifiers(instance['C'], 'C')
    q = {i: number(instance['q'][i], f'q[{i}]', positive=True) for i in C}
    U = {d: number(instance['U'][d], f'U[{d}]', positive=True) for d in D}
    y = {d: model.add_var('y', [d], binary=True,
         cost=number(instance['F'][d], f'F[{d}]')) for d in D} if with_opening else {}
    z = {(d, i): model.add_var('z', [d, i], binary=True,
         cost=number(instance['g'][d][i], f'g[{d},{i}]')) for d in D for i in C}
    for i in C:
        model.add_row('unique_assignment', [i], {z[d, i]: 1 for d in D}, lower=1, upper=1)
    for d in D:
        terms = {z[d, i]: q[i] for i in C}
        if with_opening:
            terms[y[d]] = -U[d]
            for i in C:
                model.add_row('open_before_service', [d, i], {z[d, i]: 1, y[d]: -1}, upper=0)
        model.add_row('depot_capacity', [d], terms, upper=0 if with_opening else U[d])
    model.metadata = {'D': D, 'C': C, 'cost_convention': 'g is whole-order cost; do not multiply by q',
                      'template_source': 'models/model_catalog.json', 'template_version': '1.0'}
    return model


def build_routing(instance, electric):
    model = new_model(instance, 'M4' if electric else 'M3')
    C = identifiers(instance['C'], 'C')
    depot = instance['depot_id']
    if not isinstance(depot, str) or not depot.strip() or depot in C:
        raise ValueError('depot_id must be a nonempty ID distinct from all customers')
    nodes = [depot] + C
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    q = {i: number(instance['q'][i], f'q[{i}]', positive=True) for i in C}
    U = number(instance['U'], 'U', positive=True)
    Q = number(instance['Q'], 'Q', positive=True)
    h = number(instance['h'], 'h')
    if sum(q.values()) > U + 1e-8:
        raise ValueError('Assigned customer demand exceeds depot U')
    # q > Q produces a mathematically infeasible model. Preflight validation is
    # responsible for rejecting it earlier; keeping it here enables model audits.
    if electric:
        if instance.get('distance_unit') != 'km':
            raise ValueError('M4 requires explicit distance_unit=km')
        B = number(instance['B'], 'B', positive=True)
        p, r0, r1 = (number(instance[k], k) for k in ('p', 'r0', 'r1'))
        if r1 < r0:
            raise ValueError('r1 must be >= r0')
        distance = instance['ell']
        matrix(distance, nodes, 'ell')
        a = {(i,j): distance[i][j] * r0 for i,j in arcs}
        b = {(i,j): distance[i][j] * (r1-r0) / Q for i,j in arcs}
    else:
        matrix(instance['c'], nodes, 'c')
    x, f = {}, {}
    for i, j in arcs:
        cost_x = (h if i == depot else 0) + (p*a[i,j] if electric else instance['c'][i][j])
        x[i,j] = model.add_var('x', [i,j], binary=True, cost=cost_x)
        f[i,j] = model.add_var('f', [i,j], upper=Q, cost=p*b[i,j] if electric else 0)
    for i in C:
        model.add_row('visit_out', [i], {x[i,j]:1 for j in nodes if j != i}, lower=1, upper=1)
        model.add_row('visit_in', [i], {x[j,i]:1 for j in nodes if j != i}, lower=1, upper=1)
        flow = [(f[j,i],1) for j in nodes if j != i] + [(f[i,j],-1) for j in nodes if j != i]
        model.add_row('delivery_flow', [i], flow, lower=q[i], upper=q[i])
        model.add_row('empty_return', [i], {f[i,depot]:1}, lower=0, upper=0)
    model.add_row('depot_balance', [depot], [(x[depot,j],1) for j in C] + [(x[i,depot],-1) for i in C], lower=0, upper=0)
    for i,j in arcs:
        model.add_row('vehicle_capacity', [i,j], {f[i,j]:1, x[i,j]:-Q}, upper=0)
    if electric:
        e = {i: model.add_var('e', [i], upper=B) for i in C}
        for j in C:
            model.add_row('departure_energy', [j], {e[j]:1, x[depot,j]:-a[depot,j], f[depot,j]:-b[depot,j]}, lower=0)
        for i in C:
            for j in C:
                if i != j:
                    # e[j] >= e[i] + E[i,j] - B*(1-x[i,j])
                    model.add_row('energy_propagation', [i,j],
                        {e[j]:1, e[i]:-1, x[i,j]:-a[i,j]-B, f[i,j]:-b[i,j]}, lower=-B)
            model.add_row('return_energy', [i], {e[i]:1, x[i,depot]:a[i,depot], f[i,depot]:b[i,depot]}, upper=B)
    model.metadata = {'depot_id': depot, 'C': C, 'nodes': nodes,
        'template_source': 'models/model_catalog.json', 'template_version': '1.0',
        'depot_capacity_prechecked': True, 'unlimited_fleet_up_to_customer_count': True}
    if electric:
        model.metadata.update(distance_unit='km', energy_bounds='0 <= e[i] <= B via variable bounds',
            energy_expression='ell[i,j] * (r0*x[i,j] + (r1-r0)*f[i,j]/Q)',
            energy_note='e is an auxiliary bound, not necessarily exact energy; derive route energy from x and f')
    return model
