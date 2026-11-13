"""Bind an explicit scenario to a fixed model using source values and Python arithmetic."""
from __future__ import annotations

import math
from copy import deepcopy

from .data import distance, travel_cost
from .validation import direct_route_energy, validate_instance


SCENARIO_FIELDS = {
    "scenario_id", "instance_id", "model_id", "model_version", "opened_depots",
    "assignment", "distance_scale_km", "distance_scale_source", "description",
    "data_version", "parameter_values", "model_fingerprint",
}


def bind_scenario(base: dict, scenario: dict) -> dict:
    """Return canonical instances and a separate provenance/status envelope.

    No depot opening, customer assignment, or distance scale is automatically filled.
    For M3/M4 all original customers must be assigned before per-depot decomposition.
    """
    result = {"status": "ready", "missing_fields": [], "errors": [], "warnings": [], "instances": [], "provenance": {}}

    def issue(kind, code, field, message):
        result[kind].append({"code": code, "field": field, "message": message})

    def missing(field):
        if field not in result["missing_fields"]:
            result["missing_fields"].append(field)

    def finish():
        result["status"] = "invalid" if result["errors"] else "needs_clarification" if result["missing_fields"] else "ready"
        if result["status"] != "ready":
            result["instances"] = []
        return result

    if not isinstance(scenario, dict):
        issue("errors", "SCENARIO_TYPE", "$", "Scenario must be a JSON object")
        return finish()
    for key in sorted(set(scenario) - SCENARIO_FIELDS):
        issue("errors", "UNKNOWN_FIELD", key, "Unknown scenario field; unsupported requirements must be reviewed, not ignored")
    for key in ("scenario_id", "instance_id", "model_id", "model_version", "description"):
        value = scenario.get(key)
        if value is None or value == "":
            missing(key)
        elif not isinstance(value, str):
            issue("errors", "FIELD_TYPE", key, "Expected a nonempty string")
    if result["errors"] or result["missing_fields"]:
        return finish()
    model = scenario["model_id"]
    if model not in {"M1", "M2", "M3", "M4"}:
        issue("errors", "MODEL_ID", "model_id", "Expected M1, M2, M3 or M4")
    if scenario["model_version"] != "1.0":
        issue("errors", "MODEL_VERSION", "model_version", "This package implements model version 1.0")
    if scenario["instance_id"] != base.get("instance_id"):
        issue("errors", "INSTANCE_MISMATCH", "instance_id", "Scenario instance_id differs from loaded base")
    if result["errors"]:
        return finish()

    from .workflow import fingerprint
    if scenario.get('data_version') is not None and scenario['data_version']!=fingerprint(base):
        issue('errors','STALE_DATA','data_version','数据版本已变化，必须重新确认数据。')
        return finish()
    values=scenario.get('parameter_values',{})
    if not isinstance(values,dict):
        issue('errors','PARAMETER_VALUES','parameter_values','参数覆盖必须为白名单字段对象。')
        return finish()
    allowed={field+'.'+entity for field in ['U','q']+(['F'] if model=='M1' else []) for entity in base[field]}
    allowed|={'Q','h'}
    if model=='M4':allowed|={'ev.'+field for field in ['B','p','r0','r1'] if field in base['ev']}
    changes=[]
    original_base=base
    base=deepcopy(base)
    for field,value in values.items():
        if field not in allowed or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            issue('errors','PARAMETER_FIELD',field,'参数字段不在白名单内或数值不合法。');continue
        if value<0 or (field in {'Q','ev.B'} or field.startswith(('U.','q.'))) and value==0:
            issue('errors','PARAMETER_RANGE',field,'参数值超出允许范围。');continue
        parts=field.split('.');target=base if len(parts)==1 else base[parts[0]];key=parts[-1]
        changes.append({'field':field,'source_value':target[key],'effective_value':value,'source':'validated_parameter_update'})
        target[key]=value
    if result['errors']:return finish()
    result["provenance"] = {
        "scenario_id": scenario["scenario_id"], "scenario_description": scenario["description"],
        "source": deepcopy(base["provenance"]), "source_cells": deepcopy(base["source_cells"]),
        "node_id_convention": "Original depot ID retained; customer spreadsheet integer n maps to cn.",
        "model_version": "1.0", "transformations": [], "scenario_values": deepcopy(scenario),
        "data_version":fingerprint(original_base), "parameter_updates":changes,
    }
    D, C = list(base["D"]), list(base["C"])
    opened, assignment = scenario.get("opened_depots"), scenario.get("assignment")
    if model == "M1":
        if opened is not None:
            issue("errors", "MODEL_SCOPE", "opened_depots", "M1 chooses opening decisions; opened_depots must be null")
        if assignment is not None:
            issue("errors", "MODEL_SCOPE", "assignment", "M1 chooses assignments; assignment must be null")
    else:
        if opened is None:
            missing("opened_depots")
        elif not isinstance(opened, list) or not opened or any(not isinstance(d, str) for d in opened):
            issue("errors", "OPENED_DEPOTS_TYPE", "opened_depots", "Supply a nonempty explicit list of depot IDs")
        else:
            if len(set(opened)) != len(opened):
                issue("errors", "DUPLICATE_DEPOT", "opened_depots", "Depot IDs must be unique")
            unknown = set(opened) - set(D)
            if unknown:
                issue("errors", "UNKNOWN_DEPOT", "opened_depots", f"Unknown depot IDs: {sorted(unknown)}")
            D = list(opened)
        if model == "M2" and assignment is not None:
            issue("errors", "MODEL_SCOPE", "assignment", "M2 chooses assignments; assignment must be null")
    if model in {"M3", "M4"}:
        if assignment is None:
            missing("assignment")
        elif not isinstance(assignment, dict):
            issue("errors", "ASSIGNMENT_TYPE", "assignment", "Expected customer-to-depot object")
        else:
            absent, extra = set(C) - set(assignment), set(assignment) - set(C)
            for customer in sorted(absent):
                missing(f"assignment.{customer}")
            if extra:
                issue("errors", "UNKNOWN_CUSTOMER", "assignment", f"Unknown customer IDs: {sorted(extra)}")
            for customer, depot in assignment.items():
                if depot is None:
                    missing(f"assignment.{customer}")
                elif not isinstance(depot, str) or depot not in base["D"]:
                    issue("errors", "ASSIGNMENT_DEPOT", f"assignment.{customer}", f"Unknown depot ID: {depot!r}")
                elif opened is not None and isinstance(opened, list) and depot not in opened:
                    issue("errors", "CLOSED_DEPOT", f"assignment.{customer}", f"Depot {depot} is not in opened_depots")
    if model == "M4":
        for key in ("B", "p", "r0", "r1"):
            if key not in base["ev"]:
                missing(f"data.ev.{key}")
        scale, source = scenario.get("distance_scale_km"), scenario.get("distance_scale_source")
        if scale is None:
            missing("distance_scale_km")
        elif isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            issue("errors", "DISTANCE_SCALE", "distance_scale_km", "Provide a finite positive number of km per coordinate unit")
        if source is None or (isinstance(source, str) and not source.strip()):
            missing("distance_scale_source")
        elif not isinstance(source, str):
            issue("errors", "DISTANCE_SOURCE", "distance_scale_source", "Describe the verified conversion or explicit experimental assumption")
        if isinstance(scale, (int, float)) and not isinstance(scale, bool) and math.isfinite(scale) and scale > 0 and isinstance(source, str) and source.strip():
            issue("warnings", "EXPLICIT_DISTANCE_ASSUMPTION", "distance_scale_km", "Coordinate units are unspecified in the workbook; scenario scaling and its stated source are used. EV rates are interpreted per km consistently with this explicit assumption.")
            result["provenance"]["transformations"].append({"field": "ell", "formula": "Euclidean(coordinates[i],coordinates[j]) * distance_scale_km", "distance_unit": "km", "distance_scale_km": scale, "source": source})
    else:
        if scenario.get("distance_scale_km") is not None or scenario.get("distance_scale_source") is not None:
            issue("errors", "MODEL_SCOPE", "distance_scale_km", "Distance scale fields must be null outside M4")
    if result["errors"] or result["missing_fields"]:
        return finish()

    # q <= Q is also mandatory for M1/M2's independent-direct-delivery cost convention.
    oversized = [c for c in C if base["q"][c] > base["Q"] + 1e-9]
    if oversized:
        issue("errors", "VEHICLE_CAPACITY", "data.q", f"Indivisible demand exceeds Q for {oversized}")
        return finish()
    common = {"model_id": model, "model_version": "1.0", "instance_id": base["instance_id"]}
    if model in {"M1", "M2"}:
        instance = {**common, "D": D, "C": C, "q": dict(base["q"]),
                    "U": {d: base["U"][d] for d in D},
                    "g": {d: {c: base["h"] + travel_cost(base, d, c) + travel_cost(base, c, d) for c in C} for d in D}}
        if model == "M1":
            instance["F"] = {d: base["F"][d] for d in D}
        result["instances"].append(instance)
        result["provenance"]["transformations"].append({"field": "g", "formula": "h + travel_cost(d,c) + travel_cost(c,d)", "travel_cost_formula": "floor(100*Euclidean)" if base["route_cost_type"] == 0 else "Euclidean", "interpretation": "Whole-customer direct-delivery cost, not multiplied by q again", "Q_precheck": base["Q"]})
        issue("warnings", "FEASIBILITY_NOT_CERTIFIED", "assignment", "Passing aggregate capacity checks does not prove an indivisible assignment exists; obtain a feasible assignment from a solver or a verified baseline.")
    else:
        for depot in D:
            clients = [c for c in C if assignment[c] == depot]
            if not clients:
                continue
            nodes = [depot] + clients
            instance = {**common, "depot_id": depot, "C": clients,
                        "q": {c: base["q"][c] for c in clients},
                        "U": base["U"][depot], "Q": base["Q"], "h": base["h"]}
            if model == "M3":
                instance["c"] = {a: {b: travel_cost(base, a, b) for b in nodes if b != a} for a in nodes}
            else:
                instance.update(base["ev"])
                instance["distance_unit"] = "km"
                instance["ell"] = {a: {b: distance(base, a, b) * scenario["distance_scale_km"] for b in nodes if b != a} for a in nodes}
            result["instances"].append(instance)
        result["provenance"]["transformations"].append({"field": "per_depot_instances", "rule": "Explicit full assignment decomposed by opened depot; empty customer groups skipped", "opened_depots": D, "assignment_source": "scenario.assignment"})
        if model == "M3":
            result["provenance"]["transformations"].append({"field": "c", "formula": "floor(100*Euclidean)" if base["route_cost_type"] == 0 else "Euclidean"})
    for index, instance in enumerate(result["instances"]):
        validation_errors = validate_instance(instance)
        for error in validation_errors:
            issue("errors", "INSTANCE_VALIDATION", f"instances[{index}]", error)
        if model == "M4" and not validation_errors:
            failed = [c for c in instance["C"] if direct_route_energy(instance, c) > instance["B"] + 1e-9]
            if failed:
                issue("warnings", "DIRECT_BASELINE_NOT_CERTIFIED", f"instances[{index}]", f"Direct-return energy baseline exceeds B for {failed}; this sufficient baseline failed, which is not a declaration that the original model is infeasible")
            else:
                result["provenance"].setdefault("baseline_certificates", []).append({"depot_id": instance["depot_id"], "type": "one_customer_per_route", "capacity_and_energy_feasible": True, "routes": [[instance["depot_id"], c, instance["depot_id"]] for c in instance["C"]], "route_energies": {c: direct_route_energy(instance, c) for c in instance["C"]}})
        elif model == "M3" and not validation_errors:
            result["provenance"].setdefault("baseline_certificates", []).append({"depot_id": instance["depot_id"], "type": "one_customer_per_route", "capacity_feasible": True, "routes": [[instance["depot_id"], c, instance["depot_id"]] for c in instance["C"]]})
    return finish()
