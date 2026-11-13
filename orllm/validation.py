"""JSON Schema validation followed by domain and cross-field validation."""
from __future__ import annotations

import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def get_schema(model_id: str) -> dict:
    if not isinstance(model_id, str) or model_id not in {"M1", "M2", "M3", "M4"}:
        raise ValueError(f"Unsupported model_id: {model_id!r}")
    return json.loads((SCHEMA_DIR / f"{model_id}.schema.json").read_text(encoding="utf-8"))


def _nonfinite(value, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        yield f"{path}: nonfinite numeric value is forbidden"
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _nonfinite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _nonfinite(child, f"{path}[{index}]")


def validate_instance(instance) -> list[str]:
    """Return actionable errors; an empty list means structurally valid input.

    M1/M2 necessary capacity checks do not prove unsplittable assignment feasibility.
    M4 direct-route energy is a separate sufficient-baseline warning, not an error.
    """
    if not isinstance(instance, dict):
        return ["$: instance must be a JSON object"]
    model_id = instance.get("model_id")
    if not isinstance(model_id, str) or model_id not in {"M1", "M2", "M3", "M4"}:
        return ["$.model_id: expected M1, M2, M3, or M4"]
    errors = list(_nonfinite(instance))
    schema = get_schema(model_id)
    Draft202012Validator.check_schema(schema)
    for issue in sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda e: tuple(str(x) for x in e.path)):
        path = ".".join(str(x) for x in issue.path)
        errors.append(f"$.{path}: {issue.message}")
    if errors:
        return errors
    customers = set(instance["C"])

    def exact_keys(mapping, expected, path):
        actual = set(mapping)
        if actual != expected:
            missing, extra = sorted(expected - actual), sorted(actual - expected)
            errors.append(f"{path}: keys mismatch; missing={missing}, extra={extra}")

    exact_keys(instance["q"], customers, "q")
    if model_id in {"M1", "M2"}:
        depots = set(instance["D"])
        if customers & depots:
            errors.append("D and C must have disjoint node IDs")
        exact_keys(instance["U"], depots, "U")
        exact_keys(instance["g"], depots, "g")
        if model_id == "M1":
            exact_keys(instance["F"], depots, "F")
        for depot, row in instance["g"].items():
            exact_keys(row, customers, f"g.{depot}")
        if sum(instance["q"].values()) > sum(instance["U"].values()) + 1e-9:
            errors.append("Total demand exceeds total available depot capacity (necessary condition fails)")
        if max(instance["q"].values()) > max(instance["U"].values()) + 1e-9:
            errors.append("At least one indivisible customer demand exceeds every depot capacity")
    else:
        depot = instance["depot_id"]
        if depot in customers:
            errors.append("depot_id must not be in C")
        nodes = customers | {depot}
        matrix_name = "c" if model_id == "M3" else "ell"
        matrix = instance[matrix_name]
        exact_keys(matrix, nodes, matrix_name)
        for node, row in matrix.items():
            exact_keys(row, nodes - {node}, f"{matrix_name}.{node}")
        if sum(instance["q"].values()) > instance["U"] + 1e-9:
            errors.append("Assigned total demand exceeds this depot capacity U")
        if any(value > instance["Q"] + 1e-9 for value in instance["q"].values()):
            errors.append("At least one indivisible customer demand exceeds vehicle capacity Q")
        if model_id == "M4" and instance["r1"] < instance["r0"]:
            errors.append("r1 must be greater than or equal to r0")
    return errors


def direct_route_energy(instance: dict, customer: str) -> float:
    depot = instance["depot_id"]
    return (instance["ell"][depot][customer]
            * (instance["r0"] + (instance["r1"] - instance["r0"]) * instance["q"][customer] / instance["Q"])
            + instance["ell"][customer][depot] * instance["r0"])
