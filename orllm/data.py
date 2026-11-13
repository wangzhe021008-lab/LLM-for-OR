"""Read the supplied workbook without changing it; preserve source cells and notes."""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path

from openpyxl import load_workbook


PARAMETER_LABELS = {
    "Q": "vehicle capacity",
    "h": "opening cost of a route (cost of a vehicle)",
    "route_cost_type": "route cost type",
    "p": "Unit cost of electricity(C_e)",
    "r0": "Electricity consumption per unit distance for an empty EV.(p'_0)",
    "r1": "Electricity consumption per unit distance for a fully loaded EV(p'_*)",
    "B": "EV_battery_capacity_MAX(kWh)=(QB_k)",
    "carbon_price": "Unit cost (or penalty) for carbon emission treatment(C_em)",
    "emission_coefficient": "Emission coefficie variation(G)",
    "diesel_price": "Unit cost of diesel fuel(C_f)",
    "diesel_empty_rate": "Fuel consumption per unit distance when an ICEV is empty(p_0)",
    "diesel_full_rate": "Fuel consumption per unit distance when an ICEV is fully loaded(p_*)",
}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _number(value, context: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context}: expected a finite number, got {value!r}")
    return value


def prepare_data(workbook_path, output_dir) -> dict:
    """Generate capacity-specific base instances, manifest, and original-cell snapshots.

    Unknown coordinate distance units remain unknown. No open depot or customer
    assignment is inferred. Cached formula values are preserved, not recomputed.
    """
    workbook_path, output_dir = Path(workbook_path), Path(output_dir)
    formulas = load_workbook(workbook_path, data_only=False)
    values = load_workbook(workbook_path, data_only=True)
    manifest = {
        "format_version": "1.0", "workbook": workbook_path.name,
        "workbook_sha256": hashlib.sha256(workbook_path.read_bytes()).hexdigest(),
        "sheet_count": len(formulas.sheetnames), "split_sheet_count": 0,
        "instance_count": 0, "instances": [],
        "distance_unit": None,
        "notes": [
            "Capacity alternatives such as 70/150 are separate experimental instances, not a heterogeneous fleet.",
            "travel_cost = floor(100 * Euclidean distance) for route_cost_type=0; type=1 uses Euclidean distance.",
            "Coordinate physical unit is unspecified. M4 requires an explicit distance_scale_km and assumption source.",
            "Raw workbook and all nonempty source cells (including unused columns, labels, formulas, and cached values) are retained.",
            "No assignment or open-depot configuration has been inferred from coordinates.",
        ],
    }
    for sheet in formulas:
        cached = values[sheet.title]
        snapshot = {"sheet": sheet.title, "cells": {}}
        for row in sheet:
            for cell in row:
                if cell.value is not None:
                    record = {"value": cell.value}
                    if cell.data_type == "f":
                        record["cached_value"] = cached[cell.coordinate].value
                    snapshot["cells"][cell.coordinate] = record
        _write(output_dir / "source_sheets" / f"{sheet.title}.json", snapshot)

        base = {
            "format_version": "1.0", "instance_id": None, "source_sheet": sheet.title,
            "D": [], "C": [], "coordinates": {}, "q": {}, "U": {}, "F": {},
            "Q": None, "h": None, "route_cost_type": None, "ev": {},
            "extra_parameters": {}, "coordinate_distance_unit": None,
            "source_cells": {}, "original_notes": [],
        }
        for row in sheet:
            identifier = row[0].value
            if isinstance(identifier, str) and re.fullmatch(r"d\d+", identifier):
                node = identifier
                base["D"].append(node)
                base["U"][node] = _number(cached.cell(row[0].row, 4).value, f"{sheet.title}!D{row[0].row}")
                base["F"][node] = _number(cached.cell(row[0].row, 5).value, f"{sheet.title}!E{row[0].row}")
                base["source_cells"][f"U.{node}"] = f"D{row[0].row}"
                base["source_cells"][f"F.{node}"] = f"E{row[0].row}"
                if sheet.cell(row[0].row, 6).value is not None:
                    base["extra_parameters"].setdefault("depot_carbon", {})[node] = cached.cell(row[0].row, 6).value
                    base["source_cells"][f"extra_parameters.depot_carbon.{node}"] = f"F{row[0].row}"
            elif isinstance(identifier, (int, float)) and not isinstance(identifier, bool) and float(identifier).is_integer():
                node = f"c{int(identifier)}"
                base["C"].append(node)
                base["q"][node] = _number(cached.cell(row[0].row, 4).value, f"{sheet.title}!D{row[0].row}")
                base["source_cells"][f"q.{node}"] = f"D{row[0].row}"
            else:
                continue
            if node in base["coordinates"]:
                raise ValueError(f"Duplicate node {node} in {sheet.title}")
            base["coordinates"][node] = {
                "x": _number(cached.cell(row[0].row, 2).value, f"{sheet.title}!B{row[0].row}"),
                "y": _number(cached.cell(row[0].row, 3).value, f"{sheet.title}!C{row[0].row}"),
            }
            base["source_cells"][f"coordinates.{node}"] = [f"B{row[0].row}", f"C{row[0].row}"]
            base["source_cells"][f"node_id.{node}"] = f"A{row[0].row}"

        found = {}
        for row in sheet:
            for cell in row:
                if isinstance(cell.value, str):
                    for key, label in PARAMETER_LABELS.items():
                        if cell.value.strip().casefold() == label.casefold():
                            value_cell = cached.cell(cell.row, cell.column + 1)
                            if key in found:
                                raise ValueError(f"Ambiguous label {label!r} in {sheet.title}")
                            found[key] = value_cell.value
                            base["source_cells"][key] = value_cell.coordinate
                    if "To calculate the matrix distance" in cell.value:
                        base["original_notes"].append({"cell": cell.coordinate, "text": cell.value})
        for key in ("Q", "h", "route_cost_type"):
            if key not in found:
                raise ValueError(f"Missing {key} in {sheet.title}")
        capacity_value = found["Q"]
        capacities = [float(x.strip()) for x in capacity_value.split("/")] if isinstance(capacity_value, str) else [capacity_value]
        if len(set(capacities)) != len(capacities) or any(_number(x, "Q") <= 0 for x in capacities):
            raise ValueError(f"Invalid capacity options in {sheet.title}: {capacity_value!r}")
        if len(capacities) > 1:
            manifest["split_sheet_count"] += 1
        base["h"] = _number(found["h"], "h")
        base["route_cost_type"] = found["route_cost_type"]
        if base["route_cost_type"] not in (0, 1):
            raise ValueError(f"Unsupported route cost type in {sheet.title}")
        base["ev"] = {k: _number(found[k], k) for k in ("B", "p", "r0", "r1") if k in found}
        base["extra_parameters"].update({k: v for k, v in found.items() if k not in {"Q", "h", "route_cost_type", "B", "p", "r0", "r1"}})
        base["original_vehicle_capacity"] = capacity_value
        for capacity in capacities:
            instance = deepcopy(base)
            instance["Q"] = int(capacity) if float(capacity).is_integer() else capacity
            instance_id = f"{sheet.title}__Q{capacity:g}"
            instance["instance_id"] = instance_id
            instance["provenance"] = {
                "workbook": workbook_path.name, "workbook_sha256": manifest["workbook_sha256"],
                "sheet": sheet.title, "source_snapshot": f"source_sheets/{sheet.title}.json",
                "capacity_choice": instance["Q"], "capacity_source_value": capacity_value,
                "capacity_source_cell": base["source_cells"]["Q"],
            }
            _write(output_dir / "instances" / f"{instance_id}.json", instance)
            manifest["instances"].append({
                "instance_id": instance_id, "source_sheet": sheet.title,
                "Q": instance["Q"], "depots": len(instance["D"]), "customers": len(instance["C"]),
                "has_ev_parameters": set(instance["ev"]) == {"B", "p", "r0", "r1"},
                "path": f"instances/{instance_id}.json",
            })
    manifest["instance_count"] = len(manifest["instances"])
    _write(output_dir / "manifest.json", manifest)
    formulas.close()
    values.close()
    return manifest


def load_base(root, instance_id: str) -> dict:
    """Load an ID from the manifest; root may be package root or data/processed."""
    root = Path(root)
    processed = root if (root / "manifest.json").exists() else root / "data" / "processed"
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    entry = next((item for item in manifest["instances"] if item["instance_id"] == instance_id), None)
    if entry is None:
        raise ValueError(f"Unknown instance_id: {instance_id!r}; choose an exact ID from data/processed/manifest.json")
    path = (processed / entry["path"]).resolve()
    if not path.is_relative_to(processed.resolve()):
        raise ValueError("Invalid path in manifest")
    return json.loads(path.read_text(encoding="utf-8"))


def distance(base: dict, first: str, second: str) -> float:
    a, b = base["coordinates"][first], base["coordinates"][second]
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def travel_cost(base: dict, first: str, second: str):
    value = distance(base, first, second)
    return math.floor(100 * value) if base["route_cost_type"] == 0 else value
