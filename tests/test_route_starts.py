"""Upstream routes must be valid under the complete EV model before use."""
import copy
import importlib.util
import unittest

import main
from orllm.linear_model import build_routing
from orllm.solvers import _capacity_vehicle_bound


def electric_instance(capacity=5, battery=10):
    nodes = ["d1", "c1", "c2"]
    return dict(model_id="M4", model_version="1.0", instance_id="checked_reference",
                depot_id="d1", C=nodes[1:], q={"c1": 2, "c2": 3}, U=20,
                Q=capacity, h=100, B=battery, p=1, r0=0.1, r1=0.2,
                distance_unit="km", ell={a: {b: 1 for b in nodes if b != a} for a in nodes})


class RouteStartTests(unittest.TestCase):
    def test_reconstructs_remaining_load_and_exact_arrival_energy(self):
        instance = electric_instance()
        model = build_routing(instance, True)
        start = main.reference_route_start(instance, model,
                    [{"depot_id": "d1", "nodes": ["d1", "c1", "c2", "d1"]}])
        self.assertIsNotNone(start)
        self.assertTrue(model.check_primal(start["values"])["valid"])
        decoded = {(v["family"], tuple(v["indices"])): x
                   for v, x in zip(model.variables, start["values"])}
        self.assertEqual(decoded["f", ("d1", "c1")], 5)
        self.assertEqual(decoded["f", ("c1", "c2")], 3)
        self.assertEqual(decoded["f", ("c2", "d1")], 0)
        self.assertAlmostEqual(decoded["e", ("c1",)], 0.2)
        self.assertAlmostEqual(decoded["e", ("c2",)], 0.36)
        self.assertAlmostEqual(start["objective"], 100.46)

    def test_rejects_battery_capacity_and_incomplete_reference(self):
        route = [{"depot_id": "d1", "nodes": ["d1", "c1", "c2", "d1"]}]
        for instance in (electric_instance(battery=0.1), electric_instance(capacity=4)):
            with self.subTest(instance=instance):
                self.assertIsNone(main.reference_route_start(instance, build_routing(instance, True), route))
        instance = electric_instance()
        self.assertIsNone(main.reference_route_start(instance, build_routing(instance, True),
            [{"depot_id": "d1", "nodes": ["d1", "c1", "d1"]}]))

    @unittest.skipUnless(importlib.util.find_spec("coptpy"), "optional copt unavailable")
    def test_copt_uses_implied_vehicle_bound_without_modifying_export(self):
        instance = electric_instance(capacity=4)
        model = build_routing(instance, True)
        exported = copy.deepcopy(model.to_dict())
        routes = [{"depot_id": "d1", "nodes": ["d1", c, "d1"]} for c in instance["C"]]
        start = main.reference_route_start(instance, model, routes)
        model.mip_start = start["values"]
        self.assertEqual(_capacity_vehicle_bound(model)["certificate"]["minimum_vehicles"], 2)
        result = model.solve_copt(time_limit=5, mip_gap=0.08)
        if result.get("error_type") == "license_error":
            self.skipTest("COPT license unavailable")
        self.assertEqual(result["status"], "optimal", result)
        self.assertAlmostEqual(result["objective"], 200.525)
        self.assertTrue(result["primal_check"]["valid"])
        self.assertTrue(result["mip_start_used"])
        self.assertLessEqual(result["mip_gap"], 0.08)
        self.assertEqual(result["capacity_vehicle_bound"]["minimum_vehicles"], 2)
        self.assertEqual(model.to_dict(), exported)

    def test_capacity_packing_backtracks_and_checks_the_full_ev_model(self):
        instance = electric_instance(capacity=10)
        instance["C"] = [f"c{i}" for i in range(1, 7)]
        instance["q"] = dict(zip(instance["C"], [6, 5, 3, 2, 2, 2]))
        nodes = ["d1"] + instance["C"]
        instance["ell"] = {a: {b: 1 for b in nodes if b != a} for a in nodes}
        model = build_routing(instance, True)
        start = main.packing_route_start(instance, model)
        self.assertIsNotNone(start)
        self.assertTrue(model.check_primal(start["values"])["valid"])
        departures = sum(value for v, value in zip(model.variables, start["values"])
                         if v["family"] == "x" and v["indices"][0] == "d1")
        self.assertEqual(departures, 2)
        self.assertEqual(start["source"], "capacity_packing_of_fixed_assignment")
        instance["B"] = 0.01
        self.assertIsNone(main.packing_route_start(instance, build_routing(instance, True)))

    def test_failed_minimum_capacity_packing_is_not_an_infeasibility_claim(self):
        instance = electric_instance(capacity=10)
        instance["C"] = [f"c{i}" for i in range(1, 6)]
        instance["q"] = {c: 4 for c in instance["C"]}
        nodes = ["d1"] + instance["C"]
        instance["ell"] = {a: {b: 1 for b in nodes if b != a} for a in nodes}
        model = build_routing(instance, True)
        self.assertIsNone(main.packing_route_start(instance, model))
        routes = [{"depot_id": "d1", "nodes": ["d1", *clients, "d1"]}
                  for clients in (["c1", "c2"], ["c3", "c4"], ["c5"])]
        self.assertIsNotNone(main.reference_route_start(instance, model, routes))

    def test_best_checked_start_is_retained_with_the_native_bound(self):
        instance = electric_instance()
        model = build_routing(instance, True)
        initial = main.packing_route_start(instance, model)
        two_routes = [{"depot_id": "d1", "nodes": ["d1", c, "d1"]} for c in instance["C"]]
        worse = main.reference_route_start(instance, model, two_routes)
        native = model._solution("feasible_limit", "Time limit", worse["values"], solver="copt",
                                 mip_dual_bound=100.0, mip_gap=0.5, runtime_seconds=300)
        native["objective_check"] = {"valid": True}
        result = main.best_checked_incumbent(model, native, initial=initial)
        self.assertAlmostEqual(result["objective"], 100.46)
        self.assertEqual(result["mip_dual_bound"], 100)
        self.assertLess(result["mip_gap"], 0.08)
        self.assertEqual(result["incumbent_source"], "capacity_packing_of_fixed_assignment")
        self.assertEqual(result["native_objective"], native["objective"])
        self.assertIsNone(result["objective_check"]["solver_reported"])
        self.assertTrue(result["primal_check"]["valid"])

    def test_better_previous_point_uses_new_bound_without_modifying_history(self):
        instance = electric_instance()
        model = build_routing(instance, True)
        start = main.packing_route_start(instance, model)
        previous = model._solution("feasible_limit", "Old limit", start["values"], solver="copt",
                                   mip_dual_bound=90, mip_gap=0.104, runtime_seconds=30)
        previous["objective_check"] = {"valid": True}
        original = copy.deepcopy(previous)
        two_routes = [{"depot_id": "d1", "nodes": ["d1", c, "d1"]} for c in instance["C"]]
        worse = main.reference_route_start(instance, model, two_routes)
        native = model._solution("feasible_limit", "New limit", worse["values"], solver="copt",
                                 mip_dual_bound=100, mip_gap=0.5, runtime_seconds=300)
        native["objective_check"] = {"valid": True}
        result = main.best_checked_incumbent(model, native, previous)
        self.assertEqual(result["mip_dual_bound"], 100)
        self.assertLess(result["mip_gap"], 0.08)
        self.assertEqual(result["incumbent_source"], "previous_solution")
        self.assertEqual(previous, original)

    def test_checked_point_without_a_valid_bound_is_only_feasible(self):
        instance = electric_instance()
        model = build_routing(instance, True)
        initial = main.packing_route_start(instance, model)
        result = main.best_checked_incumbent(model, {"status": "optimal"}, initial=initial)
        self.assertEqual(result["status"], "feasible_limit")
        self.assertIsNone(result["mip_gap"])


if __name__ == "__main__":
    unittest.main()
