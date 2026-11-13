"""M4: fixed per-depot electric routing, load-dependent energy, no recharging."""
from orllm.linear_model import build_routing

def build_model(instance):
    return build_routing(instance, electric=True)
