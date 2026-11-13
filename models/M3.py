"""M3: fixed per-depot capacitated vehicle routing with delivery flow."""
from orllm.linear_model import build_routing

def build_model(instance):
    return build_routing(instance, electric=False)
