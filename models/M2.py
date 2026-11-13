"""M2: fixed assignment among already-open, allowed depots."""
from orllm.linear_model import build_assignment

def build_model(instance):
    return build_assignment(instance, with_opening=False)
