"""M1: fixed capacitated facility opening and whole-customer assignment."""
from orllm.linear_model import build_assignment

def build_model(instance):
    return build_assignment(instance, with_opening=True)
