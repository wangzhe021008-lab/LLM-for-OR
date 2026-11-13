"""Deterministic fixed-model dispatch; LLM outputs never change constraints."""
from . import M1, M2, M3, M4

_BUILDERS = {'M1': M1.build_model, 'M2': M2.build_model, 'M3': M3.build_model, 'M4': M4.build_model}

def build_model(instance):
    model_id = instance.get('model_id')
    if model_id not in _BUILDERS:
        raise ValueError(f'Unknown model_id: {model_id}')
    return _BUILDERS[model_id](instance)
