from __future__ import annotations
# DiamondWorldJAX — JAX/NumPyro/Flax baseball simulation package

from .schema import PITCH_TYPE_IDX, PA_OUTCOME_IDX, N_PITCH_TYPES, N_PA_OUTCOMES


def __getattr__(name):
    # Lazy: importing the joint model pulls in JAX/NumPyro/Flax, which the
    # pure data/sim modules (and the Windows dev venv) don't need.
    if name == "diamondworld_model":
        from .model import diamondworld_model
        return diamondworld_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
