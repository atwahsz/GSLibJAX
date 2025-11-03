"""Back-transform utilities for normal scores using stored mapping."""

from __future__ import annotations

import json
from typing import Tuple

import jax.numpy as jnp

from .nscore import NScoreModel, nscore_inverse


def save_nscore_model(path: str, model: NScoreModel) -> None:
    data = {
        "sorted_values": [float(x) for x in list(model.sorted_values.tolist())],
        "quantiles": [float(x) for x in list(model.quantiles.tolist())],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_nscore_model(path: str) -> NScoreModel:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sv = jnp.array(data["sorted_values"], dtype=jnp.float32)
    q = jnp.array(data["quantiles"], dtype=jnp.float32)
    return NScoreModel(sorted_values=sv, quantiles=q)


def back_transform(z: jnp.ndarray, model: NScoreModel) -> jnp.ndarray:
    return nscore_inverse(z, model)


