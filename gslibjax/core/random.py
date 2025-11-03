"""Random number utilities with JAX PRNG.

This module provides a clean, typed interface for PRNG keys. It does not try to
bit-match GSLIB's ACORNI sequence, but exposes helpers to map integer seeds to
JAX keys deterministically.
"""

from __future__ import annotations

from typing import Tuple, Union

import jax
import jax.numpy as jnp

PRNGKey = jax.Array
PRNGKeyLike = Union[int, Tuple[int, int], PRNGKey]


def make_prng_key(seed: PRNGKeyLike) -> PRNGKey:
    """Create a JAX PRNG key from an int, pair of ints, or an existing key.

    Args:
        seed: Integer seed, pair of ints (for two-word seed), or PRNG key.

    Returns:
        JAX PRNG key.
    """
    if isinstance(seed, jax.Array):
        return seed
    if isinstance(seed, tuple):
        a, b = int(seed[0]), int(seed[1])
        return jax.random.PRNGKey(a ^ (b << 1))
    return jax.random.PRNGKey(int(seed))


def split_key(key: PRNGKey, num: int = 2) -> jax.Array:
    """Split a key into ``num`` subkeys.

    Args:
        key: A JAX PRNG key.
        num: Number of subkeys requested.

    Returns:
        Array of shape (num, 2) representing PRNG subkeys.
    """
    return jax.random.split(key, num)


def normal(key: PRNGKey, shape: Tuple[int, ...]) -> jax.Array:
    """Draw standard normal deviates, stateless.

    Args:
        key: PRNG key.
        shape: Output shape.

    Returns:
        Array of shape ``shape`` with N(0,1) samples.
    """
    return jax.random.normal(key, shape)


def uniform(key: PRNGKey, shape: Tuple[int, ...], minval: float = 0.0, maxval: float = 1.0) -> jax.Array:
    """Draw uniform samples in [minval, maxval).

    Args:
        key: PRNG key.
        shape: Output shape.
        minval: Minimum value (inclusive).
        maxval: Maximum value (exclusive).

    Returns:
        Array of shape ``shape`` with uniform samples.
    """
    return jax.random.uniform(key, shape=shape, minval=minval, maxval=maxval)


