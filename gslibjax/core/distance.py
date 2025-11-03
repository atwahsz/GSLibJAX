"""Distance and rotation utilities in JAX."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def squared_euclidean(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise squared Euclidean distances between two point sets.

    Args:
        a: Array with shape (na, d).
        b: Array with shape (nb, d).

    Returns:
        Matrix of shape (na, nb) with squared distances.
    """
    a2 = jnp.sum(a * a, axis=1, keepdims=True)
    b2 = jnp.sum(b * b, axis=1, keepdims=True).T
    return a2 + b2 - 2.0 * (a @ b.T)


def rotation_matrix_3d(angles_deg: Tuple[float, float, float]) -> jnp.ndarray:
    """Construct a 3D rotation matrix using ZYX (yaw-pitch-roll) in degrees.

    Args:
        angles_deg: (ang1, ang2, ang3) in degrees.

    Returns:
        3x3 rotation matrix.
    """
    a1, a2, a3 = [jnp.deg2rad(x) for x in angles_deg]
    ca1, sa1 = jnp.cos(a1), jnp.sin(a1)
    ca2, sa2 = jnp.cos(a2), jnp.sin(a2)
    ca3, sa3 = jnp.cos(a3), jnp.sin(a3)

    rz = jnp.array([[ca1, -sa1, 0.0], [sa1, ca1, 0.0], [0.0, 0.0, 1.0]])
    ry = jnp.array([[ca2, 0.0, sa2], [0.0, 1.0, 0.0], [-sa2, 0.0, ca2]])
    rx = jnp.array([[1.0, 0.0, 0.0], [0.0, ca3, -sa3], [0.0, sa3, ca3]])
    return rz @ ry @ rx


def rotate_points(points: jnp.ndarray, angles_deg: Tuple[float, float, float]) -> jnp.ndarray:
    """Rotate points by angles using ZYX order.

    Args:
        points: Array shape (n, 3).
        angles_deg: (ang1, ang2, ang3) degrees.

    Returns:
        Rotated points with shape (n, 3).
    """
    r = rotation_matrix_3d(angles_deg)
    return points @ r.T


