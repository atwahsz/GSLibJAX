"""Anisotropy transforms: rotate and scale coordinates to isotropic space.

Follows GSLIB convention with three rotation angles (ang1, ang2, ang3) in
degrees and three ranges (a_max, a_mid, a_min). Applies rotation then scales by
1/range along each axis.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from .distance import rotation_matrix_3d


def apply_anisotropy(
    coords: jnp.ndarray,
    ranges: Tuple[float, float, float],
    angles_deg: Tuple[float, float, float],
) -> jnp.ndarray:
    """Transform coordinates to isotropic space via rotation+scaling.

    Args:
        coords: (n, d) with d=2 or 3. If d=2, z is assumed 0.
        ranges: (a1, a2, a3) practical ranges along rotated axes.
        angles_deg: (ang1, ang2, ang3) ZYX angles in degrees.

    Returns:
        Transformed coordinates with same leading dimension. Output has d dims.
    """
    d = coords.shape[1]
    if d == 2:
        pts3 = jnp.concatenate([coords, jnp.zeros((coords.shape[0], 1))], axis=1)
    else:
        pts3 = coords
    r = rotation_matrix_3d(angles_deg)
    rot = pts3 @ r.T
    a1, a2, a3 = ranges
    scale = jnp.array([1.0 / jnp.maximum(a1, 1e-12), 1.0 / jnp.maximum(a2, 1e-12), 1.0 / jnp.maximum(a3, 1e-12)])
    tr = rot * scale[None, :]
    if d == 2:
        return tr[:, :2]
    return tr


