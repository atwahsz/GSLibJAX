"""Experimental variogram calculator (gamv) in JAX-compatible style.

Computes directional and omnidirectional experimental variograms with binning.
This is a simplified version focusing on isotropic distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from ..core.anisotropy import apply_anisotropy


@dataclass
class VariogramBin:
    center: float
    num_pairs: int
    gamma: float


def pairwise_variogram(values: jnp.ndarray, coords: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute upper-triangular pairwise distances and semivariances.

    Args:
        values: shape (n,)
        coords: shape (n, d)

    Returns:
        (distances, semivariances) for i<j stacked to 1D shape (m,)
    """
    n = values.shape[0]
    # Build all i<j indices
    ii, jj = jnp.triu_indices(n, k=1)
    dv = values[ii] - values[jj]
    gamma = 0.5 * (dv * dv)
    d = jnp.linalg.norm(coords[ii] - coords[jj], axis=1)
    return d, gamma


def experimental_variogram(
    values: jnp.ndarray,
    coords: jnp.ndarray,
    lag: float,
    nlag: int,
    ranges: Tuple[float, float, float] | None = None,
    angles_deg: Tuple[float, float, float] | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute omnidirectional experimental variogram with fixed lag and nlag.

    Args:
        values: (n,)
        coords: (n, d)
        lag: lag width
        nlag: number of lags

    Returns:
        (lag_centers, gamma, npairs)
    """
    if ranges is not None and angles_deg is not None:
        tcoords = apply_anisotropy(coords, ranges, angles_deg)
    else:
        tcoords = coords
    d, g = pairwise_variogram(values, tcoords)
    edges = jnp.arange(0.0, (nlag + 1) * lag, lag)
    centers = edges[:-1] + 0.5 * lag
    # Bin distances
    bin_idx = jnp.clip(jnp.floor(d / lag).astype(jnp.int32), 0, nlag - 1)
    npairs = jnp.bincount(bin_idx, length=nlag)
    sums = jnp.bincount(bin_idx, weights=g, length=nlag)
    gamma = jnp.where(npairs > 0, sums / npairs, 0.0)
    return centers, gamma, npairs


def _direction_unit(azimuth_deg: float) -> jnp.ndarray:
    # 2D unit vector for azimuth measured from x-axis (east), CCW.
    a = jnp.deg2rad(azimuth_deg)
    return jnp.array([jnp.cos(a), jnp.sin(a)])


def directional_variogram(
    values: jnp.ndarray,
    coords: jnp.ndarray,
    lag: float,
    nlag: int,
    azimuth_deg: float,
    atol_deg: float,
    bandwidth: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Directional experimental variogram (2D) with angular tolerance and bandwidth.

    Pairs (i,j) are accepted if the perpendicular distance of the vector
    (xj - xi) to the direction line is <= bandwidth AND the angular deviation
    from azimuth is <= atol.

    Args:
        values: (n,)
        coords: (n, 2)
        lag: lag width
        nlag: number of lags
        azimuth_deg: direction azimuth in degrees
        atol_deg: angular tolerance in degrees
        bandwidth: max perpendicular distance to direction line

    Returns:
        (lag_centers, gamma, npairs)
    """
    n = values.shape[0]
    ii, jj = jnp.triu_indices(n, k=1)
    dv = values[ii] - values[jj]
    gamma_ij = 0.5 * (dv * dv)
    h = coords[jj] - coords[ii]  # (m, 2)
    hlen = jnp.linalg.norm(h, axis=1)
    h_unit = jnp.where(hlen[:, None] > 0, h / hlen[:, None], 0.0)

    u = _direction_unit(azimuth_deg)
    # Angular deviation via dot product
    cos_dev = jnp.clip(jnp.sum(h_unit * u[None, :], axis=1), -1.0, 1.0)
    ang_dev = jnp.degrees(jnp.arccos(cos_dev))
    # Perpendicular distance to direction line: |h x u| in 2D = |hx*uy - hy*ux|
    perp = jnp.abs(h[:, 0] * u[1] - h[:, 1] * u[0])

    mask = (ang_dev <= atol_deg) & (perp <= bandwidth)

    dpar = jnp.abs(jnp.sum(h * u[None, :], axis=1))  # projection length along direction
    dpar = jnp.where(mask, dpar, -1.0)

    edges = jnp.arange(0.0, (nlag + 1) * lag, lag)
    centers = edges[:-1] + 0.5 * lag
    bin_idx = jnp.clip(jnp.floor(dpar / lag).astype(jnp.int32), 0, nlag - 1)
    valid = dpar >= 0
    bin_idx = jnp.where(valid, bin_idx, -1)

    # Use bincount by filtering invalid pairs
    def bincount_masked(idx: jnp.ndarray, weights: jnp.ndarray, length: int) -> jnp.ndarray:
        idx_clipped = jnp.clip(idx, 0, length - 1)
        w = jnp.where(idx >= 0, weights, 0.0)
        return jnp.bincount(idx_clipped, weights=w, length=length)

    npairs = bincount_masked(bin_idx, jnp.ones_like(gamma_ij), nlag)
    sums = bincount_masked(bin_idx, gamma_ij, nlag)
    gamma = jnp.where(npairs > 0, sums / npairs, 0.0)
    return centers, gamma, npairs


def directional_variogram_3d(
    values: jnp.ndarray,
    coords: jnp.ndarray,
    lag: float,
    nlag: int,
    azimuth_deg: float,
    dip_deg: float,
    atol_deg: float,
    bandwh_h: float,
    bandwh_v: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """3D directional variogram with azimuth, dip, angular tolerance, and bandwidths.

    Uses unit direction vector defined by azimuth (around Z) and dip (from horizontal).
    Accepts pairs if angular deviation <= atol and perpendicular horizontal/vertical
    distances within specified bandwidths. Bins by projected distance along direction.
    """
    n = values.shape[0]
    ii, jj = jnp.triu_indices(n, k=1)
    dv = values[ii] - values[jj]
    gamma_ij = 0.5 * (dv * dv)
    h = coords[jj] - coords[ii]

    az = jnp.deg2rad(azimuth_deg)
    dip = jnp.deg2rad(dip_deg)
    # Unit direction vector in 3D (east-north-up basis):
    # Horizontal projection
    ux = jnp.cos(az) * jnp.cos(dip)
    uy = jnp.sin(az) * jnp.cos(dip)
    uz = jnp.sin(dip)
    u = jnp.array([ux, uy, uz])

    hlen = jnp.linalg.norm(h, axis=1)
    h_unit = jnp.where(hlen[:, None] > 0, h / hlen[:, None], 0.0)
    cos_dev = jnp.clip(jnp.sum(h_unit * u[None, :], axis=1), -1.0, 1.0)
    ang_dev = jnp.degrees(jnp.arccos(cos_dev))

    # Perpendicular components: split into horizontal plane and vertical
    # Projection length along direction
    dpar = jnp.sum(h * u[None, :], axis=1)
    hperp_vec = h - dpar[:, None] * u[None, :]
    # Horizontal perpendicular distance
    hperp_h = jnp.linalg.norm(hperp_vec[:, :2], axis=1)
    # Vertical perpendicular distance
    hperp_v = jnp.abs(hperp_vec[:, 2])

    mask = (ang_dev <= atol_deg) & (hperp_h <= bandwh_h) & (hperp_v <= bandwh_v)
    dpar_abs = jnp.where(mask, jnp.abs(dpar), -1.0)

    edges = jnp.arange(0.0, (nlag + 1) * lag, lag)
    centers = edges[:-1] + 0.5 * lag
    bin_idx = jnp.clip(jnp.floor(dpar_abs / lag).astype(jnp.int32), 0, nlag - 1)
    valid = dpar_abs >= 0
    bin_idx = jnp.where(valid, bin_idx, -1)

    def bincount_masked(idx: jnp.ndarray, weights: jnp.ndarray, length: int) -> jnp.ndarray:
        idx_clipped = jnp.clip(idx, 0, length - 1)
        w = jnp.where(idx >= 0, weights, 0.0)
        return jnp.bincount(idx_clipped, weights=w, length=length)

    npairs = bincount_masked(bin_idx, jnp.ones_like(gamma_ij), nlag)
    sums = bincount_masked(bin_idx, gamma_ij, nlag)
    gamma = jnp.where(npairs > 0, sums / npairs, 0.0)
    return centers, gamma, npairs


