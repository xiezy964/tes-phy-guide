"""Project-local differentiable TRI3 geometry patch.

The reference basis and quadrature come from the installed JAX-FEM package;
the geometry map is recomputed in JAX from traced node coordinates.  This is
kept in the project so the installed jax-fem checkout is never modified.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax_fem.basis import get_shape_vals_and_grads


def tri3_geometry(points, cells, gauss_order=1):
    """Return JAX-traceable physical shape gradients and JxW."""
    _, grads_ref_np, weights_np = get_shape_vals_and_grads("TRI3", gauss_order)
    grads_ref = jnp.asarray(grads_ref_np)
    weights = jnp.asarray(weights_np)
    tri = points[cells]
    # dx/dxi = sum_a x_a ⊗ grad_ref(N_a)
    jac = jnp.einsum("eai,qaj->eqij", tri, grads_ref)
    det = jnp.linalg.det(jac)
    inv_jac = jnp.linalg.inv(jac)
    grads = jnp.einsum("qai,eqij->eqaj", grads_ref, inv_jac)
    return grads, det * weights[None, :]


def solve(points, cells, conductivity, top_mask, bottom_mask,
          top_temperature=1.0, bottom_temperature=0.0):
    """JAX-FEM TRI3 weak form with differentiable points and conductivity."""
    points = jnp.asarray(points)
    cells = jnp.asarray(cells, dtype=jnp.int32)
    k = jnp.asarray(conductivity).reshape((-1,))
    grads, jxw = tri3_geometry(points, cells)
    ke = k[:, None, None] * jnp.einsum("eqai,eqbi,eq->eab", grads, grads, jxw)
    n = points.shape[0]
    K = jnp.zeros((n, n), dtype=points.dtype)
    K = K.at[cells[:, :, None], cells[:, None, :]].add(ke)
    top = jnp.asarray(top_mask, dtype=points.dtype)
    bot = jnp.asarray(bottom_mask, dtype=points.dtype)
    bmask = jnp.clip(top + bot, 0., 1.)
    bval = top * top_temperature + bot * bottom_temperature
    free = 1. - bmask
    A = K * free[:, None] * free[None, :] + jnp.diag(bmask)
    rhs = free * (-(K @ bval)) + bval
    temperature = jnp.linalg.solve(A, rhs)
    # Unit temperature difference: energy equals effective conductance.
    conductance = bval @ (K @ temperature)
    return conductance, temperature
