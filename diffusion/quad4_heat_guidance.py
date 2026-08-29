"""Reusable original QUAD4 JAX-FEM physics guidance."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from therm.fem_heat import HeatConduction


@lru_cache(maxsize=1)
def _quad4_operator():
    nx = ny = 64
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=1.0, domain_y=1.0)
    cell_type = get_meshio_cell_type("QUAD4")
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def left(point): return jnp.isclose(point[0], 0.0, atol=1e-5)
    def right(point): return jnp.isclose(point[0], 1.0, atol=1e-5)
    def bottom(point): return jnp.isclose(point[1], 0.0, atol=1e-5)
    def top(point): return jnp.isclose(point[1], 1.0, atol=1e-5)

    bc = [[top, bottom], [0, 0], [lambda p: 500.0, lambda p: 300.0]]
    problem = HeatConduction(
        mesh, vec=1, dim=2, ele_type="QUAD4",
        dirichlet_bc_info=bc, location_fns=[left, right]
    )
    return ad_wrapper(problem)


def quad4_k_eff(level_set, beta=8.0):
    """Exact k_eff function from sample_grad_heat.py."""
    rho = jnp.clip(level_set, -1.0, 1.0)
    rho = (rho + 1.0) / 2.0
    rho = (jnp.tanh(beta / 2.0) + jnp.tanh(beta * (rho - 0.5))) / (2.0 * jnp.tanh(beta / 2.0))
    k = 1.0 + 99.0 * rho
    k_vec = k.flatten(order="F").reshape((-1, 1))
    sol = _quad4_operator()(k_vec)
    temperature = sol[0].reshape((65, 65), order="F")
    q = k * (temperature[:-1, :-1] - temperature[1:, :-1]) / (1.0 / 64.0)
    return -jnp.mean(q) / 200.0


def quad4_loss_and_grad(level_set, target=30.0, beta=8.0):
    def loss_fn(x):
        k = quad4_k_eff(x, beta)
        return (k - target) ** 2, k
    (loss, conductance), gradient = jax.value_and_grad(loss_fn, has_aux=True)(level_set)
    return loss, conductance, gradient
