"""Differentiable dynamic TRI3 thermal FEM component."""
from typing import Any
import equinox as eqx
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32
from tesseract_core.runtime.jax_recipes import jax_abstract_eval, jax_apply, jax_vjp
from phyguide_tesseract.dynamic_tri3_fem import solve

class InputSchema(BaseModel):
    points: Differentiable[Array[(None, 2), Float64]]
    cells: Array[(None, 3), Int32]
    conductivity: Differentiable[Array[(None,), Float64]]
    top_mask: Array[(None,), Float64]
    bottom_mask: Array[(None,), Float64]

class OutputSchema(BaseModel):
    conductance: Differentiable[Float64]
    temperature: Differentiable[Array[(None,), Float64]]

@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    c, t = solve(inputs["points"], inputs["cells"], inputs["conductivity"], inputs["top_mask"], inputs["bottom_mask"])
    return {"conductance": c, "temperature": t}

def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))

def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector: dict[str, Any]):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)

def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
