"""One differentiable VP-SDE reverse step.

The neural noise prediction is an input so this component can wrap any
diffusion checkpoint without coupling the Tesseract image to model files.
"""
from typing import Any
import jax
import jax.numpy as jnp
import equinox as eqx
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.jax_recipes import jax_abstract_eval, jax_apply, jax_vjp

class InputSchema(BaseModel):
    x_t: Differentiable[Array[(None, None, None, 1), Float64]]
    eps_pred: Array[(None, None, None, 1), Float64]
    physical_gradient: Array[(None, None, None, 1), Float64]
    beta_t: float
    std: float
    mean_coef: float
    dt: float
    guidance_strength: float
    noise: Array[(None, None, None, 1), Float64]

class OutputSchema(BaseModel):
    x_next: Differentiable[Array[(None, None, None, 1), Float64]]
    guided_score: Differentiable[Array[(None, None, None, 1), Float64]]

@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    x = inputs["x_t"]
    eps = inputs["eps_pred"]
    g = inputs["physical_gradient"]
    norm = jnp.linalg.norm(g)
    score = -eps / jnp.maximum(inputs["std"], 1e-8)
    guided = score - inputs["guidance_strength"] * g / (norm + 1e-8)
    drift = -0.5 * inputs["beta_t"] * x - inputs["beta_t"] * guided
    x_next = x - drift * inputs["dt"] + jnp.sqrt(inputs["beta_t"] * inputs["dt"]) * inputs["noise"]
    return {"x_next": x_next, "guided_score": guided}

def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))

def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector: dict[str, Any]):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)

def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
