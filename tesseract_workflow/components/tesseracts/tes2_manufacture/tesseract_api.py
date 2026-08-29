"""Manufacturability projection component and surrogate VJP."""
from typing import Any
import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64
from phyguide_tesseract.manufacturing import ManufacturingConfig, project_manufacturability, surrogate_vjp

class InputSchema(BaseModel):
    level_set: Differentiable[Array[(None, None), Float64]]
    threshold: float = 0.0
    filter_radius: int = 1
    projection_beta: float = 8.0
    min_component_size: int = 4
    keep_largest_component: bool = True
    supersample: int = 4
    minimum_feature_size: float = 2.0
    boundary_smoothing: float = 0.5

class OutputSchema(BaseModel):
    clean_level_set: Differentiable[Array[(None, None), Float64]]
    binary_mask: Array[(None, None), Float64]
    volume_fraction: Float64
    component_count: Float64
    largest_component_fraction: Float64

def config(i):
    return ManufacturingConfig(threshold=i.threshold, filter_radius=i.filter_radius,
        projection_beta=i.projection_beta, min_component_size=i.min_component_size,
        keep_largest_component=i.keep_largest_component, supersample=i.supersample,
        minimum_feature_size=i.minimum_feature_size,
        boundary_smoothing=i.boundary_smoothing)

def apply(inputs: InputSchema) -> OutputSchema:
    r = project_manufacturability(np.asarray(inputs.level_set), config(inputs))
    return OutputSchema(clean_level_set=r.clean_level_set, binary_mask=r.binary_mask.astype(np.float64),
        volume_fraction=r.metrics["volume_fraction"], component_count=r.metrics["component_count"],
        largest_component_fraction=r.metrics["largest_component_fraction"])

def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector: dict[str, Any]):
    if vjp_inputs != {"level_set"} or vjp_outputs != {"clean_level_set"}:
        raise ValueError("tes2_manufacture supports level_set -> clean_level_set")
    return {"level_set": surrogate_vjp(np.asarray(inputs.level_set), np.asarray(cotangent_vector["clean_level_set"]), config(inputs))}
