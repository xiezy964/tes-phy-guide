"""Strict Gmsh mesher component with an explicit frozen-topology VJP."""
from typing import Any
import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32
from phyguide_tesseract.meshing import contour_vjp, gmsh_level_set, gmsh_mesh_vjp

class InputSchema(BaseModel):
    level_set: Differentiable[Array[(None, None), Float64]]
    background_stride: int = 16
    gmsh_interface_scale: float = 12.0

class OutputSchema(BaseModel):
    points: Differentiable[Array[(None, 2), Float64]]
    cells: Array[(None, 3), Int32]
    phase_tags: Array[(None,), Int32]
    interface_segments: Array[(None, 2), Int32]

def apply(inputs: InputSchema) -> OutputSchema:
    phi = np.asarray(inputs.level_set)
    r = gmsh_level_set(phi, coarse_size=inputs.background_stride / 63.0,
        interface_size=inputs.gmsh_interface_scale / max(phi.shape),
        interface_bandwidth=inputs.background_stride / 63.0)
    return OutputSchema(points=r.points, cells=r.cells, phase_tags=r.phase_tags,
                        interface_segments=r.interface_segments)

def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector: dict[str, Any]):
    if vjp_inputs != {"level_set"} or vjp_outputs != {"points"}:
        raise ValueError("tes3_mesher supports level_set -> points")
    phi = np.asarray(inputs.level_set)
    mesh = gmsh_level_set(phi, coarse_size=inputs.background_stride / 63.0,
        interface_size=inputs.gmsh_interface_scale / max(phi.shape),
        interface_bandwidth=inputs.background_stride / 63.0)
    contour_cotangent = gmsh_mesh_vjp(
        mesh, np.asarray(cotangent_vector["points"])
    )
    grad = contour_vjp(mesh.contour, contour_cotangent)
    return {"level_set": grad}
