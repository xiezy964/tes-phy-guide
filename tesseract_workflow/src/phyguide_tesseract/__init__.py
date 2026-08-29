"""Local reference implementations for the Tesseract inverse-design pipeline."""

from .manufacturing import ManufacturingConfig, project_manufacturability, surrogate_forward, surrogate_vjp
from .meshing import (
    LevelSetMesh,
    contour_vjp,
    gmsh_level_set,
    gmsh_mesh_vjp,
    marching_squares,
)

__all__ = [
    "ManufacturingConfig",
    "project_manufacturability",
    "surrogate_forward",
    "surrogate_vjp",
    "LevelSetMesh",
    "marching_squares",
    "contour_vjp",
    "gmsh_level_set",
    "gmsh_mesh_vjp",
]
