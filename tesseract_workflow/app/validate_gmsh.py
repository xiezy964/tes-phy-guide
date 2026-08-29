"""Validate the real Gmsh backend independently of the diffusion pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phyguide_tesseract.meshing import (  # noqa: E402
    contour_vjp,
    gmsh_level_set,
    gmsh_mesh_vjp,
    sample_grid_bilinear,
)


def _interface_components(edges: np.ndarray) -> tuple[int, np.ndarray]:
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    unseen = set(adjacency)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return components, np.asarray([len(adjacency[node]) for node in adjacency])


def _point_segment_distance(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Return each point's distance to the closest segment."""

    result = np.full(len(points), np.inf)
    direction = ends - starts
    denominator = np.sum(direction * direction, axis=1)
    for offset in range(0, len(points), 256):
        block = points[offset : offset + 256]
        relative = block[:, None, :] - starts[None, :, :]
        parameter = np.sum(relative * direction[None, :, :], axis=2)
        parameter /= np.maximum(denominator[None, :], 1.0e-30)
        parameter = np.clip(parameter, 0.0, 1.0)
        projection = starts[None, :, :] + parameter[:, :, None] * direction[None, :, :]
        result[offset : offset + len(block)] = np.sqrt(
            np.min(np.sum((block[:, None, :] - projection) ** 2, axis=2), axis=1)
        )
    return result


def validate_case(
    name: str,
    phi: np.ndarray,
    expected_loops: int,
    output_dir: Path,
    *,
    expected_open: int = 0,
    coarse_size: float = 0.14,
    interface_size: float = 0.025,
    interface_bandwidth: float = 0.10,
) -> dict:
    mesh = gmsh_level_set(
        phi,
        coarse_size=coarse_size,
        interface_size=interface_size,
        interface_bandwidth=interface_bandwidth,
    )
    cell_edges = np.vstack(
        (mesh.cells[:, [0, 1]], mesh.cells[:, [1, 2]], mesh.cells[:, [2, 0]])
    )
    edge_set = {tuple(sorted(map(int, edge))) for edge in cell_edges}
    conforming = all(
        tuple(sorted(map(int, edge))) in edge_set for edge in mesh.interface_segments
    )
    components, degrees = _interface_components(mesh.interface_segments)

    vertex_values = sample_grid_bilinear(phi, mesh.points).reshape(-1)
    cell_values = vertex_values[mesh.cells]
    # Interface vertices can have interpolation noise.  A triangle is a true
    # cross-interface failure only if it owns vertices appreciably on both sides.
    sign_tolerance = 2.0e-3
    crossing = np.logical_and(
        np.min(cell_values, axis=1) < -sign_tolerance,
        np.max(cell_values, axis=1) > sign_tolerance,
    )
    centroids = mesh.points[mesh.cells].mean(axis=1)
    expected_tags = (sample_grid_bilinear(phi, centroids) >= 0.0).astype(np.int32)

    interface_edges = mesh.points[mesh.interface_segments]
    contour_error = _point_segment_distance(
        mesh.contour.points, interface_edges[:, 0], interface_edges[:, 1]
    )
    contour_limit = 1.5 / max(phi.shape)

    point_cotangent = np.column_stack(
        (np.sin(3.0 * mesh.points[:, 0]), np.cos(4.0 * mesh.points[:, 1]))
    )
    contour_cotangent = gmsh_mesh_vjp(mesh, point_cotangent)
    level_set_cotangent = contour_vjp(mesh.contour, contour_cotangent)

    checks = {
        "interface_is_tri3_edges": bool(conforming),
        "interface_vertex_degrees_valid": bool(
            np.all((degrees == 1) | (degrees == 2))
            and np.count_nonzero(degrees == 1) == 2 * expected_open
        ),
        "expected_interface_loops": components == expected_loops,
        "no_cross_interface_cells": int(np.count_nonzero(crossing)) == 0,
        "phase_tags_match_centroids": bool(np.array_equal(mesh.phase_tags, expected_tags)),
        "contour_error_within_pixel": float(np.max(contour_error)) <= contour_limit,
        "vjp_has_level_set_shape": level_set_cotangent.shape == phi.shape,
        "vjp_is_finite": bool(np.isfinite(level_set_cotangent).all()),
    }
    report = {
        "name": name,
        "nodes": int(len(mesh.points)),
        "cells": int(len(mesh.cells)),
        "interface_edges": int(len(mesh.interface_segments)),
        "interface_components": int(components),
        "cross_interface_cells": int(np.count_nonzero(crossing)),
        "max_contour_error": float(np.max(contour_error)),
        "contour_error_limit": float(contour_limit),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.tripcolor(
        mesh.points[:, 0],
        mesh.points[:, 1],
        mesh.cells,
        facecolors=mesh.phase_tags,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        edgecolors="0.65",
        linewidth=0.25,
    )
    axis.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells, color="0.65", lw=0.25)
    for first, second in interface_edges:
        axis.plot(
            [first[0], second[0]], [first[1], second[1]], color="#0066ff", lw=1.8
        )
    # Red is the zero contour of the original sampled field; blue is the
    # interface that Gmsh actually retained as TRI3 edges.
    y = np.linspace(0.0, 1.0, phi.shape[0])
    x = np.linspace(0.0, 1.0, phi.shape[1])
    axis.contour(x, y, phi, levels=[0.0], colors=["#ff2b2b"], linewidths=0.8)
    axis.set(
        aspect="equal",
        xlim=(0, 1),
        ylim=(0, 1),
        title=f"{name}: real Gmsh ({len(mesh.points)} nodes / {len(mesh.cells)} TRI3)",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=180)
    plt.close(figure)
    return report


def main() -> None:
    output_dir = ROOT / "results" / "gmsh_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0.0:1.0:129j, 0.0:1.0:129j]
    cases = [
        ("circle", 0.26 - np.hypot(xx - 0.5, yy - 0.5), 1),
        (
            "annulus",
            np.minimum(0.34 - np.hypot(xx - 0.5, yy - 0.5), np.hypot(xx - 0.5, yy - 0.5) - 0.16),
            2,
        ),
        (
            "three_inclusions",
            np.maximum.reduce(
                [
                    0.15 - np.hypot(xx - 0.28, yy - 0.30),
                    0.12 - np.hypot(xx - 0.70, yy - 0.32),
                    0.17 - np.hypot(xx - 0.55, yy - 0.72),
                ]
            ),
            3,
        ),
    ]
    reports = [validate_case(name, phi, loops, output_dir) for name, phi, loops in cases]
    summary = {"passed": all(item["passed"] for item in reports), "cases": reports}
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
