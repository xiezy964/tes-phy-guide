"""2-D level-set contouring, an unstructured prototype mesh, and VJP helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ContourPoint:
    point: np.ndarray
    cell: tuple[int, int]
    corner_indices: tuple[tuple[int, int], ...]
    derivatives: np.ndarray


@dataclass
class ContourResult:
    points: np.ndarray
    segments: np.ndarray
    point_records: list[ContourPoint]
    field_shape: tuple[int, int]


@dataclass
class LevelSetMesh:
    points: np.ndarray
    cells: np.ndarray
    phase_tags: np.ndarray
    interface_segments: np.ndarray
    contour: ContourResult
    motion_anchor_indices: np.ndarray | None = None
    motion_anchor_weights: np.ndarray | None = None


# Corner order: top-left, top-right, bottom-right, bottom-left.
_EDGE_CORNERS = ((0, 1), (1, 2), (2, 3), (3, 0))
_CASE_SEGMENTS = {
    0: (), 1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),),
    4: ((1, 2),), 5: ((3, 2), (0, 1)), 6: ((0, 2),),
    7: ((3, 2),), 8: ((2, 3),), 9: ((0, 2),),
    10: ((0, 3), (1, 2)), 11: ((1, 2),), 12: ((1, 3),),
    13: ((0, 1),), 14: ((3, 0),), 15: (),
}


def _edge_intersection(
    values: np.ndarray,
    cell_row: int,
    cell_col: int,
    edge: int,
    level: float,
    height: int,
    width: int,
) -> ContourPoint:
    a, b = _EDGE_CORNERS[edge]
    corner_offsets = ((0, 0), (0, 1), (1, 1), (1, 0))
    ra, ca = corner_offsets[a]
    rb, cb = corner_offsets[b]
    va, vb = values[a], values[b]
    denom = vb - va
    if abs(denom) < 1e-12:
        t = 0.5
        denom = 1e-12
    else:
        t = (level - va) / denom
    pa = np.array([(cell_col + ca) / (width - 1), (cell_row + ra) / (height - 1)])
    pb = np.array([(cell_col + cb) / (width - 1), (cell_row + rb) / (height - 1)])
    point = pa + t * (pb - pa)
    dta = (level - vb) / (denom * denom)
    dtb = -(level - va) / (denom * denom)
    derivatives = np.zeros((4, 2), dtype=np.float64)
    derivatives[a] = dta * (pb - pa)
    derivatives[b] = dtb * (pb - pa)
    corner_indices = tuple(
        (cell_row + dr, cell_col + dc) for dr, dc in corner_offsets
    )
    return ContourPoint(point, (cell_row, cell_col), corner_indices, derivatives)


def marching_squares(level_set: np.ndarray, level: float = 0.0) -> ContourResult:
    """Extract contour segments and retain local derivatives of each vertex."""

    field = np.asarray(level_set, dtype=np.float64)
    if field.ndim != 2 or min(field.shape) < 2:
        raise ValueError("level_set must be a 2D array with both dimensions >= 2")
    height, width = field.shape
    points: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    records: list[ContourPoint] = []
    for row in range(height - 1):
        for col in range(width - 1):
            values = np.array(
                [field[row, col], field[row, col + 1], field[row + 1, col + 1], field[row + 1, col]],
                dtype=np.float64,
            )
            case = sum((int(value >= level) << idx) for idx, value in enumerate(values))
            for edge_a, edge_b in _CASE_SEGMENTS[case]:
                first = _edge_intersection(values, row, col, edge_a, level, height, width)
                second = _edge_intersection(values, row, col, edge_b, level, height, width)
                first_idx = len(points)
                points.extend((first.point, second.point))
                records.extend((first, second))
                segments.append((first_idx, first_idx + 1))
    point_array = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    segment_array = np.asarray(segments, dtype=np.int64).reshape((-1, 2))
    return ContourResult(point_array, segment_array, records, field.shape)


def contour_vjp(contour: ContourResult, cotangent: np.ndarray) -> np.ndarray:
    """Scatter contour-point cotangents back to the level-set pixels."""

    cotangent = np.asarray(cotangent, dtype=np.float64)
    if cotangent.shape != contour.points.shape:
        raise ValueError("cotangent must have shape (number_of_contour_points, 2)")
    grad = np.zeros(contour.field_shape, dtype=np.float64)
    if not contour.point_records:
        return grad
    for point_index, record in enumerate(contour.point_records):
        local = record.derivatives @ cotangent[point_index]
        for value, (row, col) in zip(local, record.corner_indices, strict=True):
            grad[row, col] += value
    return grad


def sample_grid_bilinear(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample a pixel field at normalized ``(x, y)`` points."""

    field = np.asarray(field, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    row = np.clip(points[:, 1] * (field.shape[0] - 1), 0, field.shape[0] - 1)
    col = np.clip(points[:, 0] * (field.shape[1] - 1), 0, field.shape[1] - 1)
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1 = np.minimum(r0 + 1, field.shape[0] - 1)
    c1 = np.minimum(c0 + 1, field.shape[1] - 1)
    wr = row - r0
    wc = col - c0
    return (
        (1 - wr) * (1 - wc) * field[r0, c0]
        + (1 - wr) * wc * field[r0, c1]
        + wr * (1 - wc) * field[r1, c0]
        + wr * wc * field[r1, c1]
    )


def bilinear_sample_vjp(
    field_shape: tuple[int, int], points: np.ndarray, cotangent: np.ndarray
) -> np.ndarray:
    """Apply the exact transpose of :func:`sample_grid_bilinear`."""

    points = np.asarray(points, dtype=np.float64)
    cotangent = np.asarray(cotangent, dtype=np.float64)
    row = np.clip(points[:, 1] * (field_shape[0] - 1), 0, field_shape[0] - 1)
    col = np.clip(points[:, 0] * (field_shape[1] - 1), 0, field_shape[1] - 1)
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1 = np.minimum(r0 + 1, field_shape[0] - 1)
    c1 = np.minimum(c0 + 1, field_shape[1] - 1)
    wr = row - r0
    wc = col - c0
    gradient = np.zeros(field_shape, dtype=np.float64)
    np.add.at(gradient, (r0, c0), cotangent * (1 - wr) * (1 - wc))
    np.add.at(gradient, (r0, c1), cotangent * (1 - wr) * wc)
    np.add.at(gradient, (r1, c0), cotangent * wr * (1 - wc))
    np.add.at(gradient, (r1, c1), cotangent * wr * wc)
    return gradient


def _mesh_motion_map(points: np.ndarray, contour: ContourResult, neighbors: int = 4):
    """Build a frozen local interpolation from contour motion to mesh motion."""

    if contour.points.shape[0] == 0:
        return np.empty((len(points), 0), dtype=np.int64), np.empty((len(points), 0))
    unique_anchors, unique_to_first = np.unique(
        np.round(contour.points, decimals=12), axis=0, return_index=True
    )
    count = min(max(1, neighbors), len(unique_anchors))
    distances, local_indices = cKDTree(unique_anchors).query(points, k=count)
    if count == 1:
        distances = distances[:, None]
        local_indices = local_indices[:, None]
    weights = 1.0 / np.maximum(distances, 1.0e-6) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    # Keep the exterior square fixed during shape differentiation.
    exterior = np.isclose(points[:, 0], 0.0) | np.isclose(points[:, 0], 1.0)
    exterior |= np.isclose(points[:, 1], 0.0) | np.isclose(points[:, 1], 1.0)
    weights[exterior] = 0.0
    return unique_to_first[local_indices], weights


def gmsh_level_set(
    level_set: np.ndarray,
    level: float = 0.0,
    coarse_size: float = 0.08,
    interface_size: float = 0.012,
    interface_bandwidth: float = 0.08,
) -> LevelSetMesh:
    """Generate a constrained, interface-adaptive TRI3 mesh with Gmsh.

    Gmsh owns the discrete forward meshing operation. Its topology is frozen
    for one reverse pass; :func:`gmsh_mesh_vjp` supplies the explicit shape VJP.
    """

    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError("Gmsh Python bindings are required") from exc

    from skimage.measure import find_contours

    field = np.asarray(level_set, dtype=np.float64)
    contour = marching_squares(field, level)
    candidates = find_contours(field, level=level)
    point_blocks = []
    segment_blocks = []
    offset = 0
    for raw_candidate in candidates:
        # ``find_contours`` returns both closed loops (holes/islands) and
        # open curves that terminate on the outer square. Never close an
        # open curve artificially: that creates a non-physical chord.
        # skimage does not repeat the first sample for closed contours;
        # infer closure when the curve stays away from the image boundary.
        closed = not (
            np.any(raw_candidate[:, 0] <= 0.5)
            or np.any(raw_candidate[:, 0] >= field.shape[0] - 1.5)
            or np.any(raw_candidate[:, 1] <= 0.5)
            or np.any(raw_candidate[:, 1] >= field.shape[1] - 1.5)
        )
        candidate = np.column_stack(
            (
                raw_candidate[:, 1] / max(1, field.shape[1] - 1),
                raw_candidate[:, 0] / max(1, field.shape[0] - 1),
            )
        )
        if closed and len(candidate) > 1 and np.linalg.norm(candidate[0] - candidate[-1]) < 1.0e-10:
            candidate = candidate[:-1]
        minimum = 3 if closed else 2
        if len(candidate) < minimum:
            continue
        local = np.column_stack(
            (np.arange(len(candidate) - 1), np.arange(1, len(candidate)))
        )
        if closed:
            local = np.vstack((local, [len(candidate) - 1, 0]))
        point_blocks.append(candidate)
        segment_blocks.append(local + offset)
        offset += len(candidate)
    if not point_blocks:
        raise ValueError("level set has no zero contour for Gmsh")
    gmsh_contour_points = np.vstack(point_blocks)
    gmsh_segments = np.vstack(segment_blocks).astype(np.int64)
    rounded = np.round(gmsh_contour_points, decimals=12)
    unique_points, inverse = np.unique(rounded, axis=0, return_inverse=True)
    # Snap contour endpoints that meet the design-domain boundary.  These
    # points must be shared by the embedded interface and the outer curve
    # loop; coincident-but-distinct Gmsh points lead to unrecoverable edges.
    boundary_tolerance = 1.0e-10
    unique_points[np.isclose(unique_points, 0.0, atol=boundary_tolerance)] = 0.0
    unique_points[np.isclose(unique_points, 1.0, atol=boundary_tolerance)] = 1.0
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("phyguide_level_set")
        geo = gmsh.model.geo
        coordinate_tags: dict[tuple[float, float], int] = {}

        def point_tag(x: float, y: float, size: float) -> int:
            key = (round(float(x), 12), round(float(y), 12))
            if key not in coordinate_tags:
                coordinate_tags[key] = geo.addPoint(float(x), float(y), 0.0, size)
            return coordinate_tags[key]

        point_tags = [point_tag(x, y, interface_size) for x, y in unique_points]

        # Split each side of the square at every interface endpoint and reuse
        # the exact same point tag.  Side order is counter-clockwise.
        bottom = [(0.0, 0.0), (1.0, 0.0)]
        right = [(1.0, 0.0), (1.0, 1.0)]
        top = [(1.0, 1.0), (0.0, 1.0)]
        left = [(0.0, 1.0), (0.0, 0.0)]
        for x, y in unique_points:
            if y == 0.0:
                bottom.append((float(x), 0.0))
            if x == 1.0:
                right.append((1.0, float(y)))
            if y == 1.0:
                top.append((float(x), 1.0))
            if x == 0.0:
                left.append((0.0, float(y)))
        ordered_sides = (
            sorted(set(bottom), key=lambda p: p[0]),
            sorted(set(right), key=lambda p: p[1]),
            sorted(set(top), key=lambda p: p[0], reverse=True),
            sorted(set(left), key=lambda p: p[1], reverse=True),
        )
        outer_lines = []
        for side in ordered_sides:
            tags = [point_tag(x, y, coarse_size) for x, y in side]
            outer_lines.extend(geo.addLine(a, b) for a, b in zip(tags[:-1], tags[1:]))
        surface = geo.addPlaneSurface([geo.addCurveLoop(outer_lines)])
        curve_tags = []
        for a, b in gmsh_segments:
            ua = int(inverse[a])
            ub = int(inverse[b])
            if ua != ub:
                curve_tags.append(geo.addLine(point_tags[ua], point_tags[ub]))
        if not curve_tags:
            raise ValueError("level set has no interface curves for Gmsh")
        geo.synchronize()
        if curve_tags:
            gmsh.model.mesh.embed(1, curve_tags, 2, surface)
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance, "CurvesList", curve_tags)
            # ``Sampling`` was added after the Gmsh version shipped by Debian
            # bookworm/ARM64. It only refines evaluation of the distance field;
            # omitting it keeps the same Gmsh meshing path on older releases.
            try:
                gmsh.model.mesh.field.setNumber(distance, "Sampling", 20)
            except Exception:
                pass
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", interface_size)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", coarse_size)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", interface_size)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", interface_bandwidth)
            gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # The sampled contour contains short pixel-scale segments.  Extending
        # their edge length into the 2-D region makes the entire domain dense;
        # the explicit Distance/Threshold field already provides the desired
        # narrow-band refinement.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        # Keep the background coarse while allowing the distance field to
        # refine the interface. Setting min=max here would silently disable
        # the adaptive size field.
        gmsh.option.setNumber("Mesh.MeshSizeMin", interface_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", coarse_size)
        gmsh.model.mesh.generate(2)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))[:, :2]
        tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}
        element_types, _, element_nodes = gmsh.model.mesh.getElements(2, surface)
        triangles = []
        for element_type, nodes in zip(element_types, element_nodes, strict=True):
            _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if nodes_per_element == 3:
                raw = np.asarray(nodes, dtype=np.int64).reshape((-1, 3))
                triangles.append(
                    np.vectorize(lambda tag: tag_to_index[int(tag)], otypes=[np.int64])(raw)
                )
        if not triangles:
            raise RuntimeError("Gmsh did not generate TRI3 elements")
        cells = np.vstack(triangles)
        embedded_edges = []
        for curve_tag in curve_tags:
            line_types, _, line_nodes = gmsh.model.mesh.getElements(1, curve_tag)
            for element_type, nodes in zip(line_types, line_nodes, strict=True):
                _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
                if nodes_per_element == 2:
                    raw = np.asarray(nodes, dtype=np.int64).reshape((-1, 2))
                    embedded_edges.append(
                        np.vectorize(lambda tag: tag_to_index[int(tag)], otypes=[np.int64])(raw)
                    )
        used = np.unique(cells.ravel())
        remap = -np.ones(points.shape[0], dtype=np.int64)
        remap[used] = np.arange(used.size, dtype=np.int64)
        cells = remap[cells]
        points = points[used]
        interface = (
            remap[np.vstack(embedded_edges)]
            if embedded_edges
            else np.empty((0, 2), dtype=np.int64)
        )
    finally:
        gmsh.finalize()

    tri = points[cells]
    signed_area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flipped = signed_area < 0.0
    cells[flipped] = cells[flipped][:, [0, 2, 1]]
    centroids = points[cells].mean(axis=1)
    # The geometry was extracted from an interpolated zero contour, so phase
    # classification must use the same continuous field rather than nearest
    # pixels (which can disagree next to the interface).
    phase_tags = (sample_grid_bilinear(field, centroids) >= level).astype(np.int32)
    if interface.size == 0 or np.any(interface < 0):
        raise RuntimeError("Gmsh did not preserve the embedded level-set interface")
    # Every reported interface segment must be an actual TRI3 edge. This
    # catches the former failure mode where Gmsh meshed only the outer square.
    cell_edges = np.vstack((cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]))
    cell_edge_set = {tuple(sorted(edge)) for edge in cell_edges}
    if not all(tuple(sorted(edge)) in cell_edge_set for edge in interface):
        raise RuntimeError("embedded Gmsh interface is not conforming with TRI3 edges")
    anchor_indices, anchor_weights = _mesh_motion_map(points, contour)
    return LevelSetMesh(
        points,
        cells,
        phase_tags,
        interface,
        contour,
        anchor_indices,
        anchor_weights,
    )


def gmsh_mesh_vjp(mesh: LevelSetMesh, cotangent_points: np.ndarray) -> np.ndarray:
    """Apply the frozen-topology mesh-motion VJP for a Gmsh mesh.

    The result is a contour-point cotangent.  Use :func:`contour_vjp` to
    continue through marching squares to the high-resolution level set.
    """

    if mesh.motion_anchor_indices is None:
        raise ValueError("gmsh_mesh_vjp requires a mesh returned by gmsh_level_set")
    cotangent = np.asarray(cotangent_points, dtype=np.float64)
    if cotangent.shape != mesh.points.shape:
        raise ValueError(f"cotangent must have shape {mesh.points.shape}")
    contour_cotangent = np.zeros_like(mesh.contour.points)
    for neighbor in range(mesh.motion_anchor_indices.shape[1]):
        np.add.at(
            contour_cotangent,
            mesh.motion_anchor_indices[:, neighbor],
            cotangent * mesh.motion_anchor_weights[:, neighbor, None],
        )
    return contour_cotangent
