"""Hard manufacturing projection with a smooth surrogate VJP.

The forward path deliberately contains discrete image operations. The
surrogate derivative is kept separate: it is a contract for a Tesseract VJP,
not a claim that connected-component filtering is mathematically smooth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ManufacturingConfig:
    """Parameters shared by the hard forward path and its surrogate VJP."""

    threshold: float = 0.0
    filter_radius: int = 1
    projection_beta: float = 8.0
    target_volume_fraction: float | None = None
    min_component_size: int = 4
    keep_largest_component: bool = False
    supersample: int = 1
    # Minimum printable feature width in native diffusion pixels.  This is a
    # geometric constraint, independent of material volume.
    minimum_feature_size: float = 2.0
    # Curvature regularization of the final signed-distance geometry, also in
    # native diffusion pixels.  Scaling by ``supersample`` makes its physical
    # effect independent of the raster resolution used for manufacturing.
    boundary_smoothing: float = 0.0


@dataclass
class ManufacturingResult:
    """Outputs of the manufacturing projector."""

    raw_level_set: np.ndarray
    filtered_field: np.ndarray
    clean_level_set: np.ndarray
    binary_mask: np.ndarray
    metrics: dict[str, float]


def _upsample_bilinear(field: np.ndarray, scale: int) -> np.ndarray:
    """Upsample a field on its aligned grid using explicit bilinear interpolation."""

    if scale <= 1:
        return np.asarray(field, dtype=np.float64).copy()
    source = np.asarray(field, dtype=np.float64)
    height, width = source.shape
    high_height = (height - 1) * scale + 1
    high_width = (width - 1) * scale + 1
    rows = np.arange(high_height, dtype=np.float64) / scale
    cols = np.arange(high_width, dtype=np.float64) / scale
    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    r1 = np.minimum(r0 + 1, height - 1)
    c1 = np.minimum(c0 + 1, width - 1)
    wr = rows - r0
    wc = cols - c0
    return (
        (1.0 - wr[:, None]) * (1.0 - wc[None, :]) * source[r0[:, None], c0[None, :]]
        + (1.0 - wr[:, None]) * wc[None, :] * source[r0[:, None], c1[None, :]]
        + wr[:, None] * (1.0 - wc[None, :]) * source[r1[:, None], c0[None, :]]
        + wr[:, None] * wc[None, :] * source[r1[:, None], c1[None, :]]
    )


def _upsample_bilinear_vjp(
    source_shape: tuple[int, int], cotangent: np.ndarray, scale: int
) -> np.ndarray:
    """Exact transpose of :func:`_upsample_bilinear` for the aligned grid."""

    if scale <= 1:
        if tuple(cotangent.shape) != tuple(source_shape):
            raise ValueError("cotangent shape does not match source shape")
        return np.asarray(cotangent, dtype=np.float64).copy()
    cotangent = np.asarray(cotangent, dtype=np.float64)
    height, width = source_shape
    expected = ((height - 1) * scale + 1, (width - 1) * scale + 1)
    if cotangent.shape != expected:
        raise ValueError(f"cotangent shape {cotangent.shape} does not match {expected}")
    rows = np.arange(expected[0], dtype=np.float64) / scale
    cols = np.arange(expected[1], dtype=np.float64) / scale
    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    r1 = np.minimum(r0 + 1, height - 1)
    c1 = np.minimum(c0 + 1, width - 1)
    wr = rows - r0
    wc = cols - c0
    gradient = np.zeros(source_shape, dtype=np.float64)
    # The small loop keeps the implementation transparent and avoids constructing
    # a huge sparse matrix.
    for row in range(expected[0]):
        row_cot = cotangent[row]
        w_top = 1.0 - wr[row]
        w_bottom = wr[row]
        np.add.at(gradient[r0[row]], c0, row_cot * w_top * (1.0 - wc))
        np.add.at(gradient[r0[row]], c1, row_cot * w_top * wc)
        np.add.at(gradient[r1[row]], c0, row_cot * w_bottom * (1.0 - wc))
        np.add.at(gradient[r1[row]], c1, row_cot * w_bottom * wc)
    return gradient


def _density_from_level_set(x: np.ndarray, threshold: float) -> np.ndarray:
    """Map the diffusion model's [-1, 1] field to a material density."""

    return 0.5 * (np.clip(x - threshold, -1.0, 1.0) + 1.0)


def _smooth_projection(field: np.ndarray, beta: float, threshold: float) -> np.ndarray:
    """Smooth Heaviside projection used by the surrogate derivative."""

    a = np.tanh(beta * threshold)
    b = np.tanh(beta * (1.0 - threshold))
    return (a + np.tanh(beta * (field - threshold))) / (a + b)


def _filter(field: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return field
    size = 2 * radius + 1
    return ndimage.uniform_filter(field, size=size, mode="nearest")


def _filter_vjp(cotangent: np.ndarray, radius: int) -> np.ndarray:
    """Exact transpose of the nearest-padded uniform filter."""

    cotangent = np.asarray(cotangent, dtype=np.float64)
    if radius <= 0:
        return cotangent.copy()
    size = 2 * radius + 1
    height, width = cotangent.shape
    gradient = np.zeros_like(cotangent)
    scale = 1.0 / float(size * size)
    rows = np.arange(height)
    cols = np.arange(width)
    for dr in range(-radius, radius + 1):
        source_rows = np.clip(rows + dr, 0, height - 1)
        for dc in range(-radius, radius + 1):
            source_cols = np.clip(cols + dc, 0, width - 1)
            np.add.at(gradient, (source_rows[:, None], source_cols[None, :]), cotangent * scale)
    return gradient


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    return inside - outside


def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return (xx * xx + yy * yy) <= radius * radius


def _edge_safe_morphology(mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Remove features narrower than the footprint without point-contact bridges.

    We intentionally do not apply binary closing here: closing can create a
    one-pixel neck between nearby solids, which is topologically connected but
    not printable.
    """

    padding = max(1, footprint.shape[0] // 2)
    padded = np.pad(mask, padding, mode="edge")
    padded = ndimage.binary_opening(padded, structure=footprint)
    return padded[padding:-padding, padding:-padding]


def _remove_small_components(mask: np.ndarray, minimum_size: int, keep_largest: bool) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    if keep_largest:
        largest = int(np.argmax(sizes[1:]) + 1)
        return labels == largest
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels]


def project_manufacturability(
    level_set: np.ndarray,
    config: ManufacturingConfig | None = None,
) -> ManufacturingResult:
    """Apply hard binary manufacturing rules and return a continuous proxy.

    The returned ``clean_level_set`` is a signed-distance-like field derived
    from the hard mask. It is used for contour extraction. A Tesseract VJP
    should use :func:`surrogate_vjp` instead of differentiating this function.
    """

    cfg = config or ManufacturingConfig()
    phi = np.asarray(level_set, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError(f"level_set must be 2D, got shape {phi.shape}")

    scale = max(1, int(cfg.supersample))
    phi_high = _upsample_bilinear(phi, scale)
    # Work in [0, 1] before filtering so the surrogate has a well-scaled range.
    occupancy = _density_from_level_set(phi_high, cfg.threshold)
    high_radius = max(0, int(round(cfg.filter_radius * scale)))
    filtered = _filter(occupancy, high_radius)
    projected = _smooth_projection(filtered, cfg.projection_beta, 0.5)

    threshold = 0.5
    if cfg.target_volume_fraction is not None:
        vf = float(np.clip(cfg.target_volume_fraction, 1e-4, 1.0 - 1e-4))
        threshold = float(np.quantile(projected, 1.0 - vf))
    binary = projected >= threshold

    feature_radius = max(1, int(round(0.5 * cfg.minimum_feature_size * scale)))
    footprint = _disk(feature_radius)
    binary = _edge_safe_morphology(binary, footprint)
    binary = _remove_small_components(
        binary, cfg.min_component_size, cfg.keep_largest_component
    )
    clean_phi = _signed_distance(binary)
    smoothing_sigma = max(0.0, float(cfg.boundary_smoothing) * scale)
    if smoothing_sigma > 0.0:
        clean_phi = ndimage.gaussian_filter(clean_phi, smoothing_sigma, mode="reflect")
        # Report and mesh the geometry represented by the smooth zero contour,
        # not the pre-smoothing raster mask.  Reapply the connectivity rule in
        # case curvature smoothing removes an exceptionally narrow neck.
        smooth_binary = clean_phi >= 0.0
        clean_binary = _remove_small_components(
            smooth_binary, cfg.min_component_size, cfg.keep_largest_component
        )
        if not np.array_equal(clean_binary, smooth_binary):
            clean_phi = ndimage.gaussian_filter(
                _signed_distance(clean_binary), smoothing_sigma, mode="reflect"
            )
        binary = clean_phi >= 0.0

    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    component_sizes = np.bincount(labels.ravel())[1:] if count else np.array([])
    metrics = {
        "volume_fraction": float(binary.mean()),
        "component_count": float(count),
        "largest_component_fraction": float(component_sizes.max() / binary.size)
        if component_sizes.size
        else 0.0,
        "minimum_filter_radius": float(cfg.filter_radius),
        "minimum_feature_size": float(cfg.minimum_feature_size),
        "boundary_smoothing": float(cfg.boundary_smoothing),
        "supersample": float(scale),
        "input_height": float(phi.shape[0]),
        "input_width": float(phi.shape[1]),
        "output_height": float(phi_high.shape[0]),
        "output_width": float(phi_high.shape[1]),
    }
    return ManufacturingResult(phi, filtered, clean_phi, binary, metrics)


def surrogate_forward(
    level_set: np.ndarray,
    config: ManufacturingConfig | None = None,
) -> np.ndarray:
    """Continuous proxy whose derivative defines :func:`surrogate_vjp`."""

    cfg = config or ManufacturingConfig()
    phi = np.asarray(level_set, dtype=np.float64)
    scale = max(1, int(cfg.supersample))
    phi_high = _upsample_bilinear(phi, scale)
    occupancy = _density_from_level_set(phi_high, cfg.threshold)
    high_radius = max(0, int(round(cfg.filter_radius * scale)))
    filtered = _filter(occupancy, high_radius)
    projected = _smooth_projection(filtered, cfg.projection_beta, 0.5)
    smoothing_sigma = max(0.0, float(cfg.boundary_smoothing) * scale)
    if smoothing_sigma > 0.0:
        projected = ndimage.gaussian_filter(projected, smoothing_sigma, mode="reflect")
    return projected


def surrogate_vjp(
    level_set: np.ndarray,
    cotangent: np.ndarray,
    config: ManufacturingConfig | None = None,
) -> np.ndarray:
    """Approximate VJP for the hard projector.

    The derivative follows the smooth filter plus smooth Heaviside used as a
    proxy for thresholding and morphology. This is intentionally explicit so
    it can be exposed as a custom ``vector_jacobian_product`` endpoint.
    """

    cfg = config or ManufacturingConfig()
    phi = np.asarray(level_set, dtype=np.float64)
    cotangent = np.asarray(cotangent, dtype=np.float64)
    scale = max(1, int(cfg.supersample))
    expected = ((phi.shape[0] - 1) * scale + 1, (phi.shape[1] - 1) * scale + 1)
    if cotangent.shape != expected:
        raise ValueError(f"cotangent shape {cotangent.shape} does not match {expected}")

    phi_high = _upsample_bilinear(phi, scale)
    occupancy = _density_from_level_set(phi_high, cfg.threshold)
    high_radius = max(0, int(round(cfg.filter_radius * scale)))
    filtered = _filter(occupancy, high_radius)
    beta = cfg.projection_beta
    denom = 2.0 * np.tanh(beta * 0.5)
    # For the standard threshold=0.5 projection this is beta*sech^2(...)/denom.
    d_projection = beta / max(denom, 1e-12)
    d_projection *= 1.0 / np.cosh(np.clip(beta * (filtered - 0.5), -40, 40)) ** 2
    d_occupancy = 0.5 * (np.abs(phi_high - cfg.threshold) <= 1.0)
    smoothing_sigma = max(0.0, float(cfg.boundary_smoothing) * scale)
    # A reflect-mode Gaussian convolution is self-adjoint; apply it first in
    # reverse order to obtain the exact VJP of the smooth surrogate above.
    if smoothing_sigma > 0.0:
        cotangent = ndimage.gaussian_filter(cotangent, smoothing_sigma, mode="reflect")
    grad_filtered = cotangent * d_projection
    grad_occupancy = _filter_vjp(grad_filtered, high_radius)
    high_gradient = grad_occupancy * d_occupancy
    return _upsample_bilinear_vjp(phi.shape, high_gradient, scale)
