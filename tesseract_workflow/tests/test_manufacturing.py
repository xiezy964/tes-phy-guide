import numpy as np

from phyguide_tesseract.manufacturing import ManufacturingConfig, project_manufacturability, surrogate_forward, surrogate_vjp


def test_hard_projection_removes_small_component():
    phi = -3.0 * np.ones((24, 24))
    phi[5:18, 5:18] = 3.0
    phi[1, 1] = 3.0
    result = project_manufacturability(
        phi, ManufacturingConfig(filter_radius=1, min_component_size=5)
    )
    assert not result.binary_mask[1, 1]
    assert result.binary_mask[10, 10]


def test_surrogate_vjp_matches_directional_difference():
    rng = np.random.default_rng(4)
    phi = rng.normal(size=(12, 12))
    direction = rng.normal(size=phi.shape)
    cotangent = rng.normal(size=phi.shape)
    cfg = ManufacturingConfig(filter_radius=1, projection_beta=4.0)
    grad = surrogate_vjp(phi, cotangent, cfg)
    epsilon = 1e-5
    plus = np.sum(surrogate_forward(phi + epsilon * direction, cfg) * cotangent)
    minus = np.sum(surrogate_forward(phi - epsilon * direction, cfg) * cotangent)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert np.isclose(np.sum(grad * direction), finite_difference, rtol=8e-2, atol=2e-2)


def test_morphology_does_not_force_material_boundaries_to_void():
    phi = np.ones((16, 16))
    result = project_manufacturability(phi, ManufacturingConfig(filter_radius=2))
    assert result.binary_mask[0].all()
    assert result.binary_mask[-1].all()


def test_supersampled_surrogate_has_exact_transpose_vjp():
    rng = np.random.default_rng(11)
    phi = rng.normal(size=(8, 9))
    direction = rng.normal(size=phi.shape)
    config = ManufacturingConfig(filter_radius=1, projection_beta=4.0, supersample=4)
    high_shape = ((phi.shape[0] - 1) * 4 + 1, (phi.shape[1] - 1) * 4 + 1)
    cotangent = rng.normal(size=high_shape)
    gradient = surrogate_vjp(phi, cotangent, config)
    epsilon = 1e-6
    plus = np.sum(surrogate_forward(phi + epsilon * direction, config) * cotangent)
    minus = np.sum(surrogate_forward(phi - epsilon * direction, config) * cotangent)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert np.isclose(np.sum(gradient * direction), finite_difference, rtol=1e-6, atol=1e-6)


def test_supersample_changes_manufactured_resolution_but_not_gradient_resolution():
    phi = np.zeros((10, 12))
    result = project_manufacturability(phi, ManufacturingConfig(supersample=4))
    assert result.binary_mask.shape == (37, 45)
    cotangent = np.ones_like(result.clean_level_set)
    gradient = surrogate_vjp(phi, cotangent, ManufacturingConfig(supersample=4))
    assert gradient.shape == phi.shape
