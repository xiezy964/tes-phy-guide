import numpy as np

from phyguide_tesseract.meshing import contour_vjp, gmsh_level_set, gmsh_mesh_vjp


def test_gmsh_interface_is_conforming_and_vjp_finite():
    yy, xx = np.mgrid[0.0:1.0:65j, 0.0:1.0:65j]
    phi = 0.28 - np.hypot(xx - 0.5, yy - 0.5)
    mesh = gmsh_level_set(phi, coarse_size=0.16, interface_size=0.04, interface_bandwidth=0.12)
    edges = np.vstack((mesh.cells[:, [0, 1]], mesh.cells[:, [1, 2]], mesh.cells[:, [2, 0]]))
    edge_set = {tuple(sorted(map(int, edge))) for edge in edges}
    assert all(tuple(sorted(map(int, edge))) in edge_set for edge in mesh.interface_segments)
    tri = mesh.points[mesh.cells]
    assert np.all(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) > 0)
    cotangent = gmsh_mesh_vjp(mesh, np.ones_like(mesh.points))
    assert contour_vjp(mesh.contour, cotangent).shape == phi.shape
    assert np.isfinite(cotangent).all()
