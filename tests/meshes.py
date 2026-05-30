"""
Procedural test mesh generators — no file I/O, just numpy vertex/normal arrays.
Each function returns (vertices: np.ndarray shape (N,3), normals: np.ndarray shape (N,3)).
"""

import numpy as np


def cube(half=1.0):
    """8-corner unit cube, normals pointing outward from each face."""
    v = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
    ], dtype=np.float32) * half
    # face normals for each vertex (just assign nearest face normal)
    n = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
    ], dtype=np.float32)
    norms = n / np.linalg.norm(n, axis=1, keepdims=True)
    return v, norms


def sphere(radius=1.0, n_lat=20, n_lon=20):
    """UV sphere."""
    verts, norms = [], []
    for i in range(n_lat + 1):
        lat = np.pi * (-0.5 + i / n_lat)
        for j in range(n_lon):
            lon = 2 * np.pi * j / n_lon
            x = np.cos(lat) * np.cos(lon)
            y = np.sin(lat)
            z = np.cos(lat) * np.sin(lon)
            verts.append([x * radius, y * radius, z * radius])
            norms.append([x, y, z])
    return np.array(verts, dtype=np.float32), np.array(norms, dtype=np.float32)


def cylinder(radius=0.5, height=2.0, n_seg=24):
    """Open-ended cylinder (sides only)."""
    verts, norms = [], []
    for i in range(n_seg + 1):
        angle = 2 * np.pi * i / n_seg
        cx, cz = np.cos(angle), np.sin(angle)
        for y in [-height / 2, height / 2]:
            verts.append([cx * radius, y, cz * radius])
            norms.append([cx, 0.0, cz])
    return np.array(verts, dtype=np.float32), np.array(norms, dtype=np.float32)


def cone(radius=0.5, height=2.0, n_seg=24):
    """Cone with apex at top."""
    verts, norms = [], []
    apex = np.array([0.0, height / 2, 0.0], dtype=np.float32)
    for i in range(n_seg):
        angle = 2 * np.pi * i / n_seg
        cx, cz = np.cos(angle), np.sin(angle)
        base_pt = np.array([cx * radius, -height / 2, cz * radius], dtype=np.float32)
        # slant normal
        slant = np.array([cx, radius / height, cz], dtype=np.float32)
        slant /= np.linalg.norm(slant)
        verts.extend([apex, base_pt])
        norms.extend([slant, slant])
    return np.array(verts, dtype=np.float32), np.array(norms, dtype=np.float32)


def torus(R=1.0, r=0.3, n_major=24, n_minor=12):
    """Torus (donut shape)."""
    verts, norms = [], []
    for i in range(n_major):
        phi = 2 * np.pi * i / n_major
        for j in range(n_minor):
            theta = 2 * np.pi * j / n_minor
            x = (R + r * np.cos(theta)) * np.cos(phi)
            y = r * np.sin(theta)
            z = (R + r * np.cos(theta)) * np.sin(phi)
            nx = np.cos(theta) * np.cos(phi)
            ny = np.sin(theta)
            nz = np.cos(theta) * np.sin(phi)
            verts.append([x, y, z])
            norms.append([nx, ny, nz])
    return np.array(verts, dtype=np.float32), np.array(norms, dtype=np.float32)


def flat_plane(size=2.0, subdivisions=10):
    """Flat horizontal plane — tests degenerate (2-D) input handling."""
    pts = np.linspace(-size / 2, size / 2, subdivisions)
    xs, zs = np.meshgrid(pts, pts)
    verts = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], axis=1).astype(np.float32)
    norms = np.tile([0.0, 1.0, 0.0], (len(verts), 1)).astype(np.float32)
    return verts, norms


def pyramid(base=1.0, height=1.5):
    """Square pyramid — 5 vertices."""
    h2 = base / 2
    verts = np.array([
        [ h2, 0,  h2],
        [-h2, 0,  h2],
        [-h2, 0, -h2],
        [ h2, 0, -h2],
        [0.0, height, 0.0],
    ], dtype=np.float32)
    norms = verts / (np.linalg.norm(verts, axis=1, keepdims=True) + 1e-8)
    return verts, norms


# All shapes as a dict for parametrised tests
ALL_SHAPES = {
    "cube":       cube,
    "sphere":     sphere,
    "cylinder":   cylinder,
    "cone":       cone,
    "torus":      torus,
    "flat_plane": flat_plane,
    "pyramid":    pyramid,
}
