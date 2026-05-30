"""
Unit tests for the wire protocol and point-cloud helpers in unity_server.py.
These tests do NOT start a server and do NOT require OpenShape / MinkowskiEngine.
"""

import sys, os, socket, struct, json, threading
import numpy as np
import pytest

# ── make src importable ───────────────────────────────────────────────────────
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

# import only the pure-Python helpers — guard against missing heavy deps
import importlib, types

def _stub_heavy():
    """Replace heavy optional imports with stubs so we can import the module."""
    for name in ["MinkowskiEngine", "open_clip", "models",
                 "huggingface_hub", "param"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    # minimal stubs for things unity_server actually calls at import time
    sys.modules["open_clip"].get_tokenizer = lambda *a, **k: None
    sys.modules["open_clip"].create_model_and_transforms = lambda *a, **k: (None, None, None)
    sys.modules["huggingface_hub"].hf_hub_download = lambda *a, **k: ""
    # param.parse_args
    import argparse
    sys.modules["param"].parse_args = lambda argv: (argparse.Namespace(), [])
    # utils.*
    for sub in ["utils", "utils.data", "utils.misc"]:
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)
    sys.modules["utils.data"].normalize_pc = lambda xyz: xyz / (np.linalg.norm(xyz, axis=1, keepdims=True).max() + 1e-8)
    sys.modules["utils.misc"].load_config = lambda *a, **k: types.SimpleNamespace(model=types.SimpleNamespace(name="Mock", voxel_size=0.02))

_stub_heavy()
import unity_server as us

# ── helpers imported from server ─────────────────────────────────────────────
from unity_server import send_msg, recv_msg, vertices_to_pointcloud, NUM_POINTS
from tests.meshes import ALL_SHAPES


# ── protocol round-trip ───────────────────────────────────────────────────────

class _SocketPair:
    """Creates a connected socket pair for in-process testing."""
    def __enter__(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        self.client = socket.create_connection(("127.0.0.1", port))
        self.server_conn, _ = srv.accept()
        srv.close()
        return self.client, self.server_conn

    def __exit__(self, *_):
        self.client.close()
        self.server_conn.close()


@pytest.mark.parametrize("payload", [
    {"command": "ping"},
    {"command": "classify", "object_name": "Tree_01", "top_k": 3},
    {"status": "ok", "primary_label": "chair", "primary_confidence": 0.87,
     "top_results": [{"label": "chair", "confidence": 0.87}]},
    # large-ish payload to check length prefix
    {"data": list(range(500))},
])
def test_protocol_roundtrip(payload):
    with _SocketPair() as (c, s):
        send_msg(c, payload)
        received = recv_msg(s)
    assert received == payload


def test_protocol_multi_message():
    """Server side can receive several consecutive messages."""
    msgs = [{"i": i, "v": "x" * i} for i in range(10)]
    with _SocketPair() as (c, s):
        for m in msgs:
            send_msg(c, m)
        received = [recv_msg(s) for _ in msgs]
    assert received == msgs


def test_recv_returns_none_on_closed_socket():
    with _SocketPair() as (c, s):
        c.close()
        result = recv_msg(s)
    assert result is None


# ── vertices_to_pointcloud ────────────────────────────────────────────────────

@pytest.mark.parametrize("shape_name", list(ALL_SHAPES.keys()))
def test_pointcloud_shape(shape_name):
    verts, norms = ALL_SHAPES[shape_name]()
    xyz_t, feat_t = vertices_to_pointcloud(verts.tolist(), norms.tolist())
    assert xyz_t.shape  == (NUM_POINTS, 3)
    assert feat_t.shape == (NUM_POINTS, 6)


@pytest.mark.parametrize("shape_name", list(ALL_SHAPES.keys()))
def test_pointcloud_normalized(shape_name):
    """After normalization the point cloud should fit in [-1, 1]^3."""
    verts, _ = ALL_SHAPES[shape_name]()
    xyz_t, _ = vertices_to_pointcloud(verts.tolist())
    xyz = xyz_t.numpy()
    assert np.all(np.abs(xyz) <= 1.0 + 1e-5), f"{shape_name}: xyz out of [-1,1]"


def test_pointcloud_upsamples_small_mesh():
    """Meshes with fewer vertices than NUM_POINTS should be padded via repeat."""
    verts, norms = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], None
    xyz_t, feat_t = vertices_to_pointcloud(verts, norms)
    assert xyz_t.shape[0] == NUM_POINTS


def test_pointcloud_samples_large_mesh():
    """Meshes with more vertices than NUM_POINTS should be downsampled."""
    rng = np.random.default_rng(42)
    big = rng.standard_normal((NUM_POINTS * 3, 3)).astype(np.float32)
    xyz_t, _ = vertices_to_pointcloud(big.tolist())
    assert xyz_t.shape[0] == NUM_POINTS


def test_pointcloud_no_normals_fallback():
    """When normals are None the rgb channel should be filled with 0.4."""
    verts, _ = ALL_SHAPES["sphere"]()
    _, feat_t = vertices_to_pointcloud(verts.tolist(), normals=None)
    rgb = feat_t[:, 3:].numpy()
    assert np.allclose(rgb, 0.4)


def test_pointcloud_axis_swap():
    """Unity Y-up vertex [0,1,0] should map to approx [0,0,1] after Y<->Z swap."""
    verts = [[0.0, 1.0, 0.0]] * (NUM_POINTS + 1)
    xyz_t, _ = vertices_to_pointcloud(verts)
    # after swap, the original Y=1 goes to Z; after normalize all pts identical
    means = xyz_t.numpy().mean(axis=0)
    # y-component should be near 0, z-component should be dominant
    assert abs(means[1]) < 0.05, "Y should be ~0 after axis swap"
    assert abs(means[2]) > 0.5,  "Z should be dominant after axis swap"


def test_pointcloud_empty_raises():
    with pytest.raises((ValueError, Exception)):
        vertices_to_pointcloud([])
