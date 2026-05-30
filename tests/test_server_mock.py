"""
Integration tests: start the server in --mock mode in a background thread,
connect a real TCP client, send classify requests, verify responses.

Run:
    pytest tests/test_server_mock.py -v
"""

import sys, os, socket, struct, json, threading, time, argparse, types
import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)


# ── stub heavy deps (same as test_protocol.py) ───────────────────────────────

def _stub_heavy():
    for name in ["MinkowskiEngine", "open_clip", "models",
                 "huggingface_hub", "param"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["open_clip"].get_tokenizer = lambda *a, **k: None
    sys.modules["open_clip"].create_model_and_transforms = lambda *a, **k: (None, None, None)
    sys.modules["huggingface_hub"].hf_hub_download = lambda *a, **k: ""
    import argparse as _ap
    sys.modules["param"].parse_args = lambda argv: (_ap.Namespace(), [])
    for sub in ["utils", "utils.data", "utils.misc"]:
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)
    sys.modules["utils.data"].normalize_pc = (
        lambda xyz: xyz / (np.linalg.norm(xyz, axis=1, keepdims=True).max() + 1e-8))
    sys.modules["utils.misc"].load_config = lambda *a, **k: types.SimpleNamespace(
        model=types.SimpleNamespace(name="Mock", voxel_size=0.02))

_stub_heavy()
import unity_server as us
from unity_server import send_msg, recv_msg
from tests.meshes import ALL_SHAPES


# ── server fixture ────────────────────────────────────────────────────────────

def _find_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock OpenShapeServer once for the whole test module."""
    port = _find_free_port()
    args = argparse.Namespace(
        port=port, top_k=5, cpu=True, mock=True,
        model="unused", config="unused",
    )
    server = us.OpenShapeServer(args)

    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    # wait until accepting
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.2)
            s.close()
            break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("Mock server did not start in time")

    yield port


# ── client helper ─────────────────────────────────────────────────────────────

class _Client:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.sock.settimeout(10.0)

    def send(self, obj):
        send_msg(self.sock, obj)
        return recv_msg(self.sock)

    def close(self):
        self.sock.close()


# ── helpers ───────────────────────────────────────────────────────────────────

def _verts_and_norms(shape_name):
    v, n = ALL_SHAPES[shape_name]()
    return v.tolist(), n.tolist()


# ── tests ─────────────────────────────────────────────────────────────────────

def test_ping(mock_server):
    c = _Client(mock_server)
    resp = c.send({"command": "ping"})
    c.close()
    assert resp["status"] == "pong"


@pytest.mark.parametrize("shape_name", list(ALL_SHAPES.keys()))
def test_classify_shape(mock_server, shape_name):
    """Every shape should return a valid classification response."""
    c = _Client(mock_server)
    verts, norms = _verts_and_norms(shape_name)
    resp = c.send({
        "command":     "classify",
        "object_name": shape_name,
        "vertices":    verts,
        "normals":     norms,
        "labels":      ["chair", "tree", "rock", "barrel", "wall"],
        "top_k":       3,
    })
    c.close()

    assert resp["status"]             == "ok",          f"{shape_name}: bad status"
    assert resp["object_name"]        == shape_name,    f"{shape_name}: name mismatch"
    assert isinstance(resp["primary_label"],      str),   f"{shape_name}: label not str"
    assert isinstance(resp["primary_confidence"], float), f"{shape_name}: conf not float"
    assert len(resp["top_results"])   == 3,             f"{shape_name}: wrong top_k"


def test_classify_top_k_respected(mock_server):
    verts, norms = _verts_and_norms("sphere")
    labels = ["chair", "table", "sofa", "bed", "lamp", "door", "window"]
    for k in [1, 3, 5, 7]:
        c = _Client(mock_server)
        resp = c.send({
            "command":  "classify",
            "vertices": verts,
            "normals":  norms,
            "labels":   labels,
            "top_k":    k,
        })
        c.close()
        assert len(resp["top_results"]) == k, f"top_k={k} returned {len(resp['top_results'])}"


def test_classify_top_results_sorted(mock_server):
    """Confidence scores must be in descending order."""
    verts, norms = _verts_and_norms("cube")
    c = _Client(mock_server)
    resp = c.send({
        "command":  "classify",
        "vertices": verts,
        "normals":  norms,
        "labels":   ["a", "b", "c", "d", "e"],
        "top_k":    5,
    })
    c.close()
    confs = [r["confidence"] for r in resp["top_results"]]
    assert confs == sorted(confs, reverse=True), "Top results not sorted by confidence"


def test_classify_no_normals(mock_server):
    """Omitting normals should not crash — server fills in defaults."""
    verts, _ = _verts_and_norms("pyramid")
    c = _Client(mock_server)
    resp = c.send({
        "command":  "classify",
        "vertices": verts,
        "labels":   ["rock", "crystal"],
        "top_k":    2,
    })
    c.close()
    assert resp["status"] == "ok"


def test_classify_uses_default_labels_when_omitted(mock_server):
    """If no labels key is sent, server should use its default vocabulary."""
    verts, norms = _verts_and_norms("cylinder")
    c = _Client(mock_server)
    resp = c.send({
        "command":  "classify",
        "vertices": verts,
        "normals":  norms,
    })
    c.close()
    assert resp["status"] == "ok"
    assert isinstance(resp["primary_label"], str) and len(resp["primary_label"]) > 0


def test_classify_object_name_echoed(mock_server):
    verts, _ = _verts_and_norms("torus")
    c = _Client(mock_server)
    resp = c.send({
        "command":     "classify",
        "object_name": "MyTorus_42",
        "vertices":    verts,
        "labels":      ["ring", "wheel"],
        "top_k":       2,
    })
    c.close()
    assert resp["object_name"] == "MyTorus_42"


def test_cancel_acknowledged(mock_server):
    c = _Client(mock_server)
    resp = c.send({"command": "cancel"})
    c.close()
    assert resp["status"] == "cancelled"


def test_unknown_command_returns_error(mock_server):
    c = _Client(mock_server)
    resp = c.send({"command": "nonexistent_command"})
    c.close()
    assert resp["status"] == "error"


def test_multiple_requests_same_connection(mock_server):
    """TCP connection should handle many sequential requests."""
    verts, norms = _verts_and_norms("sphere")
    c = _Client(mock_server)
    for i in range(10):
        resp = c.send({
            "command":     "classify",
            "object_name": f"obj_{i}",
            "vertices":    verts,
            "normals":     norms,
            "labels":      ["tree", "rock", "wall"],
            "top_k":       2,
        })
        assert resp["status"] == "ok", f"Request {i} failed: {resp}"
    c.close()


def test_concurrent_clients(mock_server):
    """Multiple simultaneous clients should all get valid responses."""
    verts, norms = _verts_and_norms("cone")
    results = {}
    errors  = {}

    def worker(idx):
        try:
            c = _Client(mock_server)
            resp = c.send({
                "command":     "classify",
                "object_name": f"concurrent_{idx}",
                "vertices":    verts,
                "normals":     norms,
                "labels":      ["cone", "tower", "hat"],
                "top_k":       2,
            })
            c.close()
            results[idx] = resp
        except Exception as e:
            errors[idx] = str(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=15)

    assert not errors, f"Concurrent client errors: {errors}"
    for idx, resp in results.items():
        assert resp["status"] == "ok", f"Client {idx} bad response: {resp}"
