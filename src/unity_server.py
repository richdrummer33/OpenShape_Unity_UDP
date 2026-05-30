"""
OpenShape Unity Server
----------------------
TCP server that receives mesh vertex data from Unity, runs OpenShape inference,
and returns classification results as JSON.

Usage:
    python src/unity_server.py [--config src/configs/train.yaml] [--port 11000] [--top_k 5]
    python src/unity_server.py --cpu   # run without CUDA (slower)
"""

import sys
import os
import json
import socket
import struct
import threading
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from collections import OrderedDict
import re

sys.path.insert(0, os.path.dirname(__file__))
import models
from param import parse_args as openshape_parse_args
from utils.data import normalize_pc
from utils.misc import load_config

try:
    import MinkowskiEngine as ME
    HAS_MINKOWSKI = True
except ImportError:
    HAS_MINKOWSKI = False
    print("[WARN] MinkowskiEngine not found – Minkowski models unavailable.")

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("OpenShapeServer")

# ── protocol helpers ──────────────────────────────────────────────────────────

HEADER_FMT = ">I"   # 4-byte big-endian unsigned int = message length
HEADER_SIZE = struct.calcsize(HEADER_FMT)

def send_msg(sock: socket.socket, obj: dict):
    data = json.dumps(obj).encode("utf-8")
    header = struct.pack(HEADER_FMT, len(data))
    sock.sendall(header + data)

def recv_msg(sock: socket.socket) -> dict:
    header = _recv_exact(sock, HEADER_SIZE)
    if header is None:
        return None
    (length,) = struct.unpack(HEADER_FMT, header)
    body = _recv_exact(sock, length)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))

def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

# ── mock models (for CI / testing without GPU or HuggingFace checkpoints) ────

CLIP_DIM = 1280  # ViT-bigG-14 output dimension

class _MockShapeModel(torch.nn.Module):
    """Deterministic random-projection stand-in for OpenShape."""
    def __init__(self, dim=CLIP_DIM):
        super().__init__()
        self.proj = torch.nn.Linear(6, dim, bias=False)
    def forward(self, coords_or_xyz, feat, **kwargs):
        # feat shape: (N, 6)  or  (B, N, 6)
        f = feat.float()
        if f.dim() == 3:
            f = f.mean(dim=1)          # (B, 6)
        else:
            f = f.mean(dim=0, keepdim=True)  # (1, 6)
        return self.proj(f)            # (B, dim)

class _MockClipModel:
    """Stand-in for OpenCLIP – returns seeded random text embeddings."""
    def __init__(self, dim=CLIP_DIM):
        self._dim = dim
    def encode_text(self, tokens):
        # reproducible per token hash so label rankings are stable in tests
        out = []
        for row in tokens.cpu().numpy():
            seed = int(row.sum()) % (2**31)
            rng = np.random.RandomState(seed)
            out.append(rng.randn(self._dim).astype(np.float32))
        return torch.from_numpy(np.stack(out))

def load_mock_models(device: str):
    log.info("Mock mode: loading stub shape + CLIP models (no checkpoints needed).")
    shape_model = _MockShapeModel().to(device).eval()
    clip_model  = _MockClipModel()
    return shape_model, clip_model

# ── model loading ─────────────────────────────────────────────────────────────

def load_openshape(config, model_name: str, device: str):
    model = models.make(config)
    if device == "cuda":
        model = model.cuda()
    if config.model.name.startswith("Mink") and HAS_MINKOWSKI:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    log.info(f"Downloading / loading checkpoint from HuggingFace: {model_name}")
    checkpoint = torch.load(
        hf_hub_download(repo_id=model_name, filename="model.pt"),
        map_location=device,
    )
    state = OrderedDict()
    pattern = re.compile("module\\.")
    for k, v in checkpoint["state_dict"].items():
        key = re.sub(pattern, "", k) if re.search("module", k) else k
        state[key] = v
    model.load_state_dict(state)
    model.eval()
    log.info("OpenShape model ready.")
    return model


def load_open_clip(device: str):
    log.info("Loading OpenCLIP ViT-bigG-14 …")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
    )
    clip_model = clip_model.to(device).eval()
    log.info("OpenCLIP ready.")
    return clip_model

# ── point-cloud helpers ───────────────────────────────────────────────────────

NUM_POINTS = 10_000

def vertices_to_pointcloud(vertices: list[list[float]], normals: list[list[float]] | None = None):
    """
    Convert raw vertex list (and optional normals) to the (coords, features)
    tensors expected by OpenShape.
    vertices: [[x,y,z], ...]  (any count; will be sampled/padded to NUM_POINTS)
    normals:  [[nx,ny,nz], ...] or None  (used as RGB stand-in, clamped 0-1)
    """
    xyz = np.array(vertices, dtype=np.float32)

    n = len(xyz)
    if n == 0:
        raise ValueError("Empty vertex array")

    # sample / repeat to exactly NUM_POINTS
    if n < NUM_POINTS:
        idx = np.random.choice(n, NUM_POINTS, replace=True)
    else:
        idx = np.random.choice(n, NUM_POINTS, replace=False)
    xyz = xyz[idx]

    # swap Y↔Z so that Y-up (Unity) → Z-up (OpenShape convention)
    xyz[:, [1, 2]] = xyz[:, [2, 1]]
    xyz = normalize_pc(xyz)

    if normals is not None and len(normals) == n:
        rgb = np.array(normals, dtype=np.float32)[idx]
        rgb = (rgb + 1.0) * 0.5          # [-1,1] → [0,1]
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.full((NUM_POINTS, 3), 0.4, dtype=np.float32)

    features = np.concatenate([xyz, rgb], axis=1)
    xyz_t = torch.from_numpy(xyz)
    feat_t = torch.from_numpy(features)
    return xyz_t, feat_t


@torch.no_grad()
def classify(xyz_t, feat_t, model, clip_model, labels, device: str, voxel_size: float):
    if device == "cuda":
        xyz_t = xyz_t.cuda()
        feat_t = feat_t.cuda()

    if HAS_MINKOWSKI and "Mink" in type(model).__name__:
        coords = ME.utils.batched_coordinates([xyz_t], dtype=torch.float32)
        if device == "cuda":
            coords = coords.cuda()
        shape_feat = model(coords, feat_t, device=device, quantization_size=voxel_size)
    else:
        shape_feat = model(xyz_t.unsqueeze(0), feat_t.unsqueeze(0))

    # clip_model may be real OpenCLIP or _MockClipModel
    if isinstance(clip_model, _MockClipModel):
        # mock path: no tokeniser needed
        dummy_tokens = torch.zeros(len(labels), 77, dtype=torch.long)
        text_feat = clip_model.encode_text(dummy_tokens)
    else:
        tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
        tokens = tokenizer(labels).to(device)
        text_feat = clip_model.encode_text(tokens)

    sim = (F.normalize(shape_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T).squeeze(0)
    return sim.cpu().float().numpy()

# ── request handler ───────────────────────────────────────────────────────────

class ClientHandler(threading.Thread):
    def __init__(self, conn, addr, server):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr
        self.server = server

    def run(self):
        log.info(f"Client connected: {self.addr}")
        try:
            while True:
                msg = recv_msg(self.conn)
                if msg is None:
                    break
                cmd = msg.get("command", "")
                if cmd == "ping":
                    send_msg(self.conn, {"status": "pong"})
                elif cmd == "classify":
                    self._handle_classify(msg)
                elif cmd == "cancel":
                    # Nothing to cancel per-request (each is synchronous here),
                    # but we acknowledge so Unity can move on.
                    send_msg(self.conn, {"status": "cancelled"})
                else:
                    send_msg(self.conn, {"status": "error", "message": f"Unknown command: {cmd}"})
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            log.exception(f"Handler error: {e}")
        finally:
            self.conn.close()
            log.info(f"Client disconnected: {self.addr}")

    def _handle_classify(self, msg: dict):
        try:
            vertices = msg["vertices"]          # [[x,y,z], ...]
            normals  = msg.get("normals")       # [[nx,ny,nz], ...] | null
            labels   = msg.get("labels") or self.server.default_labels
            obj_name = msg.get("object_name", "unknown")
            top_k    = int(msg.get("top_k", self.server.top_k))

            xyz_t, feat_t = vertices_to_pointcloud(vertices, normals)

            scores = classify(
                xyz_t, feat_t,
                self.server.model,
                self.server.clip_model,
                labels,
                self.server.device,
                self.server.voxel_size,
            )

            order = np.argsort(scores)[::-1]
            top = [{"label": labels[i], "confidence": float(scores[i])} for i in order[:top_k]]

            send_msg(self.conn, {
                "status": "ok",
                "object_name": obj_name,
                "primary_label": top[0]["label"],
                "primary_confidence": top[0]["confidence"],
                "top_results": top,
            })
        except Exception as e:
            log.exception(f"Classify error for '{msg.get('object_name')}': {e}")
            send_msg(self.conn, {"status": "error", "message": str(e)})

# ── server ────────────────────────────────────────────────────────────────────

class OpenShapeServer:
    def __init__(self, args):
        self.port       = args.port
        self.top_k      = args.top_k
        self.device     = "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
        self.model_name = args.model

        self.mock = args.mock
        log.info(f"Device: {self.device}  |  mock={self.mock}")

        if self.mock:
            self.voxel_size = 0.02
            self.model, self.clip_model = load_mock_models(self.device)
        else:
            cli, extras = openshape_parse_args([])
            config = load_config(args.config, cli_args=vars(cli), extra_args=extras)
            self.voxel_size = config.model.voxel_size
            self.model      = load_openshape(config, self.model_name, self.device)
            self.clip_model = load_open_clip(self.device)

        # broad default label set – Unity can override per-request
        self.default_labels = [
            "chair", "table", "desk", "sofa", "bed", "lamp", "door", "window",
            "floor", "wall", "ceiling", "staircase", "bookshelf", "cabinet",
            "monitor", "keyboard", "mouse", "phone", "laptop", "television",
            "car", "truck", "bus", "bicycle", "motorcycle", "airplane", "boat",
            "tree", "plant", "flower", "grass", "rock", "stone", "fence",
            "building", "house", "roof", "pillar", "column", "arch", "bridge",
            "barrel", "box", "crate", "bottle", "cup", "mug", "bowl", "plate",
            "sword", "shield", "axe", "gun", "bow", "arrow", "helmet", "armor",
            "chest", "treasure", "coin", "crystal", "gem", "key", "scroll",
            "torch", "lantern", "candle", "fire", "smoke", "water", "ice",
            "mushroom", "cactus", "log", "stump", "bush", "vine", "hay",
            "anvil", "forge", "wheel", "gear", "pipe", "valve", "lever",
            "sign", "banner", "flag", "statue", "fountain", "well", "altar",
        ]

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", self.port))
            srv.listen(5)
            log.info(f"Listening on 127.0.0.1:{self.port}")
            while True:
                conn, addr = srv.accept()
                ClientHandler(conn, addr, self).start()


def main():
    ap = argparse.ArgumentParser(description="OpenShape Unity TCP Server")
    ap.add_argument("--config", default="src/configs/train.yaml")
    ap.add_argument("--port",   type=int, default=11000)
    ap.add_argument("--top_k",  type=int, default=5)
    ap.add_argument("--cpu",    action="store_true")
    ap.add_argument("--model",  default="OpenShape/openshape-spconv-all",
                    help="HuggingFace model repo id")
    ap.add_argument("--mock",   action="store_true",
                    help="Use stub models (no checkpoint download, for testing)")
    args = ap.parse_args()
    OpenShapeServer(args).serve()


if __name__ == "__main__":
    main()
