# OpenShape × Unity — Automatic Mesh Classifier

> **What is this?**
> You have a Unity scene full of meshes. You want to know what each one *is* — chair, tree, rock, barrel — without touching a label by hand. This tool does that automatically using [OpenShape](https://github.com/Colin97/OpenShape_code), a SOTA open-vocabulary 3-D shape understanding model (85.3% top-1 on ModelNet40 zero-shot).

---

## How it works

```
┌─────────────────────────────────┐        TCP :11000        ┌──────────────────────────────┐
│         Unity Editor            │  ───── vertices[] ──────► │   Python sidecar server      │
│                                 │                           │   (unity_server.py)          │
│  Tools > OpenShape >            │ ◄──── top-K labels ─────  │                              │
│    Mesh Classifier              │       + confidence        │   OpenShape  (MinkResNet34)  │
│                                 │                           │   OpenCLIP   (ViT-bigG-14)   │
│  [MeshClassification]           │                           │                              │
│   component stamped on each GO  │                           └──────────────────────────────┘
└─────────────────────────────────┘
```

- Unity samples vertex data from each `MeshRenderer` in the scene and sends it over a local TCP socket.
- The Python server converts vertices to a point cloud, runs OpenShape inference, then ranks your label vocabulary by cosine similarity.
- Results are written back to a `MeshClassification` MonoBehaviour on each GameObject — fully serialized, Undo-safe, and readable at runtime.

No UDP fragmentation, no file-path hacks, no Unity Sentis, no asset store purchase.

---

## Quick start

### Step 1 — Python environment

You need the [OpenShape dependencies](https://github.com/Colin97/OpenShape_code) (MinkowskiEngine, DGL, open3d, open_clip) plus the standard Python stdlib for networking. Once your conda env is set up:

```bash
# from this repo root
conda activate openshape

python src/unity_server.py          # GPU (recommended)
python src/unity_server.py --cpu    # CPU-only, slower but works
```

The first run downloads the checkpoint from HuggingFace (~800 MB). Subsequent starts are instant.

```
$ python src/unity_server.py
2024-06-01 12:00:00 [INFO] Device: cuda
2024-06-01 12:00:01 [INFO] Downloading / loading checkpoint ...
2024-06-01 12:00:12 [INFO] OpenShape model ready.
2024-06-01 12:00:22 [INFO] OpenCLIP ready.
2024-06-01 12:00:22 [INFO] Listening on 127.0.0.1:11000   <-- ready
```

Server flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--port` | `11000` | TCP port |
| `--top_k` | `5` | Labels returned per object |
| `--cpu` | off | Force CPU inference |
| `--mock` | off | Stub model (for testing, no download) |
| `--model` | `OpenShape/openshape-spconv-all` | HuggingFace repo |

---

### Step 2 — Copy the Unity scripts

Drop these two files into your Unity project:

```
Unity/Scripts/MeshClassification.cs  →  Assets/Scripts/MeshClassification.cs
Unity/Editor/OpenShapeScanner.cs     →  Assets/Editor/OpenShapeScanner.cs
```

That's it — no packages, no `.unitypackage`, no Assembly Definitions needed.

---

### Step 3 — Run a scan

Open the scanner window:

```
Tools  >  OpenShape  >  Mesh Classifier
```

```
┌──────────────────────────────────────────────┐
│  OpenShape Mesh Classifier                   │
├──────────────────────────────────────────────┤
│  Python host   [ 127.0.0.1              ]    │
│  Port          [ 11000                  ]    │
│  Top-K labels  [──●────────────] 5           │
│  Vertices      [────●───────────] 4096       │
│                                              │
│  ▶ Label vocabulary (one per line)           │
│  ┌────────────────────────────────────────┐  │
│  │ chair                                  │  │
│  │ table                                  │  │
│  │ tree                                   │  │
│  │ rock                                   │  │
│  │ barrel                                 │  │
│  │ ...                                    │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  Idle                                        │
│  ┌──────────────────────────────────────┐    │
│  │           Start Scan                 │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

While scanning:

```
┌──────────────────────────────────────────────┐
│  Classifying: Tree_03   (14 / 87)            │
│  ████████████░░░░░░░░░░░░░░░░░░  14 / 87     │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  Cancel (apply partial results)      │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

Cancel at any time — every result received so far is already applied. Nothing is lost.

---

### Step 4 — What you get

After the scan, every classified GameObject has a `MeshClassification` component visible in the Inspector:

```
▼ Mesh Classification (Script)
  ─ Primary Classification ──────────────────────
    Primary Label         tree
    Primary Confidence    0.84
  ─ All Top Results ─────────────────────────────
    ▼ Top Results         5 items
      [0]  label: tree        confidence: 0.84
      [1]  label: plant       confidence: 0.71
      [2]  label: stump       confidence: 0.62
      [3]  label: log         confidence: 0.58
      [4]  label: bush        confidence: 0.51
  ─ Meta ────────────────────────────────────────
    Last Classified Utc   2024-06-01T12:05:43Z
  ─ User overrides / extra tags ─────────────────
    ▼ Custom Tags         0 items               ← add your own, never overwritten
```

---

## Runtime API

```csharp
// Find every object the AI called a "tree"
List<GameObject> trees = MeshClassification.FindAllWithLabel("tree");

// Check one object (also searches customTags)
if (GetComponent<MeshClassification>().HasLabel("hazard"))
    EnableDamageZone();

// Read the ranked results directly
var mc = GetComponent<MeshClassification>();
Debug.Log($"{mc.primaryLabel}  ({mc.primaryConfidence:P0})");

foreach (var r in mc.topResults)
    Debug.Log($"  {r.label}: {r.confidence:F2}");
```

`HasLabel()` is case-insensitive and checks both `topResults` and your `customTags`, so mixing AI labels with hand-written ones works seamlessly.

---

## Label vocabulary

The vocabulary is open — write any words you want, one per line, in the scanner window. The model doesn't need retraining; it scores each label via CLIP similarity at inference time.

**Tips for low-poly VR art:**
- Use game-genre nouns: `chest`, `torch`, `barrel`, `shrine`, `crystal`, `gate`
- Be specific where it matters: `stone_wall` vs `wooden_wall` vs `fence`
- Add material cues: `mossy rock`, `metal door` (OpenCLIP handles multi-word labels)
- The server ships an 80-label default vocabulary covering common game objects — it activates whenever Unity doesn't send a `labels` array

---

## Testing

The test suite runs in **mock mode** — no GPU, no HuggingFace download, just `torch` (CPU) + `numpy` + `pytest`. It covers the full round-trip: wire protocol, point-cloud sampling, server request/response, concurrent clients, and edge cases.

```bash
pip install torch numpy pytest
pytest tests/ -v
```

GitHub Actions runs these tests automatically on every push and pull request (Python 3.10 and 3.11).

### Test shapes

Seven procedural meshes are generated in code (no asset files needed):

| Shape | Description | Why |
|-------|-------------|-----|
| `cube` | 8-vertex unit cube | minimal vertex count, degenerate sampling |
| `sphere` | 440-pt UV sphere | smooth, uniform coverage |
| `cylinder` | 50-pt open cylinder | partial symmetry |
| `cone` | 48-pt cone | apex singularity |
| `torus` | 288-pt donut | concave topology |
| `flat_plane` | 100-pt plane | near-planar / degenerate Z extent |
| `pyramid` | 5-vertex pyramid | extreme undersample (fewer than NUM_POINTS) |

---

## File layout

```
src/
  unity_server.py              ← Python TCP server (run this first)
  configs/train.yaml           ← OpenShape model config

Unity/
  Scripts/
    MeshClassification.cs      ← MonoBehaviour: data holder + runtime API
  Editor/
    OpenShapeScanner.cs        ← EditorWindow: scan UI + TCP client

tests/
  meshes.py                    ← procedural test mesh generators
  test_protocol.py             ← unit tests: wire protocol + point-cloud helpers
  test_server_mock.py          ← integration tests: full server round-trip (mock)

.github/
  workflows/
    test.yml                   ← CI: runs on every push / PR
```

---

## Dependencies

**Python** (in your OpenShape conda env):

```
torch              # deep learning
open_clip          # CLIP text encoder
MinkowskiEngine    # sparse 3D convolutions (OpenShape backbone)
open3d             # point-cloud I/O helpers
huggingface_hub    # checkpoint download
numpy              # everything else
```

All networking uses only the Python standard library (`socket`, `struct`, `json`, `threading`).

**Unity** — nothing beyond the built-in editor API. No packages, no UPM, no DLLs.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` | Python server isn't running yet, or wrong port |
| `Server ping failed` | Server still loading the model — wait for `Listening on ...` log line |
| Slow inference | Add `--cpu` if no CUDA GPU, or reduce `Vertices to sample` in the window |
| No MeshRenderers found | Make sure at least one object in the scene has a `MeshRenderer` + `MeshFilter` |
| Labels seem random | The mock server (`--mock`) produces deterministic-but-arbitrary results; use the real model for actual inference |

---

*Based on [OpenShape](https://arxiv.org/pdf/2305.10764.pdf) (NeurIPS 2023) by Colin Zhang et al.*
