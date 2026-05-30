# OpenShape Unity Integration

Automatically classifies every mesh in a Unity scene using the OpenShape model
and stamps each GameObject with a `MeshClassification` component.

## Architecture

```
Unity Editor  ──TCP:11000──►  Python server (unity_server.py)
   sends: vertices[]              │
   receives: top-K labels         ▼
                             OpenShape model
                             (MinkResNet34 + OpenCLIP ViT-bigG-14)
```

All communication uses **length-prefixed JSON** over a local TCP socket —
no UDP fragmentation issues, reliable for variable-size payloads.

## Quick start

### 1 — Start the Python server

```bash
# from repo root
conda activate openshape    # or your env with the OpenShape deps
python src/unity_server.py
# options:
#   --port 11000          TCP port (default 11000)
#   --top_k 5             how many labels to return (default 5)
#   --cpu                 run on CPU if no GPU
#   --model OpenShape/openshape-spconv-all   (default)
```

The server downloads the checkpoint from HuggingFace on first run (~800 MB).

### 2 — Open the scanner window in Unity

`Tools ▶ OpenShape ▶ Mesh Classifier`

| Setting | Description |
|---------|-------------|
| Python host | `127.0.0.1` (change if server is remote) |
| Port | Must match `--port` above |
| Top-K labels | How many ranked labels to store per object |
| Vertices to sample | Points sampled from each mesh (max 10 000) |
| Label vocabulary | One label per line — edit freely, any vocabulary |

Click **Start Scan**.  A progress bar tracks each object.
Click **Cancel** at any time — results already received are applied immediately.

### 3 — What gets added

Each classified GameObject receives a `MeshClassification` component with:

```
primaryLabel        "chair"
primaryConfidence   0.82
topResults          [ {label:"chair", confidence:0.82},
                      {label:"stool", confidence:0.71}, … ]
lastClassifiedUtc   "2024-06-01T12:34:56Z"
customTags          []   ← add your own tags here, never overwritten
```

### Runtime usage (example)

```csharp
// Find all "tree" objects at runtime
var trees = MeshClassification.FindAllWithLabel("tree");

// Check a specific object
if (GetComponent<MeshClassification>().HasLabel("enemy"))
    SpawnEnemyAI();
```

## Python dependencies

Beyond the standard OpenShape requirements:

```
open3d          # for point-cloud loading helpers
trimesh         # optional, for non-PLY mesh formats
```

All network code uses only the Python standard library.

## File layout

```
src/unity_server.py               ← Python TCP server
Unity/Scripts/MeshClassification.cs   ← MonoBehaviour (data + runtime API)
Unity/Editor/OpenShapeScanner.cs      ← EditorWindow (UI + scan logic)
```

Copy the two `.cs` files into your Unity project's `Assets` folder
(`Editor/` for the scanner, `Scripts/` for the component).
