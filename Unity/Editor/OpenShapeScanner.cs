#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Editor window that scans every MeshRenderer in the open scene,
/// sends vertex data to the Python OpenShape server, and stamps each
/// GameObject with a <see cref="MeshClassification"/> component.
///
/// Open via: Tools ▶ OpenShape ▶ Mesh Classifier
/// </summary>
public class OpenShapeScanner : EditorWindow
{
    // ── settings ──────────────────────────────────────────────────────────────

    [SerializeField] private string  host        = "127.0.0.1";
    [SerializeField] private int     port        = 11000;
    [SerializeField] private int     topK        = 5;
    [SerializeField] private int     sampleVerts = 10000;

    // custom labels the user can edit in the window (one per line)
    [SerializeField] private string  labelsText  =
        "chair\ntable\nsofa\nbed\nlamp\ndoor\nwindow\nfloor\nwall\nceiling\n" +
        "tree\nrock\nbuilding\ncar\nbarrel\nchest\ntorch\nsign\nfountain\nwell";

    // ── runtime state ─────────────────────────────────────────────────────────

    private bool   _scanning    = false;
    private bool   _cancelFlag  = false;
    private string _statusLine  = "Idle";
    private float  _progress    = 0f;
    private int    _done, _total;

    private Thread         _scanThread;
    private Queue<Action>  _mainQueue   = new();
    private readonly object _qLock      = new();

    private Vector2 _scrollLabels;
    private bool    _showLabels = false;

    // ── menu ──────────────────────────────────────────────────────────────────

    [MenuItem("Tools/OpenShape/Mesh Classifier")]
    public static void Open() => GetWindow<OpenShapeScanner>("OpenShape Classifier");

    // ── GUI ───────────────────────────────────────────────────────────────────

    private void OnGUI()
    {
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("OpenShape Mesh Classifier", EditorStyles.boldLabel);
        EditorGUILayout.Space(4);

        EditorGUI.BeginDisabledGroup(_scanning);
        host        = EditorGUILayout.TextField("Python host", host);
        port        = EditorGUILayout.IntField("Port", port);
        topK        = EditorGUILayout.IntSlider("Top-K labels", topK, 1, 20);
        sampleVerts = EditorGUILayout.IntSlider("Vertices to sample", sampleVerts, 256, 16384);
        EditorGUI.EndDisabledGroup();

        EditorGUILayout.Space(4);
        _showLabels = EditorGUILayout.Foldout(_showLabels, "Label vocabulary (one per line)");
        if (_showLabels)
        {
            EditorGUI.BeginDisabledGroup(_scanning);
            _scrollLabels = EditorGUILayout.BeginScrollView(_scrollLabels, GUILayout.Height(120));
            labelsText = EditorGUILayout.TextArea(labelsText, GUILayout.ExpandHeight(true));
            EditorGUILayout.EndScrollView();
            EditorGUI.EndDisabledGroup();
        }

        EditorGUILayout.Space(8);

        if (_scanning)
        {
            EditorGUILayout.LabelField(_statusLine, EditorStyles.helpBox);
            Rect r = GUILayoutUtility.GetRect(18, 18, GUILayout.ExpandWidth(true));
            EditorGUI.ProgressBar(r, _progress, $"{_done} / {_total}");
            EditorGUILayout.Space(4);
            if (GUILayout.Button("Cancel (apply partial results)", GUILayout.Height(28)))
                _cancelFlag = true;
        }
        else
        {
            EditorGUILayout.LabelField(_statusLine, EditorStyles.helpBox);
            if (GUILayout.Button("Start Scan", GUILayout.Height(32)))
                StartScan();
        }
    }

    private void Update()
    {
        // drain main-thread callbacks posted from the scan thread
        lock (_qLock)
        {
            while (_mainQueue.Count > 0)
                _mainQueue.Dequeue()();
        }
    }

    private void OnDisable()
    {
        _cancelFlag = true;
    }

    // ── scan orchestration ────────────────────────────────────────────────────

    private void StartScan()
    {
        var renderers = FindAllMeshRenderers();
        if (renderers.Count == 0)
        {
            _statusLine = "No MeshRenderers found in scene.";
            return;
        }

        _cancelFlag = false;
        _scanning   = true;
        _done       = 0;
        _total      = renderers.Count;
        _progress   = 0f;
        _statusLine = $"Scanning {_total} objects…";

        // capture config for the background thread
        string h    = host;
        int    p    = port;
        int    k    = topK;
        int    sv   = sampleVerts;
        var    lbls = ParseLabels();

        // snapshot mesh data on main thread before going async
        var payloads = BuildPayloads(renderers, sv);

        _scanThread = new Thread(() => ScanThread(payloads, h, p, k, lbls))
        {
            IsBackground = true,
            Name = "OpenShapeScan"
        };
        _scanThread.Start();
    }

    // ── payload building (main thread) ────────────────────────────────────────

    private static List<(GameObject go, string name, List<float[]> verts, List<float[]> norms)>
        BuildPayloads(List<MeshRenderer> renderers, int sampleVerts)
    {
        var list = new List<(GameObject, string, List<float[]>, List<float[]>)>();
        foreach (var mr in renderers)
        {
            var mf = mr.GetComponent<MeshFilter>();
            if (mf == null || mf.sharedMesh == null) continue;
            var (v, n) = SampleMesh(mf.sharedMesh, sampleVerts);
            list.Add((mr.gameObject, mr.gameObject.name, v, n));
        }
        return list;
    }

    private static (List<float[]> verts, List<float[]> norms) SampleMesh(Mesh mesh, int count)
    {
        var srcV = mesh.vertices;
        var srcN = mesh.normals;
        bool hasN = srcN != null && srcN.Length == srcV.Length;
        int n = srcV.Length;

        // simple stride-based sampling
        int step = Mathf.Max(1, n / count);
        var v = new List<float[]>(count);
        var nm = new List<float[]>(count);
        for (int i = 0; i < n && v.Count < count; i += step)
        {
            v.Add(new[] { srcV[i].x, srcV[i].y, srcV[i].z });
            nm.Add(hasN
                ? new[] { srcN[i].x, srcN[i].y, srcN[i].z }
                : new[] { 0f, 1f, 0f });
        }
        return (v, nm);
    }

    // ── background scan thread ────────────────────────────────────────────────

    private void ScanThread(
        List<(GameObject go, string name, List<float[]> verts, List<float[]> norms)> payloads,
        string host, int port, int topK, List<string> labels)
    {
        TcpClient client = null;
        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            client.SendTimeout    = 10000;
            client.ReceiveTimeout = 60000;   // inference can take a few seconds

            // ping
            SendMsg(client, new Dictionary<string, object> { ["command"] = "ping" });
            var pong = RecvMsg(client);
            if (pong == null || (string)pong["status"] != "pong")
                throw new Exception("Server ping failed.");

            foreach (var (go, name, verts, norms) in payloads)
            {
                if (_cancelFlag) break;

                PostStatus($"Classifying: {name}  ({_done + 1}/{_total})");

                var req = new Dictionary<string, object>
                {
                    ["command"]     = "classify",
                    ["object_name"] = name,
                    ["vertices"]    = verts,
                    ["normals"]     = norms,
                    ["labels"]      = labels,
                    ["top_k"]       = topK,
                };
                SendMsg(client, req);
                var resp = RecvMsg(client);
                if (resp == null) { PostStatus("Connection lost."); break; }
                if ((string)resp["status"] == "ok")
                    ApplyResult(go, resp);

                _done++;
                PostProgress((float)_done / _total);
            }
        }
        catch (Exception e)
        {
            PostStatus($"Error: {e.Message}");
        }
        finally
        {
            client?.Close();
            PostDone(_cancelFlag ? "Scan cancelled – partial results applied." :
                                   $"Done. {_done}/{_total} objects classified.");
        }
    }

    // ── result application (dispatched to main thread) ────────────────────────

    private void ApplyResult(GameObject go, Dictionary<string, object> resp)
    {
        string primary  = resp["primary_label"]      as string ?? "";
        float  conf     = Convert.ToSingle(resp["primary_confidence"]);
        var    raw      = resp["top_results"] as List<object>;
        string utc      = DateTime.UtcNow.ToString("o");

        var top = new List<MeshClassification.ClassificationResult>();
        if (raw != null)
        {
            foreach (var item in raw)
            {
                if (item is Dictionary<string, object> d)
                {
                    top.Add(new MeshClassification.ClassificationResult
                    {
                        label      = d["label"]      as string ?? "",
                        confidence = Convert.ToSingle(d["confidence"]),
                    });
                }
            }
        }

        Dispatch(() =>
        {
            if (go == null) return;
            var mc = go.GetComponent<MeshClassification>()
                     ?? Undo.AddComponent<MeshClassification>(go);
            mc.ApplyResults(primary, conf, top, utc);
        });
    }

    // ── thread helpers ────────────────────────────────────────────────────────

    private void PostStatus(string s)  => Dispatch(() => { _statusLine = s; Repaint(); });
    private void PostProgress(float p) => Dispatch(() => { _progress = p;   Repaint(); });
    private void PostDone(string msg)  => Dispatch(() =>
    {
        _statusLine = msg;
        _scanning   = false;
        _progress   = 1f;
        Repaint();
        AssetDatabase.SaveAssets();
    });

    private void Dispatch(Action a)
    {
        lock (_qLock) _mainQueue.Enqueue(a);
    }

    // ── TCP protocol (length-prefixed JSON, mirrors python side) ─────────────

    private static readonly Encoding Enc = Encoding.UTF8;

    private static void SendMsg(TcpClient c, Dictionary<string, object> obj)
    {
        string json = MiniJson.Serialize(obj);
        byte[] body = Enc.GetBytes(json);
        byte[] hdr  = BitConverter.GetBytes((uint)body.Length);
        if (BitConverter.IsLittleEndian) Array.Reverse(hdr);   // big-endian
        var s = c.GetStream();
        s.Write(hdr,  0, hdr.Length);
        s.Write(body, 0, body.Length);
    }

    private static Dictionary<string, object> RecvMsg(TcpClient c)
    {
        var s    = c.GetStream();
        var hdr  = ReadExact(s, 4);
        if (hdr == null) return null;
        if (BitConverter.IsLittleEndian) Array.Reverse(hdr);
        uint len  = BitConverter.ToUInt32(hdr, 0);
        var body = ReadExact(s, (int)len);
        if (body == null) return null;
        return MiniJson.Deserialize(Enc.GetString(body)) as Dictionary<string, object>;
    }

    private static byte[] ReadExact(System.IO.Stream s, int n)
    {
        var buf = new byte[n];
        int got = 0;
        while (got < n)
        {
            int r = s.Read(buf, got, n - got);
            if (r == 0) return null;
            got += r;
        }
        return buf;
    }

    // ── utility ───────────────────────────────────────────────────────────────

    private static List<MeshRenderer> FindAllMeshRenderers()
    {
        var all = new List<MeshRenderer>();
#if UNITY_2023_1_OR_NEWER
        all.AddRange(FindObjectsByType<MeshRenderer>(FindObjectsSortMode.None));
#else
        all.AddRange(FindObjectsOfType<MeshRenderer>());
#endif
        return all;
    }

    private List<string> ParseLabels()
    {
        var result = new List<string>();
        foreach (var line in labelsText.Split('\n'))
        {
            var t = line.Trim();
            if (!string.IsNullOrEmpty(t)) result.Add(t);
        }
        return result;
    }
}

// ── minimal JSON serialiser (no external dependency needed in Editor) ─────────

internal static class MiniJson
{
    // Serialise a Dictionary<string,object> / List<object> / primitives to JSON.
    // Only needs to handle the types we actually send.
    public static string Serialize(object obj)
    {
        var sb = new System.Text.StringBuilder();
        Write(obj, sb);
        return sb.ToString();
    }

    private static void Write(object obj, System.Text.StringBuilder sb)
    {
        if (obj == null)        { sb.Append("null"); return; }
        if (obj is bool b)      { sb.Append(b ? "true" : "false"); return; }
        if (obj is string s)    { WriteString(s, sb); return; }
        if (obj is int i)       { sb.Append(i); return; }
        if (obj is float f)     { sb.AppendFormat("{0:G9}", f); return; }
        if (obj is double d)    { sb.AppendFormat("{0:G17}", d); return; }

        if (obj is Dictionary<string, object> dict)
        {
            sb.Append('{');
            bool first = true;
            foreach (var kv in dict)
            {
                if (!first) sb.Append(',');
                WriteString(kv.Key, sb);
                sb.Append(':');
                Write(kv.Value, sb);
                first = false;
            }
            sb.Append('}');
            return;
        }

        if (obj is IEnumerable<object> listO)
        {
            sb.Append('[');
            bool first = true;
            foreach (var item in listO) { if (!first) sb.Append(','); Write(item, sb); first = false; }
            sb.Append(']');
            return;
        }

        if (obj is List<float[]> lof)
        {
            sb.Append('[');
            bool first = true;
            foreach (var arr in lof)
            {
                if (!first) sb.Append(',');
                sb.Append('[');
                for (int k = 0; k < arr.Length; k++) { if (k > 0) sb.Append(','); sb.AppendFormat("{0:G9}", arr[k]); }
                sb.Append(']');
                first = false;
            }
            sb.Append(']');
            return;
        }

        if (obj is List<string> ls)
        {
            sb.Append('[');
            bool first = true;
            foreach (var v in ls) { if (!first) sb.Append(','); WriteString(v, sb); first = false; }
            sb.Append(']');
            return;
        }

        // fallback
        sb.Append(obj.ToString());
    }

    private static void WriteString(string s, System.Text.StringBuilder sb)
    {
        sb.Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"':  sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n");  break;
                case '\r': sb.Append("\\r");  break;
                case '\t': sb.Append("\\t");  break;
                default:   sb.Append(c);      break;
            }
        }
        sb.Append('"');
    }

    // Minimal deserialiser (only what we need from server responses)
    public static object Deserialize(string json)
    {
        int idx = 0;
        return ParseValue(json, ref idx);
    }

    private static object ParseValue(string s, ref int i)
    {
        SkipWs(s, ref i);
        if (i >= s.Length) return null;
        char c = s[i];
        if (c == '{') return ParseObject(s, ref i);
        if (c == '[') return ParseArray(s, ref i);
        if (c == '"') return ParseString(s, ref i);
        if (c == 't') { i += 4; return true; }
        if (c == 'f') { i += 5; return false; }
        if (c == 'n') { i += 4; return null; }
        return ParseNumber(s, ref i);
    }

    private static Dictionary<string, object> ParseObject(string s, ref int i)
    {
        var d = new Dictionary<string, object>();
        i++; // {
        SkipWs(s, ref i);
        while (s[i] != '}')
        {
            string key = ParseString(s, ref i);
            SkipWs(s, ref i); i++; // :
            object val = ParseValue(s, ref i);
            d[key] = val;
            SkipWs(s, ref i);
            if (s[i] == ',') i++;
            SkipWs(s, ref i);
        }
        i++; // }
        return d;
    }

    private static List<object> ParseArray(string s, ref int i)
    {
        var list = new List<object>();
        i++; // [
        SkipWs(s, ref i);
        while (s[i] != ']')
        {
            list.Add(ParseValue(s, ref i));
            SkipWs(s, ref i);
            if (s[i] == ',') i++;
            SkipWs(s, ref i);
        }
        i++; // ]
        return list;
    }

    private static string ParseString(string s, ref int i)
    {
        i++; // opening "
        var sb = new System.Text.StringBuilder();
        while (s[i] != '"')
        {
            if (s[i] == '\\') { i++; sb.Append(s[i]); }
            else sb.Append(s[i]);
            i++;
        }
        i++; // closing "
        return sb.ToString();
    }

    private static object ParseNumber(string s, ref int i)
    {
        int start = i;
        while (i < s.Length && (char.IsDigit(s[i]) || s[i] == '-' || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+'))
            i++;
        string raw = s.Substring(start, i - start);
        if (raw.Contains('.') || raw.Contains('e') || raw.Contains('E'))
            return double.Parse(raw, System.Globalization.CultureInfo.InvariantCulture);
        return long.Parse(raw);
    }

    private static void SkipWs(string s, ref int i)
    {
        while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
    }
}
#endif
