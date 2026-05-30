using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Attached automatically by OpenShapeScanner to every GameObject whose
/// MeshRenderer was classified.  Stores the classification results so they
/// survive domain reloads and can be used at runtime (e.g. for spawn logic,
/// LOD, audio, shader keywords, etc.).
/// </summary>
[DisallowMultipleComponent]
public class MeshClassification : MonoBehaviour
{
    // ── serialised data ───────────────────────────────────────────────────────

    [Serializable]
    public class ClassificationResult
    {
        public string label;
        [Range(0f, 1f)] public float confidence;
    }

    [Header("Primary classification")]
    public string primaryLabel      = "";
    public float  primaryConfidence = 0f;

    [Header("All top results")]
    public List<ClassificationResult> topResults = new();

    [Header("Meta")]
    public string lastClassifiedUtc = "";   // ISO-8601 timestamp set by scanner

    // ── custom user tags ──────────────────────────────────────────────────────

    [Header("User overrides / extra tags")]
    [Tooltip("Add any number of custom category tags here. " +
             "These are never overwritten by the scanner.")]
    public List<string> customTags = new();

    // ── convenience API ───────────────────────────────────────────────────────

    /// <summary>Returns true when <paramref name="label"/> appears in
    /// topResults or customTags (case-insensitive).</summary>
    public bool HasLabel(string label)
    {
        if (string.IsNullOrEmpty(label)) return false;
        foreach (var r in topResults)
            if (string.Equals(r.label, label, StringComparison.OrdinalIgnoreCase))
                return true;
        foreach (var t in customTags)
            if (string.Equals(t, label, StringComparison.OrdinalIgnoreCase))
                return true;
        return false;
    }

    /// <summary>Finds all active GameObjects in the scene that carry a
    /// specific classification label.</summary>
    public static List<GameObject> FindAllWithLabel(string label)
    {
        var result = new List<GameObject>();
        foreach (var mc in FindObjectsByType<MeshClassification>(FindObjectsSortMode.None))
            if (mc.HasLabel(label))
                result.Add(mc.gameObject);
        return result;
    }

    // ── editor helpers ────────────────────────────────────────────────────────

#if UNITY_EDITOR
    /// <summary>Called by the scanner to set results and mark the asset dirty.</summary>
    internal void ApplyResults(string primary, float confidence,
                               List<ClassificationResult> top, string utc)
    {
        primaryLabel      = primary;
        primaryConfidence = confidence;
        topResults        = top;
        lastClassifiedUtc = utc;
        UnityEditor.EditorUtility.SetDirty(this);
    }
#endif
}
