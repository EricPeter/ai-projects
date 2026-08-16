"use client";
import { useState, useEffect } from "react";
import { Project } from "@/lib/projects";

const UGANDAN_LANGUAGES = [
  { code: "lug", name: "Luganda" },
  { code: "ach", name: "Acholi" },
  { code: "teo", name: "Ateso" },
  { code: "lgg", name: "Lugbara" },
  { code: "nyn", name: "Runyankole" },
  { code: "eng", name: "English" },
];

export default function TranslatePlayground({ project }: { project: Project }) {
  const [text, setText] = useState("");
  const [sourceLang, setSourceLang] = useState("eng");
  const [targetLang, setTargetLang] = useState("lug");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [langs, setLangs] = useState(UGANDAN_LANGUAGES);

  useEffect(() => {
    fetch(`http://localhost:${project.port}/languages`)
      .then((r) => r.json())
      .then((d) => {
        if (Array.isArray(d.languages)) setLangs(d.languages);
      })
      .catch(() => {});
  }, [project.port]);

  async function translate() {
    if (!text.trim() || loading) return;
    setLoading(true);
    setError("");
    setResult("");
    try {
      const res = await fetch(`http://localhost:${project.port}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, source_language: sourceLang, target_language: targetLang }),
      });
      const data = await res.json();
      setResult(data.translation || data.translated_text || JSON.stringify(data));
    } catch {
      setError(`Connection failed. Is the backend running on port ${project.port}?`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>From</label>
          <select
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
            className="w-full px-3 py-2 rounded-lg border text-sm outline-none"
            style={{ background: "var(--bg)", borderColor: "var(--border)", color: "var(--text)" }}
          >
            {langs.map((l) => (
              <option key={l.code} value={l.code}>{l.name}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>To</label>
          <select
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            className="w-full px-3 py-2 rounded-lg border text-sm outline-none"
            style={{ background: "var(--bg)", borderColor: "var(--border)", color: "var(--text)" }}
          >
            {langs.map((l) => (
              <option key={l.code} value={l.code}>{l.name}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>Source text</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to translate..."
            rows={6}
            className="w-full px-4 py-3 rounded-lg border text-sm outline-none resize-none focus:border-violet-600"
            style={{ background: "var(--bg)", borderColor: "var(--border)", color: "var(--text)" }}
          />
        </div>
        <div>
          <label className="block text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>Translation</label>
          <div
            className="w-full h-[148px] px-4 py-3 rounded-lg border text-sm overflow-y-auto"
            style={{ background: "var(--bg)", borderColor: "var(--border)", color: result ? "var(--text)" : "var(--muted)" }}
          >
            {loading ? (
              <span className="animate-pulse">Translating...</span>
            ) : result || "Translation will appear here..."}
          </div>
        </div>
      </div>

      <button
        onClick={translate}
        disabled={loading || !text.trim()}
        className="px-5 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 bg-violet-700 hover:bg-violet-600 text-white"
      >
        {loading ? "Translating..." : "Translate"}
      </button>

      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
}
