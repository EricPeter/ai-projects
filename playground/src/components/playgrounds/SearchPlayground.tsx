"use client";
import { useState } from "react";
import { Project } from "@/lib/projects";

interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export default function SearchPlayground({ project }: { project: Project }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sampleText, setSampleText] = useState("");
  const [upserting, setUpserting] = useState(false);
  const [upsertMsg, setUpsertMsg] = useState("");

  async function upsert() {
    if (!sampleText.trim()) return;
    setUpserting(true);
    setUpsertMsg("");
    try {
      const res = await fetch(`http://localhost:${project.port}/upsert`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: [sampleText] }),
      });
      const data = await res.json();
      setUpsertMsg(data.message || "Upserted successfully");
      setSampleText("");
    } catch {
      setUpsertMsg("Failed. Is the backend running?");
    } finally {
      setUpserting(false);
    }
  }

  async function search() {
    if (!query.trim() || loading) return;
    setLoading(true);
    setError("");
    setResults([]);
    try {
      const res = await fetch(`http://localhost:${project.port}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 5 }),
      });
      const data = await res.json();
      setResults(data.results || data);
    } catch {
      setError(`Connection failed. Is the backend running on port ${project.port}?`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <label className="block text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>
          Add a document
        </label>
        <div className="flex gap-2">
          <input
            value={sampleText}
            onChange={(e) => setSampleText(e.target.value)}
            placeholder="Any text to add to the vector store..."
            className="flex-1 px-4 py-2.5 rounded-lg border text-sm outline-none focus:border-violet-600"
            style={{ background: "var(--bg)", borderColor: "var(--border)", color: "var(--text)" }}
          />
          <button
            onClick={upsert}
            disabled={upserting || !sampleText.trim()}
            className="px-4 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 bg-slate-700 hover:bg-slate-600 text-white"
          >
            {upserting ? "Adding..." : "Add"}
          </button>
        </div>
        {upsertMsg && <p className="text-xs mt-1.5 text-emerald-400">{upsertMsg}</p>}
      </div>

      <div>
        <label className="block text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>
          Semantic search
        </label>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && search()}
            placeholder="Search the vector store..."
            className="flex-1 px-4 py-2.5 rounded-lg border text-sm outline-none focus:border-violet-600"
            style={{ background: "var(--bg)", borderColor: "var(--border)", color: "var(--text)" }}
          />
          <button
            onClick={search}
            disabled={loading || !query.trim()}
            className="px-4 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 bg-violet-700 hover:bg-violet-600 text-white"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {results.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-mono" style={{ color: "var(--muted)" }}>{results.length} results</p>
          {results.map((r, i) => (
            <div
              key={i}
              className="rounded-lg border p-3"
              style={{ background: "var(--bg)", borderColor: "var(--border)" }}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-mono text-violet-400">#{i + 1}</span>
                <span className="text-xs font-mono" style={{ color: "var(--muted)" }}>
                  score: {typeof r.score === "number" ? r.score.toFixed(4) : r.score}
                </span>
              </div>
              <p className="text-sm" style={{ color: "var(--text)" }}>{r.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
