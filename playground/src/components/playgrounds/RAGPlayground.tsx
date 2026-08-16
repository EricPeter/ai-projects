"use client";
import { useState } from "react";
import { Project } from "@/lib/projects";

interface RAGResult {
  answer: string;
  citations: number[];
  documents?: { content: string; page: number }[];
}

export default function RAGPlayground({ project }: { project: Project }) {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<RAGResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [error, setError] = useState("");

  async function uploadPDF(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`http://localhost:${project.port}/ingest`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      setUploadMsg(data.message || `Ingested: ${file.name}`);
    } catch {
      setUploadMsg("Upload failed. Is the backend running?");
    } finally {
      setUploading(false);
    }
  }

  async function query() {
    if (!question.trim() || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`http://localhost:${project.port}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setResult(data);
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
          1. Upload a PDF to ingest
        </label>
        <div
          className="border rounded-lg p-4 flex items-center gap-3"
          style={{ borderColor: "var(--border)", background: "var(--bg)" }}
        >
          <label className="cursor-pointer px-4 py-2 rounded-lg text-sm font-medium bg-violet-700 hover:bg-violet-600 text-white transition-colors">
            {uploading ? "Uploading..." : "Choose PDF"}
            <input type="file" accept=".pdf" className="hidden" onChange={uploadPDF} disabled={uploading} />
          </label>
          {uploadMsg && <p className="text-sm text-emerald-400">{uploadMsg}</p>}
        </div>
      </div>

      <div>
        <label className="block text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>
          2. Ask a question
        </label>
        <div className="flex gap-2">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && query()}
            placeholder="What does the document say about..."
            className="flex-1 px-4 py-2.5 rounded-lg border text-sm outline-none transition-colors focus:border-violet-600"
            style={{ background: "var(--bg)", borderColor: "var(--border)", color: "var(--text)" }}
          />
          <button
            onClick={query}
            disabled={loading || !question.trim()}
            className="px-4 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 bg-violet-700 hover:bg-violet-600 text-white"
          >
            {loading ? "Searching..." : "Ask"}
          </button>
        </div>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {result && (
        <div className="space-y-3">
          <div
            className="rounded-xl border p-4"
            style={{ background: "var(--bg)", borderColor: "var(--border)" }}
          >
            <p className="text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>Answer</p>
            <p className="text-sm leading-relaxed text-slate-200">{result.answer}</p>
            {result.citations?.length > 0 && (
              <p className="text-xs mt-2" style={{ color: "var(--muted)" }}>
                Pages cited: {result.citations.join(", ")}
              </p>
            )}
          </div>
          {result.documents && result.documents.length > 0 && (
            <div>
              <p className="text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>
                Retrieved chunks
              </p>
              <div className="space-y-2">
                {result.documents.slice(0, 3).map((d, i) => (
                  <div
                    key={i}
                    className="rounded-lg border p-3"
                    style={{ background: "var(--bg)", borderColor: "var(--border)" }}
                  >
                    <p className="text-xs mb-1 font-mono text-violet-400">Page {d.page}</p>
                    <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
                      {d.content.slice(0, 200)}...
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
