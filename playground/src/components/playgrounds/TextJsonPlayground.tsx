"use client";
import { useState } from "react";
import { Project } from "@/lib/projects";
import JsonOutput from "../JsonOutput";

interface Props {
  project: Project;
  placeholder?: string;
  inputLabel?: string;
  endpoint?: string;
  bodyBuilder?: (input: string) => Record<string, unknown>;
}

export default function TextJsonPlayground({
  project,
  placeholder = "Enter your query...",
  inputLabel = "Query",
  endpoint,
  bodyBuilder,
}: Props) {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    if (!input.trim() || loading) return;
    setLoading(true);
    setError("");
    setResult(null);

    const ep = endpoint || Object.values(project.endpoints || {})[0] || "/";
    const body = bodyBuilder ? bodyBuilder(input) : { query: input, question: input, text: input };

    try {
      const res = await fetch(`http://localhost:${project.port}${ep}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(`Connection failed. Is the backend running on port ${project.port}?`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-3">
      <div>
        <label className="block text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>
          {inputLabel}
        </label>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={placeholder}
          rows={4}
          className="w-full px-4 py-3 rounded-lg border text-sm outline-none transition-colors focus:border-violet-600 resize-none"
          style={{ background: "var(--bg)", borderColor: "var(--border)", color: "var(--text)" }}
        />
      </div>

      <button
        onClick={run}
        disabled={loading || !input.trim()}
        className="px-5 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 bg-violet-700 hover:bg-violet-600 text-white"
      >
        {loading ? "Running..." : "Run"}
      </button>

      {error && (
        <p className="text-sm text-red-400 mt-2">{error}</p>
      )}

      <JsonOutput data={result} />
    </div>
  );
}
