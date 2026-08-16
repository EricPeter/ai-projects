"use client";
import { useState } from "react";
import { Project } from "@/lib/projects";

interface EvalResult {
  score?: number;
  metrics?: Record<string, number>;
  results?: { name: string; score: number; passed: boolean }[];
  leaderboard?: { model: string; score: number }[];
}

export default function EvalPlayground({ project }: { project: Project }) {
  const [result, setResult] = useState<EvalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const isLeaderboard = project.id === 17;

  async function run() {
    setLoading(true);
    setError("");
    setResult(null);
    const ep = isLeaderboard ? "/leaderboard" : "/evaluate";
    try {
      const res = await fetch(`http://localhost:${project.port}${ep}`, {
        method: isLeaderboard ? "GET" : "POST",
        headers: isLeaderboard ? undefined : { "Content-Type": "application/json" },
        body: isLeaderboard ? undefined : JSON.stringify({}),
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
    <div className="space-y-4">
      <p className="text-sm" style={{ color: "var(--muted)" }}>
        {isLeaderboard
          ? "Run the benchmark against all configured models and view the leaderboard."
          : "Run the full evaluation suite against the golden dataset and see metric scores."}
      </p>

      <button
        onClick={run}
        disabled={loading}
        className="px-5 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 bg-violet-700 hover:bg-violet-600 text-white"
      >
        {loading ? "Running..." : isLeaderboard ? "Load Leaderboard" : "Run Evals"}
      </button>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {result && (
        <div
          className="rounded-xl border p-4"
          style={{ background: "var(--bg)", borderColor: "var(--border)" }}
        >
          {result.leaderboard ? (
            <div>
              <p className="text-xs font-mono mb-3" style={{ color: "var(--muted)" }}>Leaderboard</p>
              <div className="space-y-2">
                {result.leaderboard.map((row, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <span className="text-sm text-slate-200">{row.model}</span>
                    <div className="flex items-center gap-3">
                      <div className="w-32 h-1.5 rounded-full bg-slate-800">
                        <div
                          className="h-full rounded-full bg-violet-500"
                          style={{ width: `${Math.min(100, row.score * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono text-violet-300 w-12 text-right">
                        {(row.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : result.results ? (
            <div>
              <p className="text-xs font-mono mb-3" style={{ color: "var(--muted)" }}>Metrics</p>
              <div className="space-y-2">
                {result.results.map((r, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <span className="text-sm text-slate-200">{r.name}</span>
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-xs px-2 py-0.5 rounded font-mono ${
                          r.passed ? "bg-emerald-900/40 text-emerald-300" : "bg-red-900/40 text-red-300"
                        }`}
                      >
                        {r.passed ? "PASS" : "FAIL"}
                      </span>
                      <span className="text-xs font-mono text-slate-400 w-12 text-right">
                        {(r.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <pre className="text-sm font-mono text-emerald-300 overflow-x-auto">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
