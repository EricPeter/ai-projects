"use client";
import { useState } from "react";
import { Project } from "@/lib/projects";

interface ProbeResult {
  probe_id: string;
  category: string;
  response: string;
  latency_ms: number;
  keyword_score: number;
}

interface DriftAlert {
  metric: string;
  baseline: number;
  current: number;
  severity: "warning" | "critical";
}

interface DashboardProbe {
  probe_id: string;
  category: string;
  status: "healthy" | "warning" | "critical";
  stats: { latency_p50_ms?: number; latency_p95_ms?: number; count?: number; avg_tokens?: number };
  recent_alerts: DriftAlert[];
}

const STATUS_COLORS = {
  healthy: "bg-emerald-900/40 text-emerald-300 border-emerald-800",
  warning: "bg-yellow-900/40 text-yellow-300 border-yellow-800",
  critical: "bg-red-900/40 text-red-300 border-red-800",
};

export default function DriftPlayground({ project }: { project: Project }) {
  const [probeResults, setProbeResults] = useState<ProbeResult[]>([]);
  const [alerts, setAlerts] = useState<DriftAlert[]>([]);
  const [dashboard, setDashboard] = useState<DashboardProbe[]>([]);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runProbes() {
    setRunning(true);
    setError("");
    try {
      const res = await fetch(`http://localhost:${project.port}/probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      setProbeResults(data.results || []);
      setAlerts(data.drift_alerts || []);
    } catch {
      setError(`Connection failed. Is the backend running on port ${project.port}?`);
    } finally {
      setRunning(false);
    }
  }

  async function loadDashboard() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`http://localhost:${project.port}/dashboard`);
      const data = await res.json();
      setDashboard(data.probes || []);
    } catch {
      setError(`Connection failed. Is the backend running on port ${project.port}?`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex gap-2">
        <button
          onClick={runProbes}
          disabled={running}
          className="px-5 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 bg-violet-700 hover:bg-violet-600 text-white"
        >
          {running ? "Running probes..." : "Run Probes Now"}
        </button>
        <button
          onClick={loadDashboard}
          disabled={loading}
          className="px-5 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 border hover:border-violet-600 hover:text-violet-300"
          style={{ borderColor: "var(--border)", color: "var(--text)" }}
        >
          {loading ? "Loading..." : "Load Dashboard"}
        </button>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {alerts.length > 0 && (
        <div>
          <p className="text-xs font-mono mb-2" style={{ color: "var(--muted)" }}>
            Drift alerts ({alerts.length})
          </p>
          <div className="space-y-2">
            {alerts.map((a, i) => (
              <div
                key={i}
                className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm ${STATUS_COLORS[a.severity]}`}
              >
                <span className="font-mono font-medium uppercase text-xs">{a.severity}</span>
                <span>{a.metric.replace(/_/g, " ")}</span>
                <span className="font-mono text-xs">
                  {a.baseline.toFixed(3)} → {a.current.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {probeResults.length > 0 && (
        <div>
          <p className="text-xs font-mono mb-2" style={{ color: "var(--muted)" }}>
            Probe results
          </p>
          <div className="space-y-2">
            {probeResults.map((r, i) => (
              <div
                key={i}
                className="rounded-lg border p-3"
                style={{ background: "var(--bg)", borderColor: "var(--border)" }}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-mono text-violet-400">{r.probe_id}</span>
                  <div className="flex gap-3 text-xs font-mono" style={{ color: "var(--muted)" }}>
                    <span>{r.latency_ms.toFixed(0)}ms</span>
                    <span>kw: {(r.keyword_score * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <p className="text-xs leading-relaxed" style={{ color: "var(--text)" }}>
                  {r.response.slice(0, 150)}{r.response.length > 150 ? "..." : ""}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {dashboard.length > 0 && (
        <div>
          <p className="text-xs font-mono mb-2" style={{ color: "var(--muted)" }}>
            Probe health
          </p>
          <div className="grid grid-cols-2 gap-2">
            {dashboard.map((p) => (
              <div
                key={p.probe_id}
                className="rounded-lg border p-3"
                style={{ background: "var(--bg)", borderColor: "var(--border)" }}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-mono text-slate-200">{p.probe_id}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded border ${STATUS_COLORS[p.status]}`}>
                    {p.status}
                  </span>
                </div>
                <div className="text-xs font-mono space-y-0.5" style={{ color: "var(--muted)" }}>
                  {p.stats.count !== undefined && <div>samples: {p.stats.count}</div>}
                  {p.stats.latency_p50_ms !== undefined && (
                    <div>p50: {p.stats.latency_p50_ms.toFixed(0)}ms · p95: {p.stats.latency_p95_ms?.toFixed(0)}ms</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
