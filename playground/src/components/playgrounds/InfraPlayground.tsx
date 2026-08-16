"use client";
import { useState } from "react";
import { Project } from "@/lib/projects";

export default function InfraPlayground({ project }: { project: Project }) {
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const statusEndpoint = project.endpoints?.status || project.endpoints?.metrics;

  async function checkStatus() {
    if (!statusEndpoint) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`http://localhost:${project.port}${statusEndpoint}`);
      const data = await res.json();
      setStatus(data);
    } catch {
      setError(`Could not connect to port ${project.port}. Start the service first.`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      {project.dashboards && project.dashboards.length > 0 && (
        <div>
          <p className="text-xs mb-3 font-mono" style={{ color: "var(--muted)" }}>Dashboards</p>
          <div className="flex flex-wrap gap-2">
            {project.dashboards.map((d) => (
              <a
                key={d.label}
                href={d.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border text-sm transition-colors hover:border-violet-600 hover:text-violet-300"
                style={{ borderColor: "var(--border)", color: "var(--text)", background: "var(--bg)" }}
              >
                {d.label} →
              </a>
            ))}
          </div>
        </div>
      )}

      {statusEndpoint && (
        <div>
          <p className="text-xs mb-3 font-mono" style={{ color: "var(--muted)" }}>Service status</p>
          <button
            onClick={checkStatus}
            disabled={loading}
            className="px-5 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 bg-violet-700 hover:bg-violet-600 text-white"
          >
            {loading ? "Checking..." : "Check Status"}
          </button>
          {error && <p className="text-sm text-red-400 mt-2">{error}</p>}
          {status && (
            <pre
              className="mt-3 rounded-lg p-4 text-sm font-mono overflow-x-auto text-emerald-300"
              style={{ background: "var(--bg)" }}
            >
              {JSON.stringify(status, null, 2)}
            </pre>
          )}
        </div>
      )}

      <div
        className="rounded-xl border p-4"
        style={{ background: "var(--bg)", borderColor: "var(--border)" }}
      >
        <p className="text-xs mb-3 font-mono" style={{ color: "var(--muted)" }}>Run locally</p>
        <pre className="text-xs font-mono text-slate-300 overflow-x-auto leading-relaxed">
          {`cd ai-projects/${project.id.toString().padStart(2, "0")}-${project.slug}\ndocker compose up`}
        </pre>
      </div>
    </div>
  );
}
