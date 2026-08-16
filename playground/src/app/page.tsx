import Link from "next/link";
import { PROJECTS } from "@/lib/projects";

const TYPE_COLORS: Record<string, string> = {
  chat: "bg-blue-900/40 text-blue-300 border-blue-800",
  "text-in-json-out": "bg-violet-900/40 text-violet-300 border-violet-800",
  rag: "bg-emerald-900/40 text-emerald-300 border-emerald-800",
  translate: "bg-yellow-900/40 text-yellow-300 border-yellow-800",
  search: "bg-cyan-900/40 text-cyan-300 border-cyan-800",
  eval: "bg-orange-900/40 text-orange-300 border-orange-800",
  infra: "bg-slate-900/40 text-slate-300 border-slate-700",
};

const TYPE_LABELS: Record<string, string> = {
  chat: "Chat",
  "text-in-json-out": "API",
  rag: "RAG",
  translate: "Translate",
  search: "Search",
  eval: "Eval",
  infra: "Infrastructure",
};

export default function Home() {
  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      <header className="border-b" style={{ borderColor: "var(--border)" }}>
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                AI Projects Playground
              </h1>
              <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
                19 production-grade AI engineering projects · Interactive demos
              </p>
            </div>
            <a
              href="https://github.com/EricPeter/ai-projects"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm px-4 py-2 rounded-lg border transition-colors hover:border-violet-600 hover:text-violet-300"
              style={{ borderColor: "var(--border)", color: "var(--muted)" }}
            >
              GitHub →
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-10">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {PROJECTS.map((project) => (
            <Link
              key={project.id}
              href={`/projects/${project.id}`}
              className="group block rounded-xl border p-5 transition-all duration-150 hover:border-violet-700 hover:shadow-lg hover:shadow-violet-900/20"
              style={{ background: "var(--surface)", borderColor: "var(--border)" }}
            >
              <div className="flex items-start justify-between mb-3">
                <span
                  className="text-xs font-mono font-medium px-2 py-0.5 rounded"
                  style={{ color: "var(--muted)", background: "var(--bg)" }}
                >
                  {String(project.id).padStart(2, "0")}
                </span>
                <span
                  className={`text-xs px-2 py-0.5 rounded border ${TYPE_COLORS[project.playgroundType]}`}
                >
                  {TYPE_LABELS[project.playgroundType]}
                </span>
              </div>

              <h2 className="font-semibold text-white mb-2 group-hover:text-violet-300 transition-colors">
                {project.name}
              </h2>
              <p className="text-sm leading-relaxed mb-4" style={{ color: "var(--muted)" }}>
                {project.description}
              </p>

              <div className="flex flex-wrap gap-1.5">
                {project.tech.slice(0, 3).map((t) => (
                  <span
                    key={t}
                    className="text-xs px-2 py-0.5 rounded font-mono"
                    style={{ background: "var(--bg)", color: "var(--muted)" }}
                  >
                    {t}
                  </span>
                ))}
                {project.tech.length > 3 && (
                  <span
                    className="text-xs px-2 py-0.5 rounded font-mono"
                    style={{ background: "var(--bg)", color: "var(--muted)" }}
                  >
                    +{project.tech.length - 3}
                  </span>
                )}
              </div>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
