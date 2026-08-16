import { notFound } from "next/navigation";
import { PROJECTS } from "@/lib/projects";
import BackLink from "@/components/BackLink";
import RAGPlayground from "@/components/playgrounds/RAGPlayground";
import ChatPlayground from "@/components/playgrounds/ChatPlayground";
import TextJsonPlayground from "@/components/playgrounds/TextJsonPlayground";
import TranslatePlayground from "@/components/playgrounds/TranslatePlayground";
import SearchPlayground from "@/components/playgrounds/SearchPlayground";
import EvalPlayground from "@/components/playgrounds/EvalPlayground";
import InfraPlayground from "@/components/playgrounds/InfraPlayground";
import DriftPlayground from "@/components/playgrounds/DriftPlayground";

const HEALTH_BOT_LANGUAGES = [
  { code: "eng", name: "English" },
  { code: "lug", name: "Luganda" },
  { code: "ach", name: "Acholi" },
];

const ROUTER_BODY = (input: string) => ({ query: input });
const GUARDRAILS_BODY = (input: string) => ({ text: input });
const RESEARCH_BODY = (input: string) => ({ topic: input });
const HITL_BODY = (input: string) => ({ action: input, action_type: "general" });
const TENANT_BODY = (input: string) => ({
  prompt: input,
  tenant_id: "demo-tenant",
  token: "demo-token",
});

export async function generateStaticParams() {
  return PROJECTS.map((p) => ({ id: String(p.id) }));
}

export default function ProjectPage({ params }: { params: { id: string } }) {
  const project = PROJECTS.find((p) => p.id === Number(params.id));
  if (!project) notFound();

  function renderPlayground() {
    switch (project!.playgroundType) {
      case "rag":
        return <RAGPlayground project={project!} />;
      case "translate":
        return <TranslatePlayground project={project!} />;
      case "search":
        return <SearchPlayground project={project!} />;
      case "eval":
        if (project!.id === 20) return <DriftPlayground project={project!} />;
        return <EvalPlayground project={project!} />;
      case "infra":
        return <InfraPlayground project={project!} />;
      case "chat":
        if (project!.id === 19) {
          return <ChatPlayground project={project!} languages={HEALTH_BOT_LANGUAGES} />;
        }
        return <ChatPlayground project={project!} />;
      case "text-in-json-out": {
        const configs: Record<number, { placeholder: string; inputLabel: string; endpoint: string; bodyBuilder: (s: string) => Record<string, unknown> }> = {
          2: {
            placeholder: "Write a complex analysis of transformer attention mechanisms...",
            inputLabel: "Query to route",
            endpoint: "/route",
            bodyBuilder: ROUTER_BODY,
          },
          3: {
            placeholder: "Quantum computing applications in drug discovery",
            inputLabel: "Research topic",
            endpoint: "/research",
            bodyBuilder: RESEARCH_BODY,
          },
          6: {
            placeholder: "Hi, my name is John Doe, call me at 0772-123-456...",
            inputLabel: "Text to analyze",
            endpoint: "/check",
            bodyBuilder: GUARDRAILS_BODY,
          },
          10: {
            placeholder: "Explain the key benefits of RAG for enterprise AI.",
            inputLabel: "Prompt",
            endpoint: "/api/v1/complete",
            bodyBuilder: TENANT_BODY,
          },
          15: {
            placeholder: "Delete all user records from the production database",
            inputLabel: "Action to evaluate",
            endpoint: "/submit",
            bodyBuilder: HITL_BODY,
          },
        };
        const cfg = configs[project!.id] || {
          placeholder: "Enter input...",
          inputLabel: "Input",
          endpoint: Object.values(project!.endpoints || {})[0] || "/",
          bodyBuilder: (s: string) => ({ query: s }),
        };
        return (
          <TextJsonPlayground
            project={project!}
            placeholder={cfg.placeholder}
            inputLabel={cfg.inputLabel}
            endpoint={cfg.endpoint}
            bodyBuilder={cfg.bodyBuilder}
          />
        );
      }
    }
  }

  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      <header className="border-b" style={{ borderColor: "var(--border)" }}>
        <div className="max-w-4xl mx-auto px-6 py-4">
          <BackLink />
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span
              className="text-xs font-mono px-2 py-0.5 rounded"
              style={{ color: "var(--muted)", background: "var(--surface)" }}
            >
              {String(project.id).padStart(2, "0")}
            </span>
            <h1 className="text-2xl font-bold text-white">{project.name}</h1>
          </div>
          <p className="text-sm mb-4 leading-relaxed" style={{ color: "var(--muted)", maxWidth: "560px" }}>
            {project.description}
          </p>
          <div className="flex flex-wrap gap-2">
            {project.tech.map((t) => (
              <span
                key={t}
                className="text-xs px-2.5 py-1 rounded font-mono border"
                style={{ background: "var(--bg)", color: "var(--muted)", borderColor: "var(--border)" }}
              >
                {t}
              </span>
            ))}
          </div>
        </div>

        <div
          className="rounded-2xl border p-6"
          style={{ background: "var(--surface)", borderColor: "var(--border)" }}
        >
          <p className="text-xs font-mono mb-5" style={{ color: "var(--muted)" }}>
            Playground · port {project.port}
          </p>
          {renderPlayground()}
        </div>

        {project.endpoints && Object.keys(project.endpoints).length > 0 && (
          <div className="mt-6">
            <p className="text-xs font-mono mb-3" style={{ color: "var(--muted)" }}>API endpoints</p>
            <div
              className="rounded-xl border p-4 space-y-2"
              style={{ background: "var(--surface)", borderColor: "var(--border)" }}
            >
              {Object.entries(project.endpoints).map(([name, path]) => (
                <div key={name} className="flex items-center gap-3">
                  <span className="text-xs font-mono text-violet-400 uppercase w-20">{name}</span>
                  <code
                    className="text-xs font-mono"
                    style={{ color: "var(--text)" }}
                  >
                    POST http://localhost:{project.port}{path}
                  </code>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
