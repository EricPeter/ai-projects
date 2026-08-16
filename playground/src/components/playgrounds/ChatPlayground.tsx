"use client";
import { useState, useRef, useEffect } from "react";
import { Project } from "@/lib/projects";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface Props {
  project: Project;
  languages?: { code: string; name: string }[];
}

export default function ChatPlayground({ project, languages }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState("eng");
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => Math.random().toString(36).slice(2));
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);

    try {
      const endpoint = project.endpoints?.chat || project.endpoints?.respond || "/chat";
      const body: Record<string, unknown> = { message: userMsg, session_id: sessionId };
      if (languages) {
        body.language_code = language;
        body.question = userMsg;
        delete body.message;
      }

      const res = await fetch(`http://localhost:${project.port}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      const reply = data.response || data.message || data.answer || JSON.stringify(data);
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${e instanceof Error ? e.message : "Connection failed. Is the backend running on port ${project.port}?"}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-[520px]">
      {languages && (
        <div className="mb-3 flex items-center gap-2">
          <label className="text-xs" style={{ color: "var(--muted)" }}>Language:</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="text-sm px-3 py-1.5 rounded-lg border outline-none"
            style={{ background: "var(--bg)", borderColor: "var(--border)", color: "var(--text)" }}
          >
            {languages.map((l) => (
              <option key={l.code} value={l.code}>{l.name}</option>
            ))}
          </select>
        </div>
      )}

      <div
        className="flex-1 overflow-y-auto rounded-xl border p-4 mb-3 space-y-3"
        style={{ background: "var(--bg)", borderColor: "var(--border)" }}
      >
        {messages.length === 0 && (
          <p className="text-center text-sm mt-16" style={{ color: "var(--muted)" }}>
            Start a conversation...
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-violet-700 text-white"
                  : "border text-slate-200"
              }`}
              style={m.role === "assistant" ? { background: "var(--surface)", borderColor: "var(--border)" } : {}}
            >
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div
              className="rounded-xl px-4 py-2.5 text-sm border"
              style={{ background: "var(--surface)", borderColor: "var(--border)", color: "var(--muted)" }}
            >
              <span className="animate-pulse">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
          placeholder="Type a message..."
          className="flex-1 px-4 py-2.5 rounded-lg border text-sm outline-none transition-colors focus:border-violet-600"
          style={{ background: "var(--bg)", borderColor: "var(--border)", color: "var(--text)" }}
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          className="px-4 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 bg-violet-700 hover:bg-violet-600 text-white"
        >
          Send
        </button>
      </div>
    </div>
  );
}
