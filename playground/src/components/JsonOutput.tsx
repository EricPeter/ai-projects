"use client";

interface Props {
  data: unknown;
  label?: string;
}

export default function JsonOutput({ data, label = "Response" }: Props) {
  if (!data) return null;
  return (
    <div className="mt-4">
      <p className="text-xs mb-2 font-mono" style={{ color: "var(--muted)" }}>
        {label}
      </p>
      <pre
        className="rounded-lg p-4 text-sm font-mono overflow-x-auto text-emerald-300 leading-relaxed"
        style={{ background: "var(--bg)" }}
      >
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}
