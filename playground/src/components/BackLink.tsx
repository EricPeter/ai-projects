"use client";
import Link from "next/link";

export default function BackLink() {
  return (
    <Link
      href="/"
      className="inline-flex items-center gap-1.5 text-sm transition-colors hover:text-white"
      style={{ color: "var(--muted)" }}
    >
      ← All projects
    </Link>
  );
}
