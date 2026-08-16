import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Projects Playground",
  description: "Interactive playground for 19 production-grade AI engineering projects",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
