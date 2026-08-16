import { anthropic } from "@ai-sdk/anthropic";
import { streamText } from "ai";

export const runtime = "edge";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = await streamText({
    model: anthropic("claude-sonnet-4-6"),
    system: "You are a helpful, concise AI assistant.",
    messages,
  });

  return result.toDataStreamResponse();
}
