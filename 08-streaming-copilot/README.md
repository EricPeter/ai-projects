# Streaming Copilot UI

Real-time token streaming chat UI built with Next.js 14 App Router and Vercel AI SDK.

## Features
- Token-by-token streaming with animated cursor
- Optimistic UI — user messages appear instantly
- Animated loading indicator (bouncing dots) while streaming
- Error recovery with inline Retry button
- Dark theme, clean minimal design

## Setup

```bash
cd 08-streaming-copilot
cp .env.local.example .env.local   # add ANTHROPIC_API_KEY
npm install
npm run dev
```

Open http://localhost:3000.

## Architecture

```
Browser → useChat() hook (Vercel AI SDK)
        → POST /api/chat
        → streamText(claude-sonnet-4-6)
        → data stream response → tokens appear in real time
```

The API route runs on the Edge runtime for minimal cold-start latency.
