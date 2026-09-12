"use client";

import { useChat } from "ai/react";
import { useEffect, useRef } from "react";
import { Message } from "./Message";
import { StreamingIndicator } from "./StreamingIndicator";

export function ChatInterface() {
  const { messages, input, handleInputChange, handleSubmit, isLoading, error, reload } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-800">
        <h1 className="text-lg font-semibold text-gray-100">AI Copilot</h1>
        <p className="text-xs text-gray-500">Powered by Claude · Real-time streaming</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <p className="text-center text-gray-600 mt-16 text-sm">Start a conversation below.</p>
        )}
        {messages.map((m) => (
          <Message key={m.id} role={m.role as "user" | "assistant"} content={m.content} />
        ))}
        {isLoading && messages[messages.length - 1]?.role === "user" && <StreamingIndicator />}

        {error && (
          <div className="flex justify-center my-4">
            <div className="bg-red-900/30 border border-red-700 rounded-lg px-4 py-3 text-sm text-red-300 flex items-center gap-3">
              <span>Something went wrong.</span>
              <button onClick={() => reload()} className="underline hover:no-underline">Retry</button>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="px-6 py-4 border-t border-gray-800">
        <div className="flex gap-3">
          <input
            value={input}
            onChange={handleInputChange}
            placeholder="Type a message… (Enter to send)"
            disabled={isLoading}
            className="flex-1 bg-gray-800 text-gray-100 placeholder-gray-500 rounded-xl px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-blue-600 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white px-5 py-3 rounded-xl text-sm font-medium transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
