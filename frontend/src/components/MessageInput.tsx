import { useState } from "react";
import type { KeyboardEvent } from "react";

interface MessageInputProps {
  onSubmit: (message: string) => void;
  loading?: boolean;
}

const MAX_CHARS = 1000;
const WARN_THRESHOLD = 900;

export default function MessageInput({ onSubmit, loading = false }: MessageInputProps) {
  const [message, setMessage] = useState("");

  const trimmed = message.slice(0, MAX_CHARS);
  const count = trimmed.length;
  const isOverWarning = count >= WARN_THRESHOLD;
  const canSubmit = count > 0 && !loading;

  function handleChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setMessage(e.target.value.slice(0, MAX_CHARS));
  }

  function handleSubmit() {
    if (!canSubmit) return;
    onSubmit(trimmed);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <textarea
        value={trimmed}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        disabled={loading}
        placeholder="Describe your banking issue..."
        rows={4}
        className="w-full resize-none rounded-lg border border-gray-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      />
      <div className="flex items-center justify-between">
        <span
          className={`text-xs ${isOverWarning ? "text-amber-600 font-semibold" : "text-gray-400"}`}
        >
          {count} / {MAX_CHARS}
        </span>
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Classifying…" : "Classify"}
        </button>
      </div>
    </div>
  );
}
