import { useState, useEffect } from "react";
import MessageInput from "./components/MessageInput";
import TicketCard from "./components/TicketCard";
import { getHealth, classify, ApiError, NetworkError } from "./lib/api";
import type { ClassifyResponse, HealthResponse } from "./lib/api";

function ErrorBanner({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      {message}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err) => {
        const msg =
          err instanceof ApiError
            ? `Backend error: ${err.message}`
            : err instanceof NetworkError
              ? "Cannot reach backend. Is the server running?"
              : "Health check failed.";
        setError(msg);
      });
  }, []);

  async function handleSubmit(message: string) {
    setLoading(true);
    setError(null);
    try {
      const data = await classify(message);
      setResult(data);
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? `${err.code}: ${err.message}`
          : err instanceof NetworkError
            ? "Network error. Is the server running?"
            : "Classification failed.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-10 px-4">
      <div className="mx-auto max-w-2xl flex flex-col gap-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-gray-900">BERT Ticket Router</h1>
          {health && (
            <p className="mt-1 text-xs text-gray-400">
              Model: {health.model_version} · {health.device} · {health.num_labels} intents
            </p>
          )}
        </div>

        {/* Error banner */}
        {error && <ErrorBanner message={error} />}

        {/* Input */}
        <MessageInput onSubmit={handleSubmit} loading={loading} />

        {/* Result */}
        {result && <TicketCard result={result} />}
      </div>
    </div>
  );
}
