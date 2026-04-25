/**
 * Typed API client for the BERT Ticket Router backend.
 *
 * Wraps fetch with:
 * - Base URL from VITE_API_BASE_URL env var (default: http://localhost:8000)
 * - ApiError for non-2xx responses (carries status, code, message, correlationId)
 * - NetworkError for fetch-level failures (no network, DNS failure, etc.)
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Error classes
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly correlationId: string,
  ) {
    super(message);
    this.name = "ApiError";
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NetworkError";
    Object.setPrototypeOf(this, NetworkError.prototype);
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, init);
  } catch (err) {
    throw new NetworkError(err instanceof Error ? err.message : String(err));
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = (body as { error?: { code?: string; message?: string; correlation_id?: string } }).error ?? {};
    throw new ApiError(
      response.status,
      error.code ?? "UNKNOWN_ERROR",
      error.message ?? response.statusText,
      error.correlation_id ?? "",
    );
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Domain types (mirrored from backend schemas)
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  model_version: string;
  device: string;
  num_labels: number;
  uptime_seconds: number;
}

export interface IntentScore {
  intent: string;
  confidence: number;
}

export interface Ticket {
  id: string;
  department: string;
  priority: "P1" | "P2" | "P3";
  sla_hours: number;
  tags: string[];
  created_at: string;
}

export interface ClassifyResponse {
  intent: string;
  confidence: number;
  top_k: IntentScore[];
  ticket: Ticket;
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function classify(message: string): Promise<ClassifyResponse> {
  return request<ClassifyResponse>("/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
}
