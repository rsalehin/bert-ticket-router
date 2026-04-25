import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { classify, getHealth, ApiError, NetworkError } from "../lib/api";

// ---------------------------------------------------------------------------
// Mock fetch globally
// ---------------------------------------------------------------------------

function mockFetch(status: number, body: unknown): void {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      json: () => Promise.resolve(body),
    })
  );
}

function mockFetchNetworkError(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockRejectedValue(new TypeError("Failed to fetch"))
  );
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// ApiError
// ---------------------------------------------------------------------------

describe("ApiError", () => {
  it("is an Error subclass", () => {
    const err = new ApiError(503, "MODEL_NOT_LOADED", "model not ready", "abc");
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(ApiError);
  });

  it("exposes status, code, message, correlationId", () => {
    const err = new ApiError(503, "MODEL_NOT_LOADED", "model not ready", "cid123");
    expect(err.status).toBe(503);
    expect(err.code).toBe("MODEL_NOT_LOADED");
    expect(err.message).toBe("model not ready");
    expect(err.correlationId).toBe("cid123");
  });
});

// ---------------------------------------------------------------------------
// NetworkError
// ---------------------------------------------------------------------------

describe("NetworkError", () => {
  it("is an Error subclass", () => {
    const err = new NetworkError("connection refused");
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(NetworkError);
  });

  it("preserves message", () => {
    const err = new NetworkError("connection refused");
    expect(err.message).toBe("connection refused");
  });
});

// ---------------------------------------------------------------------------
// getHealth
// ---------------------------------------------------------------------------

describe("getHealth", () => {
  it("returns health data on 200", async () => {
    mockFetch(200, {
      status: "ok",
      model_version: "bert-base@v0.1.0",
      device: "cpu",
      num_labels: 77,
      uptime_seconds: 42.0,
    });
    const health = await getHealth();
    expect(health.status).toBe("ok");
    expect(health.num_labels).toBe(77);
  });

  it("throws ApiError on non-200", async () => {
    mockFetch(503, {
      error: { code: "MODEL_NOT_LOADED", message: "model not ready", correlation_id: "cid" },
    });
    await expect(getHealth()).rejects.toBeInstanceOf(ApiError);
  });

  it("throws NetworkError on fetch failure", async () => {
    mockFetchNetworkError();
    await expect(getHealth()).rejects.toBeInstanceOf(NetworkError);
  });
});

// ---------------------------------------------------------------------------
// classify
// ---------------------------------------------------------------------------

describe("classify", () => {
  const mockResponse = {
    intent: "card_arrival",
    confidence: 0.94,
    top_k: [
      { intent: "card_arrival", confidence: 0.94 },
      { intent: "card_delivery_estimate", confidence: 0.04 },
      { intent: "lost_or_stolen_card", confidence: 0.02 },
    ],
    ticket: {
      id: "01HXR3F8Z4G2N6P7Q9S0T2U4V6",
      department: "Cards",
      priority: "P3",
      sla_hours: 24,
      tags: ["card", "delivery"],
      created_at: "2026-04-25T12:00:00Z",
    },
  };

  it("returns ClassifyResponse on 200", async () => {
    mockFetch(200, mockResponse);
    const result = await classify("my card has not arrived");
    expect(result.intent).toBe("card_arrival");
    expect(result.confidence).toBe(0.94);
    expect(result.top_k).toHaveLength(3);
    expect(result.ticket.department).toBe("Cards");
  });

  it("sends message in request body", async () => {
    mockFetch(200, mockResponse);
    await classify("my card has not arrived");
    const fetchMock = vi.mocked(fetch);
    const [, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(options.body as string);
    expect(body.message).toBe("my card has not arrived");
  });

  it("sends Content-Type application/json", async () => {
    mockFetch(200, mockResponse);
    await classify("test");
    const fetchMock = vi.mocked(fetch);
    const [, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((options.headers as Record<string, string>)["Content-Type"]).toBe(
      "application/json"
    );
  });

  it("throws ApiError on 422", async () => {
    mockFetch(422, {
      error: { code: "VALIDATION_FAILED", message: "bad input", correlation_id: "cid" },
    });
    await expect(classify("")).rejects.toBeInstanceOf(ApiError);
  });

  it("throws ApiError with correct code on error", async () => {
    mockFetch(422, {
      error: { code: "VALIDATION_FAILED", message: "bad input", correlation_id: "cid" },
    });
    try {
      await classify("");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect((e as ApiError).code).toBe("VALIDATION_FAILED");
    }
  });

  it("throws NetworkError on fetch failure", async () => {
    mockFetchNetworkError();
    await expect(classify("test")).rejects.toBeInstanceOf(NetworkError);
  });
});
