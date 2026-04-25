import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";

// ---------------------------------------------------------------------------
// Mock the API module
// ---------------------------------------------------------------------------

vi.mock("../lib/api", () => ({
  getHealth: vi.fn(),
  classify: vi.fn(),
  ApiError: class ApiError extends Error {
    constructor(
      public status: number,
      public code: string,
      message: string,
      public correlationId: string,
    ) {
      super(message);
    }
  },
  NetworkError: class NetworkError extends Error {
    constructor(message: string) {
      super(message);
    }
  },
}));

import { getHealth, classify } from "../lib/api";

const mockHealth = {
  status: "ok",
  model_version: "bert-base@v0.1.0",
  device: "cpu",
  num_labels: 77,
  uptime_seconds: 10,
};

const mockClassifyResult = {
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
    priority: "P3" as const,
    sla_hours: 24,
    tags: ["card", "delivery"],
    created_at: "2026-04-25T12:00:00Z",
  },
};

beforeEach(() => {
  vi.mocked(getHealth).mockResolvedValue(mockHealth);
  vi.mocked(classify).mockResolvedValue(mockClassifyResult);
});

afterEach(() => {
  vi.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("App", () => {
  it("renders the MessageInput on load", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole("textbox")).toBeDefined());
  });

  it("calls getHealth on mount", async () => {
    render(<App />);
    await waitFor(() => expect(vi.mocked(getHealth)).toHaveBeenCalledOnce());
  });

  it("shows backend status after health check succeeds", async () => {
    render(<App />);
    await waitFor(() =>
      expect(screen.getByText(/bert-base@v0\.1\.0/)).toBeDefined()
    );
  });

  it("shows error banner when health check fails", async () => {
    vi.mocked(getHealth).mockRejectedValueOnce(new Error("connection refused"));
    render(<App />);
    await waitFor(() =>
      expect(screen.getByRole("alert")).toBeDefined()
    );
  });

  it("shows TicketCard after classify succeeds", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitFor(() => screen.getByRole("textbox"));
    await user.type(screen.getByRole("textbox"), "my card has not arrived");
    await user.click(screen.getByRole("button"));
    await waitFor(() =>
      expect(screen.getAllByText("card_arrival").length).toBeGreaterThanOrEqual(1)
    );
  });

  it("shows error banner when classify fails", async () => {
    const { ApiError } = await import("../lib/api");
    vi.mocked(classify).mockRejectedValueOnce(
      new ApiError(422, "VALIDATION_FAILED", "bad input", "cid")
    );
    const user = userEvent.setup();
    render(<App />);
    await waitFor(() => screen.getByRole("textbox"));
    await user.type(screen.getByRole("textbox"), "test");
    await user.click(screen.getByRole("button"));
    await waitFor(() => expect(screen.getByRole("alert")).toBeDefined());
  });

  it("disables input while classifying", async () => {
    let resolveClassify!: (v: typeof mockClassifyResult) => void;
    vi.mocked(classify).mockReturnValueOnce(
      new Promise((res) => { resolveClassify = res; })
    );
    const user = userEvent.setup();
    render(<App />);
    await waitFor(() => screen.getByRole("textbox"));
    await user.type(screen.getByRole("textbox"), "test");
    await user.click(screen.getByRole("button"));
    expect(screen.getByRole("textbox")).toHaveProperty("disabled", true);
    resolveClassify(mockClassifyResult);
  });
});
