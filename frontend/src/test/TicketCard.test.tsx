import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import TicketCard from "../components/TicketCard";
import type { ClassifyResponse } from "../lib/api";

const mockResult: ClassifyResponse = {
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

const p1Result: ClassifyResponse = {
  ...mockResult,
  ticket: { ...mockResult.ticket, priority: "P1" },
};

const p2Result: ClassifyResponse = {
  ...mockResult,
  ticket: { ...mockResult.ticket, priority: "P2" },
};

describe("TicketCard", () => {
  it("renders the top intent", () => {
    render(<TicketCard result={mockResult} />);
    expect(screen.getAllByText("card_arrival").length).toBeGreaterThanOrEqual(1);
  });

  it("renders the department", () => {
    render(<TicketCard result={mockResult} />);
    expect(screen.getByText("Cards")).toBeDefined();
  });

  it("renders the priority badge", () => {
    render(<TicketCard result={mockResult} />);
    expect(screen.getByText("P3")).toBeDefined();
  });

  it("renders P1 badge with red styling", () => {
    render(<TicketCard result={p1Result} />);
    const badge = screen.getByText("P1");
    expect(badge.className).toMatch(/red/);
  });

  it("renders P2 badge with amber styling", () => {
    render(<TicketCard result={p2Result} />);
    const badge = screen.getByText("P2");
    expect(badge.className).toMatch(/amber/);
  });

  it("renders P3 badge with slate styling", () => {
    render(<TicketCard result={mockResult} />);
    const badge = screen.getByText("P3");
    expect(badge.className).toMatch(/slate/);
  });

  it("renders all three top-k intents", () => {
    render(<TicketCard result={mockResult} />);
    // card_arrival appears in both header and confidence bar
    expect(screen.getAllByText("card_arrival").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("card_delivery_estimate")).toBeDefined();
    expect(screen.getByText("lost_or_stolen_card")).toBeDefined();
  });

  it("renders confidence as percentage", () => {
    render(<TicketCard result={mockResult} />);
    expect(screen.getByText("94%")).toBeDefined();
  });

  it("renders tags as pills", () => {
    render(<TicketCard result={mockResult} />);
    expect(screen.getByText("card")).toBeDefined();
    expect(screen.getByText("delivery")).toBeDefined();
  });

  it("renders sla_hours", () => {
    render(<TicketCard result={mockResult} />);
    expect(screen.getByText(/24/)).toBeDefined();
  });

  it("renders ticket id (truncated or full)", () => {
    render(<TicketCard result={mockResult} />);
    expect(screen.getByText(/01HXR/)).toBeDefined();
  });
});
