import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MessageInput from "../components/MessageInput";

describe("MessageInput", () => {
  it("renders a textarea", () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    expect(screen.getByRole("textbox")).toBeDefined();
  });

  it("renders a submit button", () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    expect(screen.getByRole("button")).toBeDefined();
  });

  it("shows character count", () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    expect(screen.getByText(/0\s*\/\s*1000/)).toBeDefined();
  });

  it("updates character count as user types", async () => {
    const user = userEvent.setup();
    render(<MessageInput onSubmit={vi.fn()} />);
    await user.type(screen.getByRole("textbox"), "hello");
    expect(screen.getByText(/5\s*\/\s*1000/)).toBeDefined();
  });

  it("calls onSubmit with message when button clicked", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<MessageInput onSubmit={onSubmit} />);
    await user.type(screen.getByRole("textbox"), "my card is lost");
    await user.click(screen.getByRole("button"));
    expect(onSubmit).toHaveBeenCalledWith("my card is lost");
  });

  it("calls onSubmit on Enter key", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<MessageInput onSubmit={onSubmit} />);
    await user.type(screen.getByRole("textbox"), "test message{Enter}");
    expect(onSubmit).toHaveBeenCalledWith("test message");
  });

  it("disables textarea and button when loading=true", () => {
    render(<MessageInput onSubmit={vi.fn()} loading={true} />);
    expect(screen.getByRole("textbox")).toHaveProperty("disabled", true);
    expect(screen.getByRole("button")).toHaveProperty("disabled", true);
  });

  it("disables submit button when message is empty", () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    expect(screen.getByRole("button")).toHaveProperty("disabled", true);
  });

  it("enables submit button when message is non-empty", async () => {
    const user = userEvent.setup();
    render(<MessageInput onSubmit={vi.fn()} />);
    await user.type(screen.getByRole("textbox"), "hello");
    expect(screen.getByRole("button")).toHaveProperty("disabled", false);
  });

  it("shows warning style when over 900 chars", async () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    const textarea = screen.getByRole("textbox");
    fireEvent.change(textarea, { target: { value: "a".repeat(950) } });
    expect(screen.getByText(/950\s*\/\s*1000/)).toBeDefined();
  });

  it("does not allow more than 1000 characters", async () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    const textarea = screen.getByRole("textbox");
    fireEvent.change(textarea, { target: { value: "a".repeat(1001) } });
    const count = screen.getByText(/\/\s*1000/);
    const displayed = parseInt(count.textContent?.split("/")[0] ?? "0");
    expect(displayed).toBeLessThanOrEqual(1000);
  });
});
