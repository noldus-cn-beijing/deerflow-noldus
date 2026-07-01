// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { StatusCard } from "./status-card";

afterEach(() => cleanup());

describe("StatusCard", () => {
  it("renders an accent bar (status color) + title + children, not a full-card fill", () => {
    const { container } = render(
      <StatusCard status="warning" title="分析已暂停">
        <p>正文</p>
      </StatusCard>,
    );
    expect(screen.getByText("分析已暂停")).toBeInTheDocument();
    expect(screen.getByText("正文")).toBeInTheDocument();
    // accent bar present (thin status strip)
    expect(container.querySelector('[class*="bg-status-warning"]')).not.toBeNull();
    // shell uses radius/shadow tokens, not raw values
    const shell = container.firstChild as HTMLElement;
    expect(shell.className).toMatch(/rounded-|shadow-/);
  });

  it("applies pulse animation only when pulse=true", () => {
    const { container, rerender } = render(<StatusCard status="danger" pulse title="x" />);
    expect(container.innerHTML).toMatch(/pulse-warm/);
    rerender(<StatusCard status="danger" title="x" />);
    expect(container.innerHTML).not.toMatch(/pulse-warm/);
  });
});
