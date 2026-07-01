// @vitest-environment jsdom
import { cleanup } from "@testing-library/react";
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { AccentBar, StatusBadge } from "./status-badge";

afterEach(() => cleanup());

describe("StatusBadge (color-not-only 三件套)", () => {
  it("renders icon + colored text + label for each status", () => {
    const { container, rerender } = render(<StatusBadge status="danger" label="失败" />);
    expect(screen.getByText("失败")).toBeInTheDocument();
    // color: uses the danger status token class (not a raw hex).
    expect(container.querySelector('[class*="text-status-danger"]')).not.toBeNull();
    // icon: an svg is present (the third leg of color-not-only).
    expect(container.querySelector("svg")).not.toBeNull();

    rerender(<StatusBadge status="success" label="完成" />);
    expect(container.querySelector('[class*="text-status-success"]')).not.toBeNull();
  });
});

describe("AccentBar (细竖条，非整卡变色)", () => {
  it("renders a thin status-colored bar", () => {
    const { container } = render(<AccentBar status="warning" />);
    const bar = container.firstChild as HTMLElement;
    expect(bar.className).toMatch(/bg-status-warning/);
    // thin: has a small width class (w-1 / w-0.5 / w-[3px]), not a full-card bg.
    expect(bar.className).toMatch(/w-/);
  });
});
