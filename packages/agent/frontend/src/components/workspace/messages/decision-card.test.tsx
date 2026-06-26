// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import { DecisionCard } from "@/components/workspace/messages/decision-card";
import {
  cleanup,
  renderWithProviders,
  screen,
  userEvent,
} from "@/test/test-utils";

/**
 * spec#5 决策卡 DecisionCard —— 渲染 / a11y / 键盘 / 已答闭环。
 *
 * 纯组件单测：i18n 走真实 provider（en-US），不挂整个 thread。MarkdownContent 真
 * 渲染（streamdown）。这里只断言决策卡特有的结构（accent / 标题 / context / 选项）。
 */

const OPTIONS = [
  "Center zone = center, edge zone = periphery",
  "the other way around",
  "neither, let me explain",
];

function setup(overrides: Partial<React.ComponentProps<typeof DecisionCard>> = {}) {
  const onSelect = vi.fn();
  const utils = renderWithProviders(
    <DecisionCard
      question="Which column is the center zone?"
      context='Detected columns "中心区" and "边缘区".'
      clarificationType="ambiguous_requirement"
      options={OPTIONS}
      answeredIndex={null}
      onSelect={onSelect}
      {...overrides}
    />,
  );
  return { ...utils, onSelect };
}

/** 取选项按钮们（按 role=button 且带 aria-keyshortcuts 精确定位）。 */
function optionButtons() {
  return screen
    .getAllByRole("button")
    .filter((b) => b.getAttribute("aria-keyshortcuts"));
}

describe("DecisionCard (spec#5)", () => {
  // The vitest setup only registers jest-dom matchers (no globals → no
  // testing-library auto-cleanup afterEach). Unmount between tests so queries
  // don't leak across renders.
  afterEach(cleanup);

  it("renders the strong-signal title, context, and options", () => {
    setup();
    // 标题（en-US：分析已暂停）
    expect(
      screen.getByText("Analysis paused · awaiting your confirmation"),
    ).toBeInTheDocument();
    // 决策依据前缀（Why we're asking）+ 内容（regex 容错全角冒号/curly apostrophe）
    expect(screen.getByText(/Why we.+re asking/)).toBeInTheDocument();
    expect(
      screen.getByText(/Detected columns "中心区" and "边缘区"/),
    ).toBeInTheDocument();
    // 三个选项都在
    for (const opt of OPTIONS) {
      expect(screen.getByText(opt)).toBeInTheDocument();
    }
    // 「或自定义输入」仍在（escape-routes）
    expect(screen.getByText(/Or type a custom reply below/)).toBeInTheDocument();
  });

  it("uses the risk title for risk_confirmation (strongest tone)", () => {
    setup({ clarificationType: "risk_confirmation" });
    expect(screen.getByText("Please confirm a risk")).toBeInTheDocument();
    expect(
      screen.queryByText("Analysis paused · awaiting your confirmation"),
    ).not.toBeInTheDocument();
  });

  it("hides the context block when context is absent", () => {
    setup({ context: undefined });
    expect(screen.queryByText(/Why we.+re asking/)).not.toBeInTheDocument();
  });

  it("marks options with aria-keyshortcuts 1..N (keyboard nav)", () => {
    setup();
    const shortcuts = optionButtons().map((b) =>
      b.getAttribute("aria-keyshortcuts"),
    );
    expect(shortcuts).toEqual(["1", "2", "3"]);
  });

  it("calls onSelect with the option text when an option is clicked", async () => {
    const { onSelect } = setup();
    await userEvent.click(optionButtons()[1]!);
    expect(onSelect).toHaveBeenCalledWith(OPTIONS[1]);
  });

  it("selects an option via digit key while awaiting (keyboard nav)", () => {
    const { onSelect } = setup();
    // Real usage: focus lands on an option button inside the options group; the
    // keydown listener is on the group container, so the event bubbles up to it.
    const target = optionButtons()[0]!;
    target.focus();
    target.dispatchEvent(
      new KeyboardEvent("keydown", { key: "2", bubbles: true }),
    );
    expect(onSelect).toHaveBeenCalledWith(OPTIONS[1]);
  });

  it("does NOT fire on digit key while focus is in a text field (input conflict guard)", () => {
    const { onSelect } = setup();
    // The listener ignores keys whose target is an <input>/<textarea> so typing
    // a custom reply still works (spec §六 risk row).
    const group = screen
      .getAllByRole("group")
      .find((g) => g.getAttribute("aria-label") === "Choose an option")!;
    const input = document.createElement("input");
    group.appendChild(input);
    input.focus();
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "1", bubbles: true }),
    );
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("enters the answered closed-loop: success title + all options disabled", () => {
    setup({ answeredIndex: 0 });
    expect(screen.getByText("Confirmed")).toBeInTheDocument();
    expect(
      screen.queryByText("Analysis paused · awaiting your confirmation"),
    ).not.toBeInTheDocument();
    // 所有选项按钮均已禁用（closed-loop）。
    expect(optionButtons().every((b) => (b as HTMLButtonElement).disabled)).toBe(
      true,
    );
  });

  it("does not pulse for suggestion tone (weak signal)", () => {
    const { container } = setup({ clarificationType: "suggestion" });
    const bar = container.querySelector(
      "section > .absolute.inset-y-0.left-0",
    );
    expect(bar).not.toBeNull();
    expect(bar?.className).not.toContain("animate-pulse-warm");
  });

  it("pulses the accent bar while awaiting a warning-tone clarification", () => {
    const { container } = setup({ clarificationType: "ambiguous_requirement" });
    const bar = container.querySelector(
      "section > .absolute.inset-y-0.left-0",
    );
    expect(bar?.className).toContain("animate-pulse-warm");
  });
});
