// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Todo } from "@/core/todos";

import { TodoList } from "./todo-list";

/**
 * spec 2026-06-29-todo-list-expand-clipping-fix：To-dos 面板展开后内容被压扁/裁切。
 *
 * 根因是展开态用固定 `h-28`（112px）作为 main 高度，但内部 QueueList 的
 * ScrollArea 是 `max-h-40`（160px）——内容超过 112px 就被外层 overflow-hidden 裁掉。
 *
 * 修法（grid 法）：展开态改 `grid-rows-[1fr]`，高度随内容自适应、由内部 max-h-40
 * 滚动兜底；折叠态 `grid-rows-[0fr]` 塌成 0。
 *
 * 本测试锁定修复的结构契约（防 h-28 固定高度回归）：
 *   - 展开态 main 用 `grid-rows-[1fr]` 且不带固定 `h-28`；
 *   - 折叠态 main 用 `grid-rows-[0fr]`；
 *   - 展开态 todo 文本可见、折叠态不可见。
 */

afterEach(() => cleanup());

const todos: Todo[] = [
  { content: "data-analyst 审核统计方法", status: "in_progress" },
  { content: "report-writer 生成报告", status: "pending" },
  { content: "已完成条目", status: "completed" },
];

// main 是渲染 todo 列表的区域（含 QueueList）。
function mainClass(container: HTMLElement): string {
  // TodoList 的 <main> 是 header 之后的第一个 <main>。
  const main = container.querySelector("main");
  if (!main) throw new Error("main 元素未渲染");
  return main.className;
}

describe("TodoList 展开高度（grid-rows 自适应，不再固定 h-28）", () => {
  it("展开态：main 用 grid-rows-[1fr] 自适应、不带固定 h-28，且 todo 文本可见", () => {
    const { container } = render(<TodoList todos={todos} collapsed={false} />);

    const cls = mainClass(container);
    expect(cls).toContain("grid-rows-[1fr]");
    expect(cls).not.toContain("h-28");
    // grid 布局本身要就位（替代旧的 flex）。
    expect(cls).toContain("grid");

    expect(screen.getByText("data-analyst 审核统计方法")).toBeVisible();
    expect(screen.getByText("report-writer 生成报告")).toBeVisible();
  });

  it("折叠态：main 用 grid-rows-[0fr] 塌成 0，固定 h-28 同样不得回归", () => {
    const { container } = render(<TodoList todos={todos} collapsed={true} />);

    const cls = mainClass(container);
    expect(cls).toContain("grid-rows-[0fr]");
    expect(cls).not.toContain("h-28");
  });

  it("动画 token 保留：transition 过的是 grid-template-rows 而非 height，沿用 duration-slow/ease-brand-in-out", () => {
    const { container } = render(<TodoList todos={todos} collapsed={false} />);
    const cls = mainClass(container);
    expect(cls).toContain("transition-[grid-template-rows]");
    expect(cls).toContain("duration-slow");
    expect(cls).toContain("ease-brand-in-out");
    // 旧的 height 过渡必须被替换掉。
    expect(cls).not.toContain("transition-[height]");
  });
});
