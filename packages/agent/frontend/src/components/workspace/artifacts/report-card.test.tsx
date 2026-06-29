// @vitest-environment jsdom
import { cleanup, fireEvent, render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ReportCard } from "./report-card";

/**
 * spec 2026-06-29-html-report-format：ReportCard 按扩展名路由渲染器。
 *
 * .html 报告 → HTMLContent（消毒后内联渲染，含 .report-html-content 容器）。
 * .md 报告 → 仍走 MarkdownContent（旧报告不回归）。
 *
 * 用 mock 的 loadArtifactContent + i18n 隔离网络/翻译，只验路由分支。
 */

vi.mock("@/core/artifacts/loader", () => ({
  loadArtifactContent: vi.fn(async ({ filepath }: { filepath: string }) => ({
    content: filepath.endsWith(".html")
      ? "<h2>实验概况</h2><script>alert(1)</script>"
      : "# 实验概况\n\n内容",
    url: "",
  })),
}));

const tGallery = {
  reportTitle: "报告",
  reportOpen: "展开",
  reportDownload: "下载",
  noArtifacts: "无",
};

vi.mock("@/core/i18n/hooks", () => ({
  useI18n: () => ({ t: { gallery: tGallery }, locale: "zh-CN" }),
}));

beforeEach(() => {
  // jsdom 无 window.matchMedia / IntersectionObserver，给空实现避免组件 mount 报错。
  const noop = () => undefined;
  window.matchMedia =
    window.matchMedia ?? ((_q: unknown) => ({ matches: false, addEventListener: noop, removeEventListener: noop }));
});
afterEach(() => cleanup());

function makeMeta(path: string) {
  return { path, kind: "report" as const, filename: path.split("/").pop() };
}

describe("ReportCard renderer routing", () => {
  it("renders .html report through HTMLContent (sanitized)", async () => {
    const { container, getByText } = render(<ReportCard meta={makeMeta("/mnt/user-data/outputs/report.html")} threadId="t1" />);
    fireEvent.click(getByText("展开"));
    // 等消毒后的标题出现（mounted 翻转 + DOMPurify 消毒后异步落 DOM）
    await waitFor(() => {
      expect(container.querySelector(".report-html-content")?.innerHTML).toContain("实验概况");
    });
    const inner = container.querySelector(".report-html-content")?.innerHTML ?? "";
    expect(inner.toLowerCase()).not.toContain("<script");
  });

  it("renders .md report through MarkdownContent (no html-content container)", async () => {
    const { container, getByText } = render(<ReportCard meta={makeMeta("/mnt/user-data/outputs/report.md")} threadId="t1" />);
    fireEvent.click(getByText("展开"));
    await waitFor(() => {
      // md 报告不产 .report-html-content 容器
      expect(container.querySelector(".report-html-content")).toBeNull();
    });
    // md 路径仍渲染出文本（MarkdownContent 容器存在）
    expect(container.textContent).toContain("实验概况");
  });
});
