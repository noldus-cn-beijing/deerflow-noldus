// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { HTMLContent } from "./html-content";

/**
 * spec 2026-06-29-html-report-format：HTMLContent 对 report.html 做 XSS 消毒。
 *
 * HTML 来自 LLM 产出，前端 dangerouslySetInnerHTML 必须先过 DOMPurify。
 * 这里验证消毒后 <script>/onerror 等危险内容被剥，安全结构（表/段落/data 图）保留。
 * DOMPurify 在 jsdom 环境可用（vitest config 用 jsdom）。
 */

afterEach(cleanup);

describe("HTMLContent sanitization", () => {
  it("strips <script> tags", async () => {
    const { container } = render(<HTMLContent content="<p>ok</p><script>alert(1)</script><p>after</p>" />);
    const html = container.querySelector(".report-html-content")?.innerHTML ?? "";
    expect(html.toLowerCase()).not.toContain("<script");
    expect(html).not.toContain("alert(1)");
    expect(html).toContain("ok");
    expect(html).toContain("after");
  });

  it("strips inline onerror/onload handlers", async () => {
    const { container } = render(
      <HTMLContent content={'<img src="x" onerror="alert(1)" onload="evil()">'} />,
    );
    const html = container.querySelector(".report-html-content")?.innerHTML.toLowerCase() ?? "";
    expect(html).not.toContain("onerror");
    expect(html).not.toContain("onload");
    expect(html).not.toContain("alert");
  });

  it("strips <iframe>", async () => {
    const { container } = render(<HTMLContent content={'<iframe src="https://evil"></iframe><p>keep</p>'} />);
    const html = container.querySelector(".report-html-content")?.innerHTML.toLowerCase() ?? "";
    expect(html).not.toContain("<iframe");
    expect(html).toContain("keep");
  });

  it("preserves safe structured content and inline data images", async () => {
    const content =
      "<h2>实验概况</h2><table><tr><td>1.0</td></tr></table><ul><li>项 A</li></ul>" +
      '<img src="data:image/png;base64,QUJD">';
    const { container } = render(<HTMLContent content={content} />);
    const html = container.querySelector(".report-html-content")?.innerHTML ?? "";
    for (const fragment of ["<h2>实验概况</h2>", "<table>", "<li>项 A</li>", "data:image/png;base64,QUJD"]) {
      expect(html).toContain(fragment);
    }
  });

  it("never injects un-sanitized script into the DOM", () => {
    // 无论首屏（mounted=false，sanitizeHtml 返回空串）还是 mount 后（DOMPurify 剥 script），
    // 不变式都是：<script> 永不进入 DOM。
    const { container } = render(<HTMLContent content="<script>alert(1)</script>" />);
    const inner = container.querySelector(".report-html-content")?.innerHTML ?? "";
    expect(inner.toLowerCase()).not.toContain("<script");
    expect(inner).not.toContain("alert");
  });

  it("applies prose class so typography styling engages", () => {
    // spec 2026-06-29：prose 是死类的根因——@tailwindcss/typography 未装时 prose 不生成
    // 任何 CSS。这里断言 className 上确有 prose（plugin 装没装另由 build grep 验收）。
    const { container } = render(<HTMLContent content="<p>ok</p>" />);
    const host = container.querySelector(".report-html-content");
    expect(host?.className).toContain("prose");
    expect(host?.className).toContain("prose-sm");
  });

  it("does not render <head>/<title> content into the body", () => {
    // 后端 seal 已把 <title> 整段删除（drop-with-content），但纵深防御：即便传入完整文档，
    // 前端也只应渲染 <body> 内容——<head>/<title> 文字不得冒进正文渲染区（dogfood 73b41dc3
    // 的裸标题 bug）。DOMPurify 默认 WHOLE_DOCUMENT=false，对合法文档只返回 body 内容。
    const { container } = render(
      <HTMLContent
        content={
          '<!DOCTYPE html><html><head><title>裸露标题不该出现</title><meta charset="utf-8"></head>' +
          "<body><h2>正文标题</h2><p>正文</p></body></html>"
        }
      />,
    );
    const inner = container.querySelector(".report-html-content")?.innerHTML ?? "";
    expect(inner).not.toContain("裸露标题不该出现");
    expect(inner).not.toContain("<title");
    expect(inner).not.toContain("<head");
    // body 正文保留
    expect(inner).toContain("正文标题");
    expect(inner).toContain("正文");
  });
});
