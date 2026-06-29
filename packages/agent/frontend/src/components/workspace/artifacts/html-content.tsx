"use client";

/**
 * HTMLContent — 渲染 report.html（spec 2026-06-29-html-report-format）。
 *
 * 报告载体从 Markdown 改 HTML 后，ReportCard 按扩展名路由：.html → 本组件，
 * .md → 仍走 MarkdownContent（旧报告不回归）。
 *
 * XSS 防御（硬要求）：HTML 来自 LLM 产出，前端 dangerouslySetInnerHTML
 * 必须先过 DOMPurify 消毒。后端 seal 时已做一层确定性消毒（剥 script / 内联 on 事件 /
 * iframe），此处是纵深防御的二次 sanitize——两层独立，任一被绕过另一层兜底。
 */
import DOMPurify from "dompurify";
import { useEffect, useMemo, useState } from "react";

import { cn } from "@/lib/utils";

export type HTMLContentProps = {
  /** 原始 HTML 字符串（来自 report.html，可能含 LLM 产出的危险内容）。 */
  content: string;
  className?: string;
};

/**
 * 在浏览器侧用 DOMPurify 消毒 HTML。SSR/首屏期间（无 window）返回空串，
 * 避免把未消毒 HTML 注入 DOM；客户端 mount 后立即消毒渲染。
 *
 * 消毒配置：禁止所有可执行内容（script / 内联 on 事件 / iframe 等），保留结构化标签与
 * 内联 data 图（report.html 的代表性图）。
 */
function sanitizeHtml(html: string): string {
  if (typeof window === "undefined") return "";
  return DOMPurify.sanitize(html, {
    // WHOLE_DOCUMENT:false（DOMPurify 默认）——传入完整文档时只渲染 <body> 内容，
    // <head>/<title> 不进正文渲染区（dogfood 73b41dc3 的裸标题 bug 防线之一；
    // 后端 seal 亦已把 <title> 整段删除，纵深防御）。
    WHOLE_DOCUMENT: false,
    // 禁止所有可执行/远程加载内容；data: 图像保留（base64 内联的代表性图）。
    FORBID_TAGS: ["script", "style", "iframe", "object", "embed", "form", "input", "button"],
    FORBID_ATTR: ["onerror", "onload", "onclick", "onmouseover", "onfocus", "onblur"],
    ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto|tel|data:image\/|\/|#))/i,
  });
}

/** 渲染已消毒的 HTML 报告。 */
export function HTMLContent({ content, className }: HTMLContentProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const clean = useMemo(() => (mounted ? sanitizeHtml(content) : ""), [mounted, content]);

  return (
    <div
      className={cn("report-html-content prose prose-sm max-w-none", className)}
      // content 已在 sanitizeHtml 中经 DOMPurify 消毒（剥 script/on*/iframe 等）。
      // 后端 seal 时亦做过一层确定性消毒（纵深防御）。
      dangerouslySetInnerHTML={{ __html: clean }}
    />
  );
}
