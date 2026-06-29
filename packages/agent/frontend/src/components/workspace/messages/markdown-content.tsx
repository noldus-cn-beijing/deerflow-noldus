"use client";

import type { AnchorHTMLAttributes, ImgHTMLAttributes } from "react";
import { memo, useMemo } from "react";

import {
  MessageResponse,
  type MessageResponseProps,
} from "@/components/ai-elements/message";
import { resolveArtifactURL } from "@/core/artifacts/utils";
import { streamdownPlugins } from "@/core/streamdown";
import { cn } from "@/lib/utils";

import { CitationLink } from "../citations/citation-link";

function isExternalUrl(href: string | undefined): boolean {
  return !!href && /^https?:\/\//.test(href);
}

export type MarkdownContentProps = {
  content: string;
  isLoading: boolean;
  rehypePlugins?: MessageResponseProps["rehypePlugins"];
  className?: string;
  remarkPlugins?: MessageResponseProps["remarkPlugins"];
  components?: MessageResponseProps["components"];
  /** Thread ID for resolving artifact image URLs (e.g. /mnt/user-data/... paths).
   *  When provided, all <img> tags in the markdown will have their src resolved
   *  through the artifact API automatically. */
  threadId?: string;
};

function MarkdownContentImpl({
  content,
  rehypePlugins = streamdownPlugins.rehypePlugins,
  className,
  remarkPlugins = streamdownPlugins.remarkPlugins,
  components: componentsFromProps,
  threadId,
}: MarkdownContentProps) {
  // spec 2026-06-29-streaming-render-perf Step 2 — `components` identity must be
  // stable so the underlying streamdown (via MessageResponse) block-level memo
  // actually engages: a new `components` object every render would defeat that
  // memo and force the whole markdown (incl. Shiki syntax highlighting) to
  // re-parse every token.
  //
  // Deps are `[threadId, componentsFromProps]`:
  //   - `threadId` is a string (value-stable across renders).
  //   - `componentsFromProps` is reference-compared. Callers MUST pass a
  //     memoized object (or none). message-list-item.tsx already memoizes its
  //     `components` on `[thread_id]`; the other call sites pass none
  //     (`undefined`, trivially stable). An inline-literal `components={...}`
  //     here would re-trigger the memo every render — that is exactly the
  //     "prop 不稳定" root cause and is why Step 3 hoists subtask-card's inline
  //     object to a module constant.
  const components = useMemo(() => {
    return {
      a: (props: AnchorHTMLAttributes<HTMLAnchorElement>) => {
        if (typeof props.children === "string") {
          const match = /^citation:(.+)$/.exec(props.children);
          if (match) {
            const [, text] = match;
            return <CitationLink {...props}>{text}</CitationLink>;
          }
        }
        const { className, target, rel, ...rest } = props;
        const external = isExternalUrl(props.href);
        return (
          <a
            {...rest}
            className={cn(
              "text-primary underline underline-offset-4 hover:text-brand transition-colors",
              className,
            )}
            target={target ?? (external ? "_blank" : undefined)}
            rel={rel ?? (external ? "noopener noreferrer" : undefined)}
          />
        );
      },
      // Resolve /mnt/user-data/… artifact paths so markdown images (e.g. from
      // present_files content, report.md, handoff summaries) get correct URLs
      // instead of raw filesystem paths that 404.
      //
      // 规范形态（SSOT，2026-06-18）：report.md 图片 src 一律是
      // /mnt/user-data/outputs/<name>.png（后端 seal 统一产出）。前端只认这一种：
      //   - /mnt/user-data/… → artifact API（保留全前缀，后端 resolve_virtual_path 命中）
      //   - http(s)://        → 外链原样
      //   - 其余              → 原样渲染让其 404 暴露（响亮失败，不猜测/兜底）
      // 详见 docs/superpowers/specs/2026-06-18-report-image-path-ssot-spec.md。
      img: (props: ImgHTMLAttributes<HTMLImageElement>) => {
        const src = props.src;
        if (typeof src === "string" && threadId && src.startsWith("/mnt/user-data/")) {
          return <img {...props} src={resolveArtifactURL(src, threadId)} />;
        }
        return <img {...props} />;
      },
      ...componentsFromProps,
    };
  }, [componentsFromProps, threadId]);

  if (!content) return null;

  return (
    <MessageResponse
      className={className}
      remarkPlugins={remarkPlugins}
      rehypePlugins={rehypePlugins}
      components={components}
    >
      {content}
    </MessageResponse>
  );
}

/**
 * Renders markdown content.
 *
 * spec 2026-06-29-streaming-render-perf Step 2 — wrapped in `React.memo` so a
 * terminal (completed) message's markdown subtree does NOT re-render when a
 * sibling streams a new token. Combined with the stable `components` identity
 * above, this lets streamdown's built-in block-level memo (committed-prefix +
 * live-tail) actually engage: historical markdown blocks are not re-parsed
 * (incl. Shiki syntax highlighting) on every streaming batch.
 *
 * Default shallow comparison is correct here: `content`/`isLoading`/`className`
 * /`threadId` are compared by value, `components`/`remarkPlugins`/`rehypePlugins`
 * by reference (stable via callers' useMemo or the module-level streamdown
 * constants). No custom comparator needed.
 */
export const MarkdownContent = memo(MarkdownContentImpl);
MarkdownContent.displayName = "MarkdownContent";
