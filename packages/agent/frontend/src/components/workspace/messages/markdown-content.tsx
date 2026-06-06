"use client";

import { useMemo } from "react";
import type { AnchorHTMLAttributes, ImgHTMLAttributes } from "react";

import {
  MessageResponse,
  type MessageResponseProps,
} from "@/components/ai-elements/message";
import { normalizeArtifactImageSrc, resolveArtifactURL } from "@/core/artifacts/utils";
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

/** Renders markdown content. */
export function MarkdownContent({
  content,
  rehypePlugins = streamdownPlugins.rehypePlugins,
  className,
  remarkPlugins = streamdownPlugins.remarkPlugins,
  components: componentsFromProps,
  threadId,
}: MarkdownContentProps) {
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
      // Resolve /mnt/… and other artifact paths so markdown images (e.g. from
      // present_files content, report.md, handoff summaries) get correct URLs
      // instead of raw filesystem paths that 404.
      img: (props: ImgHTMLAttributes<HTMLImageElement>) => {
        const src = props.src;
        if (typeof src === "string" && threadId) {
          // /mnt/user-data/… → artifact API
          if (src.startsWith("/mnt/")) {
            return <img {...props} src={resolveArtifactURL(src, threadId)} />;
          }
          // Other virtual paths (outputs/X.png, /user-data/…)
          const normalized = normalizeArtifactImageSrc(src);
          if (normalized) {
            return <img {...props} src={resolveArtifactURL(normalized, threadId)} />;
          }
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
