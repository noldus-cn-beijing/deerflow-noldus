import type { Message } from "@langchain/langgraph-sdk";
import { ChevronUp, LightbulbIcon } from "lucide-react";
import { useParams } from "next/navigation";
import { memo, useMemo, useState, type ImgHTMLAttributes } from "react";

import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { Loader } from "@/components/ai-elements/loader";
import {
  Message as AIElementMessage,
  MessageContent as AIElementMessageContent,
  MessageResponse as AIElementMessageResponse,
  MessageToolbar,
} from "@/components/ai-elements/message";
import { Task, TaskTrigger } from "@/components/ai-elements/task";
import { FeedbackButtons } from "@/components/feedback/feedback-buttons";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { resolveArtifactURL } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import {
  extractContentCached,
  extractReasoningCached,
} from "@/core/messages/extraction-cache";
import {
  extractQualityWarnings,
  parseUploadedFiles,
  stripUploadedFilesTag,
  type FileInMessage,
} from "@/core/messages/utils";
import { humanMessagePlugins } from "@/core/streamdown";
import { cn } from "@/lib/utils";

import { CopyButton } from "../copy-button";

import { MarkdownContent } from "./markdown-content";
import { MessageAttachments } from "./message-attachments";
import { QualityWarningBanner } from "./quality-warning-banner";

// Phase0#7 Step 2 — wrapped in React.memo so a parent re-render (e.g. an
// unrelated streaming token landing in a sibling message) does NOT re-render
// this completed message's subtree. Props are stable across unrelated renders
// now that MessageList memoizes the groups array (Step 1).
export const MessageListItem = memo(function MessageListItem({
  className,
  message,
  isLoading,
  isStreaming = false,
  threadId,
  messageRunIds,
}: {
  className?: string;
  message: Message;
  isLoading?: boolean;
  /**
   * Whether THIS message is the one currently receiving stream tokens (the
   * last message while the thread is loading). Drives the extraction cache:
   * terminal messages (`isStreaming=false`) are long-cached so they stop being
   * O(n)-re-scanned every deferred batch; the single in-flight message
   * (`isStreaming=true`) bypasses the cache because its content is mutating
   * this batch. Defaults to `false` (terminal) for callers that don't track
   * per-message streaming — correct and cache-friendly.
   */
  isStreaming?: boolean;
  threadId?: string;
  messageRunIds?: Map<string, string>;
}) {
  const isHuman = message.type === "human";
  return (
    <AIElementMessage
      className={cn("group/conversation-message relative w-full", className)}
      from={isHuman ? "user" : "assistant"}
    >
      <MessageContent
        className={isHuman ? "w-fit" : "w-full"}
        message={message}
        isLoading={isLoading}
        isStreaming={isStreaming}
      />
      {!isLoading && !isHuman && threadId && message.id && (() => {
        const runId = messageRunIds?.get(message.id);
        if (!runId) return null; // run_id 还没拿到时不渲染，防止误绑
        return (
          <FeedbackButtons
            threadId={threadId}
            runId={runId}
            messageId={message.id}
            className="px-1"
          />
        );
      })()}
      {!isLoading && (
        <MessageToolbar
          className={cn(
            isHuman ? "-bottom-9 justify-end" : "-bottom-8",
            "absolute right-0 left-0 z-20 opacity-0 transition-opacity delay-200 duration-base ease-brand-out group-hover/conversation-message:opacity-100",
          )}
        >
          <div className="flex gap-1">
            <CopyButton
              clipboardData={
                // CopyButton only renders when `!isLoading` (terminal), so the
                // cache is safe here and serves a cached value.
                extractContentCached(message, false) ??
                extractReasoningCached(message, false) ??
                ""
              }
            />
          </div>
        </MessageToolbar>
      )}
    </AIElementMessage>
  );
});

/**
 * Custom image component that handles artifact URLs
 */
function MessageImage({
  src,
  alt,
  threadId,
  maxWidth = "90%",
  ...props
}: React.ImgHTMLAttributes<HTMLImageElement> & {
  threadId: string;
  maxWidth?: string;
}) {
  if (!src) return null;

  const imgClassName = cn("overflow-hidden rounded-lg", `max-w-[${maxWidth}]`);

  if (typeof src !== "string") {
    return <img className={imgClassName} src={src} alt={alt} {...props} />;
  }

  const url = src.startsWith("/mnt/") ? resolveArtifactURL(src, threadId) : src;

  return (
    <a href={url} target="_blank" rel="noopener noreferrer">
      <img className={imgClassName} src={url} alt={alt} {...props} />
    </a>
  );
}

function MessageContent_({
  className,
  message,
  isLoading = false,
  isStreaming = false,
}: {
  className?: string;
  message: Message;
  isLoading?: boolean;
  isStreaming?: boolean;
}) {
  const isHuman = message.type === "human";
  const { thread_id } = useParams<{ thread_id: string }>();
  const components = useMemo(
    () => ({
      img: (props: ImgHTMLAttributes<HTMLImageElement>) => (
        <MessageImage {...props} threadId={thread_id} maxWidth="90%" />
      ),
    }),
    [thread_id],
  );

  // Cached extraction (spec 2026-06-29 Step 1.4): terminal messages are served
  // from the per-message cache; only the in-flight (`isStreaming`) message
  // re-scans. Output is byte-identical to the uncached extractor.
  const rawContent = extractContentCached(message, isStreaming);
  const reasoningContent = extractReasoningCached(message, isStreaming);

  const files = useMemo(() => {
    const files = message.additional_kwargs?.files;
    if (!Array.isArray(files) || files.length === 0) {
      if (rawContent.includes("<uploaded_files>")) {
        // If the content contains the <uploaded_files> tag, we return the parsed files from the content for backward compatibility.
        return parseUploadedFiles(rawContent);
      }
      return null;
    }
    return files as FileInMessage[];
  }, [message.additional_kwargs?.files, rawContent]);

  const contentToDisplay = useMemo(() => {
    if (isHuman) {
      return rawContent ? stripUploadedFilesTag(rawContent) : "";
    }
    return rawContent ?? "";
  }, [rawContent, isHuman]);

  const filesList =
    files && files.length > 0 && thread_id ? (
      <MessageAttachments files={files} threadId={thread_id} />
    ) : null;

  const qualityWarnings = useMemo(
    () => extractQualityWarnings(message as unknown as Record<string, unknown>),
    [message],
  );

  // Uploading state: mock AI message shown while files upload
  if (message.additional_kwargs?.element === "task") {
    return (
      <AIElementMessageContent className={className}>
        <Task defaultOpen={false}>
          <TaskTrigger title="">
            <div className="text-muted-foreground flex w-full cursor-default items-center gap-2 text-sm select-none">
              <Loader className="size-4" />
              <span>{contentToDisplay}</span>
            </div>
          </TaskTrigger>
        </Task>
      </AIElementMessageContent>
    );
  }

  if (isHuman) {
    const messageResponse = contentToDisplay ? (
      <AIElementMessageResponse
        remarkPlugins={humanMessagePlugins.remarkPlugins}
        rehypePlugins={humanMessagePlugins.rehypePlugins}
        components={components}
        parseIncompleteMarkdown={false}
      >
        {contentToDisplay}
      </AIElementMessageResponse>
    ) : null;
    return (
      <div className={cn("ml-auto flex flex-col gap-2", className)}>
        {filesList}
        {messageResponse && (
          <AIElementMessageContent className="w-fit">
            {messageResponse}
          </AIElementMessageContent>
        )}
      </div>
    );
  }

  return (
    <AIElementMessageContent className={className}>
      {filesList}
      {reasoningContent && (
        <ReasoningPanel
          isStreaming={isLoading}
          reasoningContent={reasoningContent}
        />
      )}
      {qualityWarnings.length > 0 && (
        <QualityWarningBanner warnings={qualityWarnings} />
      )}
      {contentToDisplay && (
        <MarkdownContent
          content={contentToDisplay}
          isLoading={isLoading}
          className="my-3"
          components={components}
        />
      )}
    </AIElementMessageContent>
  );
}

const MessageContent = memo(MessageContent_);

/**
 * Lead Agent 思考块 — 使用与 MessageGroup 中"思考"块同款 ChainOfThought 风格,
 * 并打上 "Lead Agent" 徽章,与 SubtaskCard 内 subagent 名字标识对齐。
 *
 * 历史: 早期用 `ai-elements/Reasoning`(Shimmer 闪烁版),与 MessageGroup 的
 * ChainOfThought 风格不一致,用户反馈"两套 thinking 让人困惑"。2026-05-21
 * 统一为单一 layout + 角色徽章。
 */
function ReasoningPanel({
  isStreaming,
  reasoningContent,
}: {
  isStreaming: boolean;
  reasoningContent: string;
}) {
  const { t } = useI18n();
  // chat-render-jank-on-open fix (Fix 3, 2026-06-26): historical messages (a
  // thread that is NOT streaming) default COLLAPSED. Most researchers read the
  // conclusion, not the lead's reasoning trace; mounting every historical
  // reasoning block expanded means N Radix Collapsible `useLayoutEffect`
  // synchronous height measurements on thread open — a confirmed 2.5% slice of
  // the open-jank CPU profile (CollapsibleContentImpl.useLayoutEffect).
  // Collapsing on mount skips that measurement entirely; the user can still
  // click to expand any block. The in-flight message keeps its streaming
  // behavior: when the thread IS streaming we default open so the live
  // thinking trace stays visible (matches MessageGroup's streaming-open rule).
  const [showThinking, setShowThinking] = useState(isStreaming);

  return (
    <ChainOfThought
      className="w-full gap-2 rounded-lg border p-0.5"
      open={true}
    >
      <Button
        className="w-full items-start justify-start text-left"
        variant="ghost"
        onClick={() => setShowThinking(!showThinking)}
      >
        <div className="flex w-full items-center justify-between">
          <div className="flex items-center gap-2">
            <ChainOfThoughtStep
              className="font-normal"
              label={t.common.thinking}
              icon={LightbulbIcon}
            />
            <Badge variant="secondary" className="text-xs font-normal">
              Lead Agent
            </Badge>
          </div>
          <ChevronUp
            className={cn(
              "text-muted-foreground size-4",
              showThinking ? "" : "rotate-180",
            )}
          />
        </div>
      </Button>
      {showThinking && (
        <ChainOfThoughtContent className="px-4 pb-2">
          <ChainOfThoughtStep
            label={
              <MarkdownContent content={reasoningContent} isLoading={isStreaming} />
            }
          />
        </ChainOfThoughtContent>
      )}
    </ChainOfThought>
  );
}
