import type { Message } from "@langchain/langgraph-sdk";
import {
  BookOpenTextIcon,
  ChevronUp,
  FolderOpenIcon,
  GlobeIcon,
  LightbulbIcon,
  ListTodoIcon,
  MessageCircleQuestionMarkIcon,
  NotebookPenIcon,
  SearchIcon,
  SquareTerminalIcon,
  WrenchIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtSearchResult,
  ChainOfThoughtSearchResults,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { CodeBlock } from "@/components/ai-elements/code-block";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/core/i18n/hooks";
import {
  extractReasoningContentFromMessage,
  findToolCallResult,
} from "@/core/messages/utils";
import { getStageBroadcastForBash } from "@/core/tools/stage-broadcast";
import { extractTitleFromMarkdown } from "@/core/utils/markdown";
import { env } from "@/env";
import { cn } from "@/lib/utils";

import { useArtifacts } from "../artifacts";
import { FlipDisplay } from "../flip-display";
import { Tooltip } from "../tooltip";

import { MarkdownContent } from "./markdown-content";

export function MessageGroup({
  className,
  messages,
  isLoading = false,
}: {
  className?: string;
  messages: Message[];
  isLoading?: boolean;
}) {
  const { t } = useI18n();
  const [showAbove, setShowAbove] = useState(
    env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true",
  );
  // 1A: 流式期间默认展开思考面板，与最终答案的 ReasoningPanel（默认展开）行为一致。
  // 用户手动折叠后不再跟随 isLoading，尊重用户操作。
  const thinkingToggledByUser = useRef(false);
  const [showLastThinking, setShowLastThinking] = useState(
    env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true" || isLoading,
  );
  useEffect(() => {
    if (isLoading && !thinkingToggledByUser.current) {
      setShowLastThinking(true);
    }
  }, [isLoading]);
  const steps = useMemo(() => convertToSteps(messages), [messages]);
  const lastToolCallStep = useMemo(() => {
    const filteredSteps = steps.filter((step) => step.type === "toolCall");
    return filteredSteps[filteredSteps.length - 1];
  }, [steps]);
  const aboveLastToolCallSteps = useMemo(() => {
    if (lastToolCallStep) {
      const index = steps.indexOf(lastToolCallStep);
      return steps.slice(0, index);
    }
    return [];
  }, [lastToolCallStep, steps]);
  const lastReasoningStep = useMemo(() => {
    if (lastToolCallStep) {
      const index = steps.indexOf(lastToolCallStep);
      return steps.slice(index + 1).find((step) => step.type === "reasoning");
    } else {
      const filteredSteps = steps.filter((step) => step.type === "reasoning");
      return filteredSteps[filteredSteps.length - 1];
    }
  }, [lastToolCallStep, steps]);
  // Guard: if there is nothing renderable inside (no tool calls, no reasoning,
  // no "above" history), the ChainOfThought wrapper would still emit an empty
  // `<div class="not-prose w-full gap-2 rounded-lg border p-0.5">` — visually a
  // 4-6px tall horizontal bar with a border. Skip the wrapper entirely in that
  // case.
  //
  // Exception — while a run is streaming (`isLoading`), NEVER collapse the
  // group to null. A frame with no renderable steps is a transient state
  // between two LLM turns (tool result just landed; next AIMessage chunk
  // hasn't arrived yet). If we return null here the entire group disappears
  // and reappears, giving the user the impression that the response vanished.
  // Render a lightweight "thinking" placeholder instead.
  const hasAnythingToRender =
    aboveLastToolCallSteps.length > 0 ||
    lastToolCallStep !== undefined ||
    lastReasoningStep !== undefined;
  if (!hasAnythingToRender) {
    if (!isLoading) {
      return null;
    }
    return (
      <div
        className={cn(
          "text-muted-foreground flex items-center gap-2 px-2 py-1 text-sm",
          className,
        )}
      >
        <LightbulbIcon className="size-4 animate-pulse" />
        <span>{t.common.thinking}</span>
      </div>
    );
  }
  return (
    <ChainOfThought
      className={cn("w-full gap-2 rounded-lg border p-0.5", className)}
      open={true}
    >
      {aboveLastToolCallSteps.length > 0 && (
        <Button
          key="above"
          className="w-full items-start justify-start text-left"
          variant="ghost"
          onClick={() => setShowAbove(!showAbove)}
        >
          <ChainOfThoughtStep
            label={
              <span className="opacity-60">
                {showAbove
                  ? t.toolCalls.lessSteps
                  : t.toolCalls.moreSteps(aboveLastToolCallSteps.length)}
              </span>
            }
            icon={
              <ChevronUp
                className={cn(
                  "size-4 opacity-60 transition-transform duration-base ease-brand-in-out",
                  showAbove ? "rotate-180" : "",
                )}
              />
            }
          ></ChainOfThoughtStep>
        </Button>
      )}
      {lastToolCallStep && (
        <ChainOfThoughtContent className="px-4 pb-2">
          {showAbove &&
            aboveLastToolCallSteps.map((step) =>
              step.type === "reasoning" ? (
                <ChainOfThoughtStep
                  key={step.id}
                  label={
                    <MarkdownContent
                      content={step.reasoning ?? ""}
                      isLoading={isLoading}
                    />
                  }
                ></ChainOfThoughtStep>
              ) : step.type === "toolCall" ? (
                <ToolCall key={step.id} {...step} isLoading={isLoading} />
              ) : null,
            )}
          {lastToolCallStep && (
            <FlipDisplay uniqueKey={lastToolCallStep.id ?? ""}>
              <ToolCall
                key={lastToolCallStep.id}
                {...lastToolCallStep}
                isLast={true}
                isLoading={isLoading}
              />
            </FlipDisplay>
          )}
        </ChainOfThoughtContent>
      )}
      {lastReasoningStep && (
        <>
          <Button
            key={lastReasoningStep.id}
            className="w-full items-start justify-start text-left"
            variant="ghost"
            onClick={() => {
              thinkingToggledByUser.current = true;
              setShowLastThinking(!showLastThinking);
            }}
          >
            <div className="flex w-full items-center justify-between">
              <div className="flex items-center gap-2">
                <ChainOfThoughtStep
                  className="font-normal"
                  label={t.common.thinking}
                  icon={LightbulbIcon}
                ></ChainOfThoughtStep>
                <Badge variant="secondary" className="text-xs font-normal">
                  Lead Agent
                </Badge>
              </div>
              <div>
                <ChevronUp
                  className={cn(
                    "text-muted-foreground size-4",
                    showLastThinking ? "" : "rotate-180",
                  )}
                />
              </div>
            </div>
          </Button>
          {showLastThinking && (
            <ChainOfThoughtContent className="px-4 pb-2">
              <ChainOfThoughtStep
                key={lastReasoningStep.id}
                label={
                  <MarkdownContent
                    content={lastReasoningStep.reasoning ?? ""}
                    isLoading={isLoading}
                  />
                }
              ></ChainOfThoughtStep>
            </ChainOfThoughtContent>
          )}
        </>
      )}
    </ChainOfThought>
  );
}

export function ToolCall({
  id,
  messageId,
  name,
  args,
  result,
  isLast = false,
  isLoading = false,
}: {
  id?: string;
  messageId?: string;
  name: string;
  args: Record<string, unknown>;
  result?: string | Record<string, unknown>;
  isLast?: boolean;
  isLoading?: boolean;
}) {
  const { t } = useI18n();
  const { setOpen, autoOpen, autoSelect, selectedArtifact, select } =
    useArtifacts();

  if (name === "web_search") {
    let label: React.ReactNode = t.toolCalls.searchForRelatedInfo;
    if (typeof args.query === "string") {
      label = t.toolCalls.searchOnWebFor(args.query);
    }
    return (
      <ChainOfThoughtStep key={id} label={label} icon={SearchIcon}>
        {Array.isArray(result) && (
          <ChainOfThoughtSearchResults>
            {result.map((item) => (
              <ChainOfThoughtSearchResult key={item.url}>
                <a href={item.url} target="_blank" rel="noopener noreferrer">
                  {item.title}
                </a>
              </ChainOfThoughtSearchResult>
            ))}
          </ChainOfThoughtSearchResults>
        )}
      </ChainOfThoughtStep>
    );
  } else if (name === "image_search") {
    let label: React.ReactNode = t.toolCalls.searchForRelatedImages;
    if (typeof args.query === "string") {
      label = t.toolCalls.searchForRelatedImagesFor(args.query);
    }
    const results = (
      result as {
        results: {
          source_url: string;
          thumbnail_url: string;
          image_url: string;
          title: string;
        }[];
      }
    )?.results;
    return (
      <ChainOfThoughtStep key={id} label={label} icon={SearchIcon}>
        {Array.isArray(results) && (
          <ChainOfThoughtSearchResults>
            {Array.isArray(results) &&
              results.map((item) => (
                <Tooltip key={item.image_url} content={item.title}>
                  <a
                    className="size-24 overflow-hidden rounded-lg object-cover"
                    href={item.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <div className="bg-accent size-24">
                      <img
                        className="size-full object-cover"
                        src={item.thumbnail_url}
                        alt={item.title}
                        width={100}
                        height={100}
                      />
                    </div>
                  </a>
                </Tooltip>
              ))}
          </ChainOfThoughtSearchResults>
        )}
      </ChainOfThoughtStep>
    );
  } else if (name === "web_fetch") {
    const url = (args as { url: string })?.url;
    let title = url;
    if (typeof result === "string") {
      const potentialTitle = extractTitleFromMarkdown(result);
      if (potentialTitle && potentialTitle.toLowerCase() !== "untitled") {
        title = potentialTitle;
      }
    }
    return (
      <ChainOfThoughtStep
        key={id}
        label={t.toolCalls.viewWebPage}
        icon={GlobeIcon}
      >
        <ChainOfThoughtSearchResult>
          {url && (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="cursor-pointer"
            >
              {title}
            </a>
          )}
        </ChainOfThoughtSearchResult>
      </ChainOfThoughtStep>
    );
  } else if (name === "ls") {
    let description: string | undefined = (args as { description: string })
      ?.description;
    if (!description) {
      description = t.toolCalls.listFolder;
    }
    const path: string | undefined = (args as { path: string })?.path;
    return (
      <ChainOfThoughtStep key={id} label={description} icon={FolderOpenIcon}>
        {path && (
          <ChainOfThoughtSearchResult className="cursor-pointer">
            {path}
          </ChainOfThoughtSearchResult>
        )}
      </ChainOfThoughtStep>
    );
  } else if (name === "read_file") {
    let description: string | undefined = (args as { description: string })
      ?.description;
    if (!description) {
      description = t.toolCalls.readFile;
    }
    const { path } = args as { path: string; content: string };
    return (
      <ChainOfThoughtStep key={id} label={description} icon={BookOpenTextIcon}>
        {path && (
          <ChainOfThoughtSearchResult className="cursor-pointer">
            {path}
          </ChainOfThoughtSearchResult>
        )}
      </ChainOfThoughtStep>
    );
  } else if (name === "write_file" || name === "str_replace") {
    let description: string | undefined = (args as { description: string })
      ?.description;
    if (!description) {
      description = t.toolCalls.writeFile;
    }
    const path: string | undefined = (args as { path: string })?.path;
    if (isLoading && isLast && autoOpen && autoSelect && path) {
      setTimeout(() => {
        const url = new URL(
          `write-file:${path}?message_id=${messageId}&tool_call_id=${id}`,
        ).toString();
        if (selectedArtifact === url) {
          return;
        }
        select(url, true);
        setOpen(true);
      }, 100);
    }

    return (
      <ChainOfThoughtStep
        key={id}
        className="cursor-pointer"
        label={description}
        icon={NotebookPenIcon}
        onClick={() => {
          select(
            new URL(
              `write-file:${path}?message_id=${messageId}&tool_call_id=${id}`,
            ).toString(),
          );
          setOpen(true);
        }}
      >
        {path && (
          <ChainOfThoughtSearchResult className="cursor-pointer">
            {path}
          </ChainOfThoughtSearchResult>
        )}
      </ChainOfThoughtStep>
    );
  } else if (name === "bash") {
    const command: string | undefined = (args as { command: string })?.command;
    const description: string | undefined = (args as { description: string })
      ?.description;
    const stageBroadcast = getStageBroadcastForBash(command ?? "", t);
    const label = stageBroadcast ?? description ?? t.toolCalls.executeCommand;
    return (
      <ChainOfThoughtStep
        key={id}
        label={label}
        icon={SquareTerminalIcon}
      >
        {command && (
          <CodeBlock
            className="mx-0 cursor-pointer border-none px-0"
            showLineNumbers={false}
            language="bash"
            code={command}
          />
        )}
      </ChainOfThoughtStep>
    );
  } else if (name === "ask_clarification") {
    return (
      <ChainOfThoughtStep
        key={id}
        label={t.toolCalls.stageBroadcast.askClarification}
        icon={MessageCircleQuestionMarkIcon}
      ></ChainOfThoughtStep>
    );
  } else if (name === "write_todos") {
    return (
      <ChainOfThoughtStep
        key={id}
        label={t.toolCalls.writeTodos}
        icon={ListTodoIcon}
      ></ChainOfThoughtStep>
    );
  } else if (name === "inspect_uploaded_file") {
    // Show filepath + compact summary of what was found (sheets, columns, treatment).
    const description: string | undefined = (args as { description: string })
      ?.description;
    const filepath: string | undefined = (args as { filepath: string })?.filepath;
    const summary = summarizeInspectResult(result);
    return (
      <ChainOfThoughtStep
        key={id}
        label={description ?? t.toolCalls.useTool(name)}
        icon={SearchIcon}
      >
        {filepath && (
          <ChainOfThoughtSearchResult className="cursor-pointer">
            {filepath}
          </ChainOfThoughtSearchResult>
        )}
        {summary && (
          <div className="text-muted-foreground mt-1 text-xs">{summary}</div>
        )}
      </ChainOfThoughtStep>
    );
  } else if (name === "prep_metric_plan") {
    // Show what was planned: paradigm, metric count, subject count.
    const description: string | undefined = (args as { description: string })
      ?.description;
    const summary = summarizePrepResult(result);
    return (
      <ChainOfThoughtStep
        key={id}
        label={description ?? t.toolCalls.useTool(name)}
        icon={ListTodoIcon}
      >
        {summary && (
          <div className="text-muted-foreground mt-1 text-xs">{summary}</div>
        )}
      </ChainOfThoughtStep>
    );
  } else if (name === "set_experiment_paradigm") {
    // Show which paradigm was set.
    const description: string | undefined = (args as { description: string })
      ?.description;
    const paradigm = (args as { paradigm?: string })?.paradigm;
    const ev19Template = (args as { ev19_template?: string })?.ev19_template;
    return (
      <ChainOfThoughtStep
        key={id}
        label={description ?? t.toolCalls.useTool(name)}
        icon={WrenchIcon}
      >
        {(paradigm ?? ev19Template) && (
          <div className="text-muted-foreground mt-1 text-xs">
            {[paradigm, ev19Template].filter(Boolean).join(" · ")}
          </div>
        )}
      </ChainOfThoughtStep>
    );
  } else {
    const description: string | undefined = (args as { description: string })
      ?.description;
    return (
      <ChainOfThoughtStep
        key={id}
        label={description ?? t.toolCalls.useTool(name)}
        icon={WrenchIcon}
      ></ChainOfThoughtStep>
    );
  }
}

interface GenericCoTStep<T extends string = string> {
  id?: string;
  messageId?: string;
  type: T;
}

export interface CoTReasoningStep extends GenericCoTStep<"reasoning"> {
  reasoning: string | null;
}

export interface CoTToolCallStep extends GenericCoTStep<"toolCall"> {
  name: string;
  args: Record<string, unknown>;
  result?: string;
}

export interface CoTTextStep extends GenericCoTStep<"text"> {
  content: string;
}

export type CoTStep = CoTReasoningStep | CoTToolCallStep | CoTTextStep;

/**
 * Lead-agent timeline: only hide tools whose calls genuinely carry zero
 * signal for the user. Showing everything else (file探索、bash 诊断、领域
 * 工具) is the visible evidence that the agent is doing real work — without
 * it, users see only thinking dots and assume the run has hung.
 *
 * `str_replace` / `write_file` stay hidden because lead uses them purely to
 * shuffle handoff JSON between subagents — opaque internal plumbing.
 */
export const LEAD_HIDDEN_TOOL_CALL_NAMES = new Set<string>([
  "str_replace",
  "write_file",
]);

/**
 * Subtask timeline (SubtaskCard expanded state): only hide pure filesystem
 * noise. KEEP ethoinsight domain tools visible — those ARE the "expert at
 * work" steps users want to see. `bash` stays visible too because subagents
 * use it for diagnostic commands worth showing.
 */
export const SUBTASK_HIDDEN_TOOL_CALL_NAMES = new Set<string>([
  "read_file",
  "write_file",
  "str_replace",
  "ls",
  "glob",
  "grep",
]);

export function convertToSteps(
  messages: Message[],
  hiddenToolNames: Set<string> = LEAD_HIDDEN_TOOL_CALL_NAMES,
  includeText = false,
): CoTStep[] {
  const steps: CoTStep[] = [];
  for (const message of messages) {
    if (message.type === "ai") {
      const reasoning = extractReasoningContentFromMessage(message);
      if (reasoning) {
        const step: CoTReasoningStep = {
          id: message.id ? `${message.id}-reasoning` : undefined,
          messageId: message.id,
          type: "reasoning",
          reasoning: extractReasoningContentFromMessage(message),
        };
        steps.push(step);
      }
      if (
        includeText &&
        typeof message.content === "string" &&
        message.content.trim()
      ) {
        steps.push({
          id: message.id ? `${message.id}-text` : undefined,
          messageId: message.id,
          type: "text",
          content: message.content,
        });
      }
      for (const tool_call of message.tool_calls ?? []) {
        // `task` is a subagent dispatch — rendered as a dedicated subtask
        // card elsewhere, not in the CoT timeline.
        if (tool_call.name === "task") {
          continue;
        }
        if (hiddenToolNames.has(tool_call.name)) {
          continue;
        }
        const step: CoTToolCallStep = {
          id: tool_call.id,
          messageId: message.id,
          type: "toolCall",
          name: tool_call.name,
          args: tool_call.args,
        };
        const toolCallId = tool_call.id;
        if (toolCallId) {
          const toolCallResult = findToolCallResult(toolCallId, messages);
          if (toolCallResult) {
            try {
              const json = JSON.parse(toolCallResult);
              step.result = json;
            } catch {
              step.result = toolCallResult;
            }
          }
        }
        steps.push(step);
      }
    }
  }
  return steps;
}

// ── ToolCall result summarisers ──────────────────────────────────────
//
// These extract human-readable one-liners from structured tool results.
// The step.result field is already populated by convertToSteps (using
// findToolCallResult → JSON.parse), so we only need to read known keys.
//
// Field shapes verified against actual tool returns (2026-06-04 audit):
// - inspect_uploaded_file → {sheets, columns, raw_metadata, …}  (NO row_count)
// - prep_metric_plan      → {status, plan_summary: {paradigm, metric_count, subject_count, …}}
//   (metric_count is nested inside plan_summary, NOT a top-level key)

function summarizeInspectResult(
  result: string | Record<string, unknown> | undefined,
): string | null {
  if (!result || typeof result !== "object" || Array.isArray(result)) return null;
  const parts: string[] = [];

  if (result.sheets && typeof result.sheets === "object") {
    const sheetNames = Object.keys(result.sheets as Record<string, unknown>);
    if (sheetNames.length > 0) {
      parts.push(`${sheetNames.length} sheet(s): ${sheetNames.join(", ")}`);
    }
  }
  if (Array.isArray(result.columns)) {
    parts.push(`${result.columns.length} columns`);
  }
  if (result.raw_metadata && typeof result.raw_metadata === "object") {
    const meta = result.raw_metadata as Record<string, unknown>;
    const treatment = meta.Treatment;
    if (typeof treatment === "string") parts.push(`Treatment: ${treatment}`);
    const groups = meta.Group ?? meta.Groups;
    if (typeof groups === "string") parts.push(`Groups: ${groups}`);
  }

  return parts.length > 0 ? parts.join(" · ") : null;
}

function summarizePrepResult(
  result: string | Record<string, unknown> | undefined,
): string | null {
  if (!result || typeof result !== "object" || Array.isArray(result)) return null;
  const planSummary = result.plan_summary;
  if (!planSummary || typeof planSummary !== "object" || Array.isArray(planSummary)) return null;
  const ps = planSummary as Record<string, unknown>;

  const parts: string[] = [];
  if (typeof ps.paradigm === "string") {
    parts.push(ps.paradigm);
  }
  if (typeof ps.metric_count === "number") {
    parts.push(`${ps.metric_count} metrics`);
  }
  if (typeof ps.subject_count === "number") {
    parts.push(`${ps.subject_count} subjects`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}
