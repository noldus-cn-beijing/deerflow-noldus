import type { AIMessage } from "@langchain/langgraph-sdk";
import {
  CheckCircleIcon,
  ChevronUp,
  ClipboardListIcon,
  Loader2Icon,
  LightbulbIcon,
  XCircleIcon,
} from "lucide-react";
import { memo, useEffect, useMemo, useRef, useState } from "react";
import { type Components, Streamdown } from "streamdown";

import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { FeedbackButtons } from "@/components/feedback/feedback-buttons";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/core/i18n/hooks";
import { hasToolCalls } from "@/core/messages/utils";
import { useActiveStageNarration } from "@/core/stages/context";
import { streamdownPlugins } from "@/core/streamdown";
import { useSubtask } from "@/core/tasks/context";
import { explainLastToolCall } from "@/core/tools/utils";
import { cn } from "@/lib/utils";

import { CitationLink } from "../citations/citation-link";
import { FlipDisplay } from "../flip-display";

import { MarkdownContent } from "./markdown-content";
import {
  convertToSteps,
  SUBTASK_HIDDEN_TOOL_CALL_NAMES,
  ToolCall,
  type CoTStep,
} from "./message-group";

// spec 2026-06-29-streaming-render-perf Step 3 — hoisted to a module constant so
// the `components` prop identity is stable across renders (an inline
// `{{ a: CitationLink }}` literal here was a new object every render, which
// would defeat streamdown's block-level memo on the prompt render).
const PROMPT_STREAMDOWN_COMPONENTS = { a: CitationLink } as Components;

// Phase0#7 Step 2 — memoized so an unrelated parent re-render (e.g. a
// streaming token landing in a sibling subagent group) does not re-render
// this card's subtree. Props (taskId/threadId/messageRunIds) are stable
// across unrelated renders now that MessageList memoizes the groups (Step 1).
// The card's own state (task status from useSubtask) still updates normally.
export const SubtaskCard = memo(function SubtaskCard({
  className,
  taskId,
  threadId,
  messageRunIds,
}: {
  className?: string;
  taskId: string;
  threadId?: string;
  messageRunIds?: Map<string, string>;
}) {
  const { t } = useI18n();
  const task = useSubtask(taskId)!;
  // A2: stage narration from A1 backend events (replaces frontend query-table
  // translation in stage-broadcast.ts). Source = backend only, never inferred.
  const activeNarration = useActiveStageNarration();
  // 2A: subagent 运行中默认展开卡片，让用户看到 timeline（含 reasoning step）。
  // 用户手动折叠后不再跟随 status 变化，尊重用户操作。
  const cardToggledByUser = useRef(false);
  const [collapsed, setCollapsed] = useState(task.status !== "in_progress");
  useEffect(() => {
    if (task.status === "in_progress" && !cardToggledByUser.current) {
      setCollapsed(false);
    }
  }, [task.status]);
  const icon = useMemo(() => {
    if (task.status === "completed") {
      return <CheckCircleIcon className="size-3" />;
    } else if (task.status === "failed") {
      return <XCircleIcon className="size-3 text-red-500" />;
    } else if (task.status === "in_progress") {
      return <Loader2Icon className="size-3 animate-spin" />;
    }
  }, [task.status]);
  return (
    <ChainOfThought
      className={cn(
        "relative w-full gap-2 rounded-lg border py-0 transition-colors",
        task.status === "in_progress" &&
          "border-brand/50 animate-pulse-soft",
        className,
      )}
      open={!collapsed}
    >
      <div className="bg-background/95 flex w-full flex-col rounded-lg">
        <div className="flex w-full items-center justify-between p-0.5">
          <Button
            className="w-full items-start justify-start text-left"
            variant="ghost"
            onClick={() => {
              cardToggledByUser.current = true;
              setCollapsed(!collapsed);
            }}
          >
            <div className="flex w-full items-center justify-between">
              <ChainOfThoughtStep
                className="font-normal"
                label={
                  task.status === "completed" ? (
                    <span className="text-muted-foreground">
                      {t.subtasks.completed}
                    </span>
                  ) : task.status === "failed" ? (
                    <span className="text-red-500/67">
                      {t.subtasks.failed}
                    </span>
                  ) : (
                    <Shimmer duration={3} spread={3}>
                      {activeNarration ?? t.subtasks.in_progress}
                    </Shimmer>
                  )
                }
                icon={<ClipboardListIcon />}
              ></ChainOfThoughtStep>
              <div className="flex items-center gap-1">
                {collapsed && (
                  <div
                    className={cn(
                      "text-muted-foreground flex items-center gap-1 text-xs font-normal",
                      task.status === "failed" ? "text-red-500 opacity-67" : "",
                    )}
                  >
                    {icon}
                    <FlipDisplay
                      className="max-w-[420px] truncate pb-1"
                      uniqueKey={task.latestMessage?.id ?? ""}
                    >
                      {task.status === "in_progress" &&
                      task.latestMessage &&
                      hasToolCalls(task.latestMessage)
                        ? explainLastToolCall(task.latestMessage, t)
                        : task.status === "completed" && task.result
                          ? // Show the first line of the result as a summary.
                            // `||` (not `??`) is intentional: an empty/whitespace
                            // first line must fall back to the status label, and
                            // `??` would only catch null/undefined, leaving "".
                            // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing
                            task.result.split("\n")[0]?.trim() || t.subtasks[task.status]
                          : t.subtasks[task.status]}
                    </FlipDisplay>
                  </div>
                )}
                <ChevronUp
                  className={cn(
                    "text-muted-foreground size-4",
                    !collapsed ? "" : "rotate-180",
                  )}
                />
              </div>
            </div>
          </Button>
        </div>
        {task.status === "completed" && task.result && (
          <div className="border-border/60 bg-muted/30 mx-3 mb-3 rounded-md border px-3 py-2">
            <div className="text-muted-foreground mb-1 flex items-center gap-1.5 text-xs font-medium">
              <LightbulbIcon className="size-3" />
              <span>{t.subtasks.taskResult}</span>
            </div>
            <MarkdownContent content={task.result} isLoading={false} />
          </div>
        )}
        <ChainOfThoughtContent className="px-4 pb-4">
          {task.prompt && (
            <ChainOfThoughtStep
              label={
                <span className="text-muted-foreground">
                  {t.subtasks.taskDescription}
                </span>
              }
            >
              <div className="pt-1">
                <Streamdown
                  {...streamdownPlugins}
                  components={PROMPT_STREAMDOWN_COMPONENTS}
                >
                  {task.prompt}
                </Streamdown>
              </div>
            </ChainOfThoughtStep>
          )}
          {task.messages.length > 0 && (
            <SubtaskCoTTimeline
              messages={task.messages}
              isLoading={task.status === "in_progress"}
            />
          )}
          {task.status === "failed" && (
            <ChainOfThoughtStep
              label={<div className="text-red-500">{task.error}</div>}
              icon={<XCircleIcon className="size-4 text-red-500" />}
            ></ChainOfThoughtStep>
          )}
        </ChainOfThoughtContent>
        {task.status === "completed" && threadId && (() => {
          const syntheticId = `subtask-${taskId}`;
          const runId = messageRunIds?.get(syntheticId);
          if (!runId) return null; // subtask 上下文暂缺 run_id，不渲染反馈
          return (
            <FeedbackButtons
              threadId={threadId}
              runId={runId}
              messageId={syntheticId}
              className="px-4 pb-3"
            />
          );
        })()}
      </div>
    </ChainOfThought>
  );
});

function SubtaskCoTTimeline({
  messages,
  isLoading,
}: {
  messages: AIMessage[];
  isLoading: boolean;
}) {
  const steps = useMemo(
    () => convertToSteps(messages, SUBTASK_HIDDEN_TOOL_CALL_NAMES, true),
    [messages],
  );
  if (steps.length === 0) return null;
  return (
    <>
      {steps.map((step) => (
        <CoTStepRenderer
          key={step.id}
          step={step}
          messages={messages}
          isLoading={isLoading}
        />
      ))}
    </>
  );
}

function CoTStepRenderer({
  step,
  messages,
  isLoading,
}: {
  step: CoTStep;
  messages: AIMessage[];
  isLoading: boolean;
}) {
  if (step.type === "reasoning") {
    return (
      <ChainOfThoughtStep
        key={step.id}
        icon={LightbulbIcon}
        label={
          <MarkdownContent
            content={step.reasoning ?? ""}
            isLoading={isLoading}
          />
        }
      />
    );
  }
  if (step.type === "text") {
    return (
      <ChainOfThoughtStep
        key={step.id}
        label={
          <MarkdownContent
            content={step.content}
            isLoading={isLoading}
          />
        }
      />
    );
  }
  // toolCall
  void messages;
  return <ToolCall key={step.id} {...step} isLoading={isLoading} />;
}
