import type { AIMessage } from "@langchain/langgraph-sdk";
import {
  CheckCircleIcon,
  ChevronUp,
  ClipboardListIcon,
  Loader2Icon,
  LightbulbIcon,
  XCircleIcon,
} from "lucide-react";
import { useMemo, useState } from "react";
import { Streamdown } from "streamdown";

import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { Button } from "@/components/ui/button";
import { ShineBorder } from "@/components/ui/shine-border";
import { useI18n } from "@/core/i18n/hooks";
import { hasToolCalls } from "@/core/messages/utils";
import { useRehypeSplitWordsIntoSpans } from "@/core/rehype";
import { streamdownPluginsWithWordAnimation } from "@/core/streamdown";
import { useSubtask } from "@/core/tasks/context";
import { explainLastToolCall } from "@/core/tools/utils";
import { cn } from "@/lib/utils";

import { CitationLink } from "../citations/citation-link";
import { FlipDisplay } from "../flip-display";

import { MarkdownContent } from "./markdown-content";
import {
  convertToSteps,
  HIDDEN_TOOL_CALL_NAMES,
  ToolCall,
  type CoTStep,
} from "./message-group";

// Placeholder until C3 introduces SUBTASK_HIDDEN_TOOL_CALL_NAMES.
const SUBTASK_HIDDEN_TOOL_CALL_NAMES = HIDDEN_TOOL_CALL_NAMES;

export function SubtaskCard({
  className,
  taskId,
  isLoading,
}: {
  className?: string;
  taskId: string;
  isLoading: boolean;
}) {
  const { t } = useI18n();
  const [collapsed, setCollapsed] = useState(true);
  const rehypePlugins = useRehypeSplitWordsIntoSpans(isLoading);
  const task = useSubtask(taskId)!;
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
      className={cn("relative w-full gap-2 rounded-lg border py-0", className)}
      open={!collapsed}
    >
      <div
        className={cn(
          "ambilight z-[-1]",
          task.status === "in_progress" ? "enabled" : "",
        )}
      ></div>
      {task.status === "in_progress" && (
        <>
          <ShineBorder
            borderWidth={1.5}
            shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
          />
        </>
      )}
      <div className="bg-background/95 flex w-full flex-col rounded-lg">
        <div className="flex w-full items-center justify-between p-0.5">
          <Button
            className="w-full items-start justify-start text-left"
            variant="ghost"
            onClick={() => setCollapsed(!collapsed)}
          >
            <div className="flex w-full items-center justify-between">
              <ChainOfThoughtStep
                className="font-normal"
                label={
                  task.status === "in_progress" ? (
                    <Shimmer duration={3} spread={3}>
                      {task.description}
                    </Shimmer>
                  ) : (
                    task.description
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
                  {...streamdownPluginsWithWordAnimation}
                  components={{ a: CitationLink }}
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
              rehypePlugins={rehypePlugins}
            />
          )}
          {task.status === "completed" && task.result && (
            <ChainOfThoughtStep
              label={
                <span className="text-muted-foreground">
                  {t.subtasks.taskResult}
                </span>
              }
              icon={<CheckCircleIcon className="size-3" />}
            >
              <div className="pt-1">
                <MarkdownContent
                  content={task.result}
                  isLoading={false}
                  rehypePlugins={rehypePlugins}
                />
              </div>
            </ChainOfThoughtStep>
          )}
          {task.status === "failed" && (
            <ChainOfThoughtStep
              label={<div className="text-red-500">{task.error}</div>}
              icon={<XCircleIcon className="size-4 text-red-500" />}
            ></ChainOfThoughtStep>
          )}
        </ChainOfThoughtContent>
      </div>
    </ChainOfThought>
  );
}

function SubtaskCoTTimeline({
  messages,
  isLoading,
  rehypePlugins,
}: {
  messages: AIMessage[];
  isLoading: boolean;
  rehypePlugins: ReturnType<typeof useRehypeSplitWordsIntoSpans>;
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
          rehypePlugins={rehypePlugins}
        />
      ))}
    </>
  );
}

function CoTStepRenderer({
  step,
  messages,
  isLoading,
  rehypePlugins,
}: {
  step: CoTStep;
  messages: AIMessage[];
  isLoading: boolean;
  rehypePlugins: ReturnType<typeof useRehypeSplitWordsIntoSpans>;
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
            rehypePlugins={rehypePlugins}
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
            rehypePlugins={rehypePlugins}
          />
        }
      />
    );
  }
  // toolCall
  void messages;
  return <ToolCall key={step.id} {...step} isLoading={isLoading} />;
}
