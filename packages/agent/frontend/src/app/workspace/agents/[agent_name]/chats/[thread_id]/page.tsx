"use client";

import { BotIcon, PlusSquare } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useState } from "react";

import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import { AgentWelcome } from "@/components/workspace/agent-welcome";
import { AnalysisRail } from "@/components/workspace/analysis-rail";
import { ArtifactTrigger } from "@/components/workspace/artifacts";
import { ChatBox, useThreadChat } from "@/components/workspace/chats";
import { ExportTrigger } from "@/components/workspace/export-trigger";
import { InputBox } from "@/components/workspace/input-box";
import {
  MessageList,
  MESSAGE_LIST_DEFAULT_PADDING_BOTTOM,
  MESSAGE_LIST_FOLLOWUPS_EXTRA_PADDING_BOTTOM,
  INPUT_BOX_PADDING_BREATHING_ROOM_PX,
} from "@/components/workspace/messages";
import { ThreadContext } from "@/components/workspace/messages/context";
import { ThreadTitle } from "@/components/workspace/thread-title";
import { TodoList } from "@/components/workspace/todo-list";
import { TokenUsageIndicator } from "@/components/workspace/token-usage-indicator";
import { Tooltip } from "@/components/workspace/tooltip";
import { RunTraceWidget } from "@/components/workspace/trace";
import { useAgent } from "@/core/agents";
import { useI18n } from "@/core/i18n/hooks";
import { useNotification } from "@/core/notification/hooks";
import { useThreadSettings } from "@/core/settings";
import { useThreadStream } from "@/core/threads/hooks";
import { textOfMessage } from "@/core/threads/utils";
import { env } from "@/env";
import { cn } from "@/lib/utils";

export default function AgentChatPage() {
  const { t } = useI18n();
  const [showFollowups, setShowFollowups] = useState(false);
  const [inputBoxHeight, setInputBoxHeight] = useState<number>(0);
  const router = useRouter();

  const { agent_name } = useParams<{
    agent_name: string;
  }>();

  const { agent } = useAgent(agent_name);

  const { threadId, isNewThread, setIsNewThread } = useThreadChat();
  const [welcomeDismissed, setWelcomeDismissed] = useState(false);
  const [settings, setSettings] = useThreadSettings(threadId);

  const isCenteredLayout = isNewThread && !welcomeDismissed;

  const { showNotification } = useNotification();
  const { thread: streamThread, sendMessage } = useThreadStream({
    threadId: isNewThread ? undefined : threadId,
    context: { ...settings.context, agent_name: agent_name },
    onStart: () => {
      setIsNewThread(false);
      // ! Important: Never use next.js router for navigation in this case, otherwise it will cause the thread to re-mount and lose all states. Use native history API instead.
      history.replaceState(
        null,
        "",
        `/workspace/agents/${agent_name}/chats/${threadId}`,
      );
    },
    onFinish: (state) => {
      if (document.hidden || !document.hasFocus()) {
        let body = "Conversation finished";
        const lastMessage = state.messages[state.messages.length - 1];
        if (lastMessage) {
          const textContent = textOfMessage(lastMessage);
          if (textContent) {
            body =
              textContent.length > 200
                ? textContent.substring(0, 200) + "..."
                : textContent;
          }
        }
        showNotification(state.title, { body });
      }
    },
  });
  const thread = streamThread;

  const handleSubmit = useCallback(
    (message: PromptInputMessage) => {
      if (
        !message.text.trim() &&
        (!message.files || message.files.length === 0)
      ) {
        return;
      }
      setWelcomeDismissed(true);
      const sendPromise = sendMessage(threadId, message, {
        agent_name,
      }).catch((err) => {
        setWelcomeDismissed(false);
        throw err;
      });
      if (message.files.length > 0) {
        return sendPromise;
      }
      void sendPromise;
    },
    [sendMessage, threadId, agent_name],
  );

  const handleStop = useCallback(async () => {
    await thread.stop();
  }, [thread]);

  const handleInputBoxHeightChange = useCallback((heightPx: number) => {
    setInputBoxHeight(heightPx);
  }, []);

  // Reserve bottom space in the message stream equal to the floating input
  // box's measured height plus breathing room, so the last message /
  // decision-card options are never covered. Before the first measurement
  // lands, fall back to the static default so a short conversation renders
  // without a flash of overlap. Mirrors the standard chats route (spec §一:
  // the #219 fix was missing here — catastrophic-forgetting across the two
  // chat routes). When followups are visible, add the dedicated extra.
  const messageListPaddingBottom =
    (inputBoxHeight > 0
      ? inputBoxHeight + INPUT_BOX_PADDING_BREATHING_ROOM_PX
      : MESSAGE_LIST_DEFAULT_PADDING_BOTTOM) +
    (showFollowups ? MESSAGE_LIST_FOLLOWUPS_EXTRA_PADDING_BOTTOM : 0);

  return (
    <ThreadContext.Provider value={{ thread }}>
      <ChatBox threadId={threadId}>
        {/* spec 2026-06-26-conversation-gallery-empty §四点五 视口高度链（同 chats 路由）：
            size-full 的 h-full 在父链无 definite height 时回退视口高，底部被裁；改 flex-1
            min-h-0 w-full 受 ResizablePanel flex 约束到真实可用高。 */}
        <div className="relative flex w-full min-h-0 flex-1 justify-between">
          <header
            className={cn(
              "absolute top-0 right-0 left-0 z-30 flex h-12 shrink-0 items-center gap-2 px-4",
              isNewThread
                ? "bg-background/0 backdrop-blur-none"
                : "bg-background/80 shadow-xs backdrop-blur",
            )}
          >
            {/* Agent badge */}
            <div className="flex shrink-0 items-center gap-1.5 rounded-md border px-2 py-1">
              <BotIcon className="text-primary h-3.5 w-3.5" />
              <span className="text-xs font-medium">
                {agent?.name ?? agent_name}
              </span>
            </div>

            <div className="flex w-full items-center text-sm font-medium">
              <ThreadTitle threadId={threadId} thread={thread} />
            </div>
            <div className="mr-4 flex items-center">
              <Tooltip content={t.agents.newChat}>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    router.push(`/workspace/agents/${agent_name}/chats/new`);
                  }}
                >
                  <PlusSquare /> {t.agents.newChat}
                </Button>
              </Tooltip>
              <TokenUsageIndicator messages={thread.messages} />
              <RunTraceWidget messages={thread.messages} />
              <ExportTrigger threadId={threadId} />
              <ArtifactTrigger />
            </div>
          </header>

          <main className="flex min-h-0 max-w-full grow flex-col">
            {/* spec#4 分析进度轨：常驻 sticky 条（同源 spec#2 useRunTrace，前端推导）。 */}
            {!isNewThread && (
              <div className="pointer-events-none sticky top-14 z-20 mt-14 w-full">
                <div className="pointer-events-auto mx-auto w-full max-w-(--container-width-md) px-4">
                  <div className="bg-background/80 rounded-md px-2 py-1.5 backdrop-blur">
                    <AnalysisRail messages={thread.messages} />
                  </div>
                </div>
              </div>
            )}
            <div className="flex size-full justify-center">
              <MessageList
                className={cn("size-full", !isNewThread && "pt-4")}
                threadId={threadId}
                thread={thread}
                paddingBottom={messageListPaddingBottom}
                onSelectClarificationOption={(optionText) => {
                  void handleSubmit({ text: optionText, files: [] });
                }}
              />
            </div>

            <div className="absolute right-0 bottom-0 left-0 z-30 flex justify-center px-4">
              <div
                className={cn(
                  "relative w-full",
                  isCenteredLayout && "-translate-y-[calc(50vh-96px)]",
                  isCenteredLayout
                    ? "max-w-(--container-width-sm)"
                    : "max-w-(--container-width-md)",
                )}
              >
                <div className="absolute -top-4 right-0 left-0 z-0">
                  <div className="absolute right-0 bottom-0 left-0">
                    <TodoList
                      className="bg-background/5"
                      todos={thread.values.todos ?? []}
                      hidden={
                        !thread.values.todos || thread.values.todos.length === 0
                      }
                    />
                  </div>
                </div>

                <InputBox
                  className={cn("bg-background/5 w-full -translate-y-4")}
                  isNewThread={isNewThread}
                  threadId={threadId}
                  autoFocus={isNewThread}
                  status={
                    thread.error
                      ? "error"
                      : thread.isLoading
                        ? "streaming"
                        : "ready"
                  }
                  context={settings.context}
                  extraHeader={
                    isCenteredLayout && (
                      <AgentWelcome agent={agent} agentName={agent_name} />
                    )
                  }
                  disabled={env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true"}
                  onContextChange={(context) => setSettings("context", context)}
                  onFollowupsVisibilityChange={setShowFollowups}
                  onHeightChange={handleInputBoxHeightChange}
                  onSubmit={handleSubmit}
                  onStop={handleStop}
                />
                {env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true" && (
                  <div className="text-muted-foreground/67 w-full translate-y-12 text-center text-xs">
                    {t.common.notAvailableInDemoMode}
                  </div>
                )}
              </div>
            </div>
          </main>
        </div>
      </ChatBox>
    </ThreadContext.Provider>
  );
}
