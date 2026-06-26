"use client";

import { useCallback, useEffect, useState } from "react";

import { type PromptInputMessage } from "@/components/ai-elements/prompt-input";
import { AnalysisRail } from "@/components/workspace/analysis-rail";
import { ArtifactTrigger } from "@/components/workspace/artifacts";
import {
  ChatBox,
  useSpecificChatMode,
  useThreadChat,
} from "@/components/workspace/chats";
import { ExportTrigger } from "@/components/workspace/export-trigger";
import { InputBox } from "@/components/workspace/input-box";
import {
  MessageList,
  MESSAGE_LIST_DEFAULT_PADDING_BOTTOM,
  MESSAGE_LIST_FOLLOWUPS_EXTRA_PADDING_BOTTOM,
} from "@/components/workspace/messages";
import { ThreadContext } from "@/components/workspace/messages/context";
import { ThreadTitle } from "@/components/workspace/thread-title";
import { TodoList } from "@/components/workspace/todo-list";
import { TokenUsageIndicator } from "@/components/workspace/token-usage-indicator";
import { RunTraceWidget } from "@/components/workspace/trace";
import { Welcome } from "@/components/workspace/welcome";
import { useI18n } from "@/core/i18n/hooks";
import { useNotification } from "@/core/notification/hooks";
import { useThreadSettings } from "@/core/settings";
import { useThreadStream } from "@/core/threads/hooks";
import { textOfMessage } from "@/core/threads/utils";
import { env } from "@/env";
import { cn } from "@/lib/utils";

// Breathing room added on top of the measured input-box height so the last
// message scrolls fully past the floating input box's top edge instead of
// sitting flush against it (spec §2.2: +1.5rem).
const INPUT_BOX_PADDING_BREATHING_ROOM_PX = 24;

export default function ChatPage() {
  const { t } = useI18n();
  const [showFollowups, setShowFollowups] = useState(false);
  const [inputBoxHeight, setInputBoxHeight] = useState<number>(0);
  const { threadId, isNewThread, setIsNewThread, isMock } = useThreadChat();
  const [settings, setSettings] = useThreadSettings(threadId);
  const [mounted, setMounted] = useState(false);
  const [welcomeDismissed, setWelcomeDismissed] = useState(false);
  useSpecificChatMode();

  const isCenteredLayout = isNewThread && !welcomeDismissed;

  useEffect(() => {
    setMounted(true);
  }, []);

  // Reset welcome dismissal when navigating to a new thread so the
  // Welcome component is shown (Next.js reuses the same component
  // instance across [thread_id] route changes, so state survives).
  useEffect(() => {
    if (isNewThread) {
      setWelcomeDismissed(false);
    }
  }, [isNewThread]);

  const { showNotification } = useNotification();

  const { thread: streamThread, sendMessage, isUploading, mergedMessageRunIds: messageRunIds } = useThreadStream({
    threadId: isNewThread ? undefined : threadId,
    context: settings.context,
    isMock,
    onStart: () => {
      setIsNewThread(false);
      // ! Important: Never use next.js router for navigation in this case, otherwise it will cause the thread to re-mount and lose all states. Use native history API instead.
      history.replaceState(null, "", `/workspace/chats/${threadId}`);
    },
    onFinish: (state) => {
      if (document.hidden || !document.hasFocus()) {
        let body = "Conversation finished";
        const lastMessage = state.messages.at(-1);
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
      const sendPromise = sendMessage(threadId, message).catch((err) => {
        setWelcomeDismissed(false);
        throw err;
      });
      if (message.files.length > 0) {
        return sendPromise;
      }
      void sendPromise;
    },
    [sendMessage, threadId],
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
  // lands (or before mount), fall back to the static default so a short
  // conversation renders without a flash of overlap. When followups are
  // visible (extra row of suggestion chips inside the stream's bottom), add
  // the dedicated extra.
  const messageListPaddingBottom =
    (inputBoxHeight > 0
      ? inputBoxHeight + INPUT_BOX_PADDING_BREATHING_ROOM_PX
      : MESSAGE_LIST_DEFAULT_PADDING_BOTTOM) +
    (showFollowups ? MESSAGE_LIST_FOLLOWUPS_EXTRA_PADDING_BOTTOM : 0);

  return (
    <ThreadContext.Provider value={{ thread, isMock }}>
      <ChatBox threadId={threadId}>
        <div
          className="relative flex size-full min-h-0 justify-between"
          data-chat-root=""
        >
          <header
            className={cn(
              "absolute top-0 right-0 left-0 z-30 flex h-14 shrink-0 items-center px-4",
              isNewThread
                ? "bg-background/0 backdrop-blur-none"
                : "bg-background/80 shadow-xs backdrop-blur",
            )}
          >
            <div className="flex w-full items-center text-sm font-medium">
              <ThreadTitle threadId={threadId} thread={thread} />
            </div>
            <div className="flex items-center gap-2">
              <TokenUsageIndicator messages={thread.messages} />
              <RunTraceWidget messages={thread.messages} />
              <ExportTrigger threadId={threadId} />
              <ArtifactTrigger />
            </div>
          </header>
          <main className="flex min-h-0 max-w-full grow flex-col">
            {/* spec#4 分析进度轨：常驻 sticky 条，数据前端推导（同源 spec#2 useRunTrace）。
                仅在有进展时渲染（AnalysisRail 内部判断）。mt-14 把初始位置推到 header(h-14) 下；
                sticky top-14 滚动时钉在 header 下，不挤压消息流（bg 半透 + backdrop-blur）。 */}
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
                messageRunIds={messageRunIds}
                paddingBottom={messageListPaddingBottom}
                onSelectClarificationOption={(optionText) => {
                  void handleSubmit({ text: optionText, files: [] });
                }}
              />
            </div>
            <div className="absolute right-0 bottom-0 left-0 z-30 flex justify-center px-4 pb-4">
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
                {mounted ? (
                  <InputBox
                    className={cn("w-full")}
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
                      isCenteredLayout && <Welcome mode={settings.context.mode} />
                    }
                    disabled={
                      env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true" ||
                      isUploading
                    }
                    onContextChange={(context) =>
                      setSettings("context", context)
                    }
                    onFollowupsVisibilityChange={setShowFollowups}
                    onHeightChange={handleInputBoxHeightChange}
                    onSubmit={handleSubmit}
                    onStop={handleStop}
                  />
                ) : (
                  <div
                    aria-hidden="true"
                    className={cn(
                      "bg-card/90 glass-card shadow-float h-32 w-full rounded-3xl",
                    )}
                  />
                )}
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
