import type { AIMessage, Message, Run } from "@langchain/langgraph-sdk";
import type { ThreadsClient } from "@langchain/langgraph-sdk/client";
import { useStream } from "@langchain/langgraph-sdk/react";
import {
  type QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";

import { getAPIClient } from "../api";
import { fetch } from "../api/fetcher";
import { getBackendBaseURL } from "../config";
import { useI18n } from "../i18n/hooks";
import { isHiddenFromUIMessage } from "../messages/utils";
import type { FileInMessage } from "../messages/utils";
import type { LocalSettings } from "../settings";
import { useUpdateSubtask } from "../tasks/context";
import type { UploadedFileInfo } from "../uploads";
import { promptInputFilePartToFile, uploadFiles } from "../uploads";

import { fetchThreadTokenUsage } from "./api";
import { getStreamErrorMessage, isRunNotOnThisWorkerError } from "./stream-error";
import { threadTokenUsageQueryKey } from "./token-usage";
import type {
  AgentThread,
  AgentThreadState,
  RunMessage,
  ThreadTokenUsageResponse,
} from "./types";

export type ToolEndEvent = {
  name: string;
  data: unknown;
};

export type ThreadStreamOptions = {
  threadId?: string | null | undefined;
  context: LocalSettings["context"];
  isMock?: boolean;
  onSend?: (threadId: string) => void;
  onStart?: (threadId: string, runId: string) => void;
  onFinish?: (state: AgentThreadState) => void;
  onToolEnd?: (event: ToolEndEvent) => void;
};

type SendMessageOptions = {
  additionalKwargs?: Record<string, unknown>;
};

function isNonEmptyString(value: string | undefined): value is string {
  return typeof value === "string" && value.length > 0;
}

function messageIdentity(message: Message): string | undefined {
  if (
    "tool_call_id" in message &&
    typeof message.tool_call_id === "string" &&
    message.tool_call_id.length > 0
  ) {
    return `tool:${message.tool_call_id}`;
  }
  if (typeof message.id === "string" && message.id.length > 0) {
    return `message:${message.id}`;
  }
  return undefined;
}

function dedupeMessagesByIdentity(messages: Message[]): Message[] {
  // One canonical copy per identity, at the position of its FIRST occurrence,
  // preferring the first VISIBLE copy over any hidden copy that shares the
  // identity (hidden-then-visible reordering during streaming must not bury a
  // visible bubble behind a hidden control twin).
  //
  // This is a UI-display dedupe rule, not a general LangChain message-stream
  // contract. Hidden messages that share an identity with a visible message are
  // treated as control messages for this merged view; hidden messages carrying
  // independent tracing/task semantics should use a distinct id or a custom
  // stream/state channel instead of relying on message dedupe preservation.
  const chosenByIndex = new Map<string, { index: number; message: Message }>();

  messages.forEach((message, index) => {
    const identity = messageIdentity(message);
    if (!identity) {
      return;
    }
    const existing = chosenByIndex.get(identity);
    if (existing) {
      // Keep the first occurrence's position, but upgrade to the first VISIBLE
      // copy if the keeper is currently a hidden twin (visible beats hidden).
      const keeperIsHidden = isHiddenFromUIMessage(existing.message);
      const candidateIsVisible = !isHiddenFromUIMessage(message);
      if (keeperIsHidden && candidateIsVisible) {
        chosenByIndex.set(identity, { index: existing.index, message });
      }
      return;
    }
    chosenByIndex.set(identity, { index, message });
  });

  // Messages without an identity (no id and no tool_call_id) always survive at
  // their original position — they cannot be deduped.
  return messages.filter((message, index) => {
    const identity = messageIdentity(message);
    if (!identity) {
      return true;
    }
    return chosenByIndex.get(identity)?.index === index;
  });
}

function findLatestUnloadedRunIndex(
  runs: Run[],
  loadedRunIds: ReadonlySet<string>,
): number {
  for (let i = runs.length - 1; i >= 0; i--) {
    const run = runs[i];
    if (run && !loadedRunIds.has(run.run_id)) {
      return i;
    }
  }
  return -1;
}

export function mergeMessages(
  historyMessages: Message[],
  threadMessages: Message[],
  optimisticMessages: Message[],
): Message[] {
  // Merge two independently-loaded views of the same canonical sequence
  // (history = useThreadHistory per-run /messages; thread = useStream
  // fetchStateHistory) plus in-flight optimistic messages, preserving canonical
  // order regardless of which source hydrates first.
  //
  // Spec 2026-06-26: the previous tail-overlap scan assumed `thread` was always
  // a superset of `history`'s tail. During the rejoin race window `thread` can
  // be STALER than `history`, which made the scan `break` at history's tail and
  // left dedupe to relocate the stale thread messages to the end ("input jumps
  // to the middle"). Instead of assuming a direction, we interleave the two
  // sources like a stable merge: walk both in order, and at each step consume
  // the history message whose identity has not already appeared (so history
  // anchors older messages thread hasn't hydrated) up until the first identity
  // that also exists in thread, after which thread becomes the backbone. This
  // can only place a message at a position adjacent to a canonical neighbor —
  // never relocate it to the opposite end.
  const merged = interleaveMessages(historyMessages, threadMessages);
  return dedupeMessagesByIdentity([...merged, ...optimisticMessages]);
}

function interleaveMessages(
  historyMessages: Message[],
  threadMessages: Message[],
): Message[] {
  // Identities the thread already has — once we reach them, the thread takes
  // over as the canonical backbone (thread state is the authoritative sequence).
  const threadIdentityOrder = new Map<string, number>();
  threadMessages.forEach((m, i) => {
    const id = messageIdentity(m);
    if (id !== undefined && !threadIdentityOrder.has(id)) {
      threadIdentityOrder.set(id, i);
    }
  });

  const result: Message[] = [];
  const seen = new Set<string>();

  // Phase 1: emit history messages that are NOT in the thread (thread hasn't
  // hydrated them yet — they precede the thread's canonical range). Keep
  // history's relative order; stop at the first history message whose identity
  // IS in the thread (that's the canonical boundary where thread takes over).
  for (const m of historyMessages) {
    const id = messageIdentity(m);
    if (id !== undefined && threadIdentityOrder.has(id)) {
      break;
    }
    if (id && seen.has(id)) continue;
    result.push(m);
    if (id) seen.add(id);
  }

  // Phase 2: the thread is the canonical backbone from here. Append all thread
  // messages in order; any remaining history messages not already covered are
  // older-but-unanchored or post-thread additions — fall back to appending them
  // after the thread (history is loaded newest-run-first, so leftover history
  // entries are the oldest prefix the thread simply doesn't cover; rendering
  // them after thread is still safer than the old reorder, and in the steady
  // state the thread already contains them).
  for (const m of threadMessages) {
    const id = messageIdentity(m);
    if (id && seen.has(id)) continue;
    result.push(m);
    if (id) seen.add(id);
  }
  for (const m of historyMessages) {
    const id = messageIdentity(m);
    if (id && seen.has(id)) continue;
    result.push(m);
    if (id) seen.add(id);
  }

  return result;
}

type RunMessagesPageResponse = {
  data: RunMessage[];
  has_more?: boolean;
  hasMore?: boolean;
};

export function runMessagesPageHasMore(result: RunMessagesPageResponse) {
  return result.has_more ?? result.hasMore ?? false;
}

export function getOldestRunMessageSeq(messages: RunMessage[]) {
  let oldestSeq: number | null = null;
  for (const message of messages) {
    if (typeof message.seq !== "number") {
      continue;
    }
    oldestSeq =
      oldestSeq === null ? message.seq : Math.min(oldestSeq, message.seq);
  }
  return oldestSeq;
}

export function getNextRunMessagesBeforeSeq(
  result: RunMessagesPageResponse,
): number | null | undefined {
  if (!runMessagesPageHasMore(result)) {
    return null;
  }
  return getOldestRunMessageSeq(result.data) ?? undefined;
}

export function buildRunMessagesUrl(
  baseUrl: string,
  threadId: string,
  runId: string,
  beforeSeq?: number,
) {
  const normalizedBaseUrl = baseUrl.replace(/\/$/, "");
  const path = `/api/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}/messages`;
  const url = new URL(
    `${normalizedBaseUrl}${path}`,
    typeof window !== "undefined" ? window.location.origin : "http://localhost",
  );
  if (beforeSeq !== undefined) {
    url.searchParams.set("before_seq", String(beforeSeq));
  }
  return normalizedBaseUrl ? url.toString() : `${url.pathname}${url.search}`;
}

function getMessagesAfterBaseline(
  messages: Message[],
  baselineMessageIds: ReadonlySet<string>,
): Message[] {
  return messages.filter((message) => {
    const id = messageIdentity(message);
    return !id || !baselineMessageIds.has(id);
  });
}

const SUMMARIZATION_MIDDLEWARE_UPDATE_KEYS = new Set([
  "SummarizationMiddleware.before_model",
  "DeerFlowSummarizationMiddleware.before_model",
]);

export function getVisibleOptimisticMessages(
  optimisticMessages: Message[],
  previousHumanMessageCount: number,
  currentHumanMessageCount: number,
): Message[] {
  if (
    optimisticMessages.some((message) => message.type === "human") &&
    currentHumanMessageCount > previousHumanMessageCount
  ) {
    return [];
  }
  return optimisticMessages;
}

export function getSummarizationMiddlewareMessages(
  data: unknown,
): Message[] | undefined {
  if (typeof data !== "object" || data === null) {
    return undefined;
  }

  for (const [key, update] of Object.entries(data)) {
    if (!SUMMARIZATION_MIDDLEWARE_UPDATE_KEYS.has(key)) {
      continue;
    }
    if (typeof update !== "object" || update === null) {
      continue;
    }

    const messages = Reflect.get(update, "messages");
    if (Array.isArray(messages)) {
      return [...messages] as Message[];
    }
  }

  return undefined;
}

export function upsertThreadInSearchCache(
  queryClient: QueryClient,
  thread: AgentThread,
) {
  queryClient.setQueriesData(
    {
      queryKey: ["threads", "search"],
      exact: false,
    },
    (oldData: Array<AgentThread> | undefined) => {
      if (!oldData) {
        return [thread];
      }

      const existingIndex = oldData.findIndex(
        (t) => t.thread_id === thread.thread_id,
      );
      if (existingIndex === -1) {
        return [thread, ...oldData];
      }

      return oldData.map((t, index) => {
        if (index !== existingIndex) {
          return t;
        }
        return {
          ...thread,
          ...t,
          metadata: {
            ...(thread.metadata ?? {}),
            ...(t.metadata ?? {}),
          },
          values: {
            ...thread.values,
            ...t.values,
          },
        };
      });
    },
  );
}

export function useThreadStream({
  threadId,
  context,
  isMock,
  onSend,
  onStart,
  onFinish,
  onToolEnd,
}: ThreadStreamOptions) {
  const { t } = useI18n();
  // Track the thread ID that is currently streaming to handle thread changes during streaming
  const [onStreamThreadId, setOnStreamThreadId] = useState(() => threadId);
  // Ref to track current thread ID across async callbacks without causing re-renders,
  // and to allow access to the current thread id in onUpdateEvent
  const threadIdRef = useRef<string | null>(threadId ?? null);
  const startedRef = useRef(false);
  const pendingUsageBaselineMessageIdsRef = useRef<Set<string>>(new Set());
  const listeners = useRef({
    onSend,
    onStart,
    onFinish,
    onToolEnd,
  });

  const {
    messages: history,
    hasMore: hasMoreHistory,
    loadMore: loadMoreHistory,
    loading: isHistoryLoading,
    appendMessages,
    messageRunIds: historyMessageRunIds,
  } = useThreadHistory(onStreamThreadId ?? "");

  // Keep listeners ref updated with latest callbacks
  useEffect(() => {
    listeners.current = { onSend, onStart, onFinish, onToolEnd };
  }, [onSend, onStart, onFinish, onToolEnd]);

  useEffect(() => {
    const normalizedThreadId = threadId ?? null;
    if (!normalizedThreadId) {
      // Reset when the UI moves back to a brand new unsaved thread.
      startedRef.current = false;
      setOnStreamThreadId(normalizedThreadId);
    } else {
      setOnStreamThreadId(normalizedThreadId);
    }
    threadIdRef.current = normalizedThreadId;
  }, [threadId]);

  const handleStreamStart = useCallback((_threadId: string, _runId: string) => {
    threadIdRef.current = _threadId;
    if (!startedRef.current) {
      listeners.current.onStart?.(_threadId, _runId);
      startedRef.current = true;
    }
    setOnStreamThreadId(_threadId);
  }, []);

  // Map<message_id, run_id> for feedback button run association (Noldus)
  const [messageRunIds, setMessageRunIds] = useState<Map<string, string>>(() => new Map());

  const queryClient = useQueryClient();
  const updateSubtask = useUpdateSubtask();

  const thread = useStream<AgentThreadState>({
    client: getAPIClient(isMock),
    assistantId: "lead_agent",
    threadId: onStreamThreadId,
    reconnectOnMount: true,
    fetchStateHistory: { limit: 1 },
    onCreated(meta) {
      handleStreamStart(meta.thread_id, meta.run_id);
      const now = new Date().toISOString();
      upsertThreadInSearchCache(queryClient, {
        thread_id: meta.thread_id,
        created_at: now,
        updated_at: now,
        metadata: context.agent_name ? { agent_name: context.agent_name } : {},
        status: "busy",
        values: {
          title: t.pages.newChat,
          messages: [],
          artifacts: [],
        },
        interrupts: {},
      });
      if (context.agent_name && !isMock) {
        void getAPIClient()
          .threads.update(meta.thread_id, {
            metadata: { agent_name: context.agent_name },
          })
          .catch(() => ({}));
      }
    },
    onLangChainEvent(event) {
      if (event.event === "on_tool_end") {
        listeners.current.onToolEnd?.({
          name: event.name,
          data: event.data,
        });
      }
      // Capture message → run_id mapping for feedback buttons
      const runId = (event as { run_id?: string }).run_id;
      const output = (event as { data?: { output?: unknown } }).data?.output;
      if (runId && output && typeof output === "object" && "id" in output) {
        const msgId = (output as { id?: string }).id;
        if (msgId) {
          setMessageRunIds((prev) => {
            if (prev.get(msgId) === runId) return prev;
            const next = new Map(prev);
            next.set(msgId, runId);
            return next;
          });
        }
      }
    },
    onUpdateEvent(data) {
      const _messages = getSummarizationMiddlewareMessages(data);
      if (_messages && _messages.length >= 2) {
        for (const m of _messages) {
          if (m.name === "summary" && m.type === "human") {
            summarizedRef.current?.add(m.id ?? "");
          }
        }
        const firstRetainedVisibleIdentity = _messages
          .filter((message) => message.type !== "remove")
          .filter((message) => !isHiddenFromUIMessage(message))
          .map(messageIdentity)
          .find(isNonEmptyString);
        const _currentMessages = [...messagesRef.current];
        const _movedMessages: Message[] = [];
        for (const m of _currentMessages) {
          if (
            firstRetainedVisibleIdentity &&
            messageIdentity(m) === firstRetainedVisibleIdentity
          ) {
            break;
          }
          if (!summarizedRef.current?.has(m.id ?? "")) {
            _movedMessages.push(m);
          }
        }
        appendMessages(_movedMessages);
        messagesRef.current = [];
      }

      const updates: Array<Partial<AgentThreadState> | null> = Object.values(
        data || {},
      );
      for (const update of updates) {
        if (update && "title" in update && update.title) {
          void queryClient.setQueriesData(
            {
              queryKey: ["threads", "search"],
              exact: false,
            },
            (oldData: Array<AgentThread> | undefined) => {
              return oldData?.map((t) => {
                if (t.thread_id === threadIdRef.current) {
                  return {
                    ...t,
                    values: {
                      ...t.values,
                      title: update.title,
                    },
                  };
                }
                return t;
              });
            },
          );
        }
      }
    },
    onCustomEvent(event: unknown) {
      if (
        typeof event === "object" &&
        event !== null &&
        "type" in event &&
        event.type === "task_running"
      ) {
        const e = event as {
          type: "task_running";
          task_id: string;
          message: AIMessage;
        };
        updateSubtask({ id: e.task_id, latestMessage: e.message });
        return;
      }

      if (
        typeof event === "object" &&
        event !== null &&
        "type" in event &&
        event.type === "llm_retry" &&
        "message" in event &&
        typeof event.message === "string" &&
        event.message.trim()
      ) {
        const e = event as { type: "llm_retry"; message: string };
        toast(e.message);
      }
    },
    onError(error) {
      setOptimisticMessages([]);
      if (isRunNotOnThisWorkerError(error)) {
        // Cross-worker re-join: content already shown via fetchStateHistory.
        // Don't alarm the user with a red toast.
      } else {
        toast.error(getStreamErrorMessage(error));
      }
      pendingUsageBaselineMessageIdsRef.current = new Set(
        messagesRef.current
          .map(messageIdentity)
          .filter((id): id is string => Boolean(id)),
      );
      if (threadIdRef.current && !isMock) {
        void queryClient.invalidateQueries({
          queryKey: threadTokenUsageQueryKey(threadIdRef.current),
        });
      }
    },
    onFinish(state) {
      listeners.current.onFinish?.(state.values);
      pendingUsageBaselineMessageIdsRef.current = new Set(
        messagesRef.current
          .map(messageIdentity)
          .filter((id): id is string => Boolean(id)),
      );
      void queryClient.invalidateQueries({ queryKey: ["threads", "search"] });
      if (threadIdRef.current && !isMock) {
        void queryClient.invalidateQueries({
          queryKey: threadTokenUsageQueryKey(threadIdRef.current),
        });
      }
    },
  });

  // Optimistic messages shown before the server stream responds
  const [optimisticMessages, setOptimisticMessages] = useState<Message[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const humanMessageCount = thread.messages.filter(
    (m) => m.type === "human",
  ).length;
  const latestMessageCountsRef = useRef({ humanMessageCount });
  const sendInFlightRef = useRef(false);
  const messagesRef = useRef<Message[]>([]);
  const summarizedRef = useRef<Set<string>>(null);
  // Track human message count before sending to prevent clearing optimistic
  // messages before the server's human message arrives (e.g. when AI messages
  // from "messages-tuple" events arrive before the input human message from
  // "values" events).
  const prevHumanMsgCountRef = useRef(humanMessageCount);

  latestMessageCountsRef.current = { humanMessageCount };
  summarizedRef.current ??= new Set<string>();

  // Reset thread-local pending UI state when switching between threads so
  // optimistic messages and in-flight guards do not leak across chat views.
  useEffect(() => {
    startedRef.current = false;
    sendInFlightRef.current = false;
    pendingUsageBaselineMessageIdsRef.current = new Set(
      messagesRef.current
        .map(messageIdentity)
        .filter((id): id is string => Boolean(id)),
    );
    prevHumanMsgCountRef.current =
      latestMessageCountsRef.current.humanMessageCount;
  }, [threadId]);

  // When streaming starts without a baseline (e.g. reconnection, run started
  // from another client, or page reload mid-stream), snapshot the current
  // messages so only *new* messages are treated as "pending" for token usage.
  useEffect(() => {
    if (
      thread.isLoading &&
      pendingUsageBaselineMessageIdsRef.current.size === 0
    ) {
      pendingUsageBaselineMessageIdsRef.current = new Set(
        thread.messages
          .map(messageIdentity)
          .filter((id): id is string => Boolean(id)),
      );
    }
  }, [thread.isLoading, thread.messages]);

  // Clear optimistic when server messages arrive.
  // For messages with a human optimistic message, wait until the server's
  // human message has arrived to avoid clearing before the input message
  // appears in the stream (the input message may arrive via "values" events
  // after individual "messages-tuple" events for AI messages).
  const optimisticMessageCount = optimisticMessages.length;
  const hasHumanOptimistic = optimisticMessages.some((m) => m.type === "human");
  useEffect(() => {
    if (optimisticMessageCount === 0) return;

    const newHumanMsgArrived = humanMessageCount > prevHumanMsgCountRef.current;

    if (!hasHumanOptimistic || newHumanMsgArrived) {
      setOptimisticMessages([]);
    }
  }, [hasHumanOptimistic, humanMessageCount, optimisticMessageCount]);

  const sendMessage = useCallback(
    async (
      threadId: string,
      message: PromptInputMessage,
      extraContext?: Record<string, unknown>,
      options?: SendMessageOptions,
    ) => {
      if (sendInFlightRef.current) {
        return;
      }
      sendInFlightRef.current = true;

      const text = message.text.trim();

      // Capture the current human message count before showing optimistic
      // messages so we can wait for the server's copy of the user input.
      prevHumanMsgCountRef.current = humanMessageCount;
      pendingUsageBaselineMessageIdsRef.current = new Set(
        thread.messages
          .map(messageIdentity)
          .filter((id): id is string => Boolean(id)),
      );

      // Build optimistic files list with uploading status
      const optimisticFiles: FileInMessage[] = (message.files ?? []).map(
        (f) => ({
          filename: f.filename ?? "",
          size: 0,
          status: "uploading" as const,
        }),
      );

      const hideFromUI = options?.additionalKwargs?.hide_from_ui === true;
      const optimisticAdditionalKwargs = {
        ...options?.additionalKwargs,
        ...(optimisticFiles.length > 0 ? { files: optimisticFiles } : {}),
      };

      const newOptimistic: Message[] = [];
      if (!hideFromUI) {
        newOptimistic.push({
          type: "human",
          id: `opt-human-${Date.now()}`,
          content: text ? [{ type: "text", text }] : "",
          additional_kwargs: optimisticAdditionalKwargs,
        });
      }

      if (optimisticFiles.length > 0 && !hideFromUI) {
        // Mock AI message while files are being uploaded
        newOptimistic.push({
          type: "ai",
          id: `opt-ai-${Date.now()}`,
          content: t.uploads.uploadingFiles,
          additional_kwargs: { element: "task" },
        });
      }
      setOptimisticMessages(newOptimistic);

      listeners.current.onSend?.(threadId);

      let uploadedFileInfo: UploadedFileInfo[] = [];

      try {
        // Upload files first if any
        if (message.files && message.files.length > 0) {
          setIsUploading(true);
          try {
            const filePromises = message.files.map((fileUIPart) =>
              promptInputFilePartToFile(fileUIPart),
            );

            const conversionResults = await Promise.all(filePromises);
            const files = conversionResults.filter(
              (file): file is File => file !== null,
            );
            const failedConversions = conversionResults.length - files.length;

            if (failedConversions > 0) {
              throw new Error(
                `Failed to prepare ${failedConversions} attachment(s) for upload. Please retry.`,
              );
            }

            if (!threadId) {
              throw new Error("Thread is not ready for file upload.");
            }

            if (files.length > 0) {
              const uploadResponse = await uploadFiles(threadId, files);
              uploadedFileInfo = uploadResponse.files;

              // Update optimistic human message with uploaded status + paths
              const uploadedFiles: FileInMessage[] = uploadedFileInfo.map(
                (info) => ({
                  filename: info.filename,
                  size: info.size,
                  path: info.virtual_path,
                  status: "uploaded" as const,
                }),
              );
              setOptimisticMessages((messages) => {
                if (messages.length > 1 && messages[0]) {
                  const humanMessage: Message = messages[0];
                  return [
                    {
                      ...humanMessage,
                      additional_kwargs: { files: uploadedFiles },
                    },
                    ...messages.slice(1),
                  ];
                }
                return messages;
              });
            }
          } catch (error) {
            const errorMessage =
              error instanceof Error
                ? error.message
                : "Failed to upload files.";
            toast.error(errorMessage);
            setOptimisticMessages([]);
            throw error;
          } finally {
            setIsUploading(false);
          }
        }

        // Build files metadata for submission (included in additional_kwargs)
        const filesForSubmit: FileInMessage[] = uploadedFileInfo.map(
          (info) => ({
            filename: info.filename,
            size: info.size,
            path: info.virtual_path,
            status: "uploaded" as const,
          }),
        );

        await thread.submit(
          {
            messages: [
              {
                type: "human",
                content: [
                  {
                    type: "text",
                    text,
                  },
                ],
                additional_kwargs: {
                  ...options?.additionalKwargs,
                  ...(filesForSubmit.length > 0
                    ? { files: filesForSubmit }
                    : {}),
                },
              },
            ],
          },
          {
            threadId: threadId,
            streamSubgraphs: true,
            streamResumable: true,
            config: {
              recursion_limit: 1000,
            },
            context: {
              ...extraContext,
              ...context,
              thinking_enabled: context.reasoning_effort !== undefined || false,
              is_plan_mode: context.mode === "flywheel",
              // Mirror the UI mode into the deerflow workflow_mode runtime
              // config. The backend uses this to gate manual-mode middlewares
              // (currently GateEnforcementMiddleware for Gate 1 paradigm
              // clarification). Without this line, the UI toggle silently has
              // no effect on those middlewares and they never activate.
              workflow_mode: context.mode === "flywheel" ? "manual" : "auto",
              subagent_enabled: true,
              reasoning_effort: context.reasoning_effort ?? (context.mode === "flywheel" ? "high" : undefined),
              thread_id: threadId,
            },
          },
        );
        void queryClient.invalidateQueries({ queryKey: ["threads", "search"] });
      } catch (error) {
        setOptimisticMessages([]);
        setIsUploading(false);
        throw error;
      } finally {
        sendInFlightRef.current = false;
      }
    },
    [thread, t.uploads.uploadingFiles, context, queryClient, humanMessageCount],
  );

  // Cache the latest thread messages in a ref to compare against incoming history messages for deduplication,
  // and to allow access to the full message list in onUpdateEvent without causing re-renders.
  if (thread.messages.length >= messagesRef.current.length) {
    messagesRef.current = thread.messages;
  }

  const visibleOptimisticMessages = getVisibleOptimisticMessages(
    optimisticMessages,
    prevHumanMsgCountRef.current,
    humanMessageCount,
  );

  const mergedMessages = mergeMessages(
    history,
    thread.messages,
    visibleOptimisticMessages,
  );
  const pendingUsageMessages = thread.isLoading
    ? getMessagesAfterBaseline(
        thread.messages,
        pendingUsageBaselineMessageIdsRef.current,
      )
    : [];

  // Merge history, live stream, and optimistic messages for display
  // History messages may overlap with thread.messages; thread.messages take precedence
  const mergedThread = {
    ...thread,
    messages: mergedMessages,
  } as typeof thread;

  // Merge stream-derived and history-derived run_id maps so the feedback
  // buttons render for both live messages and reloaded history. Stream events
  // take precedence (they reflect the most recent run for a message id), but
  // history fills in everything plain-text messages and pre-refresh state miss.
  const mergedMessageRunIds = useMemo(() => {
    if (historyMessageRunIds.size === 0) return messageRunIds;
    if (messageRunIds.size === 0) return historyMessageRunIds;
    const merged = new Map(historyMessageRunIds);
    for (const [k, v] of messageRunIds) merged.set(k, v);
    return merged;
  }, [messageRunIds, historyMessageRunIds]);

  return {
    thread: mergedThread,
    pendingUsageMessages,
    sendMessage,
    isUploading,
    mergedMessageRunIds,
    isHistoryLoading,
    hasMoreHistory,
    loadMoreHistory,
  } as const;
}

export function useThreadHistory(threadId: string) {
  const runs = useThreadRuns(threadId);
  const threadIdRef = useRef(threadId);
  const runsRef = useRef(runs.data ?? []);
  const indexRef = useRef(-1);
  const loadingRef = useRef(false);
  const pendingLoadRef = useRef(false);
  const loadingRunIdRef = useRef<string | null>(null);
  const loadedRunIdsRef = useRef<Set<string>>(new Set());
  const runBeforeSeqRef = useRef<Map<string, number>>(new Map());
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  // Per-message run_id discovered during history fetch (Noldus). Feedback buttons
  // need a run_id to bind to, and on_tool_end stream events only fire for
  // tool-using messages — plain-text answers / clarifications / reasoning
  // wouldn't get a run_id otherwise, and a page refresh empties the in-memory
  // map maintained by useThreadStream. This source is authoritative.
  const [historyMessageRunIds, setHistoryMessageRunIds] = useState<Map<string, string>>(() => new Map());

  const loadMessages = useCallback(async () => {
    if (loadingRef.current) {
      const pendingRunIndex = findLatestUnloadedRunIndex(
        runsRef.current,
        loadedRunIdsRef.current,
      );
      const pendingRun = runsRef.current[pendingRunIndex];
      if (pendingRun && pendingRun.run_id !== loadingRunIdRef.current) {
        pendingLoadRef.current = true;
      }
      return;
    }
    if (runsRef.current.length === 0) {
      return;
    }

    loadingRef.current = true;
    setLoading(true);

    try {
      do {
        pendingLoadRef.current = false;

        const nextRunIndex = findLatestUnloadedRunIndex(
          runsRef.current,
          loadedRunIdsRef.current,
        );
        indexRef.current = nextRunIndex;

        const run = runsRef.current[nextRunIndex];
        if (!run) {
          indexRef.current = -1;
          return;
        }

        const requestThreadId = threadIdRef.current;
        loadingRunIdRef.current = run.run_id;
        const beforeSeq = runBeforeSeqRef.current.get(run.run_id);
        const url = buildRunMessagesUrl(
          getBackendBaseURL(),
          requestThreadId,
          run.run_id,
          beforeSeq,
        );
        const result: RunMessagesPageResponse = await fetch(url, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "include",
        }).then((res) => {
          return res.json();
        });
        const _messages = result.data
          .filter((m) => !m.metadata.caller?.startsWith("middleware:"))
          .map((m) => m.content);
        if (threadIdRef.current !== requestThreadId) {
          return;
        }
        setMessages((prev) =>
          dedupeMessagesByIdentity([..._messages, ...prev]),
        );
        // Populate historyMessageRunIds from history fetch so every prior AIMessage
        // gets a run_id binding for the feedback buttons. (Noldus)
        setHistoryMessageRunIds((prev) => {
          let next: Map<string, string> | null = null;
          for (const msg of _messages) {
            const msgId = msg.id;
            if (!msgId || prev.get(msgId) === run.run_id) continue;
            next ??= new Map(prev);
            next.set(msgId, run.run_id);
          }
          return next ?? prev;
        });
        const nextBeforeSeq = getNextRunMessagesBeforeSeq(result);
        if (typeof nextBeforeSeq === "number") {
          runBeforeSeqRef.current.set(run.run_id, nextBeforeSeq);
          pendingLoadRef.current = true;
        } else if (nextBeforeSeq === undefined) {
          console.warn(
            `Run ${run.run_id} returned has_more without message seq values; leaving it pending for retry.`,
          );
        } else {
          runBeforeSeqRef.current.delete(run.run_id);
          loadedRunIdsRef.current.add(run.run_id);
        }
        indexRef.current = findLatestUnloadedRunIndex(
          runsRef.current,
          loadedRunIdsRef.current,
        );
      } while (pendingLoadRef.current);
    } catch (err) {
      console.error(err);
    } finally {
      loadingRef.current = false;
      loadingRunIdRef.current = null;
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    const threadChanged = threadIdRef.current !== threadId;
    threadIdRef.current = threadId;

    if (threadChanged) {
      runsRef.current = [];
      indexRef.current = -1;
      pendingLoadRef.current = false;
      loadingRunIdRef.current = null;
      loadedRunIdsRef.current = new Set();
      runBeforeSeqRef.current = new Map();
      loadingRef.current = false;
      setLoading(false);
      setMessages([]);
      setHistoryMessageRunIds(new Map());
    }

    if (runs.data && runs.data.length > 0) {
      runsRef.current = runs.data ?? [];
      indexRef.current = findLatestUnloadedRunIndex(
        runs.data,
        loadedRunIdsRef.current,
      );
    }
    loadMessages().catch(() => {
      toast.error("Failed to load thread history.");
    });
  }, [threadId, runs.data, loadMessages]);

  const appendMessages = useCallback((_messages: Message[]) => {
    setMessages((prev) => {
      return dedupeMessagesByIdentity([...prev, ..._messages]);
    });
  }, []);
  const hasMore = indexRef.current >= 0 || !runs.data;
  return {
    runs: runs.data,
    messages,
    loading,
    appendMessages,
    hasMore,
    loadMore: loadMessages,
    messageRunIds: historyMessageRunIds,
  };
}

export function useThreads(
  params: Parameters<ThreadsClient["search"]>[0] = {
    limit: 50,
    sortBy: "updated_at",
    sortOrder: "desc",
    select: ["thread_id", "updated_at", "values", "metadata"],
  },
) {
  const apiClient = getAPIClient();
  return useQuery<AgentThread[]>({
    queryKey: ["threads", "search", params],
    queryFn: async () => {
      const maxResults = params.limit;
      const initialOffset = params.offset ?? 0;
      const DEFAULT_PAGE_SIZE = 50;

      // Preserve prior semantics: if a non-positive limit is explicitly provided,
      // delegate to a single search call with the original parameters.
      if (maxResults !== undefined && maxResults <= 0) {
        const response =
          await apiClient.threads.search<AgentThreadState>(params);
        return response as AgentThread[];
      }

      const pageSize =
        typeof maxResults === "number" && maxResults > 0
          ? Math.min(DEFAULT_PAGE_SIZE, maxResults)
          : DEFAULT_PAGE_SIZE;

      const threads: AgentThread[] = [];
      let offset = initialOffset;

      while (true) {
        if (typeof maxResults === "number" && threads.length >= maxResults) {
          break;
        }

        const currentLimit =
          typeof maxResults === "number"
            ? Math.min(pageSize, maxResults - threads.length)
            : pageSize;

        if (typeof maxResults === "number" && currentLimit <= 0) {
          break;
        }

        const response = (await apiClient.threads.search<AgentThreadState>({
          ...params,
          limit: currentLimit,
          offset,
        })) as AgentThread[];

        threads.push(...response);

        if (response.length < currentLimit) {
          break;
        }

        offset += response.length;
      }

      return threads;
    },
    refetchOnWindowFocus: false,
  });
}

export function useThreadRuns(threadId?: string) {
  const apiClient = getAPIClient();
  return useQuery<Run[]>({
    queryKey: ["thread", threadId],
    queryFn: async () => {
      if (!threadId) {
        return [];
      }
      const response = await apiClient.runs.list(threadId);
      return response;
    },
    refetchOnWindowFocus: false,
  });
}

export function useThreadTokenUsage(
  threadId?: string | null,
  { enabled = true }: { enabled?: boolean } = {},
) {
  return useQuery<ThreadTokenUsageResponse | null>({
    queryKey: threadTokenUsageQueryKey(threadId),
    queryFn: async () => {
      if (!threadId) {
        return null;
      }
      return fetchThreadTokenUsage(threadId);
    },
    enabled: enabled && Boolean(threadId),
    retry: false,
    refetchOnWindowFocus: false,
  });
}

export function useRunDetail(threadId: string, runId: string) {
  const apiClient = getAPIClient();
  return useQuery<Run>({
    queryKey: ["thread", threadId, "run", runId],
    queryFn: async () => {
      const response = await apiClient.runs.get(threadId, runId);
      return response;
    },
    refetchOnWindowFocus: false,
  });
}

export function useDeleteThread() {
  const queryClient = useQueryClient();
  const apiClient = getAPIClient();
  return useMutation({
    mutationFn: async ({ threadId }: { threadId: string }) => {
      await apiClient.threads.delete(threadId);

      const response = await fetch(
        `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}`,
        {
          method: "DELETE",
        },
      );

      if (!response.ok) {
        const error = await response
          .json()
          .catch(() => ({ detail: "Failed to delete local thread data." }));
        throw new Error(error.detail ?? "Failed to delete local thread data.");
      }
    },
    onSuccess(_, { threadId }) {
      queryClient.setQueriesData(
        {
          queryKey: ["threads", "search"],
          exact: false,
        },
        (oldData: Array<AgentThread> | undefined) => {
          if (oldData == null) {
            return oldData;
          }
          return oldData.filter((t) => t.thread_id !== threadId);
        },
      );
    },
    onSettled() {
      void queryClient.invalidateQueries({ queryKey: ["threads", "search"] });
    },
  });
}

export function useRenameThread() {
  const queryClient = useQueryClient();
  const apiClient = getAPIClient();
  return useMutation({
    mutationFn: async ({
      threadId,
      title,
    }: {
      threadId: string;
      title: string;
    }) => {
      await apiClient.threads.updateState(threadId, {
        values: { title },
      });
    },
    onSuccess(_, { threadId, title }) {
      queryClient.setQueriesData(
        {
          queryKey: ["threads", "search"],
          exact: false,
        },
        (oldData: Array<AgentThread>) => {
          return oldData.map((t) => {
            if (t.thread_id === threadId) {
              return {
                ...t,
                values: {
                  ...t.values,
                  title,
                },
              };
            }
            return t;
          });
        },
      );
    },
  });
}
