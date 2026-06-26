"use client";

import type { ChatStatus } from "ai";
import {
  CheckIcon,
  PaperclipIcon,
  SparklesIcon,
  RocketIcon,
  XIcon,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentProps,
} from "react";
import { toast } from "sonner";

import {
  PromptInput,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuItem,
  PromptInputActionMenuTrigger,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  usePromptInputAttachments,
  usePromptInputController,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenuGroup,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import { fetch as fetchWithAuth } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";
import { useI18n } from "@/core/i18n/hooks";
import { lastClarificationIsAwaiting } from "@/core/messages/clarification-state";
import { useModels } from "@/core/models/hooks";
import type { AgentThreadContext } from "@/core/threads";
import { textOfMessage } from "@/core/threads/utils";
import { cn } from "@/lib/utils";

import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorInput,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorName,
  ModelSelectorTrigger,
} from "../ai-elements/model-selector";
import { Suggestion, Suggestions } from "../ai-elements/suggestion";

import { StackedAttachments } from "./attachments/stacked-attachments";
import { useThread } from "./messages/context";
import { ModeHoverGuide } from "./mode-hover-guide";
import { Tooltip } from "./tooltip";

type InputMode = "auto" | "flywheel";

// Client-side mirror of the backend `uploads.max_files` limit (config.yaml).
// Keeps the limit in sync so users get an immediate hint instead of a mid-upload
// 413 rejection. Must match the backend value when that config changes.
const MAX_UPLOAD_FILES = 50;

function getResolvedMode(
  mode: InputMode | undefined,
  _supportsThinking: boolean,
): InputMode {
  return mode ?? "auto";
}

export function InputBox({
  className,
  disabled,
  autoFocus,
  status = "ready",
  context,
  extraHeader,
  isNewThread,
  threadId,
  initialValue,
  onContextChange,
  onFollowupsVisibilityChange,
  onHeightChange,
  onSubmit,
  onStop,
  ...props
}: Omit<ComponentProps<typeof PromptInput>, "onSubmit"> & {
  assistantId?: string | null;
  status?: ChatStatus;
  disabled?: boolean;
  context: Omit<
    AgentThreadContext,
    "thread_id" | "is_plan_mode" | "thinking_enabled" | "subagent_enabled"
  > & {
    mode: "auto" | "flywheel" | undefined;
    reasoning_effort?: "minimal" | "low" | "medium" | "high";
  };
  extraHeader?: React.ReactNode;
  isNewThread?: boolean;
  threadId: string;
  initialValue?: string;
  onContextChange?: (
    context: Omit<
      AgentThreadContext,
      "thread_id" | "is_plan_mode" | "thinking_enabled" | "subagent_enabled"
    > & {
      mode: "auto" | "flywheel" | undefined;
      reasoning_effort?: "minimal" | "low" | "medium" | "high";
    },
  ) => void;
  onFollowupsVisibilityChange?: (visible: boolean) => void;
  /**
   * Fired whenever the input box's outer height changes (attachments
   * stacking, multi-line expansion, followups toggling). The page uses it to
   * reserve matching bottom space in the message stream so the floating input
   * box never covers the last message / decision-card options. Also published
   * as the `--input-box-height` CSS variable on the closest
   * `[data-chat-root]` ancestor for any direct CSS consumer.
   */
  onHeightChange?: (heightPx: number) => void;
  onSubmit?: (message: PromptInputMessage) => void | Promise<void>;
  onStop?: () => void;
}) {
  const { t } = useI18n();
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const { models } = useModels();
  const { thread, isMock } = useThread();
  const { textInput } = usePromptInputController();
  const promptRootRef = useRef<HTMLDivElement | null>(null);

  const [followups, setFollowups] = useState<string[]>([]);
  const [followupsHidden, setFollowupsHidden] = useState(false);
  const [followupsLoading, setFollowupsLoading] = useState(false);
  const lastGeneratedForAiIdRef = useRef<string | null>(null);
  const wasStreamingRef = useRef(false);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingSuggestion, setPendingSuggestion] = useState<string | null>(
    null,
  );

  // Measure the input box height and report it upward so the message stream
  // can reserve matching bottom space (the input box floats absolute over the
  // stream — without this, the last message / decision-card options get
  // covered when the box grows: attachments stacking, multi-line expansion).
  // Also published as `--input-box-height` on the closest [data-chat-root]
  // ancestor for any direct CSS consumer.
  useEffect(() => {
    const el = promptRootRef.current;
    if (!el) return;
    const sync = () => {
      const h = el.offsetHeight;
      const root = el.closest("[data-chat-root]");
      if (root instanceof HTMLElement) {
        root.style.setProperty("--input-box-height", `${h}px`);
      }
      onHeightChange?.(h);
    };
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    return () => ro.disconnect();
  }, [onHeightChange]);

  useEffect(() => {
    if (models.length === 0) {
      return;
    }
    const currentModel = models.find((m) => m.name === context.model_name);
    const fallbackModel = currentModel ?? models[0]!;
    const supportsThinking = fallbackModel.supports_thinking ?? false;
    const nextModelName = fallbackModel.name;
    const nextMode = getResolvedMode(context.mode, supportsThinking);

    if (context.model_name === nextModelName && context.mode === nextMode) {
      return;
    }

    onContextChange?.({
      ...context,
      model_name: nextModelName,
      mode: nextMode,
    });
  }, [context, models, onContextChange]);

  const selectedModel = useMemo(() => {
    if (models.length === 0) {
      return undefined;
    }
    return models.find((m) => m.name === context.model_name) ?? models[0];
  }, [context.model_name, models]);

  const resolvedModelName = selectedModel?.name;

  const supportThinking = useMemo(
    () => selectedModel?.supports_thinking ?? false,
    [selectedModel],
  );

  // spec#5 §3.4 输入框态联动：流非 streaming 且消息流止于一个未答的 ask_clarification
  // 时，输入框 placeholder 切换为「回答上面的问题…」，给研究员「agent 在等你」的强信号。
  // 信号纯前端从消息流派生（lastClarificationIsAwaiting），不改后端（spec §3.4）。
  const awaitingClarification =
    status !== "streaming" &&
    !isNewThread &&
    lastClarificationIsAwaiting(thread.messages);

  const handleModelSelect = useCallback(
    (model_name: string) => {
      const model = models.find((m) => m.name === model_name);
      if (!model) {
        return;
      }
      onContextChange?.({
        ...context,
        model_name,
        mode: getResolvedMode(context.mode, model.supports_thinking ?? false),
        reasoning_effort: context.reasoning_effort,
      });
      setModelDialogOpen(false);
    },
    [onContextChange, context, models],
  );

  const handleModeSelect = useCallback(
    (mode: InputMode) => {
      onContextChange?.({
        ...context,
        mode: getResolvedMode(mode, supportThinking),
        reasoning_effort: "high",
      });
    },
    [onContextChange, context, supportThinking],
  );

  const handleSubmit = useCallback(
    async (message: PromptInputMessage) => {
      if (status === "streaming") {
        onStop?.();
        return;
      }
      if (!message.text) {
        return;
      }
      setFollowups([]);
      setFollowupsHidden(false);
      setFollowupsLoading(false);

      // Guard against submitting before the initial model auto-selection
      // effect has flushed thread settings to storage/state.
      if (resolvedModelName && context.model_name !== resolvedModelName) {
        onContextChange?.({
          ...context,
          model_name: resolvedModelName,
          mode: getResolvedMode(
            context.mode,
            selectedModel?.supports_thinking ?? false,
          ),
        });
        setTimeout(() => void onSubmit?.(message), 0);
        return;
      }

      // `onSubmit` is `(message) => void | Promise<void>`; the form-level
      // `PromptInput onSubmit` handler is treated as a void return, so don't
      // forward the (possibly Promise) result up — `void` discards it without
      // changing behaviour (PromptInput awaits its own onSubmit internally).
      void onSubmit?.(message);
    },
    [
      context,
      onContextChange,
      onSubmit,
      onStop,
      resolvedModelName,
      selectedModel?.supports_thinking,
      status,
    ],
  );

  const requestFormSubmit = useCallback(() => {
    const form = promptRootRef.current?.querySelector("form");
    form?.requestSubmit();
  }, []);

  const handleFollowupClick = useCallback(
    (suggestion: string) => {
      if (status === "streaming") {
        return;
      }
      const current = (textInput.value ?? "").trim();
      if (current) {
        setPendingSuggestion(suggestion);
        setConfirmOpen(true);
        return;
      }
      textInput.setInput(suggestion);
      setFollowupsHidden(true);
      setTimeout(() => requestFormSubmit(), 0);
    },
    [requestFormSubmit, status, textInput],
  );

  const confirmReplaceAndSend = useCallback(() => {
    if (!pendingSuggestion) {
      setConfirmOpen(false);
      return;
    }
    textInput.setInput(pendingSuggestion);
    setFollowupsHidden(true);
    setConfirmOpen(false);
    setPendingSuggestion(null);
    setTimeout(() => requestFormSubmit(), 0);
  }, [pendingSuggestion, requestFormSubmit, textInput]);

  const confirmAppendAndSend = useCallback(() => {
    if (!pendingSuggestion) {
      setConfirmOpen(false);
      return;
    }
    const current = (textInput.value ?? "").trim();
    const next = current
      ? `${current}\n${pendingSuggestion}`
      : pendingSuggestion;
    textInput.setInput(next);
    setFollowupsHidden(true);
    setConfirmOpen(false);
    setPendingSuggestion(null);
    setTimeout(() => requestFormSubmit(), 0);
  }, [pendingSuggestion, requestFormSubmit, textInput]);

  const showFollowups =
    !disabled &&
    !isNewThread &&
    !followupsHidden &&
    (followupsLoading || followups.length > 0);

  const followupsVisibilityChangeRef = useRef(onFollowupsVisibilityChange);

  useEffect(() => {
    followupsVisibilityChangeRef.current = onFollowupsVisibilityChange;
  }, [onFollowupsVisibilityChange]);

  useEffect(() => {
    followupsVisibilityChangeRef.current?.(showFollowups);
  }, [showFollowups]);

  useEffect(() => {
    return () => followupsVisibilityChangeRef.current?.(false);
  }, []);

  useEffect(() => {
    const streaming = status === "streaming";
    const wasStreaming = wasStreamingRef.current;
    wasStreamingRef.current = streaming;
    if (!wasStreaming || streaming) {
      return;
    }

    if (disabled || isMock) {
      return;
    }

    const lastAi = [...thread.messages].reverse().find((m) => m.type === "ai");
    const lastAiId = lastAi?.id ?? null;
    if (!lastAiId || lastAiId === lastGeneratedForAiIdRef.current) {
      return;
    }
    lastGeneratedForAiIdRef.current = lastAiId;

    const recent = thread.messages
      .filter((m) => m.type === "human" || m.type === "ai")
      .map((m) => {
        const role = m.type === "human" ? "user" : "assistant";
        const content = textOfMessage(m) ?? "";
        return { role, content };
      })
      .filter((m) => m.content.trim().length > 0)
      .slice(-6);

    if (recent.length === 0) {
      return;
    }

    const controller = new AbortController();
    setFollowupsHidden(false);
    setFollowupsLoading(true);
    setFollowups([]);

    fetchWithAuth(`${getBackendBaseURL()}/api/threads/${threadId}/suggestions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: recent,
        n: 3,
        model_name: context.model_name ?? undefined,
      }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          return { suggestions: [] as string[] };
        }
        return (await res.json()) as { suggestions?: string[] };
      })
      .then((data) => {
        const suggestions = (data.suggestions ?? [])
          .map((s) => (typeof s === "string" ? s.trim() : ""))
          .filter((s) => s.length > 0)
          .slice(0, 5);
        setFollowups(suggestions);
      })
      .catch(() => {
        setFollowups([]);
      })
      .finally(() => {
        setFollowupsLoading(false);
      });

    return () => controller.abort();
  }, [context.model_name, disabled, isMock, status, thread.messages, threadId]);

  return (
    <div ref={promptRootRef} className="relative flex flex-col gap-4">
      {showFollowups && (
        <div className="flex items-center justify-center pb-2">
          <div className="flex items-center gap-2">
            {followupsLoading ? (
              <div className="text-muted-foreground bg-background/80 rounded-full border px-4 py-2 text-xs backdrop-blur-sm">
                {t.inputBox.followupLoading}
              </div>
            ) : (
              <Suggestions className="min-h-16 w-fit items-start">
                {followups.map((s) => (
                  <Suggestion
                    key={s}
                    suggestion={s}
                    onClick={() => handleFollowupClick(s)}
                  />
                ))}
                <Button
                  aria-label={t.common.close}
                  className="text-muted-foreground cursor-pointer rounded-full px-3 text-xs font-normal"
                  variant="outline"
                  size="sm"
                  type="button"
                  onClick={() => setFollowupsHidden(true)}
                >
                  <XIcon className="size-4" />
                </Button>
              </Suggestions>
            )}
          </div>
        </div>
      )}
      <PromptInput
        className={cn(
          "bg-card/90 glass-card shadow-float rounded-3xl transition-[box-shadow,border-color] duration-base ease-brand-out *:data-[slot='input-group']:rounded-3xl",
          className,
        )}
        disabled={disabled}
        globalDrop
        multiple
        onSubmit={handleSubmit}
        {...props}
        maxFiles={MAX_UPLOAD_FILES}
        onError={(err) => {
          if (err.code === "max_files") {
            toast.error(
              t.inputBox.tooManyFiles.replace(
                "{max}",
                String(MAX_UPLOAD_FILES),
              ),
            );
          }
        }}
      >
        {extraHeader && (
          <div className="absolute top-0 right-0 left-0 z-10">
            <div className="absolute right-0 bottom-0 left-0 flex items-center justify-center">
              {extraHeader}
            </div>
          </div>
        )}
        <StackedAttachments />
        <PromptInputBody className="absolute top-0 right-0 left-0 z-3">
          <PromptInputTextarea
            className={cn("size-full min-h-14 py-4 text-[15px] leading-6")}
            disabled={disabled}
            placeholder={
              awaitingClarification
                ? t.clarification.awaitingPlaceholder
                : t.inputBox.placeholder
            }
            autoFocus={autoFocus}
            defaultValue={initialValue}
          />
        </PromptInputBody>
        <PromptInputFooter className="flex">
          <PromptInputTools>
            {/* TODO: Add more connectors here
          <PromptInputActionMenu>
            <PromptInputActionMenuTrigger className="px-2!" />
            <PromptInputActionMenuContent>
              <PromptInputActionAddAttachments
                label={t.inputBox.addAttachments}
              />
            </PromptInputActionMenuContent>
          </PromptInputActionMenu> */}
            <AddAttachmentsButton className="px-2!" />
            <PromptInputActionMenu>
              <ModeHoverGuide
                mode={context.mode === "flywheel" ? "flywheel" : "auto"}
              >
                <PromptInputActionMenuTrigger className="gap-1! px-2!">
                  <div>
                    {context.mode === "flywheel" ? (
                      <SparklesIcon className="size-3" />
                    ) : (
                      <RocketIcon className="size-3" />
                    )}
                  </div>
                  <div className="text-xs font-normal">
                    {context.mode === "flywheel"
                      ? t.inputBox.flywheelMode
                      : t.inputBox.autoMode}
                  </div>
                </PromptInputActionMenuTrigger>
              </ModeHoverGuide>
              <PromptInputActionMenuContent className="w-80">
                <DropdownMenuGroup>
                  <DropdownMenuLabel className="text-muted-foreground text-xs">
                    {t.inputBox.mode}
                  </DropdownMenuLabel>
                  <PromptInputActionMenu>
                    <PromptInputActionMenuItem
                      className={cn(
                        context.mode !== "flywheel"
                          ? "text-accent-foreground"
                          : "text-muted-foreground/65",
                      )}
                      onSelect={() => handleModeSelect("auto")}
                    >
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center gap-1 font-bold">
                          <RocketIcon
                            className={cn(
                              "mr-2 size-4",
                              context.mode !== "flywheel" &&
                                "text-accent-foreground",
                            )}
                          />
                          {t.inputBox.autoMode}
                        </div>
                        <div className="pl-7 text-xs">
                          {t.inputBox.autoModeDescription}
                        </div>
                      </div>
                      {context.mode !== "flywheel" ? (
                        <CheckIcon className="ml-auto size-4" />
                      ) : (
                        <div className="ml-auto size-4" />
                      )}
                    </PromptInputActionMenuItem>
                    <PromptInputActionMenuItem
                      className={cn(
                        context.mode === "flywheel"
                          ? "text-accent-foreground"
                          : "text-muted-foreground/65",
                      )}
                      onSelect={() => handleModeSelect("flywheel")}
                    >
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center gap-1 font-bold">
                          <SparklesIcon
                            className={cn(
                              "mr-2 size-4",
                              context.mode === "flywheel" &&
                                "text-accent-foreground",
                            )}
                          />
                          {t.inputBox.flywheelMode}
                        </div>
                        <div className="pl-7 text-xs">
                          {t.inputBox.flywheelModeDescription}
                        </div>
                      </div>
                      {context.mode === "flywheel" ? (
                        <CheckIcon className="ml-auto size-4" />
                      ) : (
                        <div className="ml-auto size-4" />
                      )}
                    </PromptInputActionMenuItem>
                  </PromptInputActionMenu>
                </DropdownMenuGroup>
              </PromptInputActionMenuContent>
            </PromptInputActionMenu>
          </PromptInputTools>
          <PromptInputTools>
            <ModelSelector
              open={modelDialogOpen}
              onOpenChange={setModelDialogOpen}
            >
              <ModelSelectorTrigger asChild>
                <PromptInputButton>
                  <div className="flex min-w-0 flex-col items-start text-left">
                    <ModelSelectorName className="text-xs font-normal">
                      {selectedModel?.display_name}
                    </ModelSelectorName>
                  </div>
                </PromptInputButton>
              </ModelSelectorTrigger>
              <ModelSelectorContent>
                <ModelSelectorInput placeholder={t.inputBox.searchModels} />
                <ModelSelectorList>
                  {models.map((m) => (
                    <ModelSelectorItem
                      key={m.name}
                      value={m.name}
                      onSelect={() => handleModelSelect(m.name)}
                    >
                      <div className="flex min-w-0 flex-1 flex-col">
                        <ModelSelectorName>{m.display_name}</ModelSelectorName>
                        <span className="text-muted-foreground truncate text-[10px]">
                          {m.model}
                        </span>
                      </div>
                      {m.name === context.model_name ? (
                        <CheckIcon className="ml-auto size-4" />
                      ) : (
                        <div className="ml-auto size-4" />
                      )}
                    </ModelSelectorItem>
                  ))}
                </ModelSelectorList>
              </ModelSelectorContent>
            </ModelSelector>
            <PromptInputSubmit
              className="rounded-full bg-brand hover:bg-brand-hover text-brand-foreground border-brand"
              disabled={disabled}
              variant="default"
              status={status}
            />
          </PromptInputTools>
        </PromptInputFooter>
      </PromptInput>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t.inputBox.followupConfirmTitle}</DialogTitle>
            <DialogDescription>
              {t.inputBox.followupConfirmDescription}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              {t.common.cancel}
            </Button>
            <Button variant="secondary" onClick={confirmAppendAndSend}>
              {t.inputBox.followupConfirmAppend}
            </Button>
            <Button onClick={confirmReplaceAndSend}>
              {t.inputBox.followupConfirmReplace}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function AddAttachmentsButton({ className }: { className?: string }) {
  const { t } = useI18n();
  const attachments = usePromptInputAttachments();
  return (
    <Tooltip content={t.inputBox.addAttachments}>
      <PromptInputButton
        className={cn("px-2!", className)}
        onClick={() => attachments.openFileDialog()}
      >
        <PaperclipIcon className="size-3" />
      </PromptInputButton>
    </Tooltip>
  );
}
