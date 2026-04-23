"use client";

import { useState } from "react";

import {
  submitFeedback,
  type FeedbackVerdict,
} from "@/core/api/api-client";
import { cn } from "@/lib/utils";

interface Props {
  threadId: string;
  messageId: string;
  existingVerdict?: FeedbackVerdict;
  className?: string;
}

function labelOf(v: FeedbackVerdict): string {
  if (v === "correct") return "✅ 正确";
  if (v === "needs_fix") return "⚠️ 需修正";
  return "❌ 错误";
}

export function FeedbackButtons({
  threadId,
  messageId,
  existingVerdict,
  className,
}: Props) {
  const [verdict, setVerdict] = useState<FeedbackVerdict | null>(
    existingVerdict ?? null,
  );
  const [expandedVerdict, setExpandedVerdict] =
    useState<FeedbackVerdict | null>(null);
  const [revisedText, setRevisedText] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (verdict) {
    return (
      <div
        className={cn("mt-2 text-xs text-muted-foreground", className)}
      >
        已反馈（{labelOf(verdict)}）
      </div>
    );
  }

  const handleCorrect = async () => {
    setSubmitting(true);
    try {
      await submitFeedback(threadId, { message_id: messageId, verdict: "correct" });
      setVerdict("correct");
    } catch {
      // silently ignore — feedback failure must not block the expert
    } finally {
      setSubmitting(false);
    }
  };

  const handleExpand = (v: FeedbackVerdict) => {
    setExpandedVerdict(v);
    setRevisedText("");
  };

  const handleSubmitRevision = async () => {
    if (!expandedVerdict || !revisedText.trim()) return;
    setSubmitting(true);
    try {
      await submitFeedback(threadId, {
        message_id: messageId,
        verdict: expandedVerdict,
        revised_text: revisedText.trim(),
      });
      setVerdict(expandedVerdict);
      setExpandedVerdict(null);
    } catch {
      // silently ignore
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className={cn("mt-2 flex flex-col gap-2", className)}>
      <div className="flex gap-1">
        <button
          type="button"
          aria-label="正确"
          onClick={handleCorrect}
          disabled={submitting}
          className="rounded px-2 py-1 text-xs hover:bg-accent disabled:opacity-50"
        >
          ✅ 正确
        </button>
        <button
          type="button"
          aria-label="需修正"
          onClick={() => handleExpand("needs_fix")}
          disabled={submitting}
          className="rounded px-2 py-1 text-xs hover:bg-accent disabled:opacity-50"
        >
          ⚠️ 需修正
        </button>
        <button
          type="button"
          aria-label="错误"
          onClick={() => handleExpand("wrong")}
          disabled={submitting}
          className="rounded px-2 py-1 text-xs hover:bg-accent disabled:opacity-50"
        >
          ❌ 错误
        </button>
      </div>
      {expandedVerdict && (
        <div className="flex flex-col gap-1">
          <textarea
            value={revisedText}
            onChange={(e) => setRevisedText(e.target.value)}
            placeholder={
              expandedVerdict === "needs_fix"
                ? "请写出修正版（专家版本）"
                : "请写出正确的版本"
            }
            className="min-h-[80px] rounded border p-2 text-sm"
          />
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleSubmitRevision}
              disabled={submitting || !revisedText.trim()}
              className="rounded bg-primary px-2 py-1 text-xs text-primary-foreground disabled:opacity-50"
            >
              提交
            </button>
            <button
              type="button"
              onClick={() => setExpandedVerdict(null)}
              className="rounded px-2 py-1 text-xs hover:bg-accent"
            >
              取消
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
