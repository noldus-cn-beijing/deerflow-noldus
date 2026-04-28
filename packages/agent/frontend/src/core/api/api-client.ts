"use client";

import { Client as LangGraphClient } from "@langchain/langgraph-sdk/client";

import { getLangGraphBaseURL } from "../config";

import { sanitizeRunStreamOptions } from "./stream-mode";

function createCompatibleClient(isMock?: boolean): LangGraphClient {
  const client = new LangGraphClient({
    apiUrl: getLangGraphBaseURL(isMock),
  });

  const originalRunStream = client.runs.stream.bind(client.runs);
  client.runs.stream = ((threadId, assistantId, payload) =>
    originalRunStream(
      threadId,
      assistantId,
      sanitizeRunStreamOptions(payload),
    )) as typeof client.runs.stream;

  const originalJoinStream = client.runs.joinStream.bind(client.runs);
  client.runs.joinStream = ((threadId, runId, options) =>
    originalJoinStream(
      threadId,
      runId,
      sanitizeRunStreamOptions(options),
    )) as typeof client.runs.joinStream;

  return client;
}

const _clients = new Map<string, LangGraphClient>();
export function getAPIClient(isMock?: boolean): LangGraphClient {
  const cacheKey = isMock ? "mock" : "default";
  let client = _clients.get(cacheKey);

  if (!client) {
    client = createCompatibleClient(isMock);
    _clients.set(cacheKey, client);
  }

  return client;
}

export type FeedbackVerdict = "correct" | "needs_fix" | "wrong";

export interface FeedbackRequest {
  message_id: string;
  verdict: FeedbackVerdict;
  revised_text?: string;
  note?: string;
}

export interface FeedbackItem extends FeedbackRequest {
  submitted_at: string;
}

export async function submitFeedback(
  threadId: string,
  body: FeedbackRequest,
): Promise<{ success: boolean }> {
  const res = await fetch(`/api/threads/${threadId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`submitFeedback failed: ${res.status}`);
  return res.json() as Promise<{ success: boolean }>;
}

export async function listFeedback(
  threadId: string,
): Promise<{ items: FeedbackItem[] }> {
  const res = await fetch(`/api/threads/${threadId}/feedback`);
  if (!res.ok) throw new Error(`listFeedback failed: ${res.status}`);
  return res.json() as Promise<{ items: FeedbackItem[] }>;
}
