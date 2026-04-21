import type { AIMessage } from "@langchain/langgraph-sdk";

export interface Subtask {
  id: string;
  status: "in_progress" | "completed" | "failed";
  subagent_type: string;
  description: string;
  prompt: string;
  result?: string;
  error?: string;

  /**
   * All AI messages emitted by the subagent, accumulated from each
   * `task_running` SSE event in the order they arrived. Replay-safe: entries
   * are deduplicated on `message.id`.
   */
  messages: AIMessage[];

  /**
   * Convenience pointer to messages[messages.length - 1]. Retained so the
   * collapsed card can keep rendering `explainLastToolCall(latestMessage)`
   * without iterating the array.
   */
  latestMessage?: AIMessage;
}
