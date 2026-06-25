import type { Message, Thread } from "@langchain/langgraph-sdk";

import type { ArtifactInput } from "../artifacts/types";
import type { Todo } from "../todos";

export interface AgentThreadState extends Record<string, unknown> {
  title: string;
  messages: Message[];
  /**
   * 产物列表（spec phase0-3）：ArtifactMeta[]（chart 产物带元数据），向后兼容裸 string。
   * 消费方一律经 normalizeArtifact 兜底，勿直接当 string[] 用。
   */
  artifacts: ArtifactInput[];
  todos?: Todo[];
  /** 失败/截断 chart 摘要（后端 present_file 接出，spec §四 Step 5），无则缺省。 */
  charts_status?: {
    n_rendered?: number;
    failed?: { chart_id: string; reason: string }[];
    remaining?: { chart_id: string; reason: string }[];
  };
}

export interface AgentThreadContext extends Record<string, unknown> {
  thread_id: string;
  model_name: string | undefined;
  thinking_enabled: boolean;
  is_plan_mode: boolean;
  subagent_enabled: boolean;
  workflow_mode: "manual" | "auto";
  reasoning_effort?: "minimal" | "low" | "medium" | "high";
  agent_name?: string;
}

export interface AgentThread extends Thread<AgentThreadState> {
  context?: AgentThreadContext;
}

export interface RunMessage {
  run_id: string;
  seq?: number;
  content: Message;
  metadata: {
    caller: string;
  };
  created_at: string;
}

export interface ThreadTokenUsageResponse {
  thread_id: string;
  total_tokens: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_runs: number;
  by_model: Record<string, { tokens: number; runs: number }>;
  by_caller: {
    lead_agent: number;
    subagent: number;
    middleware: number;
  };
}
