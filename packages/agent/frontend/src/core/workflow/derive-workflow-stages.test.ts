import type { AIMessage, Message } from "@langchain/langgraph-sdk";
import { describe, expect, it } from "vitest";

import { buildRunTrace } from "@/core/trace";
import type { RunTraceTranslations, TraceEvent } from "@/core/trace";

import { deriveWorkflowStages } from "./derive-workflow-stages";
import type { StageStatus } from "./derive-workflow-stages";

/** Minimal mock translations matching RunTraceTranslations (same shape as trace tests). */
function mockT(): RunTraceTranslations {
  return {
    toolCalls: {
      moreSteps: (n: number) => `${n} more`,
      lessSteps: "less",
      executeCommand: "执行命令",
      presentFiles: "展示文件",
      needYourHelp: "need help",
      useTool: (name: string) => `use ${name}`,
      searchFor: (q: string) => q,
      searchForRelatedInfo: "search info",
      searchForRelatedImages: "search images",
      searchForRelatedImagesFor: (q: string) => q,
      searchOnWebFor: (q: string) => q,
      viewWebPage: "view web",
      listFolder: "list folder",
      readFile: "read file",
      writeFile: "write file",
      clickToViewContent: "click",
      writeTodos: "todos",
      skillInstallTooltip: "install",
      stageBroadcast: {
        dispatchSubagent: (type: string) => `dispatch ${type}`,
        parseHeaders: "parse headers",
        resolveCatalog: "resolve catalog",
        askClarification: "需要确认",
        runScript: (name: string) => `run ${name}`,
        genericBash: "generic bash",
      },
    },
    subtasks: {
      subtask: "子任务",
      executing: (n: number) => `${n}`,
      in_progress: "运行中",
      completed: "完成",
      failed: "失败",
      taskDescription: "desc",
      taskResult: "result",
      expertWorking: "专家工作过程",
    },
    runTrace: {
      triggerLabel: "运行轨迹",
      drawerTitle: "运行轨迹",
      close: "close",
      empty: "empty",
      runningSteps: (n: number) => `${n} running`,
      stepCount: (n: number) => `${n} steps`,
      hasError: "error",
      gateTitle: "数据质量关卡",
      showGateDetail: "show",
      hideGateDetail: "hide",
      showSubSteps: "show sub",
      hideSubSteps: "hide sub",
      statusRunning: "进行中",
      statusOk: "完成",
      statusWarning: "提示",
      statusFailed: "出错",
      statusWaiting: "等待",
      kindParadigm: "范式",
      kindDispatch: "派遣",
      kindTool: "工具",
      kindGate: "关卡",
      kindClarification: "确认",
      kindArtifact: "产物",
    },
  };
}

interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  type: "tool_call";
}

function aiMsg(
  id: string,
  opts: { content?: string; toolCalls?: ToolCall[]; additionalKwargs?: Record<string, unknown> } = {},
): AIMessage {
  return {
    type: "ai",
    id,
    content: opts.content ?? "",
    tool_calls: opts.toolCalls,
    additional_kwargs: opts.additionalKwargs,
  } as AIMessage;
}

function toolCall(id: string, name: string, args: Record<string, unknown> = {}): ToolCall {
  return { id, name, args, type: "tool_call" } as ToolCall;
}

function humanMsg(id: string, content = "", opts: { files?: unknown[] } = {}): Message {
  return {
    type: "human",
    id,
    content,
    additional_kwargs: opts.files ? { files: opts.files } : undefined,
  } as Message;
}

/** Helper: derive stages straight from messages (the real production path). */
function stagesFrom(messages: Message[]) {
  const events = buildRunTrace(
    { messages, subtasks: {} },
    mockT(),
  );
  return deriveWorkflowStages(events, messages);
}

function statusOf(stages: ReturnType<typeof stagesFrom>, id: string): StageStatus {
  const s = stages.find((x) => x.id === id);
  if (!s) throw new Error(`stage ${id} missing`);
  return s.status;
}

describe("deriveWorkflowStages", () => {
  it("empty thread → all 7 stages pending (no false 'done')", () => {
    const stages = stagesFrom([]);
    expect(stages).toHaveLength(7);
    for (const s of stages) {
      expect(s.status).toBe("pending");
      expect(s.anchorMessageId).toBeUndefined();
    }
  });

  it("a human message with uploaded files → ① upload done", () => {
    const stages = stagesFrom([
      humanMsg("h1", "请分析这份数据", { files: [{ name: "epm.xlsx" }] }),
    ]);
    expect(statusOf(stages, "upload")).toBe("done");
    expect(stages.find((s) => s.id === "upload")!.anchorMessageId).toBe("h1");
  });

  it("set_experiment_paradigm → ② paradigm done, upload also done (it happened first)", () => {
    const stages = stagesFrom([
      humanMsg("h1", "data", { files: [{ name: "epm.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "epm" })] }),
    ]);
    expect(statusOf(stages, "upload")).toBe("done");
    expect(statusOf(stages, "paradigm")).toBe("done");
    // ③ align is the current frontier → active
    expect(statusOf(stages, "align")).toBe("active");
  });

  it("column-alignment ask_clarification NOT yet answered → ③ align waiting (HITL)", () => {
    const stages = stagesFrom([
      humanMsg("h1", "data", { files: [{ name: "oft.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "oft" })] }),
      aiMsg("a2", {
        toolCalls: [
          toolCall("c1", "ask_clarification", {
            clarification_type: "ambiguous_requirement",
            question: "中心区和边缘区分别对应哪两列？",
            options: ["A列/B列", "C列/D列"],
          }),
        ],
      }),
    ]);
    expect(statusOf(stages, "align")).toBe("waiting");
    expect(stages.find((s) => s.id === "align")!.anchorMessageId).toBe("a2");
  });

  it("column-alignment ask_clarification answered by a later human message → ③ align done", () => {
    const stages = stagesFrom([
      humanMsg("h1", "data", { files: [{ name: "oft.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "oft" })] }),
      aiMsg("a2", {
        toolCalls: [toolCall("c1", "ask_clarification", { question: "中心区/边缘区哪两列？" })],
      }),
      humanMsg("h2", "中心区=C1，边缘区=C2"),
    ]);
    expect(statusOf(stages, "align")).toBe("done");
    // compute is now the frontier
    expect(statusOf(stages, "compute")).toBe("active");
  });

  it("non-column clarification does NOT flip stage ③ to waiting", () => {
    const stages = stagesFrom([
      humanMsg("h1", "data", { files: [{ name: "x.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "epm" })] }),
      aiMsg("a2", {
        toolCalls: [
          toolCall("c1", "ask_clarification", {
            clarification_type: "approach_choice",
            question: "用 t-test 还是 Mann-Whitney？",
          }),
        ],
      }),
    ]);
    // not a column-alignment clarification → align stays active (current frontier), not waiting
    expect(statusOf(stages, "align")).toBe("active");
  });

  it("code-executor dispatch / metric bash → ④ compute done", () => {
    const stages = stagesFrom([
      humanMsg("h1", "data", { files: [{ name: "epm.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "epm" })] }),
      aiMsg("a2", {
        toolCalls: [
          toolCall("tc1", "task", { subagent_type: "code-executor" }),
          toolCall("b1", "bash", { command: "python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio" }),
        ],
      }),
    ]);
    expect(statusOf(stages, "compute")).toBe("done");
  });

  it("gate with warning severity → ⑤ qc warning; critical+blocks → ⑤ qc failed", () => {
    const warnStages = stagesFrom([
      humanMsg("h1", "data", { files: [{ name: "epm.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "epm" })] }),
      aiMsg("a2", {
        additionalKwargs: {
          quality_warnings: [{ severity: "warning", code: "W", metric: "m", message: "w" }],
        },
      }),
    ]);
    expect(statusOf(warnStages, "qc")).toBe("warning");

    const critStages = stagesFrom([
      humanMsg("h1", "data", { files: [{ name: "epm.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "epm" })] }),
      aiMsg("a2", {
        additionalKwargs: {
          quality_warnings: [
            { severity: "critical", code: "X", metric: "m", message: "bad", blocks_downstream: true },
          ],
        },
      }),
    ]);
    expect(statusOf(critStages, "qc")).toBe("failed");
  });

  it("data-analyst dispatch (completed) → ⑥ interpret done", () => {
    const events: TraceEvent[] = [
      { id: "p1", kind: "paradigm", title: "p", status: "ok", order: 1 },
      { id: "d1", kind: "dispatch", title: "data-analyst", status: "ok", order: 2 },
    ];
    const messages: Message[] = [
      humanMsg("h1", "data", { files: [{ name: "epm.xlsx" }] }),
    ];
    const stages = deriveWorkflowStages(events, messages);
    expect(statusOf(stages, "interpret")).toBe("done");
  });

  it("data-analyst dispatch (running) → ⑥ interpret active", () => {
    const events: TraceEvent[] = [
      { id: "p1", kind: "paradigm", title: "p", status: "ok", order: 1 },
      { id: "d1", kind: "dispatch", title: "data-analyst", status: "running", order: 2 },
    ];
    const stages = deriveWorkflowStages(events, [humanMsg("h1", "data", { files: [{}] })]);
    expect(statusOf(stages, "interpret")).toBe("active");
  });

  it("report-writer dispatch or report.md artifact → ⑦ report done", () => {
    const events: TraceEvent[] = [
      { id: "p1", kind: "paradigm", title: "p", status: "ok", order: 1 },
      { id: "d1", kind: "dispatch", title: "data-analyst", status: "ok", order: 2 },
      { id: "r1", kind: "dispatch", title: "report-writer", status: "ok", order: 3 },
    ];
    const stages = deriveWorkflowStages(events, [humanMsg("h1", "data", { files: [{}] })]);
    expect(statusOf(stages, "report")).toBe("done");
  });

  it("failed data-analyst dispatch → ⑥ interpret failed", () => {
    const events: TraceEvent[] = [
      { id: "p1", kind: "paradigm", title: "p", status: "ok", order: 1 },
      { id: "d1", kind: "dispatch", title: "data-analyst", status: "failed", order: 2 },
    ];
    const stages = deriveWorkflowStages(events, [humanMsg("h1", "data", { files: [{}] })]);
    expect(statusOf(stages, "interpret")).toBe("failed");
  });

  it("anchorMessageId is set from the first relevant trace event per stage", () => {
    const stages = stagesFrom([
      humanMsg("h1", "data", { files: [{ name: "epm.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "epm" })] }),
    ]);
    expect(stages.find((s) => s.id === "paradigm")!.anchorMessageId).toBe("a1");
  });

  it("does not mutate inputs (read-only derivation)", () => {
    const events: TraceEvent[] = [
      { id: "p1", kind: "paradigm", title: "p", status: "ok", order: 1 },
    ];
    const messages: Message[] = [humanMsg("h1", "data", { files: [{}] })];
    const evSnap = JSON.stringify(events);
    const msgSnap = JSON.stringify(messages);
    deriveWorkflowStages(events, messages);
    expect(JSON.stringify(events)).toBe(evSnap);
    expect(JSON.stringify(messages)).toBe(msgSnap);
  });

  it("currentStage points at the active frontier", () => {
    const stages = stagesFrom([
      humanMsg("h1", "data", { files: [{ name: "epm.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "epm" })] }),
    ]);
    // align is the frontier (paradigm done, compute not started)
    expect(stages.find((s) => s.status === "active")?.id).toBe("align");
  });
});
