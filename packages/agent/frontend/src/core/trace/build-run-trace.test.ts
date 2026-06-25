import type { AIMessage, Message } from "@langchain/langgraph-sdk";
import { describe, expect, it } from "vitest";

import type { Subtask } from "@/core/tasks/types";

import { buildRunTrace } from "./build-run-trace";
import type { RunTraceTranslations } from "./types";
import { summarizeTrace } from "./use-run-trace";

/** langgraph-sdk 不导出 ToolCall 类型，本地镜像其调用态结构。 */
interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  type: "tool_call";
}

/**
 * Minimal mock translations matching RunTraceTranslations
 * (Pick<Translations, "toolCalls" | "subtasks" | "runTrace">).
 * Only the fields buildRunTrace touches are exercised.
 */
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

function aiMsg(
  id: string,
  opts: {
    content?: string;
    toolCalls?: ToolCall[];
    additionalKwargs?: Record<string, unknown>;
  } = {},
): AIMessage {
  return {
    type: "ai",
    id,
    content: opts.content ?? "",
    tool_calls: opts.toolCalls,
    additional_kwargs: opts.additionalKwargs,
  } as AIMessage;
}

function toolCall(
  id: string,
  name: string,
  args: Record<string, unknown> = {},
): ToolCall {
  return {
    id,
    name,
    args,
    type: "tool_call",
  } as ToolCall;
}

function toolMsg(id: string, toolCallId: string, content: string): Message {
  return {
    type: "tool",
    id,
    tool_call_id: toolCallId,
    content,
  } as Message;
}

describe("buildRunTrace", () => {
  it("returns empty for no AI messages", () => {
    expect(buildRunTrace({ messages: [], subtasks: {} }, mockT())).toEqual([]);
  });

  it("emits a paradigm node for set_experiment_paradigm", () => {
    const msg = aiMsg("m1", {
      toolCalls: [
        toolCall("tc1", "set_experiment_paradigm", {
          paradigm: "epm",
          ev19_template: "elevated_plus_maze",
        }),
      ],
    });
    const events = buildRunTrace({ messages: [msg], subtasks: {} }, mockT());
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      kind: "paradigm",
      status: "ok",
      detail: { kind: "paradigm", paradigm: "epm", ev19Template: "elevated_plus_maze" },
    });
  });

  it("emits a clarification node with waiting status", () => {
    const msg = aiMsg("m1", {
      toolCalls: [
        toolCall("tc1", "ask_clarification", {
          question: "which column?",
          options: ["colA", "colB"],
        }),
      ],
    });
    const events = buildRunTrace({ messages: [msg], subtasks: {} }, mockT());
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      kind: "clarification",
      status: "waiting",
      title: "需要确认",
      detail: { kind: "clarification", question: "which column?", options: ["colA", "colB"] },
    });
  });

  it("emits a gate node — red when critical + blocks_downstream", () => {
    const msg = aiMsg("m1", {
      additionalKwargs: {
        quality_warnings: [
          { severity: "critical", code: "X", metric: "m", message: "bad", blocks_downstream: true },
        ],
      },
    });
    const events = buildRunTrace({ messages: [msg], subtasks: {} }, mockT());
    expect(events[0]).toMatchObject({ kind: "gate", status: "failed" });
    expect((events[0]!.detail as { warnings: unknown[] }).warnings).toHaveLength(1);
  });

  it("gate node — yellow for warning severity, green for info-only", () => {
    const warn = aiMsg("mw", {
      additionalKwargs: { quality_warnings: [{ severity: "warning", code: "W", metric: "m", message: "w" }] },
    });
    const info = aiMsg("mi", {
      additionalKwargs: { quality_warnings: [{ severity: "info", code: "I", metric: "m", message: "i" }] },
    });
    expect(buildRunTrace({ messages: [warn], subtasks: {} }, mockT())[0]).toMatchObject({ status: "warning" });
    expect(buildRunTrace({ messages: [info], subtasks: {} }, mockT())[0]).toMatchObject({ status: "ok" });
  });

  it("emits a dispatch node + matches subtask by toolCall.id + attaches subEvents", () => {
    const taskId = "task-1";
    const dispatchMsg = aiMsg("m1", {
      toolCalls: [toolCall(taskId, "task", { subagent_type: "data-analyst" })],
    });
    const subtask: Subtask = {
      id: taskId,
      subagent_type: "data-analyst",
      description: "d",
      prompt: "p",
      status: "completed",
      messages: [
        aiMsg("sub-1", {
          toolCalls: [toolCall("sub-tc-1", "bash", { command: "python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio" })],
        }),
      ],
    };
    const events = buildRunTrace({ messages: [dispatchMsg], subtasks: { [taskId]: subtask } }, mockT());
    expect(events).toHaveLength(1);
    const dispatch = events[0]!;
    expect(dispatch).toMatchObject({ kind: "dispatch", status: "ok", title: "dispatch data-analyst" });
    // subEvents: bash is visible (not in SUBTASK_HIDDEN_TOOL_CALL_NAMES), stage-broadcast applied
    expect(dispatch.subEvents).toHaveLength(1);
    expect(dispatch.subEvents![0]).toMatchObject({ kind: "tool" });
    expect(dispatch.subEvents![0]!.title).toContain("compute_open_arm_time_ratio");
  });

  it("dispatch without a matching subtask is still running", () => {
    const msg = aiMsg("m1", {
      toolCalls: [toolCall("task-x", "task", { subagent_type: "code-executor" })],
    });
    const events = buildRunTrace({ messages: [msg], subtasks: {} }, mockT());
    expect(events[0]).toMatchObject({ kind: "dispatch", status: "running" });
    expect(events[0]!.subEvents).toBeUndefined();
  });

  it("emits artifact node — ok when ToolMessage result arrived, running otherwise", () => {
    const msg = aiMsg("m1", {
      toolCalls: [toolCall("pf1", "present_files", { filepaths: ["plot.png", "report.md"] })],
    });
    // no tool result yet → running
    expect(buildRunTrace({ messages: [msg], subtasks: {} }, mockT())[0]).toMatchObject({
      kind: "artifact",
      status: "running",
    });
    // with tool result → ok
    const result = toolMsg("tm1", "pf1", "presented");
    expect(
      buildRunTrace({ messages: [msg, result], subtasks: {} }, mockT())[0],
    ).toMatchObject({ kind: "artifact", status: "ok" });
  });

  it("lead tool call → tool node; bash uses stage-broadcast title", () => {
    const msg = aiMsg("m1", {
      toolCalls: [toolCall("b1", "bash", { command: "python -m ethoinsight.catalog.resolve epm" })],
    });
    const events = buildRunTrace({ messages: [msg], subtasks: {} }, mockT());
    expect(events[0]).toMatchObject({ kind: "tool" });
    expect(events[0]!.title).toBe("resolve catalog");
  });

  it("lead tool call without a result is running; with result is ok", () => {
    const msg = aiMsg("m1", { toolCalls: [toolCall("t1", "inspect_uploaded_file", { filepath: "/a" })] });
    expect(buildRunTrace({ messages: [msg], subtasks: {} }, mockT())[0]).toMatchObject({ status: "running" });
    const withResult = buildRunTrace(
      { messages: [msg, toolMsg("tm", "t1", "ok")], subtasks: {} },
      mockT(),
    );
    expect(withResult[0]).toMatchObject({ status: "ok" });
  });

  it("preserves chronological order across messages and within a message", () => {
    const m1 = aiMsg("m1", { toolCalls: [toolCall("a", "bash", { command: "ls" })] });
    const m2 = aiMsg("m2", {
      toolCalls: [
        toolCall("b", "set_experiment_paradigm", { paradigm: "oft" }),
        toolCall("c", "task", { subagent_type: "code-executor" }),
      ],
    });
    const events = buildRunTrace({ messages: [m1, m2], subtasks: {} }, mockT());
    const ids = events.map((e) => e.id);
    // m1's bash first, then m2's paradigm, then m2's dispatch
    expect(ids).toEqual(["a", "b", "c"]);
    // orders strictly ascending
    const orders = events.map((e) => e.order);
    const sorted = [...orders].sort((x, y) => x - y);
    expect(orders).toEqual(sorted);
  });

  it("hides subagent internal tools in the SUBTASK_HIDDEN set (read_file/write_file/ls)", () => {
    const taskId = "task-2";
    const dispatchMsg = aiMsg("m1", { toolCalls: [toolCall(taskId, "task", { subagent_type: "x" })] });
    const subtask: Subtask = {
      id: taskId,
      subagent_type: "x",
      description: "",
      prompt: "",
      status: "in_progress",
      messages: [
        aiMsg("s1", {
          toolCalls: [
            toolCall("h1", "read_file", { path: "/a" }),
            toolCall("h2", "bash", { command: "echo hi" }),
          ],
        }),
      ],
    };
    const events = buildRunTrace({ messages: [dispatchMsg], subtasks: { [taskId]: subtask } }, mockT());
    // read_file hidden, bash visible → 1 subEvent
    expect(events[0]!.subEvents).toHaveLength(1);
    expect(events[0]!.subEvents![0]!.id).toBe("h2");
  });

  it("does not mutate inputs (read-only derivation)", () => {
    const msg = aiMsg("m1", { toolCalls: [toolCall("a", "bash", { command: "ls" })] });
    const subtasks = {};
    const messages = [msg];
    const snapshot = JSON.stringify(messages);
    buildRunTrace({ messages, subtasks }, mockT());
    expect(JSON.stringify(messages)).toBe(snapshot);
  });
});

describe("summarizeTrace (entry button badge)", () => {
  it("idle when empty", () => {
    expect(summarizeTrace([])).toMatchObject({ total: 0, running: 0, hasError: false, triggerState: "idle" });
  });

  it("running state when a node is running or waiting", () => {
    const events = [
      { id: "1", kind: "dispatch", title: "x", status: "running", order: 1 },
      { id: "2", kind: "tool", title: "y", status: "ok", order: 2 },
    ] as const;
    expect(summarizeTrace([...events])).toMatchObject({ total: 2, running: 1, triggerState: "running" });
  });

  it("error state wins over running when any failed/gate-red present", () => {
    const events = [
      { id: "1", kind: "gate", title: "g", status: "failed", order: 1 },
      { id: "2", kind: "dispatch", title: "d", status: "running", order: 2 },
    ] as const;
    const s = summarizeTrace([...events]);
    expect(s.hasError).toBe(true);
    expect(s.triggerState).toBe("error");
  });
});
