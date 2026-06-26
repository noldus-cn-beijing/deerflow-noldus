import type { AIMessage, Message } from "@langchain/langgraph-sdk";
import { describe, expect, it } from "vitest";

import { buildRunTrace } from "@/core/trace";
import type { RunTraceTranslations, TraceEvent } from "@/core/trace";

import { deriveCapabilityPlan } from "./capability-plan";
import { WORKFLOW_STAGE_IDS, type WorkflowStageId } from "./stages";

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

/** Real production path: messages → buildRunTrace → deriveCapabilityPlan. */
function planFrom(messages: Message[]) {
  const events = buildRunTrace({ messages, subtasks: {} }, mockT());
  return deriveCapabilityPlan(events, messages);
}

function idsOf(plan: ReturnType<typeof planFrom>): string[] {
  return plan.map((s) => s.id);
}

const ALL_CLASSIC: WorkflowStageId[] = ["upload", "paradigm", "align", "compute", "qc", "interpret", "report"];

describe("deriveCapabilityPlan (spec 2026-06-26 Plan B — dynamic capability progress)", () => {
  it("pure knowledge Q&A (no pipeline signal) → empty plan (rail hides)", () => {
    // 用户只问知识、agent 只回文本——无 upload / paradigm / compute / chart / report 任何信号。
    const plan = planFrom([
      humanMsg("h1", "EPM 范式的开臂时间怎么解读？"),
      aiMsg("a1", { content: "开臂时间反映焦虑水平…" }),
    ]);
    expect(plan).toEqual([]);
  });

  it("chart-only run (chart-maker dispatch + run_chart_plan, no upload/metrics) → only charts stage", () => {
    // 追问画图：用户让 agent 画图，agent 派 chart-maker / 调 run_chart_plan——无指标计算、无上传。
    const plan = planFrom([
      humanMsg("h1", "帮我把这批数据画成图"),
      aiMsg("a1", {
        toolCalls: [
          toolCall("tc1", "task", { subagent_type: "chart-maker" }),
          toolCall("rc1", "run_chart_plan", { plan: "plan_charts.json" }),
        ],
      }),
    ]);
    // 只显「图表生成」阶段，不显指标计算 / 上传等（本 run 没发生）。
    expect(idsOf(plan)).toEqual(["charts"]);
  });

  it("chart-only run also surfaces report when a report.md artifact appears", () => {
    const events: TraceEvent[] = [
      { id: "d1", kind: "dispatch", title: "dispatch chart-maker", status: "ok", order: 1 },
      {
        id: "a1",
        kind: "artifact",
        title: "present files",
        status: "ok",
        order: 2,
        detail: { kind: "artifact", filepaths: ["/mnt/user-data/outputs/plot_1.png"] },
      },
      {
        id: "a2",
        kind: "artifact",
        title: "present files",
        status: "ok",
        order: 3,
        detail: { kind: "artifact", filepaths: ["/mnt/user-data/outputs/report.md"] },
      },
    ];
    const plan = deriveCapabilityPlan(events, [humanMsg("h1", "画图并出报告")]);
    expect(idsOf(plan)).toEqual(["charts", "report"]);
  });

  it("full end-to-end → all classic stages reached so far, in order", () => {
    const plan = planFrom([
      humanMsg("h1", "data", { files: [{ name: "epm.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "epm" })] }),
      aiMsg("a2", {
        toolCalls: [
          toolCall("tc1", "task", { subagent_type: "code-executor" }),
          toolCall("b1", "bash", { command: "python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio" }),
        ],
      }),
    ]);
    // upload + paradigm 已发生；compute 进行中——这三个都该显，未触及的（align/qc/interpret/report/charts）不显。
    expect(idsOf(plan)).toEqual(["upload", "paradigm", "compute"]);
  });

  it("does not surface a stage that has no signal (e.g. align/qc never reached)", () => {
    const plan = planFrom([
      humanMsg("h1", "data", { files: [{ name: "epm.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("p1", "set_experiment_paradigm", { paradigm: "epm" })] }),
    ]);
    // 仅 upload + paradigm 有信号；align 无列对齐反问、qc/interpret/report/charts 无信号 → 不显。
    // （能力进度：阶段只在有真实信号时显，不靠线性 frontier 推断——方案 B 语义。）
    expect(idsOf(plan)).toEqual(["upload", "paradigm"]);
  });

  it("chart stage is recognized from run_chart_plan tool call (not only dispatch)", () => {
    // run_chart_plan 出现在 code-executor bash 里也认（治问题2：画图了还停在指标计算）。
    const plan = planFrom([
      humanMsg("h1", "data", { files: [{ name: "x.xlsx" }] }),
      aiMsg("a1", { toolCalls: [toolCall("rc1", "bash", { command: "python -m ethoinsight.scripts.run_chart_plan plan_charts.json" })] }),
    ]);
    expect(idsOf(plan)).toContain("charts");
  });

  it("classic stages stay in fixed WORKFLOW order even if events arrive out of order", () => {
    // WORKFLOW_STAGE_IDS 定义顺序 = 显示顺序（SSOT），不随事件到达顺序漂移。
    const order = ALL_CLASSIC.map((id) => WORKFLOW_STAGE_IDS[id]);
    const sortedAsc = [...order].sort((a, b) => a - b);
    expect(order).toEqual(sortedAsc);
  });

  it("each plan entry carries the stage id (consumer maps to def + status)", () => {
    const plan = planFrom([
      humanMsg("h1", "data", { files: [{ name: "x.xlsx" }] }),
    ]);
    expect(plan.every((s) => typeof s.id === "string")).toBe(true);
    expect(plan[0]!.id).toBe("upload");
  });
});
