import type { Message } from "@langchain/langgraph-sdk";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  extractContentFromMessage,
  extractReasoningContentFromMessage,
  getMessageGroups,
  groupMessages,
  hasContent,
  hasReasoning,
} from "./utils";

function makeAIMsg(
  id: string,
  opts: {
    content?: string;
    reasoning?: string;
    toolCalls?: Array<{ name: string; args: Record<string, unknown> }>;
  } = {},
): Message {
  return {
    type: "ai",
    id,
    content: opts.content ?? "",
    additional_kwargs:
      opts.reasoning != null ? { reasoning_content: opts.reasoning } : {},
    tool_calls: opts.toolCalls,
  } as Message;
}

function makeHumanMsg(id: string, content = "hi"): Message {
  return { type: "human", id, content } as Message;
}

function makeToolMsg(
  id: string,
  opts: {
    toolCallId: string;
    name?: string;
    content?: string;
  },
): Message {
  return {
    type: "tool",
    id,
    name: opts.name ?? "some_tool",
    tool_call_id: opts.toolCallId,
    content: opts.content ?? "{...}",
  } as Message;
}

function groupTypes(messages: Message[]): string[] {
  return groupMessages(messages, (g) => g.type);
}

describe("groupMessages", () => {
  it("AI message: reasoning + content + no tool_calls → 1 group, type 'assistant'", () => {
    const msg = makeAIMsg("1", {
      content: "Hello",
      reasoning: "Let me think...",
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant"]);
  });

  it("AI message: reasoning + tool_calls → 1 group, type 'assistant:processing'", () => {
    const msg = makeAIMsg("1", {
      content: "Using a tool...",
      reasoning: "I need to call a tool",
      toolCalls: [{ name: "search", args: {} }],
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant:processing"]);
  });

  it("AI message: reasoning only → 1 group, type 'assistant:processing'", () => {
    const msg = makeAIMsg("1", {
      reasoning: "Let me think...",
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant:processing"]);
  });

  it("AI message: content only → 1 group, type 'assistant'", () => {
    const msg = makeAIMsg("1", {
      content: "Hello",
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant"]);
  });

  it("AI message: tool_calls only → 1 group, type 'assistant:processing'", () => {
    const msg = makeAIMsg("1", {
      toolCalls: [{ name: "search", args: {} }],
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant:processing"]);
  });

  it("2 consecutive AI messages: 1st reasoning+tool_calls, 2nd reasoning+content → 2 groups", () => {
    const msg1 = makeAIMsg("1", {
      content: "Using a tool...",
      reasoning: "I need to call a tool",
      toolCalls: [{ name: "search", args: {} }],
    });
    const msg2 = makeAIMsg("2", {
      content: "Here is the answer",
      reasoning: "Now I can answer",
    });
    const result = groupTypes([msg1, msg2]);
    expect(result).toEqual(["assistant:processing", "assistant"]);
  });
});

describe("getMessageGroups — 流式瞬态孤儿 tool message (spec 2026-06-25)", () => {
  // EPM dogfood（thread 0e72d605）console 报 `Unexpected tool message outside a
  // processing group {}`。真根因=流式分片时序：AI message 的 content/tool_calls/
  // reasoning 分片分批到达；在 tool_calls 未到、content 仍空的瞬态帧里，AI message
  // 五个分支全不命中 → 不开任何组 → 其 tool result 流到即成孤儿 → console.error + 丢弃。
  // 修法：M1 任何 AI message（含暂时全空壳）都进/开 processing 组；M2 孤儿兜底进
  // processing 组不丢弃不 error。

  beforeEach(() => {
    // 每个用例独立 spy，避免互相干扰断言「无 console.error」。
    vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  // T1（治本）：流式瞬态——空 AI message + 其 tool result，不产生孤儿
  it("T1: streaming transient — empty AI message (no content/tool_calls/reasoning) gets a processing group, its tool result attaches (no orphan, no console.error)", () => {
    const msgs = [
      makeHumanMsg("h1", "A. 是画图"),
      // 流式中途：set_viz_choice 的 tool_calls 尚未到达，content 本就空 → 空壳 AI message
      makeAIMsg("a1", {}),
      makeToolMsg("t1", {
        name: "set_viz_choice",
        toolCallId: "tc1",
        content: "viz_choice set",
      }),
    ];
    const groups = getMessageGroups(msgs);

    // a1 必须开了一个 assistant:processing 组（消除孤儿源头）
    expect(groups.some((g) => g.type === "assistant:processing")).toBe(true);

    // t1 必须挂进那个 processing 组的 messages（不再孤儿/丢弃）
    const processingGroup = groups.find(
      (g) => g.type === "assistant:processing",
    );
    expect(processingGroup?.messages.map((m) => m.id)).toContain("t1");

    // 不再 console.error 丢弃（治本 + 兜底双重保证）
    expect(console.error).not.toHaveBeenCalled();
  });

  // T2（兜底）：纯孤儿 tool message（前面是 human，没有任何 open group）不丢弃、不 error
  it("T2: orphan tool after human opens a processing group (not dropped, no console.error)", () => {
    const msgs = [
      makeHumanMsg("h1", "A"),
      makeToolMsg("t1", {
        name: "x",
        toolCallId: "tc1",
        content: "r",
      }),
    ];
    const groups = getMessageGroups(msgs);

    // human 之后多了至少一个组（兜底开的 processing 组容纳 t1）
    expect(groups.length).toBeGreaterThan(1);

    // t1 进了一个 processing 组（兜底容器，对 tool message 渲染安全）
    const orphanGroup = groups.find((g) => g.messages.some((m) => m.id === "t1"));
    expect(orphanGroup?.type).toBe("assistant:processing");

    // 不再 console.error 丢弃
    expect(console.error).not.toHaveBeenCalled();
  });

  // T3（回归）：正常 AI+tool 序列分组不变（M1 不破坏既有行为）
  it("T3: regression — normal ai-with-toolcall + tool result groups unchanged", () => {
    const msgs = [
      makeHumanMsg("h1", "分析"),
      makeAIMsg("a1", {
        content: "调用工具",
        reasoning: "思考",
        toolCalls: [{ name: "search", args: {} }],
      }),
      makeToolMsg("t1", { toolCallId: "call_a1", content: "结果" }),
      makeAIMsg("a2", { content: "最终答复" }),
    ];
    // a1(reasoning+tool_calls) → processing；t1 挂进 processing；a2(content only) → assistant
    expect(groupTypes(msgs)).toEqual([
      "human",
      "assistant:processing",
      "assistant",
    ]);
    expect(console.error).not.toHaveBeenCalled();
  });

  // T4（回归）：present_files / subagent / clarification 终结组行为不变
  it("T4: regression — present-files / subagent / clarification grouping unchanged", () => {
    const presentFilesMsg = makeAIMsg("pf", {
      toolCalls: [{ name: "present_files", args: { filepaths: ["a.png"] } }],
    });
    expect(groupTypes([presentFilesMsg])).toEqual(["assistant:present-files"]);

    const subagentMsg = makeAIMsg("sg", {
      toolCalls: [{ name: "task", args: {} }],
    });
    expect(groupTypes([subagentMsg])).toEqual(["assistant:subagent"]);

    // clarification：ask_clarification tool message 开独立 clarification 组
    const clarTool = makeToolMsg("ct", {
      name: "ask_clarification",
      toolCallId: "call_clar",
      content: "请选择",
    });
    expect(groupTypes([clarTool])).toEqual(["assistant:clarification"]);
  });

  // T5（回归）：真实 dogfood 0e72d605 序列——无孤儿、关键 set_viz_choice 瞬态覆盖
  it("T5: real dogfood 0e72d605 sequence — set_viz_choice transient (empty AI) does not orphan, grouping stays sane", () => {
    // 真实序列的最易触发点：用户回答「A.是画图」→ lead 调 set_viz_choice。
    // [19] ai content='' tool_calls=['set_viz_choice'] 在流式中途 tool_calls 未到时
    // 是 content 空的空壳 → 其 [20] set_viz_choice 的 result 成孤儿的窗口最长最稳。
    const msgs = [
      makeHumanMsg("h_q", "你希望我以图表形式展示结果吗？"),
      makeAIMsg("a_ask", { content: "你希望我以图表形式展示结果吗？" }),
      makeHumanMsg("h_a", "A. 是画图"),
      // [19] 流式中途空壳（tool_calls 未到），最易触发孤儿
      makeAIMsg("a_viz", {}),
      // [20] set_viz_choice result
      makeToolMsg("t_viz", {
        name: "set_viz_choice",
        toolCallId: "call_viz",
        content: "ok",
      }),
      makeAIMsg("a_final", { content: "好的，正在生成图表。" }),
    ];
    const groups = getMessageGroups(msgs);

    // a_viz 的空壳开了一个 processing 组，t_viz 挂进去，不孤儿
    const processingGroup = groups.find(
      (g) => g.type === "assistant:processing",
    );
    expect(processingGroup?.messages.map((m) => m.id)).toContain("t_viz");

    // 最终态：a_final 的 content 渲染为 assistant 组（最终态正常，符合「只在流式中途报错」）
    expect(groups.some((g) => g.type === "assistant")).toBe(true);

    // 全程无 console.error
    expect(console.error).not.toHaveBeenCalled();
  });
});

describe("groupMessages — streaming continuity (no flicker)", () => {
  // Streaming order from server: reasoning chunks → content chunks → tool_calls
  // chunks (if any). Each chunk re-classifies the same message. Without
  // streaming-aware classification, the SAME message id jumps between groups
  // (reasoning → assistant:processing, then +content+no tool_calls →
  // assistant, then +tool_calls → assistant:processing), unmounting and
  // remounting the React tree on every chunk = visible "flicker / reload".
  //
  // Fix: while a message is the last in the array AND isStreaming=true, keep
  // it pinned to processing instead of promoting to the final "assistant"
  // group. After the stream ends, re-classification picks the final group.

  it("isStreaming=true: last AI msg with reasoning+content (no tools yet) stays in processing", () => {
    const msg = makeAIMsg("1", {
      content: "partial answer streaming...",
      reasoning: "deciding next action",
    });
    const result = groupMessages([msg], (g) => g.type);
    expect(result).toEqual(["assistant:processing"]);
  });

  it("isStreaming=false: same message classifies as assistant (no pinning)", () => {
    const msg = makeAIMsg("1", {
      content: "complete answer",
      reasoning: "decided",
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant"]);
  });

  it("isStreaming=true: non-last messages classify normally (pinning ONLY on last)", () => {
    const msg1 = makeAIMsg("1", {
      content: "first turn final",
      reasoning: "thought 1",
    });
    const msg2 = makeAIMsg("2", {
      content: "second turn streaming",
      reasoning: "thought 2",
    });
    const result = groupMessages([msg1, msg2], (g) => g.type);
    expect(result).toEqual(["assistant", "assistant:processing"]);
  });
});

describe("inline <think> handling — streaming TTFT", () => {
  it("closed <think>...</think> in string content: reasoning extracted, content cleaned", () => {
    const msg = makeAIMsg("1", {
      content: "<think>I will analyze</think>Here is the answer",
    });
    expect(extractReasoningContentFromMessage(msg)).toBe("I will analyze");
    expect(extractContentFromMessage(msg)).toBe("Here is the answer");
  });

  it("multiple closed <think> blocks: all reasoning concatenated", () => {
    const msg = makeAIMsg("1", {
      content: "<think>first</think>visible<think>second</think> answer",
    });
    expect(extractReasoningContentFromMessage(msg)).toBe("first\n\nsecond");
    expect(extractContentFromMessage(msg)).toBe("visible answer");
  });

  it("unclosed <think> at end (mid-stream): reasoning streams live, content empty", () => {
    const msg = makeAIMsg("1", {
      content: "<think>I am still thinking about",
    });
    expect(extractReasoningContentFromMessage(msg)).toBe("I am still thinking about");
    expect(extractContentFromMessage(msg)).toBe("");
  });

  it("closed <think> + open trailing <think> (mid-stream second block): both reasonings shown", () => {
    const msg = makeAIMsg("1", {
      content: "<think>first done</think>partial answer<think>second thought streaming",
    });
    expect(extractReasoningContentFromMessage(msg)).toBe(
      "first done\n\nsecond thought streaming",
    );
    expect(extractContentFromMessage(msg)).toBe("partial answer");
  });

  it("content without any <think>: returned as-is, no reasoning", () => {
    const msg = makeAIMsg("1", {
      content: "plain answer with no thinking",
    });
    expect(extractReasoningContentFromMessage(msg)).toBeNull();
    expect(extractContentFromMessage(msg)).toBe("plain answer with no thinking");
  });

  it("unclosed <think> with empty body (just <think> typed): does not crash, no reasoning yet", () => {
    const msg = makeAIMsg("1", {
      content: "<think>",
    });
    // Open tag with nothing after it — reasoning is empty string, treat as null
    expect(extractReasoningContentFromMessage(msg)).toBeNull();
    expect(extractContentFromMessage(msg)).toBe("");
  });

  it("preamble before an unclosed <think> stays in content", () => {
    const msg = makeAIMsg("1", {
      content: "Here is part of the answer.<think>but wait, let me reconsider",
    });
    expect(extractContentFromMessage(msg)).toBe("Here is part of the answer.");
    expect(extractReasoningContentFromMessage(msg)).toBe(
      "but wait, let me reconsider",
    );
  });

  it("hasReasoning recognises an unclosed <think> tag mid-stream", () => {
    expect(hasReasoning(makeAIMsg("1", { content: "<think>thinking in progress" }))).toBe(true);
  });

  it("hasContent excludes an unclosed <think> tail when no preamble exists", () => {
    expect(hasContent(makeAIMsg("1", { content: "<think>thinking in progress" }))).toBe(false);
  });

  it("hasContent stays true when preamble precedes an unclosed <think>", () => {
    expect(hasContent(makeAIMsg("1", { content: "preamble<think>still thinking" }))).toBe(true);
  });

  it("a literal <think> inside markdown inline code is not treated as reasoning", () => {
    const msg = makeAIMsg("1", {
      content: "Use `<think>` markers to delimit reasoning sections.",
    });
    expect(extractContentFromMessage(msg)).toBe(
      "Use `<think>` markers to delimit reasoning sections.",
    );
    expect(extractReasoningContentFromMessage(msg)).toBeNull();
    expect(hasReasoning(msg)).toBe(false);
  });

  it("a backtick-prefixed <think> mid-stream is not split into reasoning", () => {
    const msg = makeAIMsg("1", {
      content: "Documentation: `<think>",
    });
    expect(extractContentFromMessage(msg)).toBe(
      "Documentation: `<think>",
    );
    expect(extractReasoningContentFromMessage(msg)).toBeNull();
  });
});

describe("extractContentFromMessage — 后端控制协议方括号信号 strip (spec 2026-06-24)", () => {
  // [intent] / [gate_signals] 是后端控制协议文法，写在 AIMessage 正文里：
  //  - [intent] 被 IntentClassificationGuardrailProvider 从历史 content 读取（W17）。
  //  - [gate_signals] 被 lead 从历史 content 读取做下游决策。
  // 它们对研究员零信息量且破坏专业观感。content 必须对后端保持完整（guardrail/lead
  // 依赖从历史读），所以只在【前端 render 入口 extractContentFromMessage】行级 strip。

  it("1. 单行 [intent] E2E_FULL → strip 后只剩自然语言", () => {
    const msg = makeAIMsg("1", {
      content:
        '查看其他 1 个步骤\n使用 "identify_ev19_template" 工具\n\n[intent] E2E_FULL\n\n好的，我看到你上传了28个EPM实验数据文件。',
    });
    expect(extractContentFromMessage(msg)).toBe(
      '查看其他 1 个步骤\n使用 "identify_ev19_template" 工具\n\n好的，我看到你上传了28个EPM实验数据文件。',
    );
  });

  it("2. 单行紧凑 [gate_signals] ... → strip 干净", () => {
    const msg = makeAIMsg("1", {
      content:
        "计划已生成：28只动物，4组各7只。\n\n[gate_signals] constitution_acknowledged: true data_quality: critical_count: 0 warning_count: 0 critical_items: [] statistical_validity: ok errors_count: 0\n\n执行摘要：分析完成。",
    });
    expect(extractContentFromMessage(msg)).toBe(
      "计划已生成：28只动物，4组各7只。\n\n执行摘要：分析完成。",
    );
  });

  it("3. 多行块 [gate_signals]\\ncharts_generated: 1\\nchart_files:\\n  - x.png → 整块 strip", () => {
    const msg = makeAIMsg("1", {
      content:
        "开始绘制图表。\n\n[gate_signals]\ncharts_generated: 3\nchart_files:\n  - plot_box_open_arm.png\n  - plot_bar_entries.png\nstatistics_status: ok\n\n绘制完成。",
    });
    expect(extractContentFromMessage(msg)).toBe(
      "开始绘制图表。\n\n绘制完成。",
    );
  });

  it("4. [intent] 夹在自然语言中间 → 只删 [intent] 行，保留前后自然语言", () => {
    const msg = makeAIMsg("1", {
      content:
        "🧮 正在计算指标，预计 30-60 秒...\n\n[intent] E2E_FULL\n\n子任务已完成，结果如下：",
    });
    expect(extractContentFromMessage(msg)).toBe(
      "🧮 正在计算指标，预计 30-60 秒...\n\n子任务已完成，结果如下：",
    );
  });

  it("5. 合法用户输入保护：user message 含 [intent]-like 文本不被误 strip", () => {
    // extractContentFromMessage 对 user message 也按 content 字符串处理；
    // strip 只删「行首 [intent]/[gate_signals]」整行，研究员句子中间提到的 [intent]
    // 不会整行以方括号标记开头，故安全保留。
    const userMsg = {
      type: "human",
      id: "u1",
      content: "研究员问：什么是 [intent] 标记？为什么对话里会冒出来？",
    } as Message;
    expect(extractContentFromMessage(userMsg)).toBe(
      "研究员问：什么是 [intent] 标记？为什么对话里会冒出来？",
    );
  });

  it("6. content 完整性：strip 仅作用于展示派生，message.content 字段本身不变（未被 mutate）", () => {
    const original =
      "[intent] E2E_FULL\n\n这是给研究员看自然语言正文。";
    const msg = makeAIMsg("1", { content: original });
    // 展示层 strip 掉了 [intent] 行
    expect(extractContentFromMessage(msg)).toBe("这是给研究员看自然语言正文。");
    // 但 message.content 原样保留（后端 guardrail/lead 从历史 content 读取依赖此不变性）
    expect(msg.content).toBe(original);
  });
});
