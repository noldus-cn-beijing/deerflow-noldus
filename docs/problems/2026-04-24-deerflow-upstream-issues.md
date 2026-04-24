# DeerFlow 上游待报告问题（DeepSeek 切换暴露）

**日期**: 2026-04-24
**上游版本**: `80a7446` (deerflow/main)
**发现场景**: 把 agent 模型从 Claude Sonnet 切换到 DeepSeek V4 Pro（走 NewAPI 的 Anthropic 协议）后，前端出现 hydration 错误，消息被渲染成逐 token 一行。

切 DeepSeek 本身是合规的：NewAPI 返回的流式响应符合 Anthropic Messages API 规范（`content_block_delta` 的 `index` 保持一致，delta 按顺序累加）。问题全在 DeerFlow 自身，只是在 Sonnet 下 delta 粒度粗被掩盖了。

以下三个问题是独立的，值得分别报给上游。

---

## Issue 1: `extractContentFromMessage` 用 `\n` 拼接 content array 导致碎片化

**文件**: `frontend/src/core/messages/utils.ts:192`

```ts
if (Array.isArray(message.content)) {
  return message.content
    .map((content) => {
      switch (content.type) {
        case "text":
          return content.text;
        ...
      }
    })
    .join("\n")   // ← 问题
    .trim();
}
```

**现象**: 当 `message.content` 是 Anthropic 风格的 content array（多个 `{type: "text", text: ...}` block）时，用 `\n` 拼接。Claude 通常只吐 1 个 text block，没问题；但细粒度 delta 提供商（DeepSeek、部分 openai-compat gateway）会把一段完整回复切成几十到几百个小 text block，每个 block 之间被强制插入换行，导致：

1. Markdown 把每个 token 当独立段落 → 解析混乱
2. 出现碎片化段落里嵌块级元素（codeblock `<div>` 在段落 `<p>` 内）→ React hydration 错误 `<div> cannot be a descendant of <p>`
3. 导出的 thread markdown 文件里每个 token 一行

**复现**:
1. 用任意 delta 粒度细的 provider（DeepSeek via NewAPI 最典型）
2. 让 agent 吐一条 ~300+ 字符的回复
3. DevTools 看 `thread.messages` 里那条消息的 content 是多 text block 数组
4. 前端渲染 / 导出 markdown 会出现每 token 一行

**修复**:
相邻 text block 在语义上是连续文本（Anthropic 协议定义如此），应该用 `""` 拼接而非 `"\n"`。`extractTextFromMessage` (utils.ts:139) 有同样的 bug。

```ts
.join("")
```

Image block 之间需要换行（让图片单独成段），在 `image_url` case 的返回值里前后加 `\n` 即可。

---

## Issue 2: `streamdownPlugins` 默认启用 `rehypeRaw` 会渲染 LLM 幻觉的 HTML 标签

**文件**: `frontend/src/core/streamdown/plugins.ts`

```ts
export const streamdownPlugins = {
  remarkPlugins: [remarkGfm, [remarkMath, ...]],
  rehypePlugins: [
    rehypeRaw,                        // ← 问题
    [rehypeKatex, { output: "html" }],
  ],
};
```

**现象**: `rehypeRaw` 把 markdown 里的原生 HTML 标签转成真实 DOM 元素。当 LLM 吐出（或 streamdown 误解析）类似 `<dt>`、`<ee>` 这种孤立或不合语境的 HTML 标签时：
- 标签被当成真 DOM 渲染
- 代码块 wrapper（`<div data-code-block-container>`）出现在 `<p>` 内
- 触发 React 19 hydration 校验错误

**上游已识别相关问题**: #2321 引入了 `reasoningPlugins` —— 专门给 `<ReasoningContent>` 用，去掉了 `rehypeRaw`。但 AI 消息主体使用的 `streamdownPlugins` 仍保留 `rehypeRaw`，同样的攻击面。

**修复建议**:
把 `rehypeRaw` 从 `streamdownPlugins` 也去掉。实际上 DeerFlow 目前没有业务场景需要在 AI 消息里渲染 HTML（检查过 repo，除了 `rehype-raw` 本身没有第二处依赖）。如果未来需要，可以新增一个 opt-in 的 `htmlEnabledPlugins`，不要作为默认。

复用 #2321 引入的 `reasoningPlugins` 别名（= 去掉 rehypeRaw 的 plugins），AI 消息和 reasoning 都走它即可。

---

## Issue 3: `SummarizationMiddleware` 的内部 LLM 调用污染对话流

**文件**: `backend/packages/harness/deerflow/agents/middlewares/summarization_middleware.py`（新版）或 langchain 的 `SummarizationMiddleware`

**现象**: `_create_summary` / `_acreate_summary` 里调用 `self.model.invoke(...)` 压缩历史。这次调用没有任何 tag 标记，运行在 LangGraph 的 middleware hook 上下文里，其 token stream 会被 `stream_mode="messages"` 的 callback handler 捕获并通过 `messages-tuple` 事件广播到客户端。

对 Sonnet 这类粗粒度 delta 模型，压缩可能只产生几百字符在几个 chunk 里，前端 SDK 合并后可能只是闪过一条短消息。但对 **DeepSeek 这类细粒度 delta 模型**，压缩过程产生几百个 delta 事件，每个 delta 都被当成一条消息 chunk 广播；前端 `useStream` 把它们累积成一条"幽灵 AI 消息"，内容是整个压缩结果本身（按 token 切碎）。叠加 Issue 1 的 `.join("\n")`，就在 UI 里看到逐字逐词一行的分析摘要。

**关键证据**：
- 后端 checkpoint 里并没有存这条幽灵消息（middleware 只把压缩结果写文件 + 注入一个 pointer `HumanMessage`）
- 但前端 `thread.messages` 里有这条消息
- 消息 content 是压缩结果的原文本

**修复**:
LangGraph 已经提供了标准的 opt-out 机制 —— `langgraph.constants.TAG_NOSTREAM`（值 `"nostream"`）。在 `pregel/_messages.py:134` 的 callback handler 里会跳过含此 tag 的 LLM 调用。

最小侵入修复：

```python
# 在 SummarizationMiddleware.__init__ 尾部
self.model = self.model.with_config(tags=["nostream"])
```

一行搞定。所有后续 `.invoke()` / `.ainvoke()` 都会带上这个 tag，`messages-tuple` 流完全不会看到压缩过程的 token。

**建议**: 这个 fix 应该在 langchain 的 `SummarizationMiddleware` 层做，还是在 DeerFlow 的 `DeerFlowSummarizationMiddleware`（#2176 引入）层做？如果在 langchain 侧，所有 LangGraph agent 都受益；如果在 DeerFlow 侧，范围更可控。

更广义问题：任何 middleware 的内部 LLM 调用都有同样的流污染风险。如果 LangGraph 认可这个诉求，可以考虑给 middleware 的 `model` 属性提供 default `nostream` tag（让使用者 opt-in 广播，而非 opt-out）。

---

## 附：为什么 Sonnet 没暴露这些问题

- Sonnet 的 delta 粒度粗：一段完整回复通常落入 1 个 text block。`.join("\n")` 作用在只有 1 个元素的数组上等于无操作。
- Sonnet 极少吐出孤立的 HTML tag。`rehypeRaw` 的风险存在但不触发。
- Summarization 污染问题在 Sonnet 下也存在，但单条广播消息长度短、分段少，合并后看起来像一条正常的短 AI 消息，不明显。

换到 DeepSeek（或任何 delta 粒度细的 provider）后，三个问题同时被放大：
- Issue 1 的 `\n` 拼接在几百个 block 上产生几百行单字
- Issue 2 的 `rehypeRaw` 被 Issue 1 放大后的 markdown 喂给 streamdown，很容易触发块级元素嵌在段落里
- Issue 3 的压缩流污染产生一条完整的幽灵摘要消息

---

## 附：我们的本地修复

- Frontend `utils.ts`: `.join("\n")` → `.join("")`（两处：`extractContentFromMessage`、`extractTextFromMessage`）
- Frontend `streamdown/plugins.ts`: 去掉 `rehypeRaw`，新增 `reasoningPlugins` 别名
- Frontend `reasoning.tsx`: `<Streamdown {...props}>` → `<Streamdown {...reasoningPlugins}>`（同 #2321）
- Backend `archiving_summarization.py`: `__init__` 里 `self.model = self.model.with_config(tags=["nostream"])`

commit: `34996c9b`

---

## 提 issue 的建议顺序

1. **Issue 3 先提**（最简单，修复一行，受益面广，langchain 和 DeerFlow 两侧都可以接受）
2. **Issue 2 其次**（和 #2321 同源，容易接受扩展）
3. **Issue 1 最后**（讨论 `.join` 选择可能引战，先把前两个合了再说）
