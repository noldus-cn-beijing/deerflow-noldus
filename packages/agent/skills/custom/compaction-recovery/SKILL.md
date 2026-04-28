---
name: compaction-recovery
description: >
  Behavior guide for handling conversation compaction. When an earlier
  segment of the dialogue has been compressed into a summary file, this
  skill tells the agent where the file lives, when to read it, and how to
  verify pending items against it before acting.
version: 1.0.0
author: noldus-insight
license: MIT
---

# Compaction Recovery

当对话过长时，harness 会自动把较早的消息压缩到沙盒文件里，保留最近几条消息不动。
你需要知道这个机制怎么运作、什么时候该读文件、怎么避免用过时信息做决定。

## 发生了什么

- 压缩触发时，较早的消息从上下文里被移除
- 一条短小的"指针"HumanMessage 会出现（内容类似 `[系统] 前序对话已压缩并追加到 ...`）
- 指针只告诉你**文件位置**，**不包含**历史内容本身

## 摘要文件位置

**固定路径**：`/mnt/user-data/workspace/conversation_summary.md`

这个文件以 markdown 追加方式积累：每次压缩追加一段带时间戳的 heading，按时间顺序排列。
同一 thread 多次压缩共享这一个文件——不会有多个文件需要挑选。

## 什么时候必须读文件

遵循 **Compaction Recovery** 原则：

> Summary 状态可能过时。Post-compaction 必须验证 pending items 的文件级真相。

具体规则：

1. **处理指针消息之前提到的未完成工作时**——先 `read_file` 读摘要，对齐之前做到哪步
2. **需要引用历史事实（文件名、参数、分组信息、之前的决策）时**——必须读文件而不是凭印象
3. **用户追问"之前说过…"时**——读文件，不要猜
4. **进行长链条推理时**（比如多步分析报告的收尾）——读文件核对前序步骤的输出

反之，**不需要**读文件的场景：

- 用户抛新问题，和前序压缩内容无关（比如"你好"、纯知识问答）
- 当前未压缩消息已经包含所需全部信息
- 只需要感谢/告别类的短回复

## 怎么读

```
使用 read_file 工具：
  path = "/mnt/user-data/workspace/conversation_summary.md"
```

如果文件不存在，说明本 thread 还没触发过压缩——这是正常情况，继续回答即可。

## 读完之后：无缝续跑未完成的工作

压缩是后台机制，**用户并没有主动发消息**——指针 HumanMessage 是系统合成的，
用户的屏幕上看不到它，你看到它只是因为模型上下文里需要一条挂载锚点。

所以 post-compaction 时你的默认行为不是"打招呼 / 汇报状态 / 问下一步"，
而是**直接从摘要里的断点继续推进**。

具体判断流程：

1. 读摘要，找"待完成步骤"（⏳ / pending / 未完成 / TODO 等任意标记形式）
2. 如果有未完成条目 → **立即调对应工具继续推进**（通常是 `task` 分派给 subagent）
3. 如果摘要里所有步骤都是 ✅ 已完成 → 当前用户问题也已回答完毕，**保持沉默**等下一轮真实用户输入
4. 如果摘要不清楚、或状态矛盾 → 用 `read_file` 读 handoff JSON / 输出文件交叉核对，不要猜

**正面示例**（摘要显示 `data-analyst 解读 ⏳`）：

> [读完摘要] → 直接调用 `task(subagent_type="data-analyst", ...)` 继续 pipeline

**反面示例**（不要这样）：

- ❌ "您好！前序对话已恢复..."（用户没发话，不需要回应）
- ❌ "我注意到系统触发了对话压缩"（用户不需要知道这个机制）
- ❌ "请问您现在有什么新的需求？"（摘要里明确写着 pending items，新需求是继续推进）
- ❌ "之前已完成了一轮完整分析"（在摘要明确列出 ⏳ 未完成时做此断言 = 幻觉）

## 验证而不是采信

摘要是**有损压缩**。如果涉及**具体数据**（文件路径、参数值、数值结果），优先找
**原始文件**（`/mnt/user-data/uploads/` 里的用户上传、`/mnt/user-data/outputs/` 里
的分析产物、`/mnt/user-data/workspace/` 里的 handoff JSON）交叉验证，不要只信摘要。

## 不要做的事

- ❌ 不要把摘要内容**复述**给用户（用户已经知道自己说过什么）
- ❌ 不要在自己的回答里**模仿**摘要的结构化 dump 格式（`## Extracted Context`、
  `**Task**: ...` 这种）——那是给压缩摘要用的，不是给用户看的
- ❌ 不要**每轮都读**摘要文件——按需读取，不要浪费 token
