# 0001. Subagent handoff 漏调用 harness 补轮兜底，不用 forced tool_choice / structured output

**状态**: accepted（2026-05-29）

ethoinsight subagent（code-executor / data-analyst / chart-maker / report-writer）的 LLM 有时跑完分析却忘记调 `seal_*_handoff` tool 就结束，导致 handoff 文件不产、下游断链。我们决定用 **harness 层"工程化收尾"** 解决——而非约束模型 API 层：

1. **5.7（已实施）**：executor 在标 COMPLETED 前校验 handoff 文件存在性，不存在则标 FAILED，lead 自动重派整个 subagent（兜底，贵）。
2. **5.8（seal-resume，spec 就绪）**：标 FAILED 前先就地补一轮——把 subagent 完整对话历史 + 一条聚焦的"现在请调 seal"指令再 astream 一次。补不上才走 5.7。
3. 这两层都**模型无关**：不碰 `tool_choice`、不碰 `with_structured_output`，纯 executor 执行流程。

## 为什么不用看起来更直接的方案（探针实证，防重提）

这个决策反直觉——多数人（含一个调研 subagent）会认为"强制 LLM 调 seal 不就行了"。我们用真实 deepseek-v4-pro 探针逐一证伪了那些方案，记录于此，**避免未来有人再提**：

- **forced `tool_choice`（强制模型必须调 seal）**：deepseek-v4-pro 经 dashscope 在 thinking mode 下 **400 拒绝**该参数（`tool_choice does not support being set to required or object in thinking mode`）；显式关 thinking 后强制生效，**但返回的 args 全空**——产出空内容 handoff，比漏调更糟（漏调至少 5.7 会响亮失败，空 handoff 静默一路绿灯产垃圾）。
- **`with_structured_output`（约束最终响应为 handoff schema）**：会 strip 掉 seal tool 的 `runtime` 注入参数（撞已知教训：自定义 args_schema 的 `model_validate` 丢弃未声明的 `ToolRuntime` 字段致 TypeError）。
- **executor 从 AIMessage 的 tool_calls 提取 payload 代调 seal**：漏调现场 LLM 把结论写在自由文本/thinking 里，`tool_calls` 为空，无结构化 payload 可提取；解析自由文本会引回不确定性 + 可能产半截 JSON。

## 关键约束：必须模型无关

线上现用 deepseek-v4-pro（dashscope OpenAI 兼容），**未来会换 Qwen3（vLLM / Fireworks）**。thinking、tool_choice、reasoning 字段行为在三个后端各不相同。任何依赖模型特定行为的方案（如 forced tool_choice）都会在切模型时重新踩坑。"给 subagent 一个聚焦的收尾轮"对任何 LLM 都成立——这是选 harness 补轮而非模型层约束的决定性理由。

## 已知未覆盖（已立案）

当前校验只查 handoff 文件**存在**，不查**内容**。空内容 handoff（核心字段为空）仍会静默通过。这是 seal tool 的普遍问题（非补轮引入），留给独立的「handoff 内容完整性校验」后续 sprint。

## 参考

- Sprint 5.7 spec: `docs/superpowers/specs/2026-05-28-sprint-5.7-handoff-emission-validator-design.md`
- Sprint 5.8 spec: `docs/superpowers/specs/2026-05-29-sprint-5.8-seal-resume-design.md`
- 探针结论与代码核验记录均在上述 5.8 spec §0 / §3。
