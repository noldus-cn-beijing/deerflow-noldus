# Spec 6 (P6): loop-detection 误伤 —— write_todos 触发 FORCED STOP 剥光无辜 task

> 日期：2026-06-17
> 类型：结构病根治（红线四：通用安全机制对正常长流程误判）
> 触发源：2026-06-17 EPM dogfood 末尾 —— 用户"好的→出报告"时，lead 的 `write_todos` 被调 5 次触发 `[FORCED STOP] Tool write_todos called 5 times`，**同一条消息里所有 tool_call 被剥光**，包括 lead 要派 report-writer 的 `task` call → 报告永远派不出去。
> 状态：待 review → 批准后 worktree 实施。
> **遵循工程实践**：[红线四 —— 通用安全机制必须区分"工具语义"，禁止对正常长流程一刀切误伤](../../refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md#红线四通用安全机制必须区分工具语义禁止对正常长流程一刀切误伤)。修法照"正模式 1/2/3"。

---

## 0. 症状（用户提供的 trace 实证）
```
[FORCED STOP] Tool write_todos called 5 times — exceeded the per-tool safety limit. All tool_calls stripped.
```
连续两次（第 5、第 6 次 write_todos），每次都把整条消息的 tool_call 剥光。lead 想派 `task(report-writer)` 的 call 被连坐剥掉 → E2E 流程在最后一步（出报告）卡死。

## 1. 根因（`loop_detection_middleware.py`，两个叠加缺陷）
`tool_freq_hard_limit`（默认 5，line 209）对**所有工具一视同仁**地数"同一工具被调几次"，超限就 strip 整条消息的 tool_calls（line 514-521）。

- **缺陷 A — 不分工具语义**：`write_todos` 是**记账工具**。长 E2E（code→data→ask viz→chart→ask report→report）每个阶段更新一次 todo，调 5 次**完全正常**，不是死循环。把它和"可能真死循环"的 bash/read_file 用同一个致命阈值，是红线四的典型一刀切。
- **缺陷 B — 熔断殃及无辜**：触发时 strip **整条消息所有 tool_calls**（line 514），把语义无关、正在推进流程的 `task(report-writer)` 一起剥。即便 write_todos 真在循环，也不该连坐杀掉派 report-writer 的 call。

## 2. 修法（照红线四正模式 1/2/3）

### 2.1 正模式 1：按工具语义分级（已有 `tool_freq_overrides` 机制，直接用）
`loop_detection_config.py:61` 已支持 `tool_freq_overrides`（per-tool 阈值）。给**记账类/编排类工具配置显著更高阈值或豁免**：
- `write_todos`：记账工具，长流程多次调用正常 → 阈值大幅上调（如 warn=15/hard=30）或豁免频率检测（仍保留"完全相同参数重复"的 hash 检测，那才是真循环）。
- `task`：编排工具，一次 E2E 合法派 4-5 个 subagent → 阈值上调。
- bash/read_file 保持原阈值（它们的高频才可能是真循环）。

默认配置（`LoopDetectionConfig` 的默认值或 config.yaml）带上这组 override，**不靠每次部署手配**。

### 2.2 正模式 2：熔断副作用最小化 —— 只 strip 超限的那个工具
触发频率硬限时，**只 strip 超限工具（write_todos）的 tool_call，保留同消息里其他工具的 call**（尤其推进流程的 task/seal/ask_clarification）。

实现：line 514 的"strip 全部 tool_calls"改为"strip `tool_calls` 中 name == 超限工具名 的那些，保留其余"。这样即便 write_todos 超限，`task(report-writer)` 仍能发出，流程不死。

### 2.3 正模式 3：熔断信息可操作
FORCED STOP message 已较好（line 184 告诉了工具名+次数），补一句"其余 tool_call 已保留"（若按 2.2 改）或"仅 <tool> 被移除"，避免模型以为全没了而放弃。

## 3. 测试
- `test_write_todos_high_freq_not_tripped`（**修复前必红**）：模拟合法 E2E 序列里 write_todos 调 5-6 次（每次参数不同，非重复）→ 断言**不触发** FORCED STOP（override 生效）。当前默认阈值 5 → 红。
- `test_strip_only_offending_tool`（**修复前必红**）：构造一条消息含 [write_todos(超限), task(report-writer)] → 断言 strip 后 `task` call 保留、write_todos call 被移除。当前 strip 全部 → 红。
- `test_real_loop_still_caught`：同参数 bash 真循环 5 次 → 仍触发熔断（守住防护没被削弱）。
- `test_long_e2e_sequence_completes`：喂 code→data→chart→report 的合法工具序列 → 断言全程不误触发熔断。

## 4. 风险边界
- 不削弱对真死循环的防护：hash-based "完全相同调用重复" 检测保留；只放宽"同工具类型高频"对记账/编排工具的判定。
- override 配置要进**默认值**（代码或 config.example.yaml），不是只在某个部署的 config.yaml 里——否则换环境又复发（参考 memory `feedback_dev_prod_behavior_alignment`：行为决策要进 compose/默认，不靠手配）。
- 改 middleware 后裸导入两生产入口（CLAUDE.md 导入环铁律）。
- strip 逻辑改动要核对 line 514-521 周边：stripped AIMessage 不再要求 tool response 的逻辑（line 514 注释）在"部分 strip"下仍要成立（保留的 task call 仍需后续 tool response，别制造 dangling tool call）。**实施第一步核实 DanglingToolCallMiddleware 与部分 strip 的交互。**
