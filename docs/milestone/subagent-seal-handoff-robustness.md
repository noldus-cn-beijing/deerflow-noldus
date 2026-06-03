# Subagent seal/handoff 鲁棒性 — 真根因澄清 + 纯 prompt 修复

**状态**：in-progress（核心卡死已解，洞察优化 spec 待实施）
**时间跨度**：2026-05-29 ~ 2026-06-03
**dev HEAD**：`5915a55f`（本地领先 origin/dev 5 commit，rebase 于 PR#83 之上，待 push）

## 做了什么

FST n=1 dogfood 反复出现 `data-analyst terminated without emitting 'handoff_data_analyst.json'`，横跨 5-29~6-02 至少 6 个 thread 复发，期间被**多次误诊**。本次会话用**两次完整 dogfood transcript 对比 + 11 次真 deepseek-v4-pro 实验**把真根因钉死，并用**纯 prompt 修复**(不碰 harness 承重墙)一次性解决——dogfood 重启后 data-analyst **第一次派遣即 seal 成功**。

真根因（三层叠加，与历次误诊不同）：
1. **触发器（主因）= data-analyst step 2.8 prompt 三条矛盾指令**（"跳过空 `{}`" vs "判据不可用记 info 降级" vs "去读 plan_metrics.json 含全量 parameters_in_use"）→ 诱导模型从 plan 捞 12 个幽灵参数硬做降级审计 → 烧 3-5 turn 叙述黑洞。
2. **放大器 = deepseek 把"在 thinking/正文里写了'封存'" ≈ "完成了封存动作"**，不区分写文本与发出 seal tool_call。
3. **放大器 = `executor.py:888-911` turn 计数按 AIMessage 条数算、不看 tool_calls**，纯叙述等价吃预算 + max_turns=12 无容错。

**被实验证伪的两个旧假设**：① "幽灵参数（compute 回显 + code-executor 从 plan 回退）修完就好"——那两层都修了生效了，卡死照旧；② "预算太少 / 关 think 致推理外漏吃预算"——`reasoning_content` 关 think 时也始终存在，开 think 不显著减 AIMessage 数（11 次实验：ON 100% / OFF 87.5%，方向对但样本太小不显著）。

## 关键节点

| 日期 | 事件 | handoff / 文档 |
|------|------|---------|
| 2026-06-03 | 上一会话交接"幽灵参数双层修复（未 commit）"——其"修完就好"结论后被本会话推翻 | [dual-root-cause-fix-handoff](../handoffs/2026-06/2026-06-03-fst-seal-deadlock-dual-root-cause-fix-handoff.md) |
| 2026-06-03 | 核实两层修复已生效但卡死照旧 → 双 transcript 对比定位 step 2.8 prompt 矛盾为真根因 | [problem-statement](../problems/2026-06-03-subagent-seal-deadlock-problem-statement.md) |
| 2026-06-03 | 独立 agent 11 次 deepseek 实验：证伪"开 think 必要"，确认 prompt 矛盾为主因 | [reply-to-four-layer-proposal](../problems/2026-06-03-reply-to-four-layer-proposal.md) |
| 2026-06-03 | 第 1 批纯 prompt 修复实施 + review 通过 | [batch1-implementation-handoff](../problems/2026-06-03-batch1-implementation-handoff.md) |
| 2026-06-03 | dogfood 重启验证：data-analyst 第一次派遣即 seal，parameter_audit_findings=0；全量回归 553+3612 绿（2 pre-existing fail 非回归）；4 commit 合 dev | — |

## 当前状态

- **完成项**：
  - 真根因澄清（prompt 矛盾，非幽灵参数/非预算/非 think），三层机制写入 memory `feedback_subagent_seal_deadlock_is_prompt_not_budget`。
  - 三层幽灵参数修复全部 commit（ethoinsight 路径化裁剪 `f085f0bb` + code-executor 真相源 `bce25a1e` + data-analyst step2.8 分流 `db598b8c`）——前两层是基础，第三层是真根因解。
  - data-analyst prompt：step 2.8 空 `{}` 真跳过（禁从 plan 捞）+ step 3 正面措辞"完成标志=发出 seal tool_call"。配 8 条 prompt 文本契约测试防 sync 回退。
  - dogfood 实证 + 全量回归绿 + rebase 吸收 PR#83 前端现代化。
- **遗留项**：
  - 5 个 commit 待 push origin/dev（rebase 后 fast-forward）。
  - §5.4「无 tool_call」卡死模式（trace b7566a33/cf80346f，data-analyst 整轮无 tool_call）本次未碰——本次幽灵参数解消除了最主要诱因，若未来在别的 step 复现需另案（不走强制 tool_choice，executor.py:712 探针证明产空 args）。
  - config.yaml 明文 API key（无关本 track，建议单独外移环境变量）。
- **下一 milestone**：data-analyst 开 thinking 提升洞察质量（spec 已就绪 `5915a55f`，**仅 data-analyst 一个**开、其余全保持关；与卡死修复正交；碰受保护文件 executor.py 1 行，待新 agent 实施 + dogfood 验洞察增益）。

## 相关 handoff / 文档

- [2026-06-03 dual-root-cause-fix handoff](../handoffs/2026-06/2026-06-03-fst-seal-deadlock-dual-root-cause-fix-handoff.md) — 上一会话交接（结论已被本会话修正）
- [problem-statement](../problems/2026-06-03-subagent-seal-deadlock-problem-statement.md) — 给独立 agent 的自包含问题说明
- [reply-to-four-layer-proposal](../problems/2026-06-03-reply-to-four-layer-proposal.md) — 对四层方案的核实 + 实验结论
- [batch1-implementation-handoff](../problems/2026-06-03-batch1-implementation-handoff.md) — 第 1 批纯 prompt 修复实施单
- [data-analyst-enable-thinking-spec](../superpowers/specs/2026-06-03-data-analyst-enable-thinking-spec.md) — 下一步：仅 data-analyst 开 think 的实施 spec
