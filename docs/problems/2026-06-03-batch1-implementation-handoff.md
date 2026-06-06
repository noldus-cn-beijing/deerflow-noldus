# 实施交接：第 1 批修复 — data-analyst seal 卡死(纯 prompt,零承重墙)

> **致实施 agent**：本文档是 FST dogfood 反复「data-analyst terminated without emitting handoff」卡死的**第 1 批修复**实施单。背景与完整根因见 [`2026-06-03-subagent-seal-deadlock-problem-statement.md`](2026-06-03-subagent-seal-deadlock-problem-statement.md) 和 [`2026-06-03-reply-to-four-layer-proposal.md`](2026-06-03-reply-to-four-layer-proposal.md)，**实施前必读**。
>
> **本批只改一个文件**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`（**不是受保护文件，但内含 Noldus 定制 system prompt，逐行编辑、不要整文件覆盖**）。**不碰 `executor.py`**（那是第 2/3 批、受保护文件，本批不动）。
>
> **验收人**：写本文档的 review agent 会按 §6 验收标准逐条核对。请严格照做，**尤其 §3 的措辞不许"优化"**。

---

## 0. 根因回顾(实验已确认,不要再重新诊断)

通过真跑 deepseek-v4-pro 实验(11 次)+ 两次 dogfood transcript + git 史交叉核实，根因结构已**钉死**：

- **触发器(主因)= step 2.8 的 prompt 矛盾指令**：同时说"跳过 parameters_used 为空 `{}` 的 metric"、又说"判据不可用即记 info"、又指引"去读 plan_metrics.json(含全量 parameters_in_use)"。模型把"空 `{}`"读成"判据不可用"→ 记 info finding → 从 plan 捞 12 个参数纠结 mismatch_kind/suggestion → 烧 3-5 个 turn 的叙述黑洞。
  - 实验铁证：v1(无降级路径) thinking OFF 也 100% 成功；v2(有降级路径) thinking OFF 掉到 87.5%；v2 三次产物 `parameter_audit_findings` 数量摇摆 **0/3/0**(同一份 n=1 空参数数据，模型有时服从"跳过"产 0 条、有时服从"降级"产 3 条)——这就是矛盾指令的指纹。
- **放大器 1 = deepseek 把"写了'封存'文本" ≈ "完成了封存动作"**，不区分"在 thinking 里写封存"和"真的发出 seal tool_call"。
  - 证据：失败 transcript 模型写"现在让我来完成分析和封存"→停止，没调工具；而 lead 重派加一句"必须调 tool，不能只写文本"就翻转成功。
- **放大器 2 = max_turns=12**：纠结完余量不足，一犹豫就撞墙。(本批不处理，留第 2 批评估。)

**本批两处改动直接打掉触发器 + 放大器 1，纯 prompt、零承重墙、已实证有效。**

---

## 改动 A(Layer 2)：消除 step 2.8 矛盾 —— 让"空 `{}`"真正跳过、且严禁从 plan 捞参数

### A.1 问题精确定位

`data_analyst.py` step 2.8 当前有三处互相打架：
- 第 115 行：`跳过 parameters_used 为空 {} 的 metric。`
- 第 116 行：`判据可用才比对；判据不可用（文档缺 / n<2 / 无足够分布数据）即记 info 跳过，不阻塞。` ← "空 `{}`"会被模型读成"判据不可用"，于是走"记 info"而非"真跳过"。
- 第 126-143 行"Phase 1 降级路径"：`per_subject 缺该 metric 条目，或跨 subject 标量值 < 2 → 记一条 info finding`，且 `used_value 从 parameters_used[参数名] 取`。← 当 `parameters_used` 是 `{}` 时根本没有参数名可取，模型转而去 §2.8 之外的 `plan_metrics.json`(第 229 行指引)捞 `parameters_in_use`。

### A.2 期望逻辑(伪代码级，必须实现成这个语义)

```
进入 step 2.8 第一件事：先看 metrics_summary 里所有 metric 的 parameters_used。

分支 1 —— 所有 metric 的 parameters_used 都是空 {}：
    → 本轮没有任何"实际调过的可调参数"，参数审计【天然完成】(无对象可审，不是"略过任务")
    → parameter_audit_findings = []（空数组）
    → gate_signals.parameter_audit_findings_count = 0
    → gate_signals.parameter_audit_critical_count = 0
    → 直接进入 step 3 调 seal_data_analyst_handoff
    → 【明令】不要从 plan_metrics.json 的 parameters_in_use 取参数来补任何 finding；
       plan 里的 parameters_in_use 是"计划要用的"，不是"实际用到的"，
       实际用到什么以 metrics_summary[*].parameters_used 为唯一真相源。

分支 2 —— 至少有一个 metric 的 parameters_used 非空：
    → 只对 parameters_used 非空的那些 metric 走下方审计流程（Phase 2 / Phase 1 降级判定照旧）
    → parameters_used 为空 {} 的 metric 不产生 finding
```

### A.3 具体改法(给 review 用的精确锚点)

把第 114-116 行那段**开头总述**改写为：先做"分支 1"判断、明示空即天然完成、明示真相源。建议替换文案（**正面措辞，符合 CLAUDE.md §6**）：

> ```
> 2.8 **参数适配性审计**（Sprint 3 — 只警告不调参，铁律。seal 是必达，审计是尽力）：
>    **第一步：判断本轮是否有可审计的参数。**
>    parameters_used 的唯一真相源是 handoff_code_executor.json 的 metrics_summary[*].parameters_used，
>    它表示"实际调用计算时真正用到的可调参数"。
>
>    - 若 metrics_summary 中所有 metric 的 parameters_used 都是空 `{}`：
>      本轮的计算路径没有用到任何可调参数（例如 immobility 走 mobility_state 路径，
>      pendulum/velocity 参数均未参与）。此时参数审计【天然完成】——
>      parameter_audit_findings 为空数组 `[]`，parameter_audit_findings_count = 0，
>      parameter_audit_critical_count = 0，随即进入 step 3 调 seal_data_analyst_handoff。
>      plan_metrics.json 的 parameters_in_use 是"计划要用的参数"，不是"实际用到的"——
>      它不能作为审计对象；以 metrics_summary 的 parameters_used 为准。
>
>    - 若至少有一个 metric 的 parameters_used 非空：只对这些非空 metric 做参数-vs-数据分布比对，
>      parameters_used 为空 `{}` 的 metric 不产生 finding。比对方法见下方 a–f。
> ```

并把下方 a 段(第 118 行起)的措辞同步收口：明确"以下 a–f 仅适用于 parameters_used 非空的 metric"。降级路径(126-143 行)保留——但它**只在 parameters_used 非空、却判据/分布不足时**才触发；要在降级判定里把"`parameters_used` 为空"这个情形剔除（因为它已被第一步分流走，根本到不了这里）。

> **review 关注点**：改完后，第 116 行原来那句"判据不可用…即记 info 跳过"绝不能再让"空 `{}`"落进"记 info"分支。空 `{}` 必须在 step 2.8 最开头就被分流到"天然完成 → 直接 seal"。

---

## 改动 B(Layer 4)：在 step 3 锁定"写文本 ≠ 调工具"的正面措辞

### B.1 现状

第 95 行 step 2.7 结尾已有半句："`下一步 step 3 必须真的调 seal_data_analyst_handoff tool,不能只在 thinking 里写"封存"`"。第 184 行 step 3 本身只说"调 seal…tool，传入…"。区分力度不够、且现有半句含否定式("不能只…")，对 deepseek 有反向激活风险。

### B.2 期望：在 step 3 开头加一段【纯正面措辞】的硬区分

**以下文案为定稿，逐字写入，标注【不要优化这段措辞】**(先例见 `executor.py:719` seal-resume prompt 的同类警告)：

> ```
> 3. **封存 handoff —— 本步骤的完成标志是"发出一次 seal_data_analyst_handoff 的 tool_call"。**
>    handoff JSON 只有在你发出 seal_data_analyst_handoff 工具调用时才会真正落库。
>    请把你已经得出的结论（key_findings / 各结构化字段）作为该工具的参数填入并发出调用——
>    这一次工具调用本身，就是"封存"这个动作；它是本次任务的最后一步，发出后任务即完成。
>    （在文字里描述"已封存""分析完成"是叙述，不会落库；真正落库靠这一次 tool_call。）
>    调 seal_data_analyst_handoff tool，传入 status/key_findings/outlier_findings/excluded_metrics/
>    method_warnings/recommendations/errors/gate_signals/quality_warnings/parameter_audit_findings，
>    工具会自动写入 /mnt/user-data/workspace/handoff_data_analyst.json 并落 manifest hash。
>    如果没有相应发现，用空数组 `[]`，不要省略字段。
> ```

> **措辞设计说明(给 review)**：主句全是正面祈使("完成标志是发出 tool_call""把结论填入并发出调用""这一次工具调用就是封存动作")。唯一带括号的对比句用"是叙述，不会落库 / 靠 tool_call 落库"陈述事实，不用"禁止/不要写封存"这类反向激活。保留原 step 3 的字段清单与"空数组不省略字段"约束。

### B.3 顺手：删掉 step 2.7 第 95 行里那半句旧的否定式提醒

第 95 行结尾 "`下一步 step 3 必须真的调 seal_data_analyst_handoff tool,不能只在 thinking 里写"封存"`" —— 这半句的意图已被 B.2 的正面版本覆盖，且它含"不能只…"否定式。改法：把这半句删掉或改为正面引导("step 3 会通过发出 seal 工具调用来落库本次分析")，避免与 B.2 重复且消除反向激活词。**不要动 step 2.7 其余关于"审计至多 2-3 轮、seal 必达"的内容。**

---

## 4. 必须配套的回归测试(TDD 强制)

测试文件：`packages/agent/backend/tests/test_data_analyst_config.py`（已存在，在其中加用例）或新建 `test_data_analyst_step28_contract.py`。

至少覆盖：
1. **空参数分流契约**：构造一个 system prompt 文本断言——data_analyst 的 prompt 中，step 2.8 包含"所有 metric 的 parameters_used 都是空 `{}` → parameter_audit_findings 为空数组 → 直接 seal"的语义，且包含"plan_metrics.json 的 parameters_in_use 不作为审计对象"的明示。（prompt 是字符串常量，可直接断言关键短语存在/不存在。）
2. **禁从 plan 捞参数**：断言 prompt 中存在"以 metrics_summary 的 parameters_used 为(唯一)真相源"类表述。
3. **Layer 4 正面措辞锁定**：断言 step 3 包含"本步骤的完成标志是发出 seal_data_analyst_handoff 的 tool_call"这一精确语义，且**断言不再包含**否定式"不能只在 thinking 里写"(确认 B.3 已清理)。
4. **既有用例不回归**：跑现有 `test_data_analyst_config.py` / `test_data_analyst_insight_contract.py` / `test_seal_data_analyst_parameter_audit.py` 全绿。

> **注意**：本批是 prompt 改动，无法用单测验证"模型真的会照做"——那靠 dogfood。单测只锁"prompt 文本契约"(关键短语在/不在)，防止后续 sync 或他人编辑把这两处改回去。

---

## 5. 验证步骤(实施 agent 自查)

1. `cd packages/agent/backend && PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/test_data_analyst_config.py tests/test_data_analyst_insight_contract.py tests/test_seal_data_analyst_parameter_audit.py -q` → 全绿。
2. 跑新增/修改的契约测试 → 绿。
3. **不要 commit**。改完通知 review agent，由其 review + 决定 commit 时机(本批要与主仓库已有的 10 个旧改动一起 commit，拆分方案见原 handoff §3)。
4. **不要重启 dev / 不要跑 dogfood**——那由用户做(改 skill 需重启 dev 生效，见原 handoff §4.4)。

---

## 6. Review 验收标准(review agent 按此逐条核对)

- [ ] **只改了 `data_analyst.py` 一个文件**，没碰 `executor.py` / 其他受保护文件。
- [ ] step 2.8 最开头有"所有 parameters_used 空 `{}` → 天然完成 → 空数组 → 直接 seal"的分流，且**明令不得从 plan_metrics.json 捞 parameters_in_use**。
- [ ] 原第 116 行"判据不可用即记 info"不再能让"空 `{}`"落进记 info 分支(空已被最开头分流走)。
- [ ] 降级路径(原 126-143)仅在 `parameters_used` 非空但判据/分布不足时触发；"空 `{}`"情形已从降级判定中剔除。
- [ ] step 3 开头有 B.2 的**逐字**正面措辞("完成标志是发出 seal…tool_call""这一次工具调用本身就是封存动作")，无擅自改写。
- [ ] step 2.7 第 95 行旧否定式半句已删/改正面(B.3)，全 prompt 无新增"不要 X/禁止 X/不能只"对模型行为的反向激活措辞(deepseek 铁律 CLAUDE.md §6)。
- [ ] 配了 prompt 文本契约测试(§4 的 4 条)，且既有 data-analyst 测试全绿。
- [ ] 未 commit、未重启 dev。

---

## 7. 边界与禁区(实施 agent 务必守住)

- ❌ **不动 `executor.py`**(thinking 开关、turn 计数、seal-resume 都是第 2/3 批，受保护文件)。
- ❌ **不改 step 2.8 a–f 段里 mismatch_kind/severity 等审计算法逻辑**——它们对"parameters_used 非空"的正常路径仍然正确，本批只动"空 `{}` 的分流"和"真相源声明"。
- ❌ **不改 data-analyst 的 max_turns**(第 2 批评估)。
- ❌ **不要"顺手优化" B.2 的措辞**——它是实证有效文案的正面化定稿。
- ❌ **不要整文件覆盖 `data_analyst.py`**——逐行 surgical 编辑，保留所有其他 Noldus 定制 prompt 内容。
- ✅ 全程正面措辞(CLAUDE.md §6)。
