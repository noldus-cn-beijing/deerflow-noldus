# EPM Dogfood 验证报告 — column_aliases + LoopDetection 修复后测试

> 日期：2026-06-08 ｜ 分支：`dev`（PR #104 已合入，含 `worktree-zone-concept-params-loopdetection-fix` 的 2 commits）
> 测试方式：Playwright 浏览器自动化，端到端走通完整 EPM 分析流程
> 测试数据：`/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/原始数据-Elevated Plus Maze XT190-Trial     1.xlsx`
> 会话 ID：`1bda1847-cddc-4508-9d6f-2fa98413b398`

---

## 1. 测试目的

验证 PR #104（`ca61303d` + `cff6a518`）合入 dev 后，EPM 端到端分析流程是否正常工作：

- **Bug A 修复**：`zone_concept_params` → `_build_zone_aliases_overrides` → compute 参数注入通用化
- **Bug B 修复**：LoopDetection subagent 局部传参 `tool_freq_hard_limit=50`

---

## 2. 通过项 ✅

### 2.1 Bug A — column_aliases 参数注入生效

**修复前**：EPM 无 `anonymous_zone_override` → `_build_zone_aliases_overrides` 永远返回 `{}` → `parameters_in_use={}` → compute 脚本拿不到 zone 列名 → 全部 140 条指标 value=None。

**修复后**：5 个 EPM 指标全部计算成功，值全部非 null：

| 指标 | 值 | parameters_used |
|------|-----|-----------------|
| `open_arm_time_ratio` | 0.0799 (8.0%) | `{"open_arm_zones": ""}` |
| `open_arm_time` | 23.56s | `{"open_arm_zones": ""}` |
| `open_arm_entry_count` | 6 | `{"open_arm_zones": ""}` |
| `open_arm_entry_ratio` | 0.2857 (28.6%) | `{"open_arm_zones": "", "closed_arm_zones": ""}` |
| `total_entry_count` | 21 | `{"open_arm_zones": "", "closed_arm_zones": ""}` |

**关键验证点**：

- `parameters_used` 按 metric 精确裁剪，不再报告全集：
  - `open_arm_time_ratio` / `open_arm_time` / `open_arm_entry_count` 只含 `open_arm_zones`
  - `open_arm_entry_ratio` / `total_entry_count` 含 `open_arm_zones` + `closed_arm_zones`
  - 与 spec v3 的 `_compute_parameters_in_use` replace-only 机制一致
- zone 列名 autodiscovery 在参数为空字符串 `""` 时正常工作（正则匹配到用户列 `open`/`closed`）
- `handoff_code_executor.json` status = **completed**（修复前 metrics 三分裂导致误标 FAILED）

### 2.2 Bug B — LoopDetection 未掐死 subagent

- code-executor 使用 batch bash（5 个指标一次 `bash -c '…& …& wait'` 并行执行）
- bash 调用次数从 ~140 降为 ~5-8 次
- subagent 未被 `tool_freq_hard_limit` 硬停止

### 2.3 report-writer 正常完成

- `handoff_report_writer.json` status = completed
- 6 个 sections 全部写入：实验概况、分析方法、结果、观察与洞察、数据质量与局限、下一步建议
- errors = []，gate_signals 全部正常

---

## 3. 新发现的问题 ⚠️

### 3.1 Guardrail 路径冲突：n=1 fast path vs E2E_FULL_ASKVIZ 序列约束

**现象**：

Lead agent 尝试派遣 chart-maker 时，被 guardrail 拒绝：

```
Guardrail denied: tool 'task' was blocked (ethoinsight.path_sequence_violation).
Reason: 按 E2E_FULL_ASKVIZ 路径，chart-maker 之前必须先完成 data-analyst。
请先 task(data-analyst) 完成该步骤。
```

**根因**：

- n=1 fast path 规则（orchestration guide）：`code → 跳过 data-analyst → lead 描述性摘要 → ask(viz?) → ask(report?)`
- E2E_FULL_ASKVIZ guardrail（path_sequence）：`code-executor → data-analyst → chart-maker → report-writer`
- 两条规则在 data-analyst 是否必须的问题上矛盾

**影响**：

- chart-maker 被阻拦，图表未生成
- Lead agent 被迫派遣 data-analyst 以满足 guardrail，但 data-analyst 在 n=1 场景下无统计基础（见 3.2）

**修复建议**：

- `path_sequence` guardrail 需要感知 n=1 fast path：当 `groups.json` 显示 n<2 时，data-analyst 应标记为 optional 而非 required
- 或者在 `IntentPostStepAskGate` 层处理：n=1 时自动降级 intent 为 `E2E_FULL_ASKVIZ_N1`（不含 data-analyst 的变体）

### 3.2 data-analyst seal deadlock 复现（n=1 专用触发路径）

**现象**：

Lead agent 派遣 data-analyst（被 guardrail 要求）后，data-analyst 陷入 seal 死锁：

- 第 1 次派遣：subagent "terminated without emitting handoff_data_analyst.json"（忘记调 seal 工具）
- 第 2 次派遣（auto-retry）：subagent 在 "partial vs completed" 状态选择 + "fast-fail 是否跳过 step 2.5-2.8" + "parameter audit 是否包含" 之间反复辩论
- 持续 5+ 分钟未封存，handoff_data_analyst.json 始终未生成

**根因分析**：

这是已知 seal deadlock 模式的 n=1 专用触发路径（与 memory `feedback_subagent_seal_deadlock_is_prompt_not_budget.md` 同根因）：

1. **prompt 矛盾**：
   - 用户要求 "解读" → data-analyst 被期望做分析
   - fast-fail 规则 `n<3 → emit partial immediately, skip 2.5-2.8` → 要求快速退出
   - 两个信号冲突，subagent 在 "做分析" 和 "快速退出" 之间反复横跳

2. **n=1 专用触发条件**：
   - 正常 n≥2 场景：data-analyst 有明确的统计基础（组间比较），workflow 步骤清晰
   - n=1 场景：所有统计检验 skipped，subagent 不知道该做什么，在 prompt 中反复自问 "partial or completed?"

**与 handoff 描述的区别**：

handoff 文档（`2026-06-08-epm-dogfood-columnaliases-loopdetection-fix-handoff.md`）中提到的 seal deadlock 是通用型的（非空参数路径 `open_zones` 走 a-f 长分支卡死）。本次发现的是 n=1 专用触发路径：

- 通用型：参数多 → step 2.8 参数审计复杂 → 吃光 turn → 忘记 seal
- n=1 型：统计全 skipped → 没有明确的 work-to-do → prompt 矛盾 → 反复辩论 → 忘记 seal

两个触发路径最终表现相同（terminated without handoff），但根因不同。

**修复建议**：

- **短期**：n=1 场景下 data-analyst prompt 增加正面指令："n=1 时你的唯一工作是：①确认 fast-fail ②写 3 句中文描述性摘要 ③立刻 seal。不需要参数审计、不需要效应量、不需要异常检测。"
- **中期**：在 data-analyst prompt 中加 "if n<3, skip directly to seal" 的早退分支，像 code-executor 的 `skip_reason` 一样提前短路
- **长期**：3.1 的 guardrail 修复（n=1 不要求 data-analyst）可以从根本上消除此触发路径

### 3.3 validate_catalog 文件权限问题（低影响）

**现象**：

code-executor seal 摘要中有一条 warning：

```
validate_catalog 因文件权限 mode 600 报不可读（不影响数据有效性）
```

**分析**：

- 指标输出 JSON 文件权限为 `600`（`-rw-------`）
- `validate_catalog`（L-B 层，catalog-driven 范围校验）需要读取这些文件
- 由于 sandbox 内进程以不同用户运行，无法读取 `600` 权限文件
- 不影响数据正确性（指标值已在 compute 脚本 stdout 的 `[result]` 行输出），但 L-B 校验被跳过

**修复建议**：

- compute 脚本输出文件时显式设置权限 `os.chmod(output_path, 0o644)`
- 或者在 `emit_result()` 函数中统一处理

---

## 4. 测试环境信息

- **Git HEAD**：`ea8c6734`（Merge PR #104 into dev）
- **修复 commits**：
  - `ca61303d` — zone_concept_params 通用化 + LoopDetection subagent 阈值修复
  - `cff6a518` — prompt 引号示例 + 恒真断言修复
- **应用**：`make dev` 启动（localhost:2026）
- **模型**：deepseek-v4-pro
- **模式**：全自动（Flywheel）
- **Intent**：E2E_FULL_ASKVIZ

---

## 5. 结论

**核心修复（Bug A + Bug B）验证通过**。EPM 指标计算从 "全部 None" 恢复到全部有合法值，handoff 状态从 FAILED 恢复到 completed。

**新发现 2 个问题**需要跟进：

| # | 问题 | 严重度 | 阻塞什么 |
|---|------|--------|----------|
| 3.1 | Guardrail 路径冲突（n=1 × E2E_FULL_ASKVIZ） | P1 | chart-maker 被阻拦 |
| 3.2 | data-analyst seal deadlock（n=1 专用触发路径） | P1 | data-analyst 永远无法 seal |
| 3.3 | validate_catalog 文件权限 600 | P2 | L-B 校验被跳过（不影响值正确性） |

3.1 和 3.2 建议一起修：3.1 修了 guardrail（n=1 不要求 data-analyst），3.2 就不会触发。

---

## 6. 相关文档

- 本次 dogfood 依据的 handoff：[2026-06-08-epm-dogfood-columnaliases-loopdetection-fix-handoff.md](2026-06-08-epm-dogfood-columnaliases-loopdetection-fix-handoff.md)
- 实施 spec：[docs/superpowers/specs/2026-06-06-zone-concept-params-and-loopdetection-fix-spec.md](../superpowers/specs/2026-06-06-zone-concept-params-and-loopdetection-fix-spec.md)
- seal deadlock 通用分析：[feedback_subagent_seal_deadlock_is_prompt_not_budget.md](../../../.claude/projects/-home-wangqiuyang/memory/feedback_subagent_seal_deadlock_is_prompt_not_budget.md)
- n=1 orchestration 规则：lead_agent prompt 中 `n=1 fast path` 段
- E2E_FULL_ASKVIZ guardrail：path_sequence provider
