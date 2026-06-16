# Handoff：by-experiment skill 指标与 EV19 真实数据对齐（标注算不出的人工评分指标）

**日期**：2026-06-16
**会话主题**：用户问"重构 skill 后 code-executor 拼指标的 skill 要不要也改？"——核查后结论是**不用改 code-executor skill**，但发现 by-experiment skill 引入了 EV19 真实数据算不出的指标，已修正
**commit**：`38168293`（分支 `feat/skill-metric-tristate-fix`，基于 origin/dev `f5752d65`，**已 push 待建 PR**）
**状态**：3 个 skill 修正 + 1 份同事 review 文档已 commit+push；待建 PR + 跑 dogfood + 同事审核

---

## 一、当前任务目标

用户在前一会话把《动物行为实验指南》重构进 6 范式 by-experiment skill（见 [docs/handoffs/2026-06/2026-06-16-book-to-skill-refactor-and-mineru-install-handoff.md](docs/handoffs/2026-06/2026-06-16-book-to-skill-refactor-and-mineru-install-handoff.md)）。本会话回答两个追问：

1. **code-executor 拼指标的 skill 要不要也重写？** → 不用（见下"关键发现 #1"）
2. **重构的 skill 里有些指标变了，catalog 要不要跟着改？** → catalog 不动，但**skill 里有 EV19 算不出的指标需标注**（已修）

---

## 二、当前进展 ✅

- ✅ 核查 `ethoinsight-code` skill（code-executor 用）→ 确认**不受影响**（只读 plan 执行，不含范式判读）
- ✅ 用 `/home/wangqiuyang/DemoData` **真实 EV19 数据**经 `parse_header` 提取 6 范式真实列（非 head -1）
- ✅ 逐指标核对 skill「指标与判读」段 vs 真实列可算性
- ✅ 修正 3 个 skill（EPM/OFT/LDB），标注算不出的指标"本系统不产出、报告不得编造数值"
- ✅ 产出同事 review 证据表 [docs/review-packages/2026-06-16-skill-catalog-metric-tristate.md](docs/review-packages/2026-06-16-skill-catalog-metric-tristate.md)
- ✅ commit `38168293` + push 到 origin（用 /tmp/wt-skill-tristate worktree 隔离，未污染主仓 sync 分支）

---

## 三、关键发现（核心结论，供理解）

### #1 code-executor 的 skill 不用改（两类 skill 职责正交）
- `by-experiment/*.md`（EPM/OFT/...）= **why**（设计原理/指标判读），消费者是 data-analyst/report-writer/knowledge-assistant
- `ethoinsight-code/SKILL.md` = **how**（读 plan_metrics.json → 跑脚本 → seal），消费者是 code-executor
- 真正的"拼指标"逻辑在 **catalog/*.yaml（SSOT）+ catalog/resolve.py**，不在任何 skill。改 skill 的 why 不影响 plan schema，所以 code-executor skill 不动。**往它灌叙述内容反而会触发 LLM 脑补回归**（memory: `feedback_skill_describing_tool_output_enables_hallucination`）。

### #2 用真实数据验证后的可算性判定（关键事实，已多次自我纠错）
| 指标 | EV19 真实列证据 | 可算? |
|---|---|---|
| **伸展注意 SAP** | 6 范式**全有** `elongation` 列 | ✅ 全可算 |
| **低头探索/探出** | EPM 有 `nose_over_edge_open_arms`、OFT 有 `nose_over_wall_zone`、三迷宫有 `heading` | ✅ EPM/OFT/O-Maze 可算 |
| **LDB peek-out** | LDB 真实列**只有 `in_zone`**，无 nose_over/heading | ❌ **LDB 算不出** |
| **rearing 后腿直立** | **无任何范式有 Z 轴/高度列**（2D 俯视） | ❌ **全算不出** |
| **Zero Maze stretch-attend/head-dip** | `hesitation_count`（catalog default）是其轨迹代理 | ✅ 一致，未动 |

### #3 旧版同事 LDB skill 本就没有 peek-out/rearing
LDB 的 peek-out/鼻触/rearing 是**本次从书新加的**（旧版 96d6e2c4 同事 MVP 只有明箱%/穿梭/潜伏三件套）→ 标注不与同事基线冲突。

### #4 处理原则：保留方法论叙述 + 标注算不出（不无痕删除）
用户拍板"保留+标注算不出"，与 EPM/OFT 统一。既守住书的洞察，又防下游幻觉。

---

## 四、本次具体改动（3 skill）

| 文件 | 改动 |
|---|---|
| [epm.md](packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md) | 风险评估指标拆分：SAP/低头探索标可算（点名 elongation/heading/nose_over_edge_open_arms）；rearing 标算不出 |
| [open_field.md](packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/open_field.md) | rearing 整段标算不出（倒 U 型/性别差异并入标注）；补 SAP 可算；删解读段/脱险段重复的 rearing 条目 |
| [light_dark_box.md](packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/light_dark_box.md) | peek-out/鼻触/rearing 标算不出；SAP 保留 |

**未动**：catalog（SSOT）、ethoinsight-code skill、FST/TST/Zero Maze skill。

---

## 五、未完成事项（按优先级）

### 高优先级
- [ ] **建 PR**：https://github.com/noldus-cn-beijing/noldus-insight/pull/new/feat/skill-metric-tristate-fix （分支已 push）
- [ ] **跑 EPM/OFT dogfood**：确认标注措辞不会让 LLM 反而尝试调用不存在的指标（沿用前会话 handoff 的 dogfood 待办）

### 中优先级
- [ ] **行为学同事审核** [docs/review-packages/2026-06-16-skill-catalog-metric-tristate.md](docs/review-packages/2026-06-16-skill-catalog-metric-tristate.md) 第四节 4 个待决问题：
  1. rearing 是否真算不出（有无 3D/PhenoTyper 客户数据）
  2. SAP(elongation) 值不值得补登进更多范式 catalog
  3. head_direction 圆统计量能否当"低头探索"讲（精度差）
  4. LDB peek-out 值不值得专门写 nose-in-zone 检测脚本

### 低优先级（独立工作，"补能力"非"修 skill"）
- [ ] 若要让 SAP/nose-over 真正进 plan：catalog 补登记 optional 条目（EPM/OFT 的 nose_over_* 还**缺 compute 脚本**需新写带测试）

---

## 六、建议接手路径

1. 先读本文档 + [上一会话 handoff](docs/handoffs/2026-06/2026-06-16-book-to-skill-refactor-and-mineru-install-handoff.md)
2. 读证据表 [docs/review-packages/2026-06-16-skill-catalog-metric-tristate.md](docs/review-packages/2026-06-16-skill-catalog-metric-tristate.md)
3. 若建 PR：`gh pr create --base dev --head feat/skill-metric-tristate-fix`
4. 若跑 dogfood：用 `/home/wangqiuyang/DemoData/newdemodata/` 真实数据跑 EPM/OFT

---

## 七、风险与注意事项

1. **⚠️ 主仓在 sync 分支且 index 有大量 sync 暂存改动**：当前 `/home/wangqiuyang/noldus-insight` 在 `chore/sync-deerflow-d2cc991d` 分支，工作区有别人的 deerflow sync 暂存内容（与本次无关）。本次 commit **用 /tmp/wt-skill-tristate worktree 隔离**完成，**没碰主仓 sync 状态**。接手者别把那些 sync 改动误并进 skill PR。
2. **临时 worktree 未清理**：`/tmp/wt-skill-tristate`（分支已 push，可安全 `git worktree remove /tmp/wt-skill-tristate`）
3. **提列名必须走 parser**：EV19 文件 UTF-16/分号/标题多行，`head -1` 拿不到正确列名（memory: `feedback_ev19_head1_unusable_columns_from_parser`），用 `ethoinsight.parse._core.parse_header`
4. **不要往 catalog 加"占位指标"**：catalog 只收可算指标是其边界；算不出的标在 skill，别污染 SSOT
5. **会话中我多次自我纠错**：先误判"SAP/head-dip 算不出"，被用户两次纠正（先指出 hesitation_count 是代理、再要求用真实 DemoData 验证）后才得出 #2 的准确结论。**任何"算不出"判定必须以真实数据列为准，不可凭文档/脚本推断**。

---

## 八、下一位 Agent 的第一步建议

**建 PR**（最高优先，分支已就绪）：
```bash
cd /tmp/wt-skill-tristate   # 或在主仓 fetch 后操作
gh pr create --base dev --head feat/skill-metric-tristate-fix \
  --title "修正 EPM/OFT/LDB skill：标注 EV19 算不出的人工评分指标"
```
然后清理临时 worktree，并提醒用户把证据表第四节给行为学同事。

---

## milestone 建议

本会话是"behavioral-knowledge-from-book" track 的一个小 checkpoint：前会话把书重构进 skill 后，本会话用真实 EV19 数据**校准了 skill 的指标可算性边界**，防住下游幻觉。建议在该 track 的 milestone 记录：skill 已与真实数据列对齐（commit 38168293），下一步 = dogfood 验证 + 同事审核 4 个待决问题。不解除 CLAUDE.md 第13条的两个真实阻塞（#98 结构聚合 + #90 golden-case）。
