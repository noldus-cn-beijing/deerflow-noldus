# Handoff：by-experiment skill 指标取舍与行为学同事裁决对齐

**日期**：2026-06-16
**会话主题**：用户问"重构 skill 后 code-executor 拼指标的 skill 要不要也改？"——核查后结论是**不用改 code-executor skill**；但发现 by-experiment skill 从书引入了若干非常用/不支持的指标，先用真实数据核查、再经**行为学同事（曲若衡）裁决**修订
**分支**：`feat/skill-metric-tristate-fix`（基于 origin/dev `f5752d65`，**已 push 待建 PR**），3 个 commit：
- `38168293` 初版（基于真实数据列推断，**结论已被同事裁决覆盖**）
- `c7836942` 本会话 handoff
- `2c79cc3d` **按同事裁决修订（最终，以此为准）**
**状态**：4 个 skill（EPM/OFT/LDB/Zero Maze）+ 证据文档已 commit+push；待建 PR + 跑 dogfood

---

## ⭐ 最终结论（同事裁决，以此为准 — 覆盖 38168293 的"算不出"措辞）

可算性 ≠ 该算。指标取舍以行为学同事实践判断为准，不以"数据列在不在"为准：

| 指标 | 同事裁决 | skill 处置 |
|---|---|---|
| **rearing** | 不支持：精细行为分类器**信效度不佳** + 焦虑/旷场范式内**非必须**（能从 elongation 推但没必要） | 标"本系统不支持"+真实理由 |
| **SAP / elongation** | **很少使用** | 四范式降级"很少用、默认不算、用户要求才算" |
| **head-dipping** | 只适用 **EPM / 未来巴恩斯**（有无墙开放边缘）；OFT/LDB 带墙不存在 | EPM 保留（走 `nose_over_edge_open_arms`）；OFT/LDB 不提 |
| **Zero Maze 低头/探出** | **一般不测，优先 FewZones**；测则用**鼻尖点在"探头区"分析区**；stretch/elongation 很少用 | 撤"核心洞察/最敏感"拔高；`hesitation_count`（catalog default）保留为辅助 |
| **LDB peek-out** | 不测 nose-in-zone，以**身体中心点**为准 | 标"封闭箱无开放边缘+中心点判定" |
| **FST/TST velocity** | 默认导出带 velocity（sample 手动删了） | catalog 的 `activity_intensity` 依赖合理，**不改** |

详见证据表 [docs/review-packages/2026-06-16-skill-catalog-metric-tristate.md](docs/review-packages/2026-06-16-skill-catalog-metric-tristate.md) 第零节。

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

### #2 ~~用真实数据验证后的可算性判定~~（⚠️ 已被同事裁决覆盖，见顶部"最终结论"）
> 初版基于"数据列在不在"推断可算性（SAP 全可算、EPM/OFT 探出可算、LDB/rearing 算不出）。**此判断已被同事裁决推翻**——可算 ≠ 该算。保留此条仅为记录推理轨迹，**实际取舍以顶部最终结论表为准**。

### #3 旧版同事 LDB skill 本就没有 peek-out/rearing
LDB 的 peek-out/鼻触/rearing 是**本次从书新加的**（旧版 96d6e2c4 同事 MVP 只有明箱%/穿梭/潜伏三件套）→ 标注不与同事基线冲突。

### #4 处理原则：保留方法论叙述 + 按同事裁决标注取舍
不无痕删除——保留书的叙述完整性，但按同事裁决明确标注"不支持/很少用/默认不算/不产出"，既守洞察又防下游幻觉。

---

## 四、本次具体改动（4 skill，最终态）

| 文件 | 最终改动（commit 2c79cc3d） |
|---|---|
| [epm.md](packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md) | head-dipping 保留（EPM 有开臂无墙，走 `nose_over_edge_open_arms`）；SAP 标"很少用"；rearing 标"不支持"+信效度理由 |
| [open_field.md](packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/open_field.md) | rearing 标"不支持"；SAP 标"很少用"；带墙范式不提 head-dipping |
| [light_dark_box.md](packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/light_dark_box.md) | peek-out 标"封闭箱+中心点判定、不测 nose"；SAP"很少用"；rearing"不支持" |
| [zero_maze.md](packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/zero_maze.md) | 撤"stretch-attend 核心洞察/最敏感"拔高；低头改"一般不测优先 FewZones，测则用鼻尖点在探头区"；`hesitation_count` 保留为辅助 |

**未动**：catalog（SSOT）、ethoinsight-code skill、FST/TST skill。

---

## 五、未完成事项（按优先级）

### 高优先级
- [ ] **建 PR**：https://github.com/noldus-cn-beijing/noldus-insight/pull/new/feat/skill-metric-tristate-fix （分支已 push）
- [ ] **跑 EPM/OFT/Zero Maze dogfood**：确认标注措辞不会让 LLM 反而尝试调用不存在的指标（沿用前会话 handoff 的 dogfood 待办）

### 中优先级
- [x] **行为学同事审核**：曲若衡 2026-06-16 已裁决，结论并入证据表第零节 + 顶部最终结论表，已落 commit 2c79cc3d
- [ ] （可选）若未来要让 Zero Maze 的"探头区"指标进 plan：需实验时专门划探头分析区 + catalog 登记 + 写脚本（同事说一般不测，**低优先**）
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
4. **不要往 catalog 加"占位指标"**：catalog 只收可算指标是其边界；不支持/不常用的标在 skill，别污染 SSOT
5. **⭐ 本会话最大教训：可算 ≠ 该算**。我先凭"数据列在不在"判可算性（被用户纠正两次：hesitation_count 是代理、要求用真实 DemoData 验证），得出"SAP/探出可算"；但**行为学同事最终裁决又推翻了"可算就该写进 skill"**——SAP 列虽在但很少用、head-dipping 受物理范式约束、rearing 能从 elongation 推但信效度差+非必须。**指标取舍的最终权威是行为学同事的实践判断，不是数据列、也不是书。** 数据列只回答"能不能算"，同事回答"该不该算"。
6. **rearing 措辞别写成"算不出"或"需精细行为模块"**：准确说法是"信效度不佳 + 范式内非必须"（同事原话），写"需精细行为模块"像在替 EthoVision 推销那个信效度不好的模块。

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
