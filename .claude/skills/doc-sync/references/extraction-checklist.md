# extraction-checklist（doc-sync Step 2 参考）

Step 2 从 handoff / plan / spec 提取"已发生的事实和决策"。本文件列出**要提取的信号**
和**必须丢弃的反例**。

## 要提取的信号（已发生事实，每条标来源 handoff 路径+日期）

### PR 合入 / commit 落地
- "PR #N 已合 dev"、"merge commit `<hash>` 进 dev"、"Sprint X 已合"
- 提取：PR 号、commit hash(短)、合入分支、合入日期、覆盖范围(哪些文件/能力)
- 例：`[FACT] PR #218-#227 dogfood bug 修复批全合 dev (2026-06-26)`

### issue 状态切换
- "Issue #N CLOSED"、"Issue #N OPEN"、"新立 Issue #N"
- 提取：issue 号、旧态→新态、切换日期
- 例：`[FACT] Issue #98 结构聚合 CLOSED (2026-06-18)`

### feature track 阶段切换
- "Sprint N done"、"Phase 0 完成"、"立项(spec 入库)"、"Sprint N 实施中"
- 提取：track 名、旧阶段→新阶段、坐实日期
- 例：`[FACT] EV19 列语义对齐 Sprint 1 + Sprint 2 全已合 dev`

### 阻塞变化
- "阻塞解除"、"新阻塞"、"卡 X，需 Y"
- 提取：阻塞项、解除/新增、依赖对象
- 例：`[FACT] Golden Cases 仍为零，卡微调路线(不卡 v0.1 端到端)`

### 方案/架构决策锁定
- "锁定 X"、"决定用 Y 不用 Z"、"决策拍板：直接执行无需再问"
- 提取：决策内容、被否决的备选、锁定日期、决策依据(简)
- **关键**：这类往往是新建 ADR 的候选(见 classification-rules.md 的"新建判定门")

### skill / 能力状态变化
- "新增 skill X"、"skill X 重命名为 Y"、"X 启用/禁用"
- 提取：skill 名、变化类型、落点(extensions_config.json / skills/custom/)

## 必须丢弃的反例（不是"已发生事实"）

- ❌ **待办**："下一步要做 X"、"未完成事项" → 不是事实，丢弃(除非对应"未完成"本身就是当前真相，进入 milestone 的"遗留项")
- ❌ **计划中(未实施)**："spec 已写、尚未动手"、"实施计划就绪" → 这是"计划存在"，不是"已实施"。可回流为"📋 立项"状态，但**不能**写成"已完成"
- ❌ **推测/假设**："可能是因为 X"、"建议这样改" → 丢弃
- ❌ **问题描述**："发现 bug：现象是…" → bug 本身不是事实；只有"bug 已修(带 commit)"才是
- ❌ **会话过程描述**："本会话做了 A 然后 B" → 过程不是事实，提取其中的落地结论
- ❌ **被推翻的中间结论**：handoff A 说"方案 X"，handoff B(更晚)说"改用方案 Y" → 只取最新的 Y，X 已过期

## 时序冲突的处理

多个 handoff 提到同一事实但状态不同(如"已合 dev" vs "发现回归已 revert")：
- **取最晚日期 handoff 的状态**为当前真相
- 在事实清单里标注时序冲突 + 择取依据
- Step 4 git 校准是最终仲裁：git 里在不在 dev 是硬证据

## 与"完成标志"的关系

提取的事实清单是后续所有步骤的输入。**没有干净的事实清单，漂移检测和分类都是空谈。**
完成标志：每条事实带来源路径+日期，反例全部丢弃，时序冲突已标注。
