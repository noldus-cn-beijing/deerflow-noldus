# 2026-04-23 飞轮 join 精修 + Golden-case 路径对齐 — 交接文档

> **给下一个 AI Agent：** 你无法访问本次会话上下文。这份文档让你快速理解 2026-04-23 下半场做了什么，以及接下来怎么继续推进。
>
> **读取顺序**：本文档 → [CLAUDE.md](../../CLAUDE.md) → [docs/sop/golden-case-sop.md](../sop/golden-case-sop.md) → [golden-cases/SCHEMA.md](../../golden-cases/SCHEMA.md) → 上一份交接 [2026-04-23-training-data-flywheel-done.md](2026-04-23-training-data-flywheel-done.md)

---

## 1. 会话概览

**用户**：Qiuyang（Noldus 软件开发工程师）

**本次会话主题**：
1. 完成训练数据飞轮的 A 任务：反馈 ↔ 样本按 message_id 精确 join
2. 讨论 B 任务（范式补全）的战略方向 —— 专家知识注入路径
3. **发现并修正"重复造轮子"错误**：用户同事已在推进 `golden-cases/` 系统，AI 差点起草 `docs/domain/epm.md` 作为平行系统
4. 用户确认判读哲学："按组间比较，不用绝对阈值"
5. 把 golden-case 确立为专家知识注入的**正式唯一途径**，更新 3 份文档
6. 清理 `docs/` 下的明显冗余文件

**会话成果**：4 个独立 commit 全部落地，1 条记忆规则加入个人 memory 防止重犯错误。

---

## 2. 本次会话完成的工作 ✅

### 2.1 飞轮 join 精确匹配（commit `8e0f4427` "修复了一些飞轮问题"）

前提：上一份交接 `2026-04-23-training-data-flywheel-done.md` §4.1 明确提到"反馈-样本关联精度是 v0.1 限制"——前端已传 message_id 但后端粗粒度匹配。

**改动**：

| 文件 | 说明 |
|---|---|
| `packages/harness/deerflow/agents/middlewares/training_data_middleware.py` | Lead 样本加 `message_id: AIMessage.id`；Subagent 样本加 `message_id: f"subtask-{tool_call_id}"`——对齐前端发反馈的 key 格式 |
| `scripts/extract_e2e_sessions.py` | join 索引从 `{thread_id: [items]}` 改为 `{thread_id: {message_id: latest_feedback}}`。同 message_id 多次反馈取最新 `submitted_at` 胜出。无 message_id 的旧样本仍按"无反馈路径"保留 |
| `tests/test_training_data_middleware.py` | +2 测试：`test_lead_sample_records_ai_message_id` + `test_subagent_sample_records_subtask_prefixed_id` |
| `tests/test_extract_e2e_sessions.py` | +3 测试：多消息精确匹配、最新反馈覆盖、旧样本兼容 |
| `docs/sop/training-data-flywheel-sop.md` | 技术说明段加"反馈↔样本 Join 规则"小节 |

**测试基线**：1694 → **1699 passed**，14 skipped，0 failed。

### 2.2 Golden-case 作为专家知识注入的正式途径（commit `38b5f030` "路径修正"）

**起因**：讨论 B 任务（焦虑/抑郁范式补全）时，AI 起草了 `docs/domain/epm.md` 作为"专家填写的领域知识模板"。用户指出后发现仓库里**已有 `golden-cases/` 系统**（2026-04-22 落地，`SCHEMA.md` + `TEMPLATE/` + `case-001-shoaling-baseline/` + `scripts/validate_golden_case.py` 334 行），且他的行为学同事正在推进填写。

用户原话："你差点把我引入歧途了！"

**修正动作**：
- **删除** `docs/domain/epm.md` 和 `docs/domain/` 目录
- **更新 CLAUDE.md**：
  - 仓库结构加 `golden-cases/` 目录 + `scripts/validate_golden_case.py`
  - "重要注意事项"加第 8 条（golden-case 是专家知识注入正式途径，**不要另建 `docs/domain/`**）
  - "重要注意事项"加第 9 条（判读哲学：组间比较，不用绝对阈值）
  - "相关文档"加 SCHEMA + SOP 链接
- **新建 [docs/sop/golden-case-sop.md](../sop/golden-case-sop.md)**：给行为学同事的使用 SOP（仿 training-data-flywheel-sop.md 风格），含：
  - 1 个 case 包含什么、专家填哪两个文件
  - 参与流程（一个 case 15-60 分钟）
  - 三个原则（推理过程>结论、组间比较、写明不该说什么）
  - 与训练数据飞轮的关系（飞轮量大质不均，golden-case 量少质高，互补）
- **更新 [docs/sop/training-data-flywheel-sop.md](../sop/training-data-flywheel-sop.md)** 开头加 golden-case 关联说明

### 2.3 docs/ 冗余清理（commit `8633b356` "clean up"）

用户选择"只清理明显垃圾"，**不动 plans/ 和 handoffs/**。

| 动作 | 对象 |
|---|---|
| 删除 | `docs/e2e_tests/` 整个目录（8 份 fish shoaling fix1-fix4 迭代样本） |
| 移动+改名 | `docs/lession_from_claude_code_design.md` → `docs/refs/claude-code-design-notes.md`（修 typo） |
| 移动 | `docs/image.png` → `docs/refs/image.png`（跟随引用者） |
| 移动 | `docs/ethoinsight-architecture.html` → `docs/presentations/ethoinsight-architecture.html` |

共删除 3902 行，新增 2 个目录（`docs/refs/`、`docs/presentations/`）。

### 2.4 记忆系统更新（个人 memory，不在 repo 内）

在 `~/.claude/projects/-home-qiuyangwang/memory/` 加入：
- `feedback_check_existing_architecture_first.md` — 提新架构/目录前必须先 grep/ls，项目已有 golden-cases 系统

MEMORY.md 索引已更新。下一个 AI 会话会自动加载。

---

## 3. 未决问题 / 接下来的工作

### 3.1 **最紧迫**：推同事填完 case-001 的 TODO

**位置**：`golden-cases/case-001-shoaling-baseline/`

**状态**：工程侧初稿完成，**8 个 `TODO(行为学同事)` 待填**
- 3 个在 `expected-analysis.yaml`（reasoning 字段）
- 5 个在 `notes.md`（实验背景、指标正常范围、离群阈值、异常分类确认、结论措辞、参考文献）

**AI 能帮的**：起草一份"15 分钟清单"消息给同事（本次会话已起草原型，用户可直接发送）。示例见对话记录最后一段。

**为什么重要**：case-001 是第一个完整 golden case，一旦填完就是**所有后续 case 的样板**。流程没跑通前不要铺开做 case-002。

### 3.2 为 EPM 准备 case-002-epm-* 骨架（等 case-001 跑通后）

**前置条件**：需要专家确认 `demo-data/DemoData/高架十字迷宫/` 的 13 个文件如何分组（control vs treatment）。没有这个信息无法填 `metadata.yaml`。

**工程侧动作**（case-001 流程走通后）：
1. 挑一组 EPM demo 数据
2. 跑一遍现有分析流水线拿到真实 metrics 数值
3. 填好 `metadata.yaml` + `expected-analysis.yaml` 的 `expected_metrics` 段
4. 标清 `TODO(行为学同事)`，发给同事填

### 3.3 **工程待建**：`scripts/run_golden_case_regression.py`

当前 `validate_golden_case.py` 只做 schema 校验。**真正的回归测试还没实现**——让 agent 对 `raw-data/` 实际跑一遍分析，比对 agent 输出与 `expected-analysis.yaml` 的 required_keywords/forbidden_claims/expected_range。

**关键决策**：**等 case-001 完整填好后再写**。现在写等于用假数据写测试框架，容易走歪。SOP 里已经注明这一点。

### 3.4 持续推进：飞轮数据收集

- 累计 3 条 SFT 已录制；目标两个月内 800 条
- 每周一跑 `make training-stats` 看进度
- 每周一跑 `scripts/extract_e2e_sessions.py` 合成当周数据集
- 云部署时的数据收集方案详见 [2026-04-23-training-data-flywheel-done.md §4.2](2026-04-23-training-data-flywheel-done.md)

### 3.5 **不紧急但重要的架构债**：范式系统的"真 general"重构

本次会话讨论过但**明确搁置**的方向：
- 把 `metrics.py` 里 `if paradigm == "shoaling": elif paradigm == "epm":` 的硬编码改为注册表 + 声明式 yaml 配置（`packages/ethoinsight/ethoinsight/paradigms/<name>.yaml`）
- 拆出通用 primitives（`compute_zone_time_ratio`, `compute_zone_entries` 等）让多范式复用

**为什么搁置**：用户在决策过程中意识到应优先让 golden-case 专家工作流跑顺——等有了 3-5 份完整 case，对专家实际工作流的认知会更准，那时再决定架构更靠谱。

**不要现在做**：任何 `paradigms/*.yaml` 声明式配置的抽象。

---

## 4. 关键决策与易混淆点

### 4.1 判读哲学（项目级共识）

**不按绝对阈值**。行为学同事 explicit 确认：

> EPM/OFT 等焦虑范式的解读只看 control vs treatment 是否有显著差异，不按"正常范围 15-25%，小于 10% 就是高焦虑"这种绝对阈值判断。

**影响**：
- `ethoinsight/assess.py` 里的 `_DEFAULT_THRESHOLDS` 保留作为"批次质检参考"，**不作为判读依据**
- Agent 给出的结论必须基于统计检验 + 效应量
- Golden case 的 `expected_findings` 里的 reasoning 应按组间比较措辞

### 4.2 Golden-case vs 训练数据飞轮（不是竞争关系）

| | Golden-case | 训练数据飞轮 |
|---|---|---|
| 数据来源 | 离线人工标注 | 真实使用中自动采集 |
| 数量 | 量少（目标 16 个范式×3-5 case） | 量大（2 个月目标 800 SFT） |
| 质量 | 黄金标准 | 质量不均 |
| 用途 | 知识源 + 回归测试 + SFT 种子 | 规模化训练 |

**不要把两套系统合并**。互补共存。

### 4.3 Golden-case 的文件分工

专家**只填两个文件**：
- `expected-analysis.yaml` — 机器用的断言（required_keywords / forbidden_claims / expected_range）
- `notes.md` — 推理链（6 段大纲，300-800 字，未来作为 SFT CoT 源）

工程填：
- `metadata.yaml` — case 身份
- `raw-data/` — 准备数据
- `expected-analysis.yaml` 的 `expected_metrics` 段（跑流水线拿数值）
- 在需专家判断处标 `TODO(行为学同事)`

### 4.4 本次会话的最大教训（已存 memory）

**提新架构/目录前必须先 grep/ls 检查是否已有相关系统**。本次差点平行造 `docs/domain/`，被用户纠正。

个人 memory 已加规则。下一个 AI 会话会自动加载。但如果你是**新环境、没有 memory 访问权限**的 AI，请务必遵守：
- 提 `docs/X/` / `新的 Y 系统` 前先 `ls docs/`、`grep -r "X" CLAUDE.md`
- 发现已有系统时：扩展/对接，不另起炉灶
- 如果确实要新建，**明确说明"已检查 A/B/C 不存在或不适用"**

---

## 5. 关键文件索引

### 本次会话直接改动
- [packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py) — 加 message_id
- [packages/agent/backend/scripts/extract_e2e_sessions.py](../../packages/agent/backend/scripts/extract_e2e_sessions.py) — 精确 join
- [CLAUDE.md](../../CLAUDE.md) — 加第 8、9 条重要注意事项
- [docs/sop/golden-case-sop.md](../sop/golden-case-sop.md) — 新建
- [docs/sop/training-data-flywheel-sop.md](../sop/training-data-flywheel-sop.md) — 加关联说明

### Golden-case 系统（已存在，本次会话确立为正式途径）
- [golden-cases/SCHEMA.md](../../golden-cases/SCHEMA.md) — 字段字典
- [golden-cases/TEMPLATE/](../../golden-cases/TEMPLATE/) — 空白模板
- [golden-cases/case-001-shoaling-baseline/](../../golden-cases/case-001-shoaling-baseline/) — 第一个 case（待专家填 TODO）
- [scripts/validate_golden_case.py](../../scripts/validate_golden_case.py) — schema 校验

### 相关历史文档（保持新鲜）
- [docs/handoffs/2026-04-22-golden-case-and-auth-design.md](2026-04-22-golden-case-and-auth-design.md) — golden-case 系统落地的会话
- [docs/handoffs/2026-04-23-training-data-flywheel-done.md](2026-04-23-training-data-flywheel-done.md) — 飞轮 Phase A-E 完成交接
- [docs/plans/2026-04-23-training-data-flywheel.md](../plans/2026-04-23-training-data-flywheel.md) — 飞轮设计

---

## 6. 下一位 Agent 的第一步建议

### 用户最可能的下一句话及对应的起手动作

**"帮我写给同事的 case-001 消息"**
→ 读 `golden-cases/case-001-shoaling-baseline/notes.md` 和 `expected-analysis.yaml`，找出 8 个 `TODO(行为学同事)` 标记。为每个标记列一条简短的问题（"Subject 3 属于 A/B/C/D 哪种异常？"），凑成一份 15 分钟能完成的清单。

**"case-001 已经填完了，下一步？"**
→ 先跑 `python3 scripts/validate_golden_case.py` 验证 schema 通过。然后：
  1. 建议建 `scripts/run_golden_case_regression.py`（现在有真数据了，可以写了）
  2. 或开始 case-002-epm-* 骨架（需要先问同事 demo-data/高架十字迷宫/ 13 个文件的分组方案）

**"开始做 EPM"**
→ 停！问清楚用户是想：
  - (a) 建 case-002-epm-* 骨架（推荐，走 golden-case 流程）
  - (b) 直接改 `ethoinsight/metrics.py`（不推荐，架构债方向）
默认应走 (a)。如果走 (b)，提醒用户本会话 §3.5 已讨论过搁置理由。

**"清理剩下的 handoffs/plans"**
→ 本次会话用户选择了"只清理明显垃圾"。如果用户要推进深度清理：需要**逐份读 plans/ 和 handoffs/** 判断"已落实/过时/仍活跃"。这会消耗大量 token，估 20-40 分钟。先和用户确认规模。

**"继续 B 任务（范式补全）"**
→ B 任务 = 补 EPM/OFT 等焦虑抑郁范式的 agent 能力。正确路径是 golden-case 驱动：每个范式 2-5 个 case 定义"agent 应该做成什么样"，工程侧按需补齐 `metrics.py` 的 primitive 和 `assess.py` 的规则。**不要**直接扩 `_DEFAULT_THRESHOLDS`——判读走组间比较。

### 安全检查清单（第一步前必做）

- [ ] 读 CLAUDE.md 新增的第 8、9 条（golden-case 是正式途径 + 判读哲学）
- [ ] 扫 `golden-cases/` 目录结构，确认 SCHEMA + TEMPLATE + case-001 都在
- [ ] 如果用户要加新的"领域知识文档系统"，**拒绝并指向 golden-cases/**
- [ ] 如果用户要扩 `_DEFAULT_THRESHOLDS` 作为判读依据，**指出判读哲学已改为组间比较**

---

## 7. 测试与健康度基线

| 项 | 值 |
|---|---|
| 后端测试 | **1699 passed, 14 skipped, 0 failed**（本次 +5：2 middleware + 3 extract） |
| 前端 typecheck | 未运行（本次未改前端） |
| Golden-case schema 校验 | ✅ case-001 PASS（0 errors, 0 warnings） |
| Git 工作区 | 干净，所有改动已 commit 到 `dev` 分支 |

## 8. 本次会话的 4 个 commit

```
8633b356 clean up                  # docs 清理：删 e2e_tests，移 refs/presentations
38b5f030 路径修正                   # 删 docs/domain/epm.md + 建 golden-case-sop.md + 改 CLAUDE.md
96ec33b8 epc                       # 错误方向：起草 docs/domain/epm.md（已被 38b5f030 回滚）
8e0f4427 修复了一些飞轮问题           # message_id 精确 join + 5 新测试
```

注：`96ec33b8` 是被 `38b5f030` 完全回滚的方向，保留仅作为"走过弯路"的历史痕迹。实际有效改动是 `8e0f4427 → 38b5f030 → 8633b356` 这个序列。
