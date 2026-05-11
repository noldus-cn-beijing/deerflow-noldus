# 2026-05-11 SOTA 架构 grill-me 完成 + 实施 plan 待写 交接

> **前置文档**：
> - [v2 MVP 范式知识搬入 skill](./2026-05-09-mvp-paradigm-knowledge-into-skill-handoff-v2.md) — 上次会话搬入 + SKILL.md 修订完整记录
> - 本文档 = 2026-05-09 / 05-11 grill-me 会话产出，**SOTA 架构决策**敲定 + plan 待写 + **有平行 agent 走错路径需处理**

## ✅ Working tree 已清理（2026-05-11 本次会话末尾）

平行 agent 走错路径的产物已经处理完：

| 文件 | 处理 | 进 commit `16eeac9b` |
|---|---|---|
| `packages/ethoinsight/ethoinsight/metrics.py` +147 行 | **保留**（高质量算法实现，是 SOTA 路径需要的 EPM 新指标） | ✅ |
| `packages/ethoinsight/tests/test_template_epm.py` | **拆分**：前 415 行（5 个 TestCompute* 测试 metrics 函数）保留并重命名为 `test_metrics_epm.py`；后 110 行（TestEpmTemplateModule / TestEpmTemplateSoftGate 测脚本模板）删除 | ✅ |
| `packages/ethoinsight/ethoinsight/templates/epm.py` | **删除**（脚本模板模式，被 grill-me 否决） | — |

**验证**：20 个 EPM metrics 测试全过；ethoinsight 全部测试无新增 fail（9 个 test_parse 失败是 pre-existing 环境问题）。

下一位接手 agent 不需要处理 working tree，直接进入 P0 第 2 项（写 SOTA 架构切换 plan）即可。

未处理项：`docs/sop/aliyun-deployment-guide.md` 仍是 untracked（非本任务范围）。

## 当前任务目标

写一份**6 范式 SOTA 架构切换的细颗粒度 plan**，保存到 `docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md`（grill-me 会话定的路径）。

Plan 要求（用户明确）：
- 颗粒度**很细**（writing-plans skill 标准：每 step 2-5 分钟可执行）
- 用 superpowers:writing-plans skill 撰写
- 路径放 `docs/superpowers/plans/`
- 是"多 agent 接力的主进度板"，agent 做完 P1 接手 agent 看 plan 就知道 P2 怎么走

## 当前进展（已完成）

✅ **grill-me 会话定 SOTA 架构**（2026-05-09 / 05-11 跨日，30+ 个判断点）

✅ **核心架构决策**（按 grill-me 顺序）：

1. **6 范式实现路径** = "指标级函数 + LLM 写胶水代码"（A 路径，否决 B 工具调用、C dispatch 模板）
2. **指标颗粒度** = 按"指标公式"切，每个指标一个有命名的函数（如 `compute_open_arm_time_pct`）
3. **指标物理位置** = 按学术范式分文件：`packages/ethoinsight/ethoinsight/metrics/<范式>.py`（不按 EV19 大类分，因为内部分析路径走学术范式）
4. **code-executor 调用方式** = 通过 `bash` 写胶水脚本 `import` 指标函数 + 调用，**不**调 langchain 工具
5. **废弃的 langchain 工具**（7 个）：`parse_trajectories`、`compute_metrics`、`run_statistics`、`generate_charts`、`assess_and_handoff`、`run_paradigm_analysis`、`get_analysis_template`
6. **图表生成职责** = code-executor 出图（不单独建 chart-maker subagent；ethoinsight-charts skill 移入 code-executor 白名单修复隐性 bug）
7. **paradigm-knowledge skill** 不进 code-executor 白名单，code-executor **自包含**（信息从 by-experiment 搬到新 ethoinsight-code skill 的 by-paradigm/<范式>.md）
8. **ethoinsight-code skill** = 工程师维护的执行手册：范式 → 函数清单 + 调用范例 + handoff schema
9. **行为学决策点处理** = raw data 已经预处理（Activity_State / In zone 等列已固化阈值），工程师写代码时 read raw txt 前几行确认列名后实现函数，**不**要求同事补阈值
10. **穿梭次数定义** = 单向（明 → 暗 / 暗 → 明 任一跨越算 1 次），不是双向
11. **数据可靠性兜底** = 利用 deerflow 现成 middleware（`tool_error_handling_middleware` / `loop_detection_middleware` / `summarization_middleware` / `view_image_middleware` / `subagent_limit_middleware`），LLM 写胶水代码出错的风险全部由这些自动兜底

✅ **6 个 custom skill 命运定**（grill-me 问题 9 用户全确认）：
- `compaction-recovery` — 保留不动
- `ethoinsight` — 保留 + 重定位（给 data-analyst + report-writer 看判读方法论）；`paradigm-interpretation.md` 改指针文件
- `ethoinsight-analysis` — 重命名为 `ethoinsight-code` + 大改（删 5 工具流程描述、改写为 by-paradigm/<范式>.md）
- `ethoinsight-charts` — 保留 + 移入 code-executor 白名单 + description 改为给 code-executor 用
- `ethoinsight-planning` — 保留不动
- `ethovision-paradigm-knowledge` — 保留不动（5月9日刚改过的）

✅ **subagent 白名单确定**：
```python
# code-executor:
skills=["ethoinsight-code", "ethoinsight-charts"]
# data-analyst / report-writer / knowledge-assistant:
skills=None  # 默认全继承不变
```

✅ **5 Phase 实施顺序定**（grill-me 问题 10 用户选 B 路径）：
- **Phase 1**: EPM 端到端打通（最小通路）
- **Phase 2**: OFT 端到端验证 SOTA 架构
- **Phase 3**: Zero Maze / LDB / TST / FST 批量
- **Phase 4**: 旧工具清理（删 7 个 langchain 工具 + shoaling.py + _gate.py）
- **Phase 5**: review checklist 同步并行（不阻塞 Phase 1-4）

✅ **每 Phase 完成标志**：单测 + e2e（同事 review 异步走 Phase 5）

## 关键上下文

### Skill 系统机制（CLAUDE.md backend 第 252 行）
- SKILL.md 全文自动注入到所有 enabled agent 的 system prompt
- references/ 不自动加载，agent 主动 `read_file` 才加载（渐进披露）
- subagent 的 `available_skills` 白名单可独立配置（None = 默认全继承）

### 现有 metrics.py 现状
位置：`packages/ethoinsight/ethoinsight/metrics.py`
- `compute_distance_moved` / `compute_velocity_stats` — 范式无关
- shoaling: `compute_inter_individual_distance` / `compute_nearest_neighbor_distance` / `compute_group_polarity`
- OFT: `compute_center_time_ratio` / `compute_thigmotaxis_index`
- EPM: `compute_open_arm_time_ratio` / `compute_open_arm_entry_count` / `compute_open_arm_entry_ratio` / `compute_open_arm_time` / `compute_total_entry_count`
- LDB / TST / FST / Zero Maze: **完全没写**
- `compute_paradigm_metrics(paradigm)` 是当前的派发器（行 446-525）

### Phase 1 涉及的具体任务（开 plan 时要细化）

```
T1 (EPM): 建 metrics/ 目录 + metrics/epm.py（迁移 5 个现有 EPM 函数）
T2 (EPM): 检查 5 个函数是否够用 / 补函数（同事填的"开臂进臂时间"对应函数可能要新写）
T3 (EPM): pytest 覆盖（参考 tests/test_metrics.py 现有测试）
T6: ethoinsight-charts skill 移入 code-executor 白名单
T7: paradigm-interpretation.md 改指针文件 + ethoinsight SKILL.md 改引用
T5 (EPM): 改名 ethoinsight-analysis → ethoinsight-code + SKILL.md 大改 + by-paradigm/epm.md 撰写
T8: code-executor 配置切换（code_executor.py 改 skills 白名单 + 改 system prompt）
T9 (EPM): 端到端跑通（用 demodata 或 case-002 EPM raw 数据）
```

### 关键文件路径

| 用途 | 路径 |
|---|---|
| 上次会话 v2 交接 | `docs/handoffs/2026-05-09-mvp-paradigm-knowledge-into-skill-handoff-v2.md` |
| 5月8日地基 spec | `docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md` |
| 5月8日地基 plan（已完成） | `docs/superpowers/plans/2026-05-08-ev19-template-skill-foundation-plan.md` |
| 产品级 EV19 双层设计 | `docs/plans/2026-04-29-ev19-template-paradigm-design.md` |
| 行为学 review 包 | `docs/review-packages/2026-04-29-ev19-templates/` |
| ethoinsight 库根 | `packages/ethoinsight/ethoinsight/` |
| 待新建 plan 路径 | `docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md` |
| code-executor 配置 | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` |

## 未完成事项（按优先级）

### P0 — 必须先做

1. ~~处理 working tree 未 commit 改动~~ ✅ **2026-05-11 本次会话末尾已完成**（见本文档顶部 ✅ 段）

2. **写 SOTA 架构切换 plan** 到 `docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md`
   - 用 superpowers:writing-plans skill
   - 颗粒度极细（每 step 2-5 分钟）
   - 5 Phase 完整任务清单
   - 含核心架构决策记录（grill-me 沉淀 11 条 + skill 命运 6 条 + 白名单矩阵）
   - 含"当前阶段状态字段"（让多 agent 接力可 track）
   - **Phase 1 EPM 部分一部分工作已做完**（commit `16eeac9b`）：EPM 4 个新指标 + 测试已就位。plan Phase 1 应该体现这点，从"建 metrics/<范式>.py 目录拆分"和"改名 ethoinsight-analysis → ethoinsight-code"开始

### P1 — Phase 1 启动后做

3. 按 plan Phase 1 task-by-task 实施（subagent-driven-development 或 executing-plans 推动）

### P2 — 异步

4. **review checklist 给同事 review ethoinsight 方法论 skill**（grill-me 已经定方向：建新 review 包）—— 在 Phase 5 启动；之前还有问题待用户决策（review 包结构 / 同事返工 / 阈值是否要补）

5. **streaming PR 上游 review 跟进**（不阻塞 SOTA 架构切换）

6. **`docs/sop/aliyun-deployment-guide.md` untracked 文件** working tree 里有这个 untracked 文件，不是本任务范围；下次会话先确认归属再决定 add 或 .gitignore

## 关键发现 / 决策记录

### 多 agent 框架的核心理由（贯穿 grill-me 全程）
1. **上下文隔离** — subagent 拿专属 system prompt + 任务 prompt，避免 lead 历史污染
2. **并行执行** — 多 subagent 同时干活而非串行

**反模式**：lead 抢着自己做（如 read by-experiment 抽段拷进 task() prompt），把多 agent 退化成单 agent。

### code-executor 真实职责（SOTA 修正后）
```
code-executor:
- 输入: lead 派遣 task() prompt（paradigm + ev19_template + groups + workspace_path + 用户特殊需求）
- 工作: 
  1. read ethoinsight-code/by-paradigm/<范式>.md 知道有哪些指标函数
  2. read ethoinsight-charts skill 知道按数据特性怎么选图
  3. write_file 写胶水脚本（import 指标函数 + 算指标 + 跑统计 + 出图 + 写 handoff）
  4. bash 执行胶水脚本
  5. tool_error_handling middleware 自动给 traceback → LLM 改代码重跑（用户接受 1-2 次重试）
- 输出: handoff_code_executor.json + chart PNG 文件（落 workspace + outputs）
```

### 工程师写代码 SOP（grill-me 校准后）
- **"预制"代码前** read 一份真实 raw txt 前几行（demo-data 或 golden-cases）确认列名 / 单位
- raw data 列名固定，确认一次就在函数实现里固化
- **运行时 agent 不需要再 read raw data 列结构** —— 函数已经知道找什么列

### 行为学同事 review 包 + 工程衍生
- 同事 **single source of truth** = `docs/review-packages/2026-04-29-ev19-templates/by-experiment/<范式>.md`（**领域知识**）
- 工程师 **衍生文档** = `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/<范式>.md`（**API 视角**）
- 两份文件内容有重叠（必算指标 / 数据质量风险），但服务不同 agent 的不同视角
- **同事不需要回去补阈值**（grill-me 问题 6 校准）—— raw data 已经把阈值固化

### deerflow 兜底机制
| Middleware | 作用 | 帮助 |
|---|---|---|
| `tool_error_handling_middleware` | 工具抛错自动给 LLM | bash 脚本崩溃自动给 traceback |
| `loop_detection_middleware` | 同 hash tool call ≥3 警告 ≥5 strip | LLM 反复跑同失败脚本自动中断 |
| `view_image_middleware` | 图片 base64 注入 LLM context | code-executor 出图后 lead/data-analyst 能"看图"判读 |
| `subagent_limit_middleware` | 限制 ≤3 并发 subagent | 多范式并行时自动限流 |
| `summarization_middleware` | 长 thread 自动压缩 | 多次重试导致的 context 膨胀自动管理 |

## 建议接手路径

### 第一步：读这份交接

Working tree 已经清理完（见本文档顶部 ✅ 段），直接进入下一步。

```bash
# 1. 上次会话完整交接
cat docs/handoffs/2026-05-09-mvp-paradigm-knowledge-into-skill-handoff-v2.md

# 2. 看 EV19 双层设计（产品级）
cat docs/plans/2026-04-29-ev19-template-paradigm-design.md

# 3. 看 SKILL.md 最终形态（grill-me 修订后）
cat packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md

# 4. 看现有 metrics.py 结构
grep -n "^def" packages/ethoinsight/ethoinsight/metrics.py
```

### 第二步：写 plan

用 `superpowers:writing-plans` skill 写 plan，保存到 `docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md`。

plan 必须包含：
- header（writing-plans skill 规定的标准 header）
- "## 当前阶段"字段（让多 agent 接力 track）
- "## 全景图（5 Phase）"
- "## 核心架构决策记录"（grill-me 沉淀的 11 条 + 6 个 skill 命运）
- "## Phase 1: EPM 端到端打通" 含完整 task + step（每 step 2-5 分钟）
- "## Phase 2-5" 每 phase 任务清单（与 Phase 1 同颗粒度）
- "## 验收清单"（每 phase 完成标志）

## 风险与注意事项

### 容易混淆的点

1. **shoaling.py 是 demo 玩具，不要复制其脚本模板模式** —— 平行 agent 已经犯过这个错。任何"复制 shoaling.py + 改 paradigm 名"的工作都是错的。
2. **SOTA 架构不写 templates/<范式>.py** —— 指标算法在 `metrics/<范式>.py`，code-executor 在 sandbox 里写胶水代码 import 函数。
3. **不要把 raw data 列结构写进 skill reference** —— 那是工程师写代码时的一次性参考，不是 agent 运行时知识。
4. **不要给 code-executor 加 paradigm-knowledge skill** —— code-executor 用 ethoinsight-code 自包含；paradigm-knowledge 给 lead / data-analyst / report-writer。
5. **同事的 review 包不要工程侧改** —— single source of truth 在同事手里。

### 不建议的方向

1. ❌ 复制 `shoaling.py` 脚本模板模式写新范式
2. ❌ 给 code-executor 加 langchain @tool 工具
3. ❌ 让 lead 在 task() prompt 里抽 by-experiment 段拷进给 code-executor
4. ❌ 让 LLM 现场写指标算法（如 `compute_immobility_latency` 的 run-length encoding）
5. ❌ 强制同事补阈值（行为学决策点已固化在 raw data 列里）
6. ❌ 把 by-experiment / by-template 物理切分到两个 skill（一份内容两份引用）

## 下一位 Agent 的第一步建议

```bash
# 1. 读交接（你正在读的这份）

# 2. 读上次会话 v2 交接（补充 grill-me 前的所有上下文）
cat docs/handoffs/2026-05-09-mvp-paradigm-knowledge-into-skill-handoff-v2.md

# 3. 确认 working tree 干净（应该看不到本任务相关的未 commit 改动）
cd /home/wangqiuyang/noldus-insight
git status --short

# 4. 启动 writing-plans skill
# (在 Claude Code 里调用 Skill 工具，invoke superpowers:writing-plans)

# 5. 按本交接 "建议接手路径 - 第二步" 的要求写 plan
# 路径：docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md
# Phase 1 应该体现 commit 16eeac9b 已经做完的 EPM 算法 + 测试
```

---

## 关联 commit

```
16eeac9b EPM 4 个新指标算法 + 数据质量警告 + 单元测试（本次会话末尾整理平行 agent 产物）
aed045ac v2 交接文档：MVP 范式知识搬入 skill 全程（含 grill-me 修订）
8e6a6c1c 按 grill-me 讨论修订 ethovision-paradigm-knowledge SKILL.md：场景对照表替代分阶段流程描述
474fcf04 MVP 范式知识搬入 skill 交接文档
60cccd02 ethovision-paradigm-knowledge SKILL.md 加分析阶段渐进披露引导（含反模式，被 8e6a6c1c 修订）
96d6e2c4 搬入行为学同事 MVP 6 范式领域知识到 ethovision-paradigm-knowledge skill
```

## 关联文档

- [docs/handoffs/2026-05-09-mvp-paradigm-knowledge-into-skill-handoff-v2.md](2026-05-09-mvp-paradigm-knowledge-into-skill-handoff-v2.md) — v2 上次会话交接
- [docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md](../superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md) — EV19 地基设计
- [docs/plans/2026-04-29-ev19-template-paradigm-design.md](../plans/2026-04-29-ev19-template-paradigm-design.md) — 产品级 EV19 双层设计
- [docs/review-packages/2026-04-29-ev19-templates/README.md](../review-packages/2026-04-29-ev19-templates/README.md) — 行为学同事 review 包
- [CLAUDE.md](../../CLAUDE.md) 第 10 条 — EV19 模板体系重构架构决策
- [packages/agent/backend/CLAUDE.md](../../packages/agent/backend/CLAUDE.md) 第 252 行起 — Skill System 机制
