# EthoInsight LLM Strategy 设计评审 — 交接文档

> 日期: 2026-04-09
> 上一份交接: `docs/handoffs/2026-04-09-glm5-instruction-following-handoff.md`

---

## 1. 当前任务目标

**问题**: EthoInsight 的 LLM 策略需要从 GLM-5.1 迁移到更强的模型，同时需要明确 agent 多智能体架构是否正确。

**预期产出**: 已完成的设计评审（/office-hours → /plan-eng-review → /plan-ceo-review），确认架构方向和实施路线。

**完成标准**: 设计文档通过工程评审，CEO 评审确认产品方向正确。

---

## 2. 当前进展

### /office-hours 设计文档 ✅ 全部完成

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | 产出设计文档 "Ship Now, Tune Later" | ✅ APPROVED |
| 2 | 两轮对抗性评审 (7/10 → 8/10) | ✅ |
| 3 | 设计文档路径: `~/.gstack/projects/noldus-cn-beijing-noldus-insight/qiuyangwang-feature-etho-skills-design-20260409-151642.md` | ✅ |

### /plan-eng-review 工程评审 ✅ 全部完成

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | Step 0: 范围评估 | ✅ 范围接受 |
| 2 | Section 1: 架构评审 (5 issues) | ✅ 全部解决 |
| 3 | Section 2: 代码质量 (2 issues) | ✅ 全部解决 |
| 4 | Section 3: 测试评审 (1 issue: E2E 测试) | ✅ 全部解决 |
| 5 | Section 4: 性能评审 (2 issues, 用户推迟) | ✅ |
| 6 | Outside Voice (Claude subagent, 7 findings) | ✅ |
| 7 | 设计文档已更新: Phase 0 重新排序为 0c→0a→0b | ✅ |
| 8 | TODOS.md 已创建 | ✅ |
| 9 | 评审日志已记录 (CLEAR, 0 critical gaps) | ✅ |

### /plan-ceo-review CEO 评审 ⏳ 进行到一半，被重定向

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | 前置检查 (preamble, context recovery) | ✅ |
| 2 | 系统审计 | ✅ |
| 3 | Step 0: 前提挑战 + 模式选择 (SELECTIVE EXPANSION) | ✅ |
| 4 | Cherry-pick 提案 (#1-#4 全部被跳过/拒绝) | ✅ |
| 5 | **核心讨论: 是否应该做 agent 框架？** | ✅ 已确认方向正确 |
| 6 | Sections 1-11 评审 | ❌ 未开始 |
| 7 | Outside Voice | ❌ 未开始 |
| 8 | 输出 (NOT in scope, TODOS, Failure modes 等) | ❌ 未开始 |
| 9 | 评审日志 + Dashboard | ❌ 未开始 |
| 10 | Telemetry | ❌ 未开始 |

---

## 3. 关键上下文

### 架构决策已确认

用户在 CEO 评审中明确了多智能体架构的合理性。核心论点：

1. **code-executor** 不是简单调用 ethoinsight 函数，是根据不同实验设计**生成分析代码**
2. **data-analyst** 需要 noldus-kb 的行为学领域知识来**解读**统计结果（p=0.03 在 EPM 里意味着什么）
3. **report-writer / knowledge-assistant** 综合生成报告和回答问题
4. **主从架构原因: 上下文窗口管理** — 一个模型塞不下所有 context（数据、知识库、代码、结果、模板），分 subagent 每个只需要自己的 context
5. **未来本地部署 7B-14B 小模型** context window 更小，主从架构是为了适配这个限制

### ethoinsight Python 库（核心价值所在）

路径: `packages/ethoinsight/ethoinsight/`

| 文件 | 功能 |
|------|------|
| `parse.py` | EthoVision XT 数据解析器（UTF-16 LE, 中文表头） |
| `metrics.py` | 行为指标计算（距离、速度、区域时间等） |
| `statistics.py` | 自动选择统计检验（Shapiro-Wilk → t-test/Mann-Whitney, ANOVA/Kruskal-Wallis, Bonferroni 校正, 效应量） |
| `charts.py` | 出版级图表（色盲友好 Okabe-Ito 调色板） |
| `assess.py` | 领域特定解读（参考阈值，如 OFT center time ratio < 0.1 = 高焦虑） |
| `templates/` | 范式特定分析模板（目前有 shoaling） |

### 设计文档核心决策

- **方法**: Ship Now, Tune Later — 先用 API 模型 + noldus-kb，收集数据，有证据再微调
- **Phase 0 重新排序** (工程评审后): 0c 架构稳定化 → 0a 模型选择 → 0b 日志管道
- **Phase 0c 稳定性标准**:
  1. Lead agent 正确路由到对应 subagent
  2. Code-executor 首次 tool call 是 `get_analysis_template` (>90%)
  3. 15 次连续运行无 `GraphRecursionError`
  4. 端到端完成率 >80%
- **自动化 E2E 测试**: 使用 DeerFlowClient (pytest) 实现，无 UI 依赖
- **日志管道**: 基于中间件的集成（DeerFlow 中间件链中新增）
- **model=inherit 保持不变**: 所有 subagent 继承 lead agent 模型，Phase 2 再评估是否需要按 subagent 分配不同模型

### 用户明确表达的优先级

- "先不考虑本地部署模型的事情，我想先搞清楚 agent 的架构，先用 api 来运行"
- "与其一直切换模型，不如先构建好 agent 工程，之后我们可以直接切换不同的模型测试效果"
- "最终 agent 的效果评估，将由行为学分析人员测试后决定"
- 成本和延迟目前不是优先项
- GDPR 技术执行推迟（测试阶段用合成数据）

### 工程评审中的关键 Outside Voice 发现（已解决）

1. model=inherit 强制一个模型做 5 种任务 → 用户接受，Phase 2 再评估
2. Phase 0c/0a 并行执行会导致无法隔离问题 → 用户选择先稳定架构再换模型
3. 200 交互数据统计上不够微调决策 → Phase 2 定性为质性判断
4. 没有 API 模型全部失败的后备方案 → 用户跳过
5. noldus-kb 质量无人负责 → 已加入 TODOS.md

---

## 4. 关键发现

### 已存在的文件和系统

| 系统 | 状态 | 复用情况 |
|------|------|----------|
| DeerFlow model factory + config.yaml | 工作正常 | 模型切换只需改配置 |
| noldus-kb MCP (5 个工具) | 工作正常 | 保持不变 |
| 4 个 Noldus subagent | 工作但 GLM-5.1 下不稳定 | Phase 0c 稳定化 |
| 12 中间件链 | 工作正常 | Phase 0b 扩展日志中间件 |
| DeerFlowClient (77 单元测试) | 工作正常 | 用于 E2E 测试自动化 |
| 92 个测试文件 (backend/tests) | 工作正常 | 扩展行为分析测试 |
| noldus-kb 测试目录 | **空** (无测试文件) | 需关注 |
| config.yaml 中有硬编码的 ZhiPu API key | 安全风险 | 用户已知，会处理 |

### CEO 评审 gstack 状态

- gstack 版本: v0.16.1.0 (v0.16.2.0 可用，用户选择暂不升级)
- Telemetry: anonymous
- Proactive: true
- Cross-project learnings: false (project-scoped only)
- `_TEL_START=1775723513`, `_SESSION_ID=2133415-1775723513` (CEO 评审 telemetry 变量)

---

## 5. 未完成事项

### 高优先级

1. **继续 /plan-ceo-review** — Sections 1-11 评审未开始
   - 模式已选: SELECTIVE EXPANSION
   - Cherry-picks 全部被拒绝，无范围扩展
   - 需要从 Section 1 (Architecture) 开始
   - 注意: 很多内容与 /plan-eng-review 重叠，CEO 评审应聚焦产品/战略层面而非重复工程细节

2. **完成 CEO 评审后续步骤**:
   - Outside Voice
   - 输出 (NOT in scope, What already exists, Dream state delta, Error map, Failure modes, TODOS)
   - 评审日志 (`gstack-review-log`)
   - Review Readiness Dashboard
   - Plan File Review Report (更新设计文档的 GSTACK REVIEW REPORT)
   - Telemetry

### 中优先级

3. **开始 Phase 0c 实现** — 这是用户最想做的事:
   - 稳定 agent 架构
   - 构建 E2E 测试 (pytest + DeerFlowClient)
   - 让 3 个范式（OFT, EPM, MWM）端到端跑通

### 低优先级

4. **TODOS.md 中的 noldus-kb 质量负责人** — Pre-Phase 1 交付物

---

## 6. 建议接手路径

### 第一步：决定是否继续 CEO 评审

用户可能更想直接开始实现 Phase 0c，而非继续评审。问用户：
- A) 继续 /plan-ceo-review（从 Section 1 开始）
- B) 跳过剩余评审，直接开始 Phase 0c 实现
- C) 做一个精简版 CEO 评审（只覆盖产品方向相关的 sections）

### 如果继续 CEO 评审

```bash
# 读取设计文档
cat ~/.gstack/projects/noldus-cn-beijing-noldus-insight/qiuyangwang-feature-etho-skills-design-20260409-151642.md

# 读取现有评审日志
export PATH="$HOME/.bun/bin:$PATH"
~/.claude/skills/gstack/bin/gstack-review-read
```

从 Section 1 (Architecture Review) 开始，但聚焦于：
- 产品/用户价值层面的架构评估
- 不要重复工程评审已经覆盖的技术细节
- CEO 评审特有的: Error & Rescue Map, Security, Observability, Deployment, Long-Term Trajectory

### 如果开始 Phase 0c

```bash
cd /home/qiuyangwang/noldus-insight
# 查看当前 agent 状态
cat packages/agent/config.yaml | head -30
# 查看 subagent 定义
ls packages/agent/backend/packages/harness/deerflow/subagents/builtins/
# 查看最近的修改
git log --oneline -10
```

Phase 0c 的第一步:
1. 读取本交接文档 + 设计文档了解全貌
2. 定义 E2E 测试用例（基于 DeerFlowClient）
3. 运行现有 agent，记录当前稳定性基线
4. 按照稳定性标准逐项修复

---

## 7. 风险与注意事项

1. **CEO 评审的 telemetry 未完成**: `_TEL_START=1775723513`, `_SESSION_ID=2133415-1775723513`。如果继续 CEO 评审，需要在结束时运行 telemetry 命令。如果跳过，运行 abort telemetry。

2. **gstack 工具需要 bun 在 PATH 中**: `export PATH="$HOME/.bun/bin:$PATH"` — 否则 `gstack-review-log` 等命令会因为 JSON 校验失败而报错。

3. **不在 git 仓库根目录**: 用户的 CWD 是 `/home/qiuyangwang`，不是 `/home/qiuyangwang/noldus-insight`。需要 `cd` 到项目目录。

4. **用户偏好正面指令**: GLM-5.1 对"禁止X"会反向激活，已记录到 memory (`feedback_positive_prompting.md`)。虽然要换模型，但这个偏好值得保留。

5. **用户是工程师思维**: 更关注"能不能跑通"而非"评审文档是否完善"。不要在评审流程上花过多时间，除非用户主动要求。

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. 问用户: 继续 CEO 评审，还是直接开始 Phase 0c 实现？
3. 如果继续评审: 从 Section 1 开始，保持精简，聚焦产品决策
4. 如果开始实现: 读取设计文档中的 Phase 0c 稳定性标准，构建 E2E 测试框架
