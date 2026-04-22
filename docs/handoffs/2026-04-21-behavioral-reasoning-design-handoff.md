# 2026-04-21 行为学判断能力 + 微调策略设计会话 — 交接

> 给下一位 Agent：上下文延续点。读这份文档 + 两份新建的 design 文档即可无缝接手。
> 本轮会话类型：**设计文档会话**（非代码实装）。全程在 brainstorming 模式下和用户对齐需求、沉淀设计决策。

---

## 1. 当前任务目标

延续 2026-04-21 fix5 闭环后的下一阶段工作——**从"流程层"跑通到"内容层"深化**。具体是：

1. 更新项目文档反映 fix5 的闭环状态（roadmap、CLAUDE.md、handoff-done 文档）
2. 为 v0.1 的"行为学判断能力"制定可执行蓝图（range M0.2/M0.3 范式补全 + M2.3 异常诊断整合）
3. 澄清微调的技术选型（SFT vs DPO vs PPO、蒸馏策略）并定下分步执行计划

**用户定位**：Noldus 软件开发工程师，正在做 EthoInsight 项目，下一步要和行为学同事协作推进微调 + golden-case 数据收集。

---

## 2. 当前进展

**本轮已完成（全部未 commit、未 push）**：

| # | 动作 | 文件 | 状态 |
|---|---|---|---|
| 1 | 更新 roadmap 当前状态表 + Phase 0 进展 | `docs/roadmap.md`（M） | ✅ |
| 2 | 移除 CLAUDE.md 第 6 条 429 TODO（用户确认已解决） | `CLAUDE.md`（M） | ✅ |
| 3 | 写 fix5 完成记录 | `docs/handoffs/2026-04-21-subtask-visibility-done.md`（新建） | ✅ |
| 4 | 新建：行为学判断能力 v0.1 蓝图（11 章 + 2 附录，1036 行） | `docs/plans/2026-04-21-behavioral-reasoning-design.md`（新建） | ✅ |
| 5 | 新建：微调策略更新 + 分步执行（7 章，318 行） | `docs/plans/2026-04-21-finetuning-strategy-update.md`（新建） | ✅ |

**git 状态**：
- HEAD: `cd2d6aba v0.1`（与 origin/dev 同步）
- 未追踪：3 个新建文件（2 plan + 1 handoff）
- 已修改未暂存：`CLAUDE.md`、`docs/roadmap.md`
- **用户尚未决定是否 commit**——等用户明确指令再做

---

## 3. 关键上下文

### 3.1 项目与分支

- 仓库根：`/home/qiuyangwang/noldus-insight`（WSL2）
- 分支：`dev`（与 origin/dev 同步）
- HEAD：`cd2d6aba v0.1`（v0.1 tag 已打但未发布）
- 用户 memory：
  - Noldus 软件开发工程师、偏实用工程化
  - GLM-5.1 对"禁止 X"反向激活，必须用正面指令（但当前 lead 跑 claude-sonnet-4-6，这条主要用于微调数据设计时注意）
  - 微调方向锁定 Qwen2.5-7B / Qwen3-8B + 两层架构 + RAG

### 3.2 本轮核心设计决策（用户明确 OK 的）

这些决策写进了两份 design 文档，变更前必须和用户商量：

1. **优化函数转向** — `accuracy + 可见思考过程 > 调用时间`。是面向行为学研究员场景的价值排序，不是通用优化
2. **四层能力分层** — L1 代码 / L2 配置 / L3 RAG / L4 模型；微调只让 L4 更稳，不能替代 L1-L3
3. **判断能力两层模型** — Layer A（数值输出，已基本就绪）vs Layer B（洞察判断，v0.1 重点）
4. **"给素材不给规则"原则** — lead agent 不能写死 if/else 路由，只提供能力菜单 + 场景参考表 + ask_clarification，路径选择交给 lead reasoning
5. **Quality-reviewer subagent** — v0.1 新增，由 lead 编排（DeerFlow 架构禁止 subagent 嵌套，见 `subagents/config.py:25`）
6. **两个独立 checkpoint** — data-analyst 和 quality-reviewer 在 Phase 1+ 微调后独立，避免盲点相同
7. **SFT → DPO 两阶段、不做 PPO** — DPO 推迟到 v0.1 后，用 quality-reviewer 互审数据做
8. **蒸馏作为主数据源** — claude-sonnet-4-6 的生产 E2E 会话是 on-policy 蒸馏的金数据
9. **所有训练数据带 `<think>` CoT traces** — Qwen3 原生支持
10. **Golden-case 回归是微调硬前置** — 没有 3 个 golden-case 不动微调

### 3.3 本轮留下的开放问题（用户没明确表态，design 文档里记为"待落地时决定"）

- Quality-reviewer 模型选择：v0.1 起步用同模型（claude-sonnet-4-6）还是不同模型互审？（design §6.8 Q1）
- 审核循环轮数：默认单轮，多轮由 lead 判断触发（design §6.8 Q2）
- Reviewer metadata UI 展示：noldus-kb 不可用时要不要前端新组件展示降级标识？（design §6.8 Q3 / §11.4 已倾向不做）

### 3.4 关键文件路径

**本轮新建的设计文档**（重点读物）：
- [docs/plans/2026-04-21-behavioral-reasoning-design.md](docs/plans/2026-04-21-behavioral-reasoning-design.md) — 行为学判断能力设计（11 章）
  - §3 能力分层框架（L1-L4）
  - §4 Layer A/B 模型（**整份文档的基础词汇**）
  - §5 Lead agent 能力菜单扩展（场景参考表 A-E）
  - §6 Quality-reviewer 完整设计（schema + checklist + 降级路径 + §6.3a 具体例子）
  - §7 范式补全八件套
  - §8 异常模式库 A/B/C/D（§8.8 标注"待行为学同事优化"）
  - §9 Golden-case 验收体系（含 expected-analysis.yaml schema）
  - §11 v0.1 不做的事
- [docs/plans/2026-04-21-finetuning-strategy-update.md](docs/plans/2026-04-21-finetuning-strategy-update.md) — 微调策略 + 分步计划（7 章）
  - §1 术语速查表（SFT/DPO/PPO/LoRA/蒸馏）
  - §2 对 04-13 老文档的 6 条更新
  - §3 **6 个 Step 的分步执行计划**（行为学同事协作版）
  - §4 同事价值递增路径（22-34 小时总投入）
  - §7 **用户的 3 个 immediate actions**

**本轮修改的文档**（diff 小）：
- [CLAUDE.md](CLAUDE.md) — 移除第 6 条 429 TODO（用户说已解决）
- [docs/roadmap.md](docs/roadmap.md) — 当前状态表 + Phase 0 M0.1 状态

**本轮新建的交接文档**：
- [docs/handoffs/2026-04-21-subtask-visibility-done.md](docs/handoffs/2026-04-21-subtask-visibility-done.md) — fix5 闭环记录

**本轮会话里反复引用的既有文档**（没改）：
- [docs/handoffs/2026-04-21-subtask-visibility-handoff.md](docs/handoffs/2026-04-21-subtask-visibility-handoff.md) — 本会话起点的上轮交接
- [docs/plans/2026-04-13-fine-tuning-small-model-design.md](docs/plans/2026-04-13-fine-tuning-small-model-design.md) — 微调基础架构（本轮新文档是对它的叠加更新，不替代）
- [packages/agent/backend/packages/harness/deerflow/subagents/config.py](packages/agent/backend/packages/harness/deerflow/subagents/config.py) — L25 有 `disallowed_tools=["task"]` 证据（subagent 不能嵌套）
- [packages/ethoinsight/ethoinsight/templates/shoaling.py](packages/ethoinsight/ethoinsight/templates/shoaling.py) — 范式模板标杆（217 行）
- [packages/ethoinsight/ethoinsight/assess.py](packages/ethoinsight/ethoinsight/assess.py) — 阈值 YAML 入口

### 3.5 常用命令

```bash
# 启动 / 停止服务
cd /home/qiuyangwang/noldus-insight/packages/agent
make dev          # localhost:2026
make stop

# 测试
cd packages/agent/backend && source .venv/bin/activate && make test
cd packages/ethoinsight && source /home/qiuyangwang/noldus-insight/packages/agent/backend/.venv/bin/activate && python -m pytest tests/ -q

# 前端
cd packages/agent/frontend && pnpm check && SKIP_ENV_VALIDATION=1 pnpm build
```

---

## 4. 关键发现

### 4.1 用户提出的两个决定性思想

本轮会话里用户推翻了我的两次"过度工程化"设计，这些决定性 pushback 必须保留在未来设计里：

**决定 1 — "accuracy + 可见思考过程 > 调用时间"**（写进 design §2）
- 原始上下文：我问"多 subagent 互审延迟+token 成本值不值"
- 用户回答：用户更在乎准确度和思考过程暴露
- 推论：承担互审开销是正确的；思考过程是价值本身；可以主动打断问用户

**决定 2 — "给素材不给规则，不要规定死 route"**（写进 design §5）
- 原始上下文：我设计了"Lead agent 从 pipeline 到 router"章节，规定 raw→主路 / analyzed→审核路
- 用户反问：如果规定死就失去了用 agent 的意义
- 推论：代码里不写路由 if/else，只扩展 lead 的能力菜单（新工具 + prompt 场景素材），路径选择归 lead

### 4.2 架构硬约束（来自代码，用户独立发现）

**DeerFlow 禁止 subagent 嵌套调用**：
- 证据：`subagents/config.py:25` 默认 `disallowed_tools=["task"]`；所有 builtin subagent 显式 `disallowed_tools=["task", ...]`；`general_purpose.py:47` 注释 `# Prevent nesting`
- 影响：quality-reviewer **不能**被 data-analyst 调用，必须由 lead agent 编排
- 这一点 design §5.5 和 §6.1 明确写了

### 4.3 v0.1 的硬闸门

从 design §9 + finetuning §6 综合：

1. 5 个范式 × 5 份 golden-case 全绿
2. Qwen3-8B SFT 模型 ≥ claude-sonnet baseline（相同 golden-case）
3. 行为学同事 rubric 每维度 ≥ 4/5
4. 训练数据 ≥ 800 条（带 `<think>`、合格率 > 80%）
5. quality-reviewer 降级版稳定运行

**没有 golden-case 不动微调** — 硬前置。

### 4.4 用户已确认不做的事（design §11）

- PPO 强化学习
- 全参微调（用 LoRA）
- 多轮互审强制化
- 异常模式 A-D 之外的新类别（YAGNI）
- Reviewer metadata 的前端 UI 组件
- 跨范式推理 / 实验设计指导 / 文献综述生成（留给 v1.0+）

---

## 5. 未完成事项（按优先级）

### 5.1 【高】用户的 3 个 immediate actions（finetuning §7）

用户已决定要做，尚未启动：

1. **今天**：拷贝 fix5 的 shoaling E2E 数据（thread `6f046cc7-775a-4eb9-9027-2022e50781ca`）到 `golden-cases/case-001-shoaling-baseline/`，按 design §9.3 的目录结构组织
2. **本周**：用 finetuning §3.1 的信息模板联系行为学同事，约 30 min 启动会
3. **启动会**：发两份 design 文档 + 过 finetuning §3 的 5 步计划和同事的时间线

**下一位 Agent 遇到这些时不用主动推进**——等用户明确问才做。

### 5.2 【高】Commit 决策（本轮结束后马上要决定）

用户还没说是否 commit 本轮 5 个文件改动。如果用户说"commit"：

- **建议 commit 内容**：
  - `CLAUDE.md`（M）
  - `docs/roadmap.md`（M）
  - `docs/handoffs/2026-04-21-subtask-visibility-done.md`（new）
  - `docs/plans/2026-04-21-behavioral-reasoning-design.md`（new）
  - `docs/plans/2026-04-21-finetuning-strategy-update.md`（new）
- **不要 commit**：`docs/plans/2026-04-21-subtask-visibility-and-language.md`（上轮 fix5 的执行计划，一直保持未追踪）、`docs/e2e_tests/*`（惯例）
- 用户未明示前**不要 push**

### 5.3 【中】当行为学同事启动后的后续动作

按 finetuning §3 的 Step 1 → Step 6 执行：

- Step 2（5 月）：EPM 范式补全（roadmap M0.2）+ 抽 `templates/_base.py`
- Step 3（6 月）：Golden-case runner + OFT + quality-reviewer 降级版原型
- Step 4（7 月）：SFT 数据采集 + Fireworks 训练
- Step 5（8 月）：模型渐进切换 + v0.1 收尾
- Step 6（9 月+）：DPO 迭代

每步的"工程线"和"行为学线"分项都在 finetuning §3 详细写了。

### 5.4 【中】需要行为学同事 review 后才定稿的部分

- design §8 异常模式库（§8.8 已标注状态）
- design §9.3 expected-analysis.yaml schema（case-001 作样板给同事看）
- design §6.3a 的 reviewer 示例（是否符合行为学实战措辞）

这些地方任何未来改动需要同事 review 反馈作为依据。

### 5.5 【低】Roadmap 和两份 design 的交叉引用是否足够

design 两份文档互相引用完整（附录 B）。但 roadmap.md 没引用这两份新文档——如果用户未来问"Phase 0 M0.2 怎么做"，应该指向 design §7 + finetuning §3 Step 2。可以在 roadmap 对应位置加 link，但这是 nice-to-have，不紧急。

---

## 6. 建议接手路径

### 6.1 第一步：看用户当前状态

```bash
cd /home/qiuyangwang/noldus-insight
git log --oneline -3       # 确认 HEAD 是 cd2d6aba（或用户刚 commit 的新 hash）
git status -s              # 看本轮的 2 M + 3 ?? 是否还在，或已被 commit
ls docs/plans/2026-04-21-*.md   # 确认三个 plan 文件都在
ls docs/handoffs/2026-04-21-*.md  # 确认三个 handoff 文件都在
```

### 6.2 第二步：读核心文件（按重要性）

1. 本文档
2. `docs/plans/2026-04-21-finetuning-strategy-update.md` — 执行计划和用户接下来做什么
3. `docs/plans/2026-04-21-behavioral-reasoning-design.md` §1-4（前 4 章是整份文档的基础词汇）+ §11（明确不做什么）
4. `docs/handoffs/2026-04-21-subtask-visibility-handoff.md` + `docs/handoffs/2026-04-21-subtask-visibility-done.md`（本轮上游的两个交接，了解整体脉络）

### 6.3 第三步：根据用户反馈分支

- **用户说 "commit"**：按 §5.2 的 5 个文件做 commit（不 push 除非明示）
- **用户说"开始 Step 1 / 拷贝 case-001 数据"**：按 finetuning §7 第 1 动作做，创建 `golden-cases/case-001-shoaling-baseline/` + 从 thread 目录拷数据
- **用户已经和行为学同事沟通过、开始 Step 2**：进入 EPM 实装路径（roadmap M0.2 + design §7 + finetuning §3.2），**这时进入 brainstorming 模式重新对齐具体实装方案**，不要直接动代码
- **用户问设计里某个决定的理由**：直接引用 design 对应章节回答，不重新设计
- **用户说"重新讨论 X 决策"**：用 brainstorming 模式，参考 design §11 的总原则（"不做的诱惑三问"）做决策

---

## 7. 风险与注意事项

### 7.1 不要做的事

- **不要 git add** `docs/plans/2026-04-21-subtask-visibility-and-language.md`（上轮 fix5 的执行计划草稿，刻意不追踪）
- **不要 git add** `docs/e2e_tests/`（惯例未追踪）
- **不要**自己 push，必须问用户
- **不要**改 DeerFlow middleware 框架（StreamBridge / 消息路由 / agent graph 结构）
- **不要**在 subagent 里写 `task` 工具调用（架构禁止，见 §4.2）
- **不要**把 429 重试 TODO 加回 CLAUDE.md（用户说已解决）
- **不要**在 design 文档里加"不做的事"清单外的新能力，除非用户明确同意（YAGNI + scope creep）
- **不要**把判断能力文档改成"规定死 route"的形式（违反本轮核心设计原则 §4.1 决定 2）

### 7.2 容易误判

- **Lead 当前跑的是 claude-sonnet-4-6，不是 GLM-5.1**——但 user memory 里提到的"GLM 对禁止 X 反向激活"规则主要用于未来微调数据设计时避免负面指令
- **Design §6.3a 的 reviewer 示例是我编的剧情**——用 shoaling Subject 3 NND=70.02 数据真实、但"Miller et al., 2024"是编的文献。用户没反对但后续应该让行为学同事替换成真实文献
- **"过时的 04-13 微调文档"并没有被替代**——finetuning §0 明确说是"叠加更新不替代"。未来做决策时两份都要读
- **"两个独立 checkpoint"是 v0.1+ 的事**——v0.1 阶段 data-analyst 和 quality-reviewer 共享 base model 用不同 prompt，design §4.5 和 §6.6 都明确了
- **Golden-case runner 还没实装**——design §9 是设计，`tests/test_golden_cases.py` 和 `make test-golden` 是 Step 3（6 月）要做的

### 7.3 已知的 pre-existing issue（不要顺手"修"）

- 沿用上轮 handoff §7.3 的判断：Pydantic warning、test_granular_tools skip、ethoinsight-architecture.html 的 M 都不是本轮范围
- `llm_error_handling_middleware.py:73-74` 默认值还是 `retry_base_delay_ms=1000, retry_cap_delay_ms=8000`——用户说 429 问题已解决（可能是外部 API 稳定了或 circuit breaker 接住了），文档层面我们清了 TODO，代码没动。如果未来 429 重现再回来看

---

## 8. 下一位 Agent 的第一步建议

**第一个动作**：读用户的新 message，判断他在哪一步。

**如果用户说"commit"**：
1. 按 §5.2 清单 stage 5 个文件（不含 2 个白名单排除文件）
2. commit message 用中文，简洁（示例：`docs: 行为学判断能力设计 + 微调策略更新`）
3. 不 push，除非用户明示

**如果用户问"接下来做什么 / 怎么开始"**：
1. 指向 `finetuning-strategy-update.md §7` 的 3 个 immediate actions
2. 问用户要先做哪个（拷 case-001 / 联系同事 / 启动会准备）
3. 按选择的动作执行

**如果用户说"拷贝 case-001 数据"**：
1. 按 design §9.3 的目录结构：
   ```
   golden-cases/case-001-shoaling-baseline/
   ├── raw-data/     # 从 .deer-flow/threads/6f046cc7.../user-data/uploads/ 拷
   ├── metadata.yaml  # 新写：物种 zebrafish、范式 shoaling、分组
   ├── expected-analysis.yaml  # 新写：填你能填的部分
   └── notes.md（可选）
   ```
2. `metadata.yaml` 和 `expected-analysis.yaml` 的 schema 直接照 design §9.3 的例子
3. 数据和 schema 准备好后，建议用户把 case-001 发给行为学同事作样板

**如果用户说"开始 Step 2 / EPM 实装"**：
1. 这是**代码工作不是设计工作**，进入 plan 模式或 brainstorming 重新对齐具体实装方案
2. 参考：design §7.2 八件套、§7.3 模板骨架抽象、existing [shoaling.py](packages/ethoinsight/ethoinsight/templates/shoaling.py)
3. 确认优先抽 `templates/_base.py`，还是先直接照抄 shoaling.py 写 epm.py 再重构

**如果用户问"XX 章节为什么这样设计"**：
1. 直接引用 design 对应章节回答
2. 不要重新推导或修改，设计已经对齐过

**如果用户说"重新讨论 X 决定"**：
1. 用 brainstorming 模式（不要直接动手改文档）
2. 参考 design §11.5 的决策三问："不做会让哪个硬指标失败？"
3. 确定改动后才去改 design 文档

---

**签名**：Claude Opus 4.7 (1M context)，2026-04-21 设计会话 handoff
