# 2026-05-14 Handoff 双层协议 Spec 落地交接

> **状态**：spec 草案已写完落盘、dogfood-fix 修复已被另一 agent 全部跑完
> **核心交付**：[2026-05-14-handoff-protocol-and-runtime-isolation-spec.md](../../superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md)
> **下一步阻塞**：用户做端到端测试 → 同事 review spec → 定稿 → 写 plan → 派新 agent

---

## 当前任务目标

把 thread `5046a6e6-4bfc-4ca9-9650-b674ec3734cf` dogfood 暴露的架构问题 + 同事 PR #7 反馈 + 用户对 handoff 双层协议的诉求，沉淀成一份**架构设计 spec**，作为下一阶段实施计划的输入。

**核心阻塞已解除**：上一会话留下的 dogfood 修复 plan（10 个 Task）已由另一执行 agent **跑完 11 个 commit**——dev 分支领先 origin 11 个 commit。

---

## 当前进展

### ✅ 已完成

#### 1. Spec 草案落盘

`docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md`（~860 行）

涵盖 12 章：
- 设计原则（3 条新硬约束）
- 1. 背景与触发问题
- 2. 顶层设计：双层 handoff 协议（L1 message history + L2 workspace 文件）
- 3. Run-scoped 路径绝缘
- 4. 用户感知工作流（14 步 happy path）
- 5. EthoInsight 输出宪法（output-constitution.md 设计）
- 6. Catalog Modifier（仅指方向，不实施）
- 7. 与现有 Issue / Plan 关系
- 8. 实施路径概览（阶段 1-4）
- 9. 验收标准（7 条）
- 10. 未决项与风险
- 11/12. 修订记录 + 评审邀请

**关键设计判断**（用户已拍板）：
- L1 是所有 agent 默认契约（不只是给 lead 用）
- L2 是审计 + 例外升级路径
- `handoff_suffix` 由 lead 显式命名
- L1 < 5 KB 硬上限，先用规范约定不做强校验
- 双层 handoff 走 deerflow 原生通道：L1 经 task return value → ToolMessage、L2 经 workspace 文件 + `{{handoff://}}` 占位符

#### 2. 同事 PR #7 已 review 并融入 spec

PR #7 (commit `b5927e1`) 内容：
- [Noldus 新视觉方案总结](../../review-packages/Noldus新视觉方案总结.md)（独立工作，不在本 spec）
- [指标外分析要求](../../review-packages/2026-05-13-常见行为学指标外分析要求.md) → spec 第 6 章 modifier 方向
- [Gap 清单](../../review-packages/2026-05-13-还差哪些with细颗粒度.md) → spec 第 5 章 output-constitution + 第 9 章验收标准
- 两份 skill references 改动（effect-size-guide.md / statistics-decision-tree.md）— 与本 spec 无冲突

#### 3. Dogfood 修复 plan 已被另一 agent 全部跑完

执行 agent 跑完了 [2026-05-13-dogfood-fix-iteration-plan.md](../../superpowers/plans/2026-05-13-dogfood-fix-iteration-plan.md) 全部 10 个 Task，共 11 个 commit（dev 分支领先 origin 11 个 commit）：

| commit | Task | 内容 |
|--------|------|------|
| `555db882` | Task 1 | fix(reload): `.deer-flow/**` 递归 exclude |
| `24715250` | Task 2 | fix(lead): 角色边界 prompt（禁判读/品系/常模/金标准）|
| `8e53d064` | Task 3 | test(thinking): 复现 400 错误的单测 |
| `5d071e9a` | Task 4 | fix(claude_provider): strip malformed thinking blocks |
| `356c3de9` | Task 5 | fix(lead): 阶段播报强制清单 |
| `658f78a1` | Task 6 | fix(frontend): 受控 open 拦 reasoning 折叠 |
| `2eb1532a` | Task 7 | fix(catalog): 强制虚拟路径 + virtual_workspace_dir 参数 |
| `49414a19` | Task 8 | fix(code-executor): 禁止 ls 后重跑 |
| `423d8760` | Task 9 | docs: 验证结果交接 |
| `c745748c` | Task 10 | feat(checkpointer): adelete_for_runs |
| `f9f636d4` | 额外 | fix(lead): thinking 用用户语言（中文）|

**验证状态**（[2026-05-14-dogfood-followup-handoff.md](2026-05-14-dogfood-followup-handoff.md)）：
- ✅ #1 gateway reload 次数 = 0（自动验证）
- ✅ #10 checkpointer warning 消失（自动验证）
- ⏳ #2/#3/#4/#5/#6/#7/#8 等 9 项**需要人工 dogfood 验证**

---

## 关键上下文

### 项目状态

- **当前会话身份**：架构 spec 讨论 + 撰写
- **执行 agent**：已完成 dogfood-fix 全部 Task，在另一会话已停止
- **dev 分支**：领先 origin 11 个 commit + 本会话新增 spec 文件
- **关键提醒**：本会话**没有 commit 任何东西**——spec 文件是 untracked，由用户决定何时 commit

### 仓库结构关键路径

```
docs/superpowers/specs/
├── 2026-05-11-subagent-file-is-facts-design.md          # 前置：files-are-facts 哲学
├── 2026-05-13-metric-catalog-architecture-design.md     # 前置：catalog single source of truth
└── 2026-05-14-handoff-protocol-and-runtime-isolation-spec.md   # ★ 本会话核心交付

docs/superpowers/plans/
└── 2026-05-13-dogfood-fix-iteration-plan.md             # 已被另一 agent 跑完的 plan

docs/handoffs/2026-05/
├── 2026-05-13-catalog-dogfood-iteration-handoff.md      # 上一会话起点
├── 2026-05-14-dogfood-followup-handoff.md               # 执行 agent 写的验证交接
├── 2026-05-14-thinking-field-diagnosis-notes.md         # 执行 agent 写的诊断笔记
└── 2026-05-14-spec-handoff-protocol-handoff.md          # ★ 本文件

docs/review-packages/
├── 2026-05-13-常见行为学指标外分析要求.md                 # 同事 PR：modifier 需求
├── 2026-05-13-还差哪些with细颗粒度.md                    # 同事 PR：Gap 清单
└── Noldus新视觉方案总结.md                              # 同事 PR：品牌（独立工作）
```

### 关键概念锁定

为了避免下次会话再绕弯，把本次会话讨论沉淀的**术语**列清楚：

| 术语 | 精确含义 |
|------|---------|
| **thread** | 对话窗口的上下文容器，永久存在直到删除 |
| **runtime** | Agent 进程的运行实例（gateway 进程、langgraph 进程，reload 后是新 runtime）|
| **run** | LangGraph 的最小执行单元，用户每发一条消息触发一次（thread 5046a6e6 有 5 个 run）|
| **task_id** | lead 每次调 `task()` 工具的唯一 ID |
| **trace_id** | SubagentExecutor 创建时生成的 8 字符 ID |
| **L1 摘要** | 进入 message history 的简练 handoff（< 5 KB），所有 agent 默认消费 |
| **L2 hard fact** | 落到 workspace 文件的完整 handoff（不限大小），审计 + 例外升级路径 |
| **handoff_suffix** | lead 派遣 subagent 时显式命名的字符串（如 `epm_basic`、`time_segment_5bins`），用于 L2 文件名空间隔离 |

### 用户的 4 条核心架构原则（不能违反）

1. **handoff 简练、格式化、是 subagent → lead 的标准契约**
2. **runtime / thread / subagent / lead / run 路径绝缘**
3. **在 deerflow 框架内自洽，不绕开框架**
4. **handoff 双层职责分离**——L1 摘要给所有 agent 默认消费、L2 hard fact 给审计 + 例外升级

### 用户的 AgentCore 经验（关键启发）

> "文件 handoff 会很大，但不漂"

经反推，AgentCore 之所以"不漂"是因为 **lead 实际只消费文件中的摘要部分**，本 spec 把这条隐含约束显式化为双层 handoff。

### 同事 PR #7 反馈核心信号

- **Gap #1 "进一步落实不要参考常模或基线的默认原则"**——"落实"两字 = 之前承诺过但 thread 5046a6e6 没做到
- **Gap #4 "5 分钟内出报告"**——dogfood 跑了 10+ 分钟太慢
- **"编写代码耗时过久"**——指 lead 自己 write_file 写脚本的反模式
- **指标外分析需求**：时间分段、分区、事件/状态分段三类——是 Issue #9 ad-hoc 路径的 ground truth 输入

---

## 关键发现

### A. dogfood 修复 plan 已全部完成，不再阻塞 spec 落地

11 个 commit 涵盖 Issue #1-#10 全部修复 + 额外的 thinking 字段 strip 修复。**dev 分支领先 origin 11 个 commit、本地未 push**。

### B. Spec 本身**不可直接执行**

Spec 是设计文档，**不**含可执行 Task 步骤。下一步必须由 Claude（或新会话 agent）用 `superpowers:writing-plans` skill **将 spec 第 8 章实施路径展开为 plan**。

不要把 spec 直接交给执行 agent——会重蹈"agent 自行编实施细节、踩 import 名错误的坑"的覆辙（上次 review 已经命中 3 个阻塞 bug）。

### C. 端到端测试必须在写新 plan **之前**完成

理由：
- spec 第 2-3 章核心假设（L1 < 5 KB 不漂、lead 不 read L2）**未经过实证**
- 如果 dogfood 走出来发现 L1 摘要确实让 lead 漂了——spec 需要调整
- 在错误假设上写 plan = 浪费工程

### D. 用户原话："开启新 agent 去执行这份计划"——含义需澄清

本会话末尾用户说要"开启一个新的 agent 去执行这份计划"。但已澄清：
- Spec 不是 plan，agent 不能直接执行
- 必须先：端到端测试 → spec 定稿 → 写 plan → agent 执行

**新会话第一件事就是和用户确认这条流程，避免误启动**。

---

## 未完成事项

### 🔴 阻塞 — 必须先做

#### 1. 端到端测试 dogfood 修复

执行 agent 已完成所有代码改动，但 [dogfood-followup-handoff](2026-05-14-dogfood-followup-handoff.md) 显示 **9/11 检查项还需要人工 dogfood 验证**。

具体操作：
1. 重启服务：`cd packages/agent && make stop && make dev`
2. 打开 `http://localhost:2026`，新建 thread
3. 上传 `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/` 任意一份 EPM 单只数据
4. 跟着流程走（反问 → 单只描述 → 是否要洞察）
5. 完整跑完后回到终端 grep 验证（命令见 dogfood-followup-handoff Step 3）
6. 把 11 项检查清单的实测填进 dogfood-followup-handoff

**重点关注 Issue #3** —— 同事 PR Gap #1 的"落实"诉求是否真做到了。看 lead 在第一次呈现指标时**有没有再写"7.99% 偏低/远低于典型值"或编品系**。

#### 2. 邀请同事 review spec

把 spec 链接发给同事 Ruoheng QU（PR #7 作者），重点请他 review：
- 第 5 章 EthoInsight 输出宪法 10 条草案
- 第 6 章 catalog modifier 方向是否合他"指标外分析"的真实预期
- 第 9 章 7 条验收标准是否完整

#### 3. Spec 定稿（draft → final）

基于上面两条的反馈，调整 spec 后改 status 为 `final`，更新 §11 修订记录。

### 🟡 中优先级 — 定稿后做

#### 4. 写新 plan（基于定稿 spec）

用 `superpowers:writing-plans` skill 严格生成。预期 plan 结构：

- 阶段 1：双层 handoff 协议（核心，~1-2 周）
  - task tool 签名加 `handoff_suffix` 必填
  - 4 个 subagent prompt 改造（L1/L2 双写、catalog 字段投影）
  - lead prompt 改造（必须显式命名 suffix、inline L1 给下游）
- 阶段 2：run-scoped 路径绝缘（~1 周）
  - paths.py 加 runs/<run_id>/ 子目录
  - SandboxMiddleware 注入 RUN_DIR
  - HandoffIsolationProvider 扩展跨 run 解析
- 阶段 3：output-constitution.md（~3-5 天）
- 阶段 4：端到端验证 + spec final

**写 plan 时严格 review 真实代码**——上次 review 命中 3 个阻塞 bug（symbol 名、函数签名、违反 ai-elements 禁改约束）的教训不能再犯。

#### 5. 派新 agent 执行 plan

写完 plan 后参照本会话**之前已经写过的提示词**（在 spec 落盘前一段对话）发给新会话。

### 🟢 低优先级 — 与本 spec 平行

#### 6. dogfood 修复的几个 commit 何时 push 到 origin

执行 agent 写的 11 个 commit 都在 dev 本地、未 push origin。何时 push 由用户决定（建议端到端测试通过后 push）。

#### 7. Catalog modifier（指标外分析）

Spec 第 6 章列了方向，**不在本 plan 范围**——等 handoff 双层协议落地稳定后再开独立 plan。

#### 8. 前端品牌视觉统一（同事 PR Gap #2-3）

Logo / 配色 / 字体按 Noldus 新视觉方案统一——独立工作，与本 spec 无关。

---

## 建议接手路径

### 第一步：和用户对齐流程

新会话开始时**首先**要做的事——和用户确认：

```
"上次会话给你写了 handoff 双层协议 spec（draft）。下一步应该是：
1. 你做端到端测试验证 dogfood 修复（dogfood-followup-handoff 里有 11 项清单）
2. 看 spec 是否需要调整
3. 同事 review spec
4. spec 定稿
5. 我用 writing-plans skill 写 plan
6. 派新 agent 执行 plan

我们从哪一步开始？"
```

**不要**直接跳到写 plan 或派 agent——必须先验证 dogfood 修复 + spec 假设是否成立。

### 第二步：阅读关键文档（按优先级）

```bash
# 1. 本交接文档（你正在做）
cat docs/handoffs/2026-05/2026-05-14-spec-handoff-protocol-handoff.md

# 2. spec 草案（核心交付）
cat docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md

# 3. 执行 agent 留下的验证交接
cat docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md

# 4. 已 merged 的 dogfood plan（已执行完）
cat docs/superpowers/plans/2026-05-13-dogfood-fix-iteration-plan.md

# 5. 同事 PR 反馈
cat "docs/review-packages/2026-05-13-还差哪些with细颗粒度.md"
cat "docs/review-packages/2026-05-13-常见行为学指标外分析要求.md"
```

### 第三步：检查 git 状态

```bash
# 看 dev 是否领先 origin（应该领先 11 个 commit + 本会话新增 spec untracked）
git status
git log --oneline origin/dev..dev

# 确认 spec 是否在工作树里 untracked（应在 docs/superpowers/specs/）
git status docs/superpowers/specs/
```

### 第四步：阅读前置 spec（如有必要）

```bash
# 5-11 spec：files-are-facts 哲学（本 spec 直接前身之一）
cat docs/superpowers/specs/2026-05-11-subagent-file-is-facts-design.md

# 5-13 spec：catalog 架构（本 spec 直接前身之二）
cat docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md
```

### 第五步（如果用户决定做端到端测试）：协助验证

`dogfood-followup-handoff` 的 Step 3 有完整命令清单：
- grep gateway.log 看 reload 次数
- grep langgraph.log 看 thinking 400 错误
- grep `SubagentExecutor initialized` 看 report-writer 是否被派
- 看 thread workspace 的 metric_plan.json 看输出路径是物理还是虚拟

---

## 风险与注意事项

### 🚨 千万不要做的事

1. **不要把 spec 直接交给 agent 执行**——它不是 plan、不含可执行步骤
2. **不要在端到端测试之前写新 plan**——spec 假设未经过实证，可能要调整
3. **不要 commit spec 文件**（用户未明确授权 commit）——它当前是 untracked 状态
4. **不要 push dev 到 origin**（11 个 dogfood 修复 commit 是别人写的、用户也未授权 push）
5. **不要重新讨论已锁定的设计决策**——下面这些用户已经拍板：
   - L1 是所有 agent 默认契约（不只是 lead 用）
   - handoff_suffix 由 lead 显式命名
   - L1 不做 JSON Schema 强校验（先约定）
   - 双层 handoff（L1 message history + L2 文件）+ run-scoped 路径绝缘 + Output Constitution + modifier 留方向不实施
6. **不要改 `ai-elements/` 目录**——Frontend CLAUDE.md 明确禁止（registry-generated）
7. **不要碰这 3 个无关工作树文件**（与本会话无关）：
   - `docs/specs/llm-finetuning-strategy.md`
   - `docs/plans/2026-05-13-base-model-decision-memo.md`
   - `packages/agent/frontend/src/app/page.tsx`

### ⚠️ 容易混淆的点

#### 1. Run / Runtime / Thread 的精确含义

很多人（包括上一会话最初的 Claude）会混用这三个词。**严格按这套定义**：

- **thread** = 对话窗口
- **runtime** = agent 进程实例（gateway 重启 = 新 runtime）
- **run** = 用户发一条消息触发的一次 LangGraph 推理流程（thread 5046a6e6 有 5 个 run）

#### 2. Spec vs Plan 的差别

- **Spec** = 设计文档（"应该长这样"）—— 本会话产物
- **Plan** = 实施清单（"Task 1: 改 file:line → ... Task 2: ..."）—— 下一会话产物

不能跳过 plan 直接让 agent 干 spec。

#### 3. 当前会话**没动代码**

本会话**只**：
- 写了 1 份 spec 文档（`2026-05-14-handoff-protocol-and-runtime-isolation-spec.md`）
- 没写代码、没 commit、没 push、没动 task list

执行 agent 在**另一会话**写了 11 个 commit——那些是 dogfood-fix-plan 的产物，不是本会话产物。

#### 4. 用户的"两个 agent"心智模型

用户心里有：
- **当前 agent**（你/Claude）—— 讨论架构、写 spec、写 plan
- **执行 agent**（已结束）—— 跑 dogfood-fix-plan，11 个 commit 是它写的
- **未来 agent**（待派遣）—— 跑你即将写的新 plan

不要把这三个混在一起。

### 已被推翻的判断 / 不要回头做的事

- ❌ 不要再讨论"是否做 run 级隔离"—— 已拍板做
- ❌ 不要再讨论"L1 走 message history 还是 inline"—— 已拍板 message history 通道（task return value → ToolMessage）
- ❌ 不要再讨论"L2 文件命名"—— 已拍板 `handoff_<type>__<suffix>.json` + lead 显式命名 suffix
- ❌ 不要再讨论"是否做 JSON Schema 强校验"—— 已拍板先不做
- ❌ 不要再纠结 "drift 风险"—— 已用 AgentCore 经验 + L1 < 5 KB 上限 + ArchivingSummarizationMiddleware 兜底论证完毕
- ❌ 不要再讨论"是否在本 spec 实施 modifier"—— 已拍板留方向、不实施

### 已发现的认知陷阱

#### 陷阱 1：上一次写 plan 时没核对真实代码踩了 3 个 bug

具体是：`LEAD_AGENT_SYSTEM_PROMPT_TEMPLATE` 错（真名 `SYSTEM_PROMPT_TEMPLATE`）、`resolve()` 测试缺 `columns` 必填参数、Task 6 改 ai-elements/ 违反 CLAUDE.md。下次写 plan 时必须 **每个 import/函数签名/文件路径都 Read 确认**。

#### 陷阱 2：上一次推荐方案 A "走 message history" 时没考虑跨 run 累积

被用户问"那 message history 累积会不会漂"才意识到漏算。最终用 L1 < 5 KB 硬上限 + 未来 subagent 数量有限 + ArchivingSummarizationMiddleware 三层保险解决。这条经验体现在 spec 第 2.3 章"为什么这套设计抗漂移"——下次必须保留这个论证。

#### 陷阱 3：用户对"deerflow 缺陷 vs 我们设计缺陷"非常敏感

绝缘问题用户明确认了"是我们设计缺陷，没完全理解 deerflow 框架"——而**不是** deerflow 的缺陷。下次讨论时**不要再说**"deerflow 没考虑"，要说"我们在 deerflow 之上加层"。

---

## 下一位 Agent 的第一步建议

```bash
# 1. 读本交接 + 看 git 状态
cat docs/handoffs/2026-05/2026-05-14-spec-handoff-protocol-handoff.md
git status
git log --oneline origin/dev..dev   # 应看到 11 个执行 agent 的 commit

# 2. 读核心 spec
cat docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md

# 3. 读执行 agent 的验证交接
cat docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md

# 4. 和用户对齐下一步（关键）
```

完成 1-3 步后，向用户提议第 4 步：

> "我已掌握上次会话状态：
> 1. handoff 双层协议 spec 已写完（draft，860 行，未 commit）
> 2. dogfood-fix-plan 已被另一 agent 跑完 11 个 commit（未 push）
> 3. dogfood-followup-handoff 显示 11 项检查中 2 项自动验证通过、9 项待人工 dogfood
>
> 建议下一步：你做一次完整人工 dogfood 验证 9 项检查 → 我们看 spec 是否调整 → 同事 review → spec 定稿 → 写 plan → 派新 agent 执行 plan。
>
> 从哪一步开始？"

**不要**未经用户确认就直接写 plan 或派 agent。
