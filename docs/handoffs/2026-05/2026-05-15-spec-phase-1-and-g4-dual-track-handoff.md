# 2026-05-15 EthoInsight 双轨执行交接（G4 方案 C + Spec 阶段 1 并行进行中）

> **状态**：本会话职责完结——三份 plan 已写好，FIRST-TOKEN 回退已落地，另外两份 plan（G4 方案 C + Spec 阶段 1）**正在被另外两个 agent 在其它会话执行**，本会话等待它们回来 review。
> **交接目的**：让新会话 agent 快速接手"两个并行执行 agent 回来后的 review + 后续规划"工作。

---

## 当前任务目标

监督 + review 两个并行执行 agent 跑完后的产出，并视情况推进 EthoInsight dogfood-fix 闭环到 spec §9 完整验收。具体三件事：

1. **等 G4 方案 C agent 回报**（执行 plan `docs/superpowers/plans/2026-05-15-frontend-stage-broadcast-plan.md`）—— 预期产出 4-6 个 commit、纯前端改动
2. **等 Spec 阶段 1 agent 回报**（执行 plan `docs/superpowers/plans/2026-05-15-spec-phase-1-dual-layer-handoff-plan.md`）—— 预期产出 8-9 个 commit、跨 ethoinsight + agent backend
3. 两个都回来后给用户做 review 总结、决定下一步（spec 阶段 2 run-scoped 路径绝缘 / 同事 review / 收尾）

---

## 当前进展

### ✅ 已完成（本会话 + 之前两个执行 agent 的产出，已全部 push origin/dev）

**dev 当前 HEAD: `fca62e33` (revert: 回退 FIRST-TOKEN emoji 规则)** —— origin 与 dev 完全同步、0 ahead

本轮迭代闭环（从最早到最新，按 push origin 时间）：
- 11 个 dogfood-fix-plan commit（`555db882..f9f636d4`，Issue #1-10 + thinking 中文）
- 4 个 LeadAgentExecutionBoundaryProvider commit（`00792c3b..c8bf630b`，阶段 1.5 早期落地）
- 3 个端到端 dogfood 测试 + G5 根因诊断 commit（`3dcba32a / dd523503` 等）
- 5 个 G5 catalog 路径回归修复 commit（`b41d9c81..5b1d29c2`）
- 2 个 G1+G4 修复 commit（`007fb390 / 854adea0` —— 输出宪法落地 + FIRST-TOKEN 规则尝试）
- 1 个 FIRST-TOKEN 回退 commit（`fca62e33` —— **本会话最后一个**）

**Batch A/B 8 项检查**：7 ✅ + 1 ⚠️（G4 阶段播报；正由 G4 方案 C agent 走前端机制层修复中）

### 🟡 进行中（其他会话的两个执行 agent，本会话不可见进度）

| 任务 | 执行 plan | 预期产出 | 触动文件范围 |
|---|---|---|---|
| **G4 方案 C 前端自动播报** | `docs/superpowers/plans/2026-05-15-frontend-stage-broadcast-plan.md` | 4-6 个 commit | 仅 `packages/agent/frontend/`（i18n + stage-broadcast.ts + SubtaskCard + ToolCall） |
| **Spec 阶段 1 双层 handoff** | `docs/superpowers/plans/2026-05-15-spec-phase-1-dual-layer-handoff-plan.md` | 8-9 个 commit | `packages/agent/backend/` + `packages/ethoinsight/`（task_tool + 4 subagent + lead prompt + catalog projection/summarize CLI） |

**两个 plan 改动的文件完全不重叠**——并行不冲突。

**已观察到的旁证**（说明它们真在跑）：
- 工作树有 `packages/agent/scripts/serve.sh` modified（gateway 启动超时 30s → 90s）—— **未 commit**，疑似 G4 方案 C agent 跑 dogfood QA 时遇到 gateway 启动慢、自己改了脚本兜底。这是合理改动。

---

## 关键上下文

### 项目状态

- 项目根：`/home/wangqiuyang/noldus-insight/`
- 分支：`dev`，与 `origin/dev` 完全同步（HEAD `fca62e33`）
- 项目愿景：EthoInsight 行为学 AI 分析助手，9 月 v0.1 硬指标
- 详细架构：`CLAUDE.md`（项目根）

### 关键文件路径速查

| 用途 | 路径 |
|---|---|
| **三份 plan**（本会话产出） | `docs/superpowers/plans/2026-05-15-*.md`（revert-first-token / frontend-stage-broadcast / spec-phase-1-dual-layer-handoff） |
| **G5 修复 plan**（已执行完毕） | `docs/superpowers/plans/2026-05-14-g5-catalog-virtual-path-fix-plan.md` |
| **LeadExecBoundary plan**（已执行完毕） | `docs/superpowers/plans/2026-05-14-lead-execution-boundary-guardrail-plan.md` |
| **dogfood-fix 老 plan**（已执行完毕） | `docs/superpowers/plans/2026-05-13-dogfood-fix-iteration-plan.md` |
| **主 spec draft v2** | `docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md`（~1100 行，含 §2 L1/L2 schema、§5.5 GuardrailProvider 设计、§10.3 长期 backlog 论证） |
| **spec 修订草稿留档** | `docs/superpowers/specs/2026-05-14-handoff-protocol-spec-revision-draft.md` |
| **G5 根因诊断**（已 commit） | `docs/problems/2026-05-14-G5-catalog-virtual-path-regression.md` |
| **G1+G4 修复 handoff** | `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md`（Batch A/B 状态表 + dogfood 复测记录） |
| **E2E 测试 checklist 模板** | `docs/handoffs/2026-05/2026-05-14-e2e-test-checklist.md` |
| **dogfood 思路根源** | `docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md`（thread b0d3a611 卡死分析） |
| **G1 修复产物（共享宪法）** | `packages/agent/skills/custom/ethoinsight/references/output-constitution.md` |
| 上一会话交接 | `docs/handoffs/2026-05/2026-05-14-spec-handoff-protocol-handoff.md` |

### 关键 commit 速查

| Commit | 内容 |
|---|---|
| `fca62e33` | revert FIRST-TOKEN emoji 规则（dev HEAD）|
| `854adea0` | dogfood G1+G4 修复交接 doc |
| `007fb390` | G1（输出宪法）+ G4（FIRST-TOKEN 尝试，**已回退**）|
| `5b1d29c2..b41d9c81` | G5 修复 5 个 commit |
| `c8bf630b..00792c3b` | LeadAgentExecutionBoundaryProvider 4 个 commit |
| `f9f636d4..555db882` | dogfood-fix-plan 11 个 commit |

### 工作树未 commit 的文件（这些不是任何执行 agent 的预期产物，需 review）

```
 M docs/specs/llm-finetuning-strategy.md            ← spec-handoff 交接文档明令"不要碰"
 M packages/agent/frontend/src/app/page.tsx          ← 同上"不要碰"
 M packages/agent/scripts/serve.sh                   ← gateway timeout 30→90，疑似 G4 agent 跑 QA 时改的
 M packages/agent/skills/public/bootstrap/SKILL.md   ← 不知来源、不在 plan 范围

?? 多份未 commit 的 plan/doc（本会话产物）          ← 用户决定何时 commit
?? docs/superpowers/specs/2026-05-14-teacher-model-selection-design.md  ← 不知来源、与本工作流无关
```

### 关键 4 条用户约束（不能违反）

1. **不要 push 任何东西**——除非用户明确说"push"。当前 dev 与 origin 同步是干净状态
2. **不要 commit 未明确授权的文件**——尤其上面 4 个 M 文件，spec-handoff 交接文档已 flag 它们"不要碰"
3. **不要修代码 / 改 prompt / 调 spec**——本会话职责是 **review + 写方案**，不直接动手
4. **不要打断正在跑的两个 agent**——它们在另外两个会话独立运行，本会话无可见性、不要假设它们到哪一步

---

## 关键发现

### A. 双轨并行设计成立的关键理由

- G4 方案 C 只动 `frontend/`、Spec 阶段 1 只动 `backend/` + `ethoinsight/`——文件零重叠
- 两者无 API 边界交互——前端的 tool_call 渲染层不关心后端 handoff schema 怎么改
- 两个 agent 都不会修同一个测试文件（前端无测试框架；后端 spec 阶段 1 改 task_tool 相关测试）

### B. Spec 阶段 1 已知风险点（plan 文档 §"实施完成后"段已 flag）

如果 Spec 阶段 1 agent 在 Task 4 报"catalog schema 缺 `display_name_zh / unit_zh / one_liner / direction_for_anxiety` 字段"——**必须停下来让用户拍板**：
- 选项 A：扩 schema + 4 个 paradigm YAML 加这些字段（真闭环、工作量翻倍）
- 选项 B：projection 函数对缺字段返回 None 不抛错（先 ship、不彻底）

不要让 agent 自己选——这是 single source of truth 哲学的核心决策。

Task 6（lead prompt 改造 + 修旧测试）是最复杂步骤，可能旧测试级联失败，要监督。

### C. G4 方案 C 已知约束

- frontend "No test framework is configured"（frontend CLAUDE.md 第 12 行）——plan 已明确不写 unit test，只 manual QA
- Task 6 需要起 backend 服务做端到端验证——可能要用 90s gateway timeout（已观察到 serve.sh 被改）

### D. G4 方案 C 改动 serve.sh 的 gateway timeout——需要 review

工作树里 `packages/agent/scripts/serve.sh` 有未 commit 改动（30s → 90s）。原因可能是：
- G4 方案 C agent 跑 Task 6 manual QA 时遇到 gateway 启动慢、自己改了
- **这个改动是合理的**——之前会话也观察到 gateway 偶尔启动需要 > 30s

如果 G4 方案 C agent 回来时 commit 列表里有这一行，没问题；如果没 commit，要主动问用户/agent 是不是漏 commit。

### E. spec §9 验收 7 条中 §9.2（跨 run handoff 引用）依赖阶段 2，本阶段 1 plan **不验收**这条

监督完成后做 review checklist 时记得：spec 阶段 1 dogfood 复测 7 条里只能跑 6 条。

---

## 未完成事项

### 🔴 阻塞 — 等两个 agent 回来

**第 1 优先级**：等 G4 方案 C agent + Spec 阶段 1 agent 都回报。两个都回来 OR 任一回来超时（>1 天没动静），都要触发后续动作。

可以先做的不阻塞工作：
- 准备 review checklist（见"建议接手路径"）
- 评估是否要把本会话产出的 4 份 untracked plan/doc 一次性 commit 入库

### 🟡 中优先级 — 等阶段都跑完后

**第 2 优先级**：两个 agent 都回报后做 review。具体动作见"建议接手路径"段。

**第 3 优先级**：根据 review 结果决定下一步。两种主要候选：
- A. **spec 阶段 2**（run-scoped 路径绝缘）——主线持续推进
- B. **同事 review**——把 spec draft v2 + 阶段 1 落地结果发给 Ruoheng QU（PR #7 作者）

### 🟢 低优先级

- 评估 G4 阶段 1.5 的 `HandoffPendingActionsProvider` 是否值得做（spec §5.5.2，需阶段 1 落地后）
- 评估 `_ABSOLUTE_PATH_PATTERN` 误报修复（spec §10.3.2 长期 backlog）是否值得 promote 上线

---

## 建议接手路径

### 第一步：检查两个 agent 是否已回报（30 秒）

```bash
# 1. 看 dev 上 G4 / 阶段 1 的 commit 是否落地
cd /home/wangqiuyang/noldus-insight
git log --oneline origin/dev..dev 2>&1 | head -20  # 本地 ahead 的 commit
git log --oneline fca62e33..origin/dev 2>&1 | head -20  # origin 比本会话基线多的 commit

# 2. 看工作树新增了什么
git status --short

# 3. 看是否有新文档（dogfood 验证报告）
ls -t docs/handoffs/2026-05/ | head -5
ls -t docs/superpowers/specs/ | head -5
```

**判断**：
- 如果**两个 agent 都已 push** → origin 应该多了 ~12-15 个 commit（G4 的 4-6 个 + 阶段 1 的 8-9 个），跳到"第三步：review"
- 如果**只回来一个** → review 已回来的、等另一个
- 如果**都没回来** → 跳到"第二步：等待期可做的事"

### 第二步：等待期可做的事（不阻塞两个 agent）

1. **commit 本会话产出的 plan/doc 入库**（如果用户同意）—— 4 份 untracked：
   - `docs/superpowers/plans/2026-05-15-revert-first-token-rule-plan.md`（已执行完）
   - `docs/superpowers/plans/2026-05-15-frontend-stage-broadcast-plan.md`（执行中）
   - `docs/superpowers/plans/2026-05-15-spec-phase-1-dual-layer-handoff-plan.md`（执行中）
   - 加之前未 commit 的：
     - `docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md`
     - `docs/superpowers/plans/2026-05-14-{g5-catalog-virtual-path-fix,lead-execution-boundary-guardrail,2026-05-13-dogfood-fix-iteration}-plan.md`
     - `docs/superpowers/specs/2026-05-14-handoff-protocol-spec-revision-draft.md`

   **不要 add** 这 4 个 M / untracked 文件（spec-handoff 交接文档明令"不要碰"）：
   - `docs/specs/llm-finetuning-strategy.md`（M）
   - `docs/plans/2026-05-13-base-model-decision-memo.md`（??）
   - `docs/plans/2026-05-14-fireworks-meeting-questions.md`（??）
   - `packages/agent/frontend/src/app/page.tsx`（M）
   - `packages/agent/skills/public/bootstrap/SKILL.md`（M）
   - `docs/superpowers/specs/2026-05-14-teacher-model-selection-design.md`（??）

   建议 commit message 用 "docs: 入库本轮迭代的 plan / 诊断材料留档" 之类，不 push。

2. **写 review checklist**（针对两个 agent 回报后要看的点，提前列清单）

   - **G4 方案 C 验收**：
     - [ ] commit 数 4-6 个，文件改动范围限 `packages/agent/frontend/`
     - [ ] `pnpm check` 全绿
     - [ ] dogfood 截图能看到 5 类 tool_call 触发时显示业务播报（code-executor / data-analyst / report-writer / parse / catalog）
     - [ ] `dogfood-followup-handoff.md` 已追加 G4 方案 C 复测段、Batch A/B 表格 G4 行从 ⚠️ 改为 ✅
     - [ ] 没碰 `ui/` `ai-elements/` 目录（frontend CLAUDE.md 禁改）
     - [ ] 没碰 backend prompt 文件

   - **Spec 阶段 1 验收**：
     - [ ] commit 数 ~8-9 个
     - [ ] `make test` 全绿、`make lint` 无错
     - [ ] `pytest packages/ethoinsight/tests/` 全绿
     - [ ] dogfood 复测：spec §9.1/§9.3/§9.4/§9.5/§9.6/§9.7 通过（§9.2 跨 run 引用本阶段不验）
     - [ ] **关键看点**：task_tool 签名是否真的加了 `handoff_suffix` 必填、4 个 subagent 是否真的写 L1+L2、HandoffIsolationProvider 是否真的识别 suffix 路径
     - [ ] catalog schema 字段是否扩了（如果 plan Task 4 触发了 schema 缺失问题）—— 如扩了，让用户判断是否同意"single source of truth 真闭环"路径

### 第三步：两个都回来后的 review 流程

1. 跑：`git log --oneline fca62e33..HEAD` 看完整 commit 列表
2. 按上面 checklist 逐项验
3. 跑一次 dogfood 验证：
   ```bash
   cd /home/wangqiuyang/noldus-insight/packages/agent
   make stop && make dev
   # 等 ready 后用 Playwright 跑完整 EPM 单只分析
   # 同时验：G4 前端业务播报 + spec 阶段 1 双层 handoff
   ```
4. 给用户回报。回报模板：

```
# 双轨执行结果 review

## G4 方案 C 验收
- commits: <N> 个
- dogfood: <截图描述> ✅ / ❌
- 状态: <成功/部分/失败>

## Spec 阶段 1 验收
- commits: <N> 个
- spec §9 验收: <6 条结果，跳 §9.2>
- 关键观察: <如 catalog schema 是否扩了>
- 状态: <成功/部分/失败>

## 整体 Batch A/B 状态
- 8/8 ✅ / N/8 ✅
- 是否可 push: 是 / 否（已在执行 agent 完成时 push）

## 建议下一步
- 阶段 2 run-scoped 路径绝缘 / 同事 review / 其他
```

### 第四步：用户拍板下一步

不要自己决定方向。把 review 结果交给用户。

---

## 风险与注意事项

### 🚨 千万不要做的事

1. **不要 push** —— 当前 dev 与 origin 同步（HEAD `fca62e33`）；两个执行 agent 会自己 push 它们的 commit
2. **不要 add 那 4-6 个"不要碰"文件**（`llm-finetuning-strategy.md` / `page.tsx` / `base-model-decision-memo.md` / `fireworks-meeting-questions.md` / `bootstrap/SKILL.md` / `teacher-model-selection-design.md`）
3. **不要修任何代码**（包括 prompt / spec）——本会话职责仅 review，新方向先写方案让用户拍板
4. **不要 force push / 不要 git reset --hard origin/dev**（会丢两个 agent 的 commit）
5. **不要假设两个 agent 已到哪一步**——本会话无可见性，必须用 `git log` 或新文档作证据
6. **不要 commit serve.sh 的改动**直到搞清楚是哪个 agent 改的、为什么没自己 commit
7. **不要重新写已经写过的 plan**——三份 plan 已落档 `docs/superpowers/plans/2026-05-15-*.md`

### ⚠️ 容易混淆的点

#### 1. 三份 plan 的执行状态

- `2026-05-15-revert-first-token-rule-plan.md` —— **已执行完毕**（commit `fca62e33`）
- `2026-05-15-frontend-stage-broadcast-plan.md` —— **正在被另一 agent 执行**
- `2026-05-15-spec-phase-1-dual-layer-handoff-plan.md` —— **正在被另一 agent 执行**

不要重新派 agent 跑前一个、不要假设后两个还没开始。

#### 2. "G4 方案 C" vs "G4 修复尝试"

- 旧 G4 修复（commit `007fb390` 的 FIRST-TOKEN emoji 规则）——**已回退**（commit `fca62e33`）
- 新 G4 修复（方案 C 前端自动播报）——**正在执行中**

review 时如果 agent 报"G4 修复完成"，要分清是哪种方案。

#### 3. spec 修订草稿 vs 主 spec

- 主 spec `2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` —— **draft v2，已落档**，含 §2 L1/L2、§5.5 GuardrailProvider、§10.3 backlog
- spec 修订草稿 `2026-05-14-handoff-protocol-spec-revision-draft.md` —— **历史 audit trail**，draft v1 → v2 的过渡产物

引用 spec 时用主 spec，不用草稿。

#### 4. dogfood-followup-handoff.md 的多个版本

这一份 handoff 文档已经被改过 3-4 次（dogfood Task 9 写入 → G5 修复追加 → G1+G4 修复追加 → 即将被 G4 方案 C agent + Spec 阶段 1 agent 追加）。review 时看末尾追加的"YYYY-MM-DD G4 方案 C 复测"段或"Spec 阶段 1 验证"段。

### 已被推翻的判断 / 不要回头做的事

- ❌ **不要再尝试用 prompt 强制 first-token emoji**——已验证 deepseek 多步推理不稳定，方案 C（前端 UI）才是正确路径
- ❌ **不要修 sandbox `replace_virtual_paths_in_command`**（spec §10.3.1 论证不动 sandbox，G5 用 env-var fallback 修了）
- ❌ **不要给 G5 加 sandbox 路径不对称的根本性修复**——backlog 项，等 ad-hoc subagent 需求出现再做
- ❌ **不要尝试解决 `//` 误报**（spec §10.3.2 backlog，lead 路径已堵）

### 已发现的认知陷阱

#### 陷阱 1：单测和生产路径脱节（G5 教训）

commit `2eb1532a` 单测过但生产挂——单测直接调 Python 函数，没经过 sandbox bash 替换层。**修同类 bug 时，必须加一条经过完整生产路径（lead → bash → sandbox → CLI）的单测**。

#### 陷阱 2：prompt 自觉 vs 机制层（G1 / G4 / G5 共同教训）

三个修复都验证了：prompt 自觉约束 → 不够 → 退到机制层（共享文档 + acknowledge / env-var key 反推 / UI 自动渲染）才稳定。**未来修复时优先评估能否走机制层**，prompt 自觉作为兜底而非主防线。

#### 陷阱 3：commit message 应含来源（用户偏好）

本轮所有 commit 都用中文 + 写明 issue / thread 出处。审 review 时如果 commit message 是英文或不含来源，要给用户 flag。

---

## 下一位 Agent 的第一步建议

```bash
# 1. 读本交接 + 看 git 状态
cat docs/handoffs/2026-05/2026-05-15-spec-phase-1-and-g4-dual-track-handoff.md
git status
git log --oneline origin/dev..dev   # 看本地 ahead 的 commit（应为 0 或 G4 / 阶段 1 已 push 的）
git log --oneline fca62e33..origin/dev  # 看 origin 比本会话基线多的 commit

# 2. 看两个 agent 是否回报
# 看 docs/handoffs/2026-05/ 是否多了"G4 方案 C 复测"或"spec 阶段 1 验证"段
ls -t docs/handoffs/2026-05/
git log --oneline | grep -E "G4 方案 C|双层 handoff|stage broadcast|handoff_suffix" | head -10

# 3. 如果两个都没回来，先做"第二步等待期可做的事"
#    如果至少一个回来，跑 review checklist
```

**问用户**：

> "本会话三份 plan 已写完：FIRST-TOKEN 回退已执行（commit fca62e33 入 origin），G4 方案 C + Spec 阶段 1 正由另外两个 agent 在其他会话执行。
> 我现在能做：
> 1. 等两个 agent 回来后做 review
> 2. 等待期间把本会话留下的 untracked plan/doc 入库（4 份 plan + 1 份诊断 + 1 份 spec 草稿留档）
> 3. 等待期间写"两个 agent 回报后的 review checklist"（已包含在交接文档里，可拆出来作为独立文档）
> 你倾向哪条？"

**不要**未经用户确认就开 commit / 派新 agent / 改代码。
