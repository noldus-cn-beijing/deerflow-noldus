# 2026-05-18 Lead Handoff 消费 + 角色拆分 设计讨论交接

> **状态**：未实施。仅诊断 + 设计方案讨论完毕，已通过 grill-me 走完 7 个设计分支。等待下一会话写 spec + 实施 plan。
>
> **优先级**：🔴 P0 — 这是 2026-05-14 同类故障 4 天后**再次复现**的根本性架构问题，证明 prompt-only fix 不够。
>
> **预读**：本文件 + [memory: 2026-05-18-lead-not-reading-handoff](../../.claude/projects/-home-wangqiuyang/memory/project_2026-05-18_lead_not_reading_handoff.md) + [memory: 2026-05-14-e2e-trajectory-failure](../../.claude/projects/-home-wangqiuyang/memory/project_2026-05-14_e2e_trajectory_failure.md)

---

## 1. 这次 dogfood 暴露了什么

### 1.1 事故现场

- **Thread**: `141b4b98-e574-4a41-88f2-e70f1405d2d7`
- **前端报告**: `docs/e2e/Analyzing Mouse Elevated Plus Maze Data.md`（408 行）
- **后端日志**: `packages/agent/logs/langgraph.log`（676 行，含 39 次 SandboxAudit）
- **后端 thread workspace**: `packages/agent/backend/.deer-flow/users/cd95effa-.../threads/141b4b98-.../user-data/workspace/`

### 1.2 用户操作 + agent 行为

**第一阶段：分析数据** ✅ 正常
1. 用户："我刚做完小鼠高架十字迷宫实验，你可以帮我分析一下实验数据吗"
2. 用户上传 1 个 EPM 数据文件（Drug 组 Subject 1）
3. Lead 用 `set_experiment_paradigm` + `prep_metric_plan`（P0 fix 生效，不再 bash）
4. Lead 派 `task(code-executor)`，subagent 算出 5 个 EPM 指标
5. Lead 给用户漂亮的 markdown 报告，明确标注"仅 1 个被试，不能推断 Drug 效应"
6. **关键**：code-executor 写了 `handoff_code_executor.json`（含完整指标 + 中间数据列名 + 数据质量元信息）

**第二阶段：画图表** ❌ 严重失败
1. 用户："帮我画一些分析图表吧"
2. Lead **没读** `handoff_code_executor.json`（这是 root cause）
3. Lead 6 次 `read_file` 找 skill 文档 → 触发 LOOP DETECTED（第一次）
4. Lead 派 `task(code-executor)` 让 subagent 画图，prompt 大概率写"生成 EPM 图表"
5. Subagent code-executor：
   - 看 `metric_plan.json` 的 `charts` 字段 = `[]`（**单被试，box_open_arm 的 when: n_per_group >= 3 不满足**）
   - 不知道还有 `_common/plot_trajectory.py`（catalog 没注册）
   - 不知道哪些指标已经算过（没读上一次的 handoff）
   - **开始疯狂瞎猜脚本路径**：尝试 20+ 种命名变体 (`epm.trajectory_plot`、`epm.plot_trajectory`、`charts.trajectory_plot`、`epm.chart_trajectory`、`epm.visualize_trajectory`、`epm.draw_trajectory`、`scripts.visualize.trajectory_plot`、`scripts.plot.trajectory` 等)
   - **39 次 SandboxAudit**，全部 ModuleNotFoundError
6. Lead 最终违反边界，自己 `write_file` 写一次性 Python 脚本（截断在报告末尾，第 408 行）

### 1.3 五重叠加根因

| # | 根因 | 层 | 5-14 是否已诊断 |
|---|---|---|---|
| 1 | Lead prompt 无 "必读 handoff JSON" 铁律 | prompt | ✅ 5-14 已诊断 + deepseek 强调，**未落地** |
| 2 | code-executor 角色过载（既算指标又画图） | 架构 | ❌ 新发现 |
| 3 | `catalog/_common.yaml` 不存在，`_common/plot_trajectory.py` 是孤儿 | catalog | ❌ 新发现 |
| 4 | `resolve.py` plan.charts=[] 时不告知"为什么空"、不给 fallback 候选 | catalog 编排 | ❌ 新发现 |
| 5 | `ethoinsight-code` skill 禁 subagent 写代码但没列可用脚本清单 → 只能瞎猜 | skill 文档 | ❌ 新发现 |

**关键洞察**：根因 1 在 4 天前已经诊断过，但只做了"prompt 加禁令"（commit `24715250` 的"禁判读/品系/常模/金标准"三条禁令），**没真正强制 lead 读 handoff**。所以 5-18 又复现。

**结论**：prompt-only fix 在这类问题上不足以约束 LLM 行为，需要 deerflow harness 级**硬约束**（`HandoffEnforcementGuardrailProvider` 拦截）。

---

## 2. 用户提出的元洞察（关键引用）

> **洞察 A**: "lead agent 大包大揽了太多，它最应该的就是一个 planner，知道根据用户的需求做什么。lead agent 本身当然也能完成一些用户的任务，但是在这种画图的场景，或者出数据的场景，需要 hard fact 的地方，它找不到数据啊！！它不理解 code executor 之前跑完的数据的 Handoff！！"

> **洞察 B**: "我们的 agent 现在就像是一个精密的 demo 仪器，一点沙子和不理想的流程都能吹倒"

> **洞察 C**: "我们应该最大化的利用 deerflow 的框架解决问题的同时，在这个 deerflow 的基础上，参考其他好的 agent（比如 claude code）好的地方，然后加以利用来构建"

**对应到 Claude Code 哲学的三条对照**：

| EthoInsight 问题 | Claude Code 等价机制 |
|---|---|
| Lead 不读 handoff，凭印象决策 | Claude Code system prompt "trust but verify" + "Before recommending from memory: ... verify first" |
| code-executor 既跑指标又画图 | Claude Code: subagent 不承担决策权，只承担 information gathering 或 mechanical execution |
| 单被试边界情况 catalog 没规划 | Claude Code: fallback 写在最显眼位置（"Executing actions with care" 整段） |

---

## 3. 上次会话走完的 7 个设计分支决策（grill-me 7 轮）

| # | 决策点 | 选项 | 用户选择 | 理由要点 |
|---|---|---|---|---|
| 1 | charts=[] 时 lead 做什么 | A 直接降级 / **B 反问用户** / C 中间态 | **B** | 单被试场景多 1 次确认成本极低；信息不透明 = bad UX；现有 ask_clarification 机制天然契合 |
| 2 | 反问话术由谁产生 | A resolve 硬编码 / B lead 拼装 / C skill 文档 / **D facts+policy 分层** | **D** | resolve.py 出 facts、skill 出 policy；分层符合 single source of truth |
| 3 | `charts_fallback_available` 由谁按什么规则生成 | **A catalog/_common.yaml** / B 各范式自声明 / C chart 反向链接 / D resolve 硬编码 | **A** | `_common/` 脚本本就范式无关；DRY；与 CLAUDE.md 第 10 条"通用 + 范式双层"对仗 |
| 4 | `_common.yaml` 第一版放几个图 | A 只 trajectory_plot / **B trajectory + timeseries** / C 全部 8 个 | **B** | 用户"一些图"是复数；timeseries 是单被试高价值图；charts.py 库函数已存在；C 过度工程 |
| 5 | fallback 信息怎么暴露给 agent | **A' 复用现有 skipped 字段 + 新加 charts_fallback_available** / B 统一 charts_decision / C notes / D 单独工具 | **A'** | 现有 plan.skipped 字段已存在；不破坏现 schema；结构化数据 > 自然语言 |
| 6 | plot_trajectory.py 现有签名能否对齐 catalog 契约 | 实地确认契约 | **契约对齐，不改 plot_trajectory.py**；timeseries 用 A1（单 y_col）+ 范式默认值映射（`_DEFAULT_Y_COL_BY_PARADIGM`） | catalog `_chart_to_plan` 给 input/output 两参数；timeseries CLI 包装可选 `--y-col`，未传时按 paradigm 查默认 |
| 7 | fallback trigger 怎么定 | A charts 全空才显示 / B 永远显示 / C n_per_group 硬编码 / **D3 _common.yaml when 表达存在性 + resolve.py 编排层判定 trigger** | **D3** | 不扩 `_evaluate_when`（避免引入 `<`/`or`）；only 加 `total_subjects` 一个变量；触发逻辑住编排层语义对 |

### 3.1 具体 schema 变更（决策落地）

**resolve.py 输出的 plan.json schema 1.0 兼容扩展**：

```json
{
  "schema_version": "1.0",
  "metrics": [...],
  "statistics": {...},
  "charts": [],
  "charts_fallback_available": [
    {
      "id": "trajectory_plot",
      "script": "ethoinsight.scripts._common.plot_trajectory",
      "input": "<raw_files[0]>",
      "output": "<workspace>/plot_trajectory.png",
      "kind": "single_subject_exploration",
      "rationale": "n=1 时不能做组间对比，但可做单体探索"
    },
    {
      "id": "timeseries_plot",
      "script": "ethoinsight.scripts._common.plot_timeseries",
      "input": "<raw_files[0]>",
      "output": "<workspace>/plot_timeseries.png",
      "kind": "single_subject_exploration",
      "rationale": "n=1 时不能做组间对比，但可看时间动态"
    }
  ],
  "skipped": [
    {"id": "box_open_arm", "reason": "when_not_satisfied: n_per_group >= 3 (got 1)"}
  ],
  "notes": [...]
}
```

**新文件 `packages/ethoinsight/ethoinsight/catalog/_common.yaml`**：

```yaml
# 通用 charts — 范式无关，作为单被试/组间不可对比场景的 fallback
common_charts:
  - id: trajectory_plot
    script: ethoinsight.scripts._common.plot_trajectory
    kind: single_subject_exploration
    when: total_subjects >= 1
    rationale: 单被试或组间数据不全时的轨迹可视化

  - id: timeseries_plot
    script: ethoinsight.scripts._common.plot_timeseries
    kind: single_subject_exploration
    when: total_subjects >= 1
    rationale: 单被试或组间数据不全时的时间动态
```

**resolve.py 编排逻辑增量**（伪代码）：

```python
# Step 4.5（新增）: 评估是否注入 fallback
all_main_charts_skipped = len(cat.charts) > 0 and len(plan_charts) == 0
fallback_available: list[PlanChart] = []

if all_main_charts_skipped:
    common_cat = load_common_catalog()  # 读 _common.yaml
    for ch in common_cat.common_charts:
        if _evaluate_when(
            ch.when,
            n_per_group=n_per_group,
            n_groups=n_groups,
            total_subjects=total_subjects,  # 新增变量
        ):
            fallback_available.append(
                _chart_to_plan(ch, raw_files, workspace_dir, ...)
            )
```

**`_evaluate_atomic_when` 扩展**：加一个 `total_subjects` 变量分支（约 3 行代码）。

**新文件 `packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py`**：模式与 `plot_trajectory.py` 完全相同，约 40 行 CLI 包装 + paradigm → default y_col 映射。

---

## 4. 用户最终扩展的更大问题

讨论到第 7 轮后，用户提出比"画图 bug"更深的问题（见 §2 洞察 A/B/C），把视野拉到**整体架构**：

### 4.1 双层修复方案

**Layer 1（地基修复，0.5 天）— 强制 lead 消费 handoff**

- **铁律加入 lead prompt**：
  > 收到任何 subagent task 完成消息后，**必须 `read_file` 该 subagent 产出的 handoff JSON 文件**，然后才能 reasoning / 回答用户 / 派下一个 task。绝不基于 subagent 的自然语言总结做决策。

- **统一 handoff 文件命名约定**：`handoff_{subagent_type}_{turn_index}.json`（现有 `handoff_code_executor.json` 已经接近这个约定）

- **harness 级硬约束**：新建 `HandoffEnforcementGuardrailProvider`（用现成 `GuardrailMiddleware` + `GuardrailProvider` 协议），拦截 lead 派下一个 task 前如果上一个 task 完成了但它没读 handoff → 注入 reminder ToolMessage 强制 lead 先读

**这一层修的是 §1.3 根因 1**，是普适规则不限 EthoInsight，应入 deerflow harness。

**Layer 2（针对画图的精确修复，1 天）— Subagent 角色拆分 + catalog 扩展**

- 之前讨论的 4 个 schema 变更（见 §3.1）
- **新增**：抽出 `chart-executor` subagent 跟 `code-executor` 并列
  - `code-executor` = "given metric_plan.json, run metrics + statistics"
  - `chart-executor` = "given chart spec, run plot script + write chart handoff"
  - 两 subagent 职责单一，互不污染
  - Lead 是唯一编排者

**这一层修的是 §1.3 根因 2-5**。

### 4.2 用户当前态度：先 (b) 做 Layer 1 还是 (a) 先打 dogfood

用户在 grill-me 第 7 轮回完 D3 后切换话题，说出 §2 三条洞察后**直接要求"写一个 handoff，我新开一个 agent 继续讨论"**。

**当前未明确**：Layer 1 / Layer 2 / 两层一起 / 先写设计 spec — 用户给的 4 个选项 (a)(b)(c)(d) 没选。

我（上一会话 Claude）推荐 **(b) 先做 Layer 1**，理由：Layer 1 不做，Layer 2 的 fallback 机制会被 lead 凭直觉绕过；Layer 1 成本极低（prompt + 一个 middleware）但解锁所有后续修复的真实效果。

---

## 5. 下一会话的具体起步动作建议

### 5.1 必读（按顺序）

1. **本 handoff**（你正在读）
2. `~/.claude/projects/-home-wangqiuyang/memory/MEMORY.md` 索引
3. `~/.claude/projects/-home-wangqiuyang/memory/project_2026-05-18_lead_not_reading_handoff.md` 含 2026-05-14 关联
4. `~/.claude/projects/-home-wangqiuyang/memory/project_2026-05-14_e2e_trajectory_failure.md` 4 天前同类故障
5. `~/.claude/projects/-home-wangqiuyang/memory/project_2026-05-13_catalog_architecture_brainstorm.md` catalog 架构原 brainstorm
6. `docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md` 4 天前完整诊断 + deepseek co-analysis
7. `docs/e2e/Analyzing Mouse Elevated Plus Maze Data.md` 本次 e2e 报告（408 行）
8. 本次 langgraph.log（`packages/agent/logs/langgraph.log` 676 行，含 39 次 SandboxAudit）

### 5.2 第一句话向用户问什么

用户已经在上一会话明确给出了 §2 三条洞察。新会话**不要重新讨论这些洞察**，直接问：

> "我读了 handoff + 5-14 memory + e2e 报告 + langgraph log。
> 现在的状态：上次 grill-me 走完 7 个设计分支但只到了"画图 bug 精确修复"层级，
> 你后来扩展到"lead 不读 handoff" 的更深层问题，并指出 prompt-only fix 不够、需要 harness 级硬约束。
>
> 我建议的下一步是：**先做 Layer 1（强制 lead 消费 handoff）**，因为 Layer 2 的所有 fallback 机制都依赖 lead 真的会读 hard fact。
>
> 你认同 Layer 1 优先吗？还是想：
> (a) 跳过 Layer 1 直接做 Layer 2（接受下次还会爆雷）
> (b) Layer 1 + Layer 2 一起设计写完整 spec（2-3 天）
> (c) 先写一份独立 spec 让团队 review 再实施"

### 5.3 不要做的事

- ❌ **不要重新走 grill-me 7 轮**——决策已经做完，记录在 §3
- ❌ **不要先写代码**——这是个架构决策问题，先把 spec/plan 对齐，再实施
- ❌ **不要碰前端**（streamdown 升级已 commit `5ecd0863` 并 push，下次 dogfood 重启时生效）
- ❌ **不要直接拉上游 deerflow**——sync 第 3 轮已完成，剩下的 Plan B 等 spec-phase-1 合 dev 后再做
- ❌ **不要假设 5-14 的 prompt fix 已经生效**——5-18 复现就是证据

---

## 6. 当前仓库状态

### 6.1 Git

- 分支：`dev`
- HEAD：`5ecd0863 fix(frontend): streamdown 1.4.0 → 2.5.0 修流式吞字 bug`
- origin/dev 同步：✅ 已 push

最近 5 个 commit（按时间倒序）：

```
5ecd0863 fix(frontend): streamdown 1.4.0 → 2.5.0 修流式吞字 bug
33140a6f deps: 加 dashscope>=1.25.18（阿里云百炼 API SDK）
272c7570 merge: deerflow 上游同步第 3 轮（14 commit）
fe9acb40 docs(handoff): 2026-05-18 deerflow sync 进度记录
0ce7529a sync(deerflow): JWT secret 持久化到 .jwt_secret（上游 #2933 / 6d611c2b）
```

### 6.2 Worktree

```
/home/wangqiuyang/noldus-insight                                         5ecd0863 [dev]
/home/wangqiuyang/noldus-insight/.claude/worktrees/spec-phase-1-handoff  21713076 [worktree-spec-phase-1-handoff]
```

`spec-phase-1` 仍在使用，本次任务**不要碰**它。

### 6.3 当前服务状态（如果用户没重启）

- gateway / langgraph / frontend / nginx 在主 repo 跑（端口 2024 / 8001 / 3000 / 2026）
- 用户已经做过一次 dogfood（thread `141b4b98-...`）
- 重启 frontend 才能让 streamdown 2.5.0 生效

---

## 7. 关键文件路径速查

| 文件 | 用途 |
|---|---|
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | Lead prompt — Layer 1 必改 |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` | Middleware 链 — Layer 1 加 HandoffEnforcement |
| `packages/agent/backend/packages/harness/deerflow/guardrails/` | Layer 1 新建 HandoffEnforcementGuardrailProvider 的目录 |
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py` | P0 fix 引入的 Python 工具 |
| `packages/ethoinsight/ethoinsight/catalog/epm.yaml` | EPM catalog（含 charts: [box_open_arm]，需保留）|
| `packages/ethoinsight/ethoinsight/catalog/_common.yaml` | **本次新建** — 通用 charts 注册 |
| `packages/ethoinsight/ethoinsight/catalog/resolve.py` | **本次扩展** — Step 4.5 fallback 注入 + _evaluate_when 加 total_subjects |
| `packages/ethoinsight/ethoinsight/scripts/_common/plot_trajectory.py` | 已存在，CLI 契约对齐，不改 |
| `packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py` | **本次新建** — CLI 包装 timeseries_plot |
| `packages/ethoinsight/ethoinsight/charts.py` | `timeseries_plot` 库函数已存在（第 298 行）|
| `packages/agent/skills/custom/ethoinsight-charts/` | 选择指南 skill，不动 |
| `packages/agent/skills/custom/ethoinsight-code/SKILL.md` | code-executor skill，Layer 2 拆分时改 |

---

## 8. 风险登记

| 风险 | 严重 | 缓解 |
|---|---|---|
| Layer 1 prompt 改了不生效（5-14 prompt fix 已证 prompt-only 不够）| 🔴 高 | 必须配 HandoffEnforcementGuardrailProvider 硬拦截，不能只改 prompt |
| 抽 chart-executor subagent 改动面大 | 🟡 中 | Layer 2 范围；先 Layer 1 做完确认有效，再讨论是否抽分 |
| 单被试默认 y_col 选错（用户期望开臂占比，实际给 distance_moved）| 🟢 低 | 范式默认值映射写在 plot_timeseries.py 内部，可单独调整 |
| `catalog/_common.yaml` 新文件破坏现有测试 | 🟢 低 | resolve.py 找不到 _common.yaml 时 fallback 到空 list；现有 epm/oft test 不受影响 |
| dogfood 期间用户不耐烦 | 🟡 中 | streamdown 已修，至少下次 dogfood 流式渲染正常；Layer 1 修完后画图能"明确反问"也比 LOOP DETECTED 强 |
| 同一问题第三次复现 | 🔴 高 | 这就是为什么必须 harness 硬约束，不能再 prompt-only |

---

## 9. 上一会话的 5 个心智模型（带给下一会话）

1. **EthoInsight 不是只有 demo 路径**：单被试、用户中途换话题、handoff 不完整 = first-class 边界情况，应在 prompt 里**显式枚举**对应路径
2. **同一问题 4 天内复现 = prompt-only fix 已证伪**：未来类似问题先想"能不能 middleware/guardrail 硬约束"，再考虑 prompt
3. **single source of truth 是硬性原则**：用户 memory 里第一条就是这个；任何"两份知识同步"的方案都会被否
4. **catalog 是范式知识的唯一住所**：fallback、降级路径、单被试 chart 都应注册到 catalog，不应硬编码 Python
5. **Lead 是 planner + decision authority，不是 executor**：用户原话洞察 A 的精确化版本

---

## 10. 元方法论（用户洞察 C 的精确化）

> "最大化利用 deerflow 框架 + 参考好 agent（claude code）的优点 + 加以利用"

**操作化**：每次类似 bug 出现时，按顺序问：

1. 看 deerflow 现有有没有解决这个问题的机制（现成的 middleware、provider、skill 系统）→ CLAUDE.md 第 12 条已经在写这个原则
2. 如果有，复用它
3. 如果没有，看 Claude Code 等成熟 agent 是怎么做的
4. 把那个范式抽象成"我们 deerflow 的版本"，加进框架，而不是写一次性补丁

**对应到本次**：

- 强制读 handoff = Claude Code 的 "trust but verify"
- deerflow 等价机制 = `GuardrailMiddleware` + `GuardrailProvider` 协议
- 抽象出来 = 新 provider `HandoffEnforcementGuardrailProvider`
- 这样每次解决具体 bug，框架都积累普适能力，下次别的 PR / 范式 / agent 都能用

---

**最后**：本 handoff 不是结束，是上下文转移。下一会话目标是**对齐 Layer 1 / Layer 2 实施顺序 → 写 spec → 实施 → dogfood 验证**。
