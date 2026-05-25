# 2026-05-22 tool-vs-script 架构 grill 交接

> **状态**：grill 完成 40%，Q10 美观/前端子树全部敲定可立即实施。剩余 4 棵子树待 grill。

## 起点问题

> "看到 Claude Code 用 git bash 的方式，能不能仿照 git，将指标的代码、图表生成的代码，也做成 tooluse，而不是脚本（更重）？"

## grill 路径与核心结论

### 结论：不做 metric tool 化

完整走过所有支持"tool 化指标"的理由，全部塌陷：

| 支持理由 | 塌陷原因 |
|---|---|
| 跨范式借用指标（β3） | 行为学同事确认不存在 |
| 中间结果可观察（ε） | v0.1 客群不需要 |
| 失败后诊断（δ） | 重定向为"先发式导出指导"，不是事后诊断 |
| 脚本更重 | 重的是 LLM 试错 + Gate 死锁 + reasoning_effort=high，不是 process 冷启动 |
| 美观（git 风格） | 前端 stage-broadcast 已有半成品，补折叠+语义化即可 |
| 5/9-5/12 grill | 30+ 判断点已否决"指标当 tool"和"LLM 写胶水代码" |

### 结论：真正要做的是 7 项（按优先级）

| # | 项目 | 解决什么 | 估时 | grill 状态 |
|---|---|---|---|---|
| **1** | (Q) 脚本契约一致 + skill 签名表 | plot_trajectory 试错 narration | 2-3 天 | 未 grill |
| **2** | δ 导出指导工具 | 用户少返工 | 中 | 未 grill |
| **3** | (P) 前端语义化展示（Q10 子树） | 美观，消除 python 命令行 | 2-3 天 | **已 grill 完成** |
| **4** | β1 通用指标暴露（_common.yaml + metrics） | "顺便看看运动量" | 2-3 天 | 未 grill |
| **5** | β2 多范式 context 重构 | 一次会话多范式分析 | 1-2 周 | 未 grill |
| **6** | (S) 读动词 tool（peek_csv_columns / inspect_workspace） | Gate 1 提速 + 反复 read_file | 中 | 未 grill |
| **7** | 5/22 P0/P1 修复（FST E2E） | Gate 2 死锁 + write_todos + reasoning_effort | ~440 行 | spec 已写完，未 grill 实施细节 |

---

## ✅ Q10 子树：前端语义化展示（已完全敲定，可立即实施）

### 决策总表

| 问题 | 决策 |
|---|---|
| 展示程度 | 仅语义化，不显示 python 命令行 |
| 映射来源 | catalog yaml 的 display_name_zh = single source of truth |
| 传递路径 | prep_metric_plan 写 plan_metrics.json（已有） → 新端点 `/api/threads/{id}/plan-labels` 读并返回 `{script: {display, kind}}` → 前端一次 fetch 缓存 |
| fallback | 不需要——code-executor 受 L1(system prompt)+L2(ScriptInvocationOnlyProvider)+L3(plan 编排) 三层约束，所有脚本调用 100% 在 plan 内 |
| metric/chart/stats 区分 | 三种 kind：metric→"正在计算"、chart→"正在绘制"、stats→"正在运行" |
| 数据结构 | `{"script_path": {"display": "中文名", "kind": "metric|chart|stats"}}` |
| 脚本名展示 | (G2) 单行 `📊 正在计算 开放臂时间比例 · compute_open_arm_time_ratio` |
| 非脚本 bash | 走现有 description fallback，不额外处理 |

### API 设计

```
GET /api/threads/{thread_id}/plan-labels

Response:
{
  "ethoinsight.scripts.epm.compute_open_arm_time_ratio": {
    "display": "开放臂时间比例",
    "kind": "metric"
  },
  "ethoinsight.scripts.epm.plot_box_open_arm": {
    "display": "开放臂盒图",
    "kind": "chart"
  },
  "ethoinsight.scripts.epm.run_groupwise_stats": {
    "display": "组间统计检验",
    "kind": "stats"
  }
}
```

### 实施改动清单

| 文件 | 操作 | 行数 |
|---|---|---|
| 新建 `app/gateway/routers/plan_labels.py` | 新 router，读 workspace 的 plan_metrics.json，遍历 metrics[]/charts[]/statistics 提取 script→display→kind | ~30 |
| `app/gateway/app.py` | `include_router(plan_labels.router)` | +2 |
| `frontend/src/core/tools/stage-broadcast.ts` | `getStageBroadcastForBash` 接受可选 labelMap 参数，命中返回语义标签；`detectEthoinsightCli` 保留但不再用于生成文案 | ~15 |
| `frontend/src/components/workspace/messages/message-group.tsx` | 首次看到 prep_metric_plan 成功→fetch `/api/threads/{id}/plan-labels`→缓存；bash 渲染时传入 labelMap；脚本名以 inline code 样式展示在标签后 | ~40 |

### 注意事项

- 不改 `prep_metric_plan_tool.py`（plan_metrics.json 已含 display_name_zh）
- 不改 `plan_metrics.json` 的 schema
- single source of truth 链条：catalog yaml → prep_metric_plan → plan_metrics.json → API → 前端，display_name_zh 始终来自 catalog
- 前端缓存 per-thread，同 thread 的 run 之间 plan 不变（除非 lead 重调 prep_metric_plan）

---

## 未 grill 子树（下一个 agent 继续）

按推荐 grill 顺序排列：

### 1. (Q) 脚本契约一致 + skill 签名表（优先级最高）

**背景**：用户证据——code-executor 跑 `plot_trajectory --paradigm fst` 失败，因为 `plot_trajectory` 不接受 `--paradigm` 但 `plot_timeseries` 接受。LLM 试错-反思-重试全过程暴露给用户。

**关键文件**：
- `packages/ethoinsight/ethoinsight/scripts/_common/plot_trajectory.py`（不接受 --paradigm）
- `packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py`（接受 --paradigm）
- `packages/ethoinsight/ethoinsight/scripts/_cli.py`（统一 CLI helper）
- `packages/agent/skills/custom/ethoinsight-code/SKILL.md`（code-executor 执行手册）

**待 grill 问题**：
- 统一所有脚本签名（都接 --paradigm 默认忽略？还是精确声明谁接谁不接？）
- skill 文档里要不要加 `references/script-signatures.md` 列出每个脚本的精确参数？
- 还是更激进：让 plan_metrics.json 里每个 entry 带 `args_template` 字段，code-executor 直接拼不用猜？

### 2. δ 导出指导工具

**背景**：γ + δ 合并诉求——EthoVision 有成对指标（如 Speed-Movement / Mobility-Mobility State / Activity-Activity State），EthoInsight 只用 EV 导出的状态列。用户缺列时，系统不应只说 columns_missing，应精确告诉用户缺哪个 EV 因变量 + 在 EV 里怎么设置。

**待确认**：EV19 template → required EV variables + 配置路径的映射表，是否已在 `docs/review-packages/2026-04-29-ev19-templates/` 里？还是需要新建？

### 3. β1 通用指标暴露

**背景**：`_common.yaml` 只有 charts（trajectory + timeseries），缺 metrics。`compute_distance_moved.py` / `compute_velocity_stats.py` 脚本已存在，但 catalog 不暴露给编排。需要扩 `_common.yaml` 加 metrics 段 + `prep_metric_plan` 合入逻辑。

### 4. β2 多范式 context 重构

**背景**：`experiment_context.py` 的 `paradigm` 是单数 str，不支持一次会话多范式分析。需要 paradigms 数组化 + Gate 1/2 适配。这是最大改动项。

### 5. 5/22 P0/P1 FST E2E 修复实施细节

**spec 已完整**：[docs/problems/2026-05-22-fst-e2e-issues-and-solutions.md](../problems/2026-05-22-fst-e2e-issues-and-solutions.md)，4 阶段 ~440 行。需要 grill 的是实施顺序、是否拆 PR、dogfood 验证方式。

### 6. (S) 读动词 tool

**背景**：lead 没有轻量 inspect 工具（`peek_csv_columns` 等），Gate 1 范式识别阶段只能靠 `prep_metric_plan`（大动作）或反问用户。加 1-2 个只读 tool 可提速。

---

## 关键上下文（给接手 agent）

### 已排除的方向（不要再花时间）

- ❌ 把指标做成 LangChain tool——5/9-5/12 完整否决
- ❌ 把指标做成 MCP tool——同上
- ❌ 放开 code-executor bash 白名单（允许 python -c）——5/15 lead-bash-removal 原则
- ❌ 新建 free-analyst subagent——β3 被行为学同事否决
- ❌ 给指标脚本加 --threshold 等参数——v0.1 只用 EV 导出的状态列
- ❌ lead 恢复 write_file——破坏工具隔离

### 核心原则（不可违反）

1. **single source of truth**：display_name_zh 等元数据只在 catalog yaml 一份
2. **复用 deerflow 现成 infra 优先**：GuardrailMiddleware、LoopDetectionMiddleware、files-are-facts handoff 协议
3. **deepseek 正面指令**：不用"不要 X / 禁止 X"
4. **判读哲学**：组间比较不用绝对阈值
5. **lead 无 bash/write_file/str_replace**——5/15 P0 已锁定

### 关键文件速查

| 用途 | 路径 |
|---|---|
| 项目 CLAUDE.md | `/home/wangqiuyang/noldus-insight/CLAUDE.md` |
| code-executor 配置 | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` |
| script CLI helper | `packages/ethoinsight/ethoinsight/scripts/_cli.py` |
| stage-broadcast（前端语义化） | `packages/agent/frontend/src/core/tools/stage-broadcast.ts` |
| message-group（前端 tool_call 渲染） | `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` |
| catalog epm.yaml（示例） | `packages/ethoinsight/ethoinsight/catalog/epm.yaml` |
| _common.yaml | `packages/ethoinsight/ethoinsight/catalog/_common.yaml` |
| 5/22 FST E2E 问题 spec | `docs/problems/2026-05-22-fst-e2e-issues-and-solutions.md` |
| ethoinsight-code skill | `packages/agent/skills/custom/ethoinsight-code/SKILL.md` |
| Gateway routers | `packages/agent/backend/app/gateway/routers/` |
| 5/15 P0 lead-bash-removal handoff | `docs/handoffs/2026-05/2026-05-15-p0-lead-bash-removal-handoff.md` |
| 5/12 script-per-metric handoff | `docs/handoffs/2026-05/2026-05-12-script-per-metric-epm-completed-handoff.md` |
| 5/11 files-are-facts handoff | `docs/handoffs/2026-05/2026-05-11-files-are-facts-handoff.md` |
| 5/11 SOTA architecture grill | `docs/handoffs/2026-05/2026-05-11-paradigm-sota-architecture-grill-handoff.md` |
