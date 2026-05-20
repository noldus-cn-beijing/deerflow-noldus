# 2026-05-20 FST E2E 双 bug 修复 Handoff

> **前置上下文**:承接 [2026-05-20-e2e-3-fix-plus-ttft-plus-test-cleanup-handoff.md](2026-05-20-e2e-3-fix-plus-ttft-plus-test-cleanup-handoff.md) (PR #18)。
> 用户当天的 FST 强迫游泳 E2E 暴露出 2 个 bug，本次会话全部修复。
>
> **状态**:修复完成，测试全绿，待下一轮 E2E 验证。
>
> **分支**:`dev`
> **Commits**:
> - `a33a07cf` fix(lead-intent): 复合语义判定改为动词类别 ≥2 类 → E2E_FULL
> - `9fce7715` docs updated **(commit message 不准，实际含 Bug 2 代码改动——见下文警告)**
> - `e9876515` test(catalog): 补 multi-subject 守护测试

---

## 诊断起点：用户 FST E2E 失败 trace

用户上传 2 个 Arena 数据文件(实验组 + 对照组)，说"帮我做大鼠强迫游泳的描述性分析和可视化"。
最终结果：**用户只看到两张通用图 + `[FORCED STOP]` 红字，没有数值解读、没有报告**。

关键 log (来自 `packages/agent/logs/langgraph.log`)：
- L82-L83: `Guardrail denied: tool=ls policy=intent_classification code=ethoinsight.intent_not_declared` —— lead 在 intent 声明前调 ls 被拦截
- L145: `Tool frequency warning — too many calls to same tool type count=3 tool_name=read_file`
- L166: `prep_metric_plan success: paradigm=fst, metric_count=3` —— **只传了 1 个文件**
- L188-L204: code-executor 完成，6 AI messages，handoff_code_executor.json 写盘
- L212-L253: chart-maker 完成（max_turns=15），8 AI messages，handoff_chart_maker.json 写盘
- L261: `Tool frequency hard limit reached — forcing stop count=5 tool_name=read_file` —— **LoopDetectionMiddleware Layer 2 tool_freq_hard_limit=5**
- L263: TrainingDataMiddleware 落盘 5 samples（但撞硬限那 turn 的 tool_calls 已被 strip）
- 全程 **code-executor → chart-maker → FORCED STOP**，**data-analyst 和 report-writer 从未被派**

thread workspace 物证 (`~/.deer-flow/users/.../threads/9a1a.../user-data/workspace/`)：
- plan_metrics.json: `raw_files` 数组长度 1，只有 Arena 1，Arena 2 没出现
- handoff_code_executor.json: metrics 只有 Arena 1 的 3 个指标值，data_quality_warnings 里 LLM 自己写了 `"不动时间仅5.52秒(1.84%)异常低，可能存在检测阈值问题"` —— **LLM 的 warning 写对了，但用户没看到**
- handoff_chart_maker.json: 2 张图(trajectory + timeseries)

---

## Bug 1: 意图分类 E2E_MIN 错判 → 缺失 data-analyst

### 根因

[packages/agent/skills/custom/ethoinsight-lead-interaction/references/intent-decision-tree.md](packages/agent/skills/custom/ethoinsight-lead-interaction/references/intent-decision-tree.md) L5-L6 原 trigger：

```
Yes + "分析并画图/报告/全套" → E2E_FULL
Yes + "分析一下"/"看看" → E2E_MIN
```

用户原话 "描述性分析和可视化" 是两个动词类别(ANALYZE + VISUALIZE)，但 lead 只匹到"分析"模式 → 判 E2E_MIN。E2E_MIN 流水线是 `code-executor → ask(four-choice)`，chart-maker 根本不该被派——但 lead 脱离脚本自己派了 chart-maker；chart-maker 跑完后没有下一步脚本，lead 只好自己读 workspace JSON 汇总 → 撞 LoopDetection 硬限。

### 修复 (commit `a33a07cf`)

3 处改 + 6 个守护测试：

- [intent-decision-tree.md](packages/agent/skills/custom/ethoinsight-lead-interaction/references/intent-decision-tree.md) — 引入 4 动词类别 CALC/ANALYZE/VISUALIZE/REPORT，**≥2 类 = 复合语义 = E2E_FULL**，5 个正例 + 4 个反例，歧义偏 E2E_FULL 兜底
- [prompt.py:257-258](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L257-L258) — 主表状态机后追加复合语义判定段(lead 不读 skill 也能判对)
- [prompt.py:758](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L758) — 简表加一行复合语义提示
- [test_lead_prompt_intent_composite_semantics.py](packages/agent/backend/tests/test_lead_prompt_intent_composite_semantics.py) 6 个守护断言

---

## Bug 2: 多 subject 在 plan 层被静默丢弃

### 根因 (双层串联)

1. **prep_metric_plan_tool 签名 `uploaded_file: str`** — 单字符串，lead 只传第一个文件（别人也没法传多个）
2. **catalog.resolve 内部 `_metric_to_plan` / `_chart_to_plan` 永远 `raw_files[0]`** — 即便 resolve_metrics 收到 len>1 的 raw_files list，也只用第一个

完整调用链：
```
lead (传1个文件 Arena 1) 
  → prep_metric_plan_tool(uploaded_file: str)          [bug #1: 单文件 schema]
    → resolve_metrics(raw_files=[uploaded_file])
      → _metric_to_plan(raw_files)                     [bug #2: raw_files[0]]
        → PlanMetric(input=raw_files[0], ...)
          → plan_metrics.json 只有 Arena 1 的数据
```

### 修复 (commit `9fce7715` 代码 + `e9876515` 测试)

4 个文件 + 8 个守护测试：

- [prep_metric_plan_tool.py](packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py)
  - `uploaded_file: str` → `uploaded_files: list[str]`
  - 加 `no_files_provided` 错误码 + `failed_file` 字段定位失败文件
  - docstring 强调"把所有 <uploaded_files> 里相关文件全传进来"
  - `plan_summary` 加 `subject_count` 字段
- [schema.py](packages/ethoinsight/ethoinsight/catalog/schema.py)
  - PlanMetric + PlanChart 加 `subject_index: int = 0` 字段
- [resolve.py](packages/ethoinsight/ethoinsight/catalog/resolve.py)
  - `_metric_to_plan` → 返回 `list[PlanMetric]`，按 raw_files 展开 N 个
  - `_chart_to_plan` → 返回 `list[PlanChart]`，同上
  - 调用处 `append` → `extend`
  - output 路径：多文件时 `m_<id>_s<idx>.json`，单文件时保持 `m_<id>.json` 兼容旧产物
  - plan_to_dict / plan_metrics_to_dict / plan_charts_to_dict 序列化 subject_index
- [test_prep_metric_plan_tool.py](packages/agent/backend/tests/test_prep_metric_plan_tool.py) `TestPrepMetricPlanToolMultipleFiles` 4 个 case
- [test_catalog_resolve_multi_subject.py](packages/ethoinsight/tests/test_catalog_resolve_multi_subject.py) 4 个 case

---

## ⚠️ Commit History 异常

`9fce7715` commit message 写的是 "docs updated"，但实际 diff 包含 4 个文件 294 insertions（prep_metric_plan_tool / resolve / schema / test_prep_metric_plan_tool）。这是用户在自己另一个终端里手工 commit 时不小心把未 commit 的脏区改动一并扫走了。

`e9876515` 补上了剩下未 tracked 的 `test_catalog_resolve_multi_subject.py`。

**如果还没 push 到 origin 可以 rebase 修正 message；如果已 push 则可以 squash。由用户决定。**

---

## 测试结果

| 套 | passed | failed | 
|---|---|---|
| ethoinsight | 331 | 0 |
| backend (excl live) | 2678 | 0 |
| backend test_client_live | - | 1 pre-existing (intent guardrail 反弹路径断言不匹配) |

---

## 明确不动的东西 (不要回退)

1. **LoopDetectionMiddleware 阈值 5** — 不是 root cause，修意图后 lead 不会撞
2. **IntentClassificationGuardrailProvider** — 只校验 `[intent]` 是否输出，不校验 X 选对(选对靠 prompt)
3. **immobility <5% sanity check (脚本级)** — code-executor LLM 已在 handoff 里写了正确 warning。意图修对后 data-analyst 会被派，自然会读到 warning 并融入解读。加脚本兜底违反 CLAUDE.md §9 不用绝对阈值
4. **FST 单样本 chart catalog 空** — PR #18 handoff 已标"等行为学同事 PR"
5. **PR #18 的 5 个 commit** — 不要回退

---

## 已知仍存在的 gap (本次未处理，下次会话可做)

1. **FST/LDB/TST/zero_maze 4 个范式单样本 chart catalog 仍空** — 等行为学同事 PR 后补
2. **Gate 1 零数据时的 4 个问题反问** — UX 体验不佳但不是 bug
3. **statistical_validity == "skipped" wire-up** — PR #18 handoff 已标记 prompt 改了但未验证 LLM 是否真写 skipped
4. **worktree + uv editable install 盲区** — backend pytest 在 worktree 里导旧代码，MEMORY 已记

---

## 下次 E2E 验证点 (必跑)

用同一句 "**帮我做大鼠强迫游泳的描述性分析和可视化**" + **上传 2 个 Arena 文件**，验证：

| # | 信号 | 期望值 | 如果不符合 |
|---|---|---|---|
| 1 | lead 第一个 non-read tool call 前 `[intent]` | `E2E_FULL` | Bug 1 未修好；读 intent-decision-tree.md 看复合语义判定 |
| 2 | `prep_metric_plan` 的 `uploaded_files` 字段 | list 含 2 个虚拟路径 | Bug 2 未修好；看 tool schema 是否被 langgraph 正确暴露 |
| 3 | plan_metrics.json `metrics` 数组 | `M个指标 × 2个文件` 条，每条带 `subject_index: 0/1` | catalog.resolve 展开逻辑有 bug |
| 4 | workspace 文件 | `m_<id>_s0.json` 和 `m_<id>_s1.json` 成对 | output 路径生成错 |
| 5 | 派遣顺序 | `code-executor → data-analyst → chart-maker` | Bug 1 未修好；data-analyst 不出现则说明意图仍错 |
| 6 | data-analyst 解读内容 | 提 data_quality_warnings、强调 1.84% 可能因检测阈值问题不可信 | code-executor 的 warning 写的对但 data-analyst 没读到 |
| 7 | 最终用户消息 | 无 `[FORCED STOP]` | LoopDetection 再次触发，需排查 lead 是否仍有不合理的 read 频率 |
| 8 | 最终用户消息 | 有 data-analyst 判读 + chart-maker 出图 + 中文总结 | 流水线完整 |
| 9 | log 中 read_file frequency warning | **不应出现** (lead 不该再自己批量读 workspace JSON) | Lead 仍在试图自己汇总，需再查 prompt |

---

## 关键文件速查 (接手直接打开这些)

- [prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) L242-265 — 意图状态机 + 复合语义判定
- [intent-decision-tree.md](packages/agent/skills/custom/ethoinsight-lead-interaction/references/intent-decision-tree.md) — 复合语义判定细则
- [prep_metric_plan_tool.py](packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py) — 多文件 support
- [resolve.py](packages/ethoinsight/ethoinsight/catalog/resolve.py) L367-405 — `_metric_to_plan` / `_chart_to_plan` 展开逻辑
- [schema.py](packages/ethoinsight/ethoinsight/catalog/schema.py) L87-101, L113-119 — PlanMetric + PlanChart 新字段
- [test_lead_prompt_intent_composite_semantics.py](packages/agent/backend/tests/test_lead_prompt_intent_composite_semantics.py) — 意图守护测试
- [test_prep_metric_plan_tool.py](packages/agent/backend/tests/test_prep_metric_plan_tool.py) — prep_metric_plan 多文件守护测试
- [test_catalog_resolve_multi_subject.py](packages/ethoinsight/tests/test_catalog_resolve_multi_subject.py) — catalog 多 subject 守护测试
- [langgraph.log](packages/agent/logs/langgraph.log) L82-L267 — 本次 E2E 故障的完整 log

## 跑测试命令

```bash
# ethoinsight
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/python -m pytest tests/ -q

# backend (excl live)
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q --ignore=tests/test_client_live.py --ignore=tests/test_metric_catalog_live.py
```
