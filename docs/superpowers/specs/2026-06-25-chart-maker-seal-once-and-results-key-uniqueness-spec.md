# Spec：chart-maker 封存「只允许一次」+ run_chart_plan results key 唯一化

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-25
> 代码基线：dev HEAD `51f00191`
> 性质：🔴 高 · 封存确定性不变式（reward hacking 结构防御）+ 计数 bug 修复。
> **方向（用户拍板 2026-06-25）**：EPM dogfood（thread `a6e3775c`，112/112 画图成功）取证发现——本次 handoff `sealed_by=model`（应为 `run_plan`），根因是 chart-maker **手调 `seal_chart_maker_handoff` 覆盖**了 `run_chart_plan` 工具的确定性封存。同一取证顺带坐实 `_execute_tasks` results dict 的 id 多对一覆盖 bug。两件同属「确定性产物被后写入覆盖」族，合并一份 spec 修。
> **范式归属**：本 spec 是 `run_chart_plan` 确定性化 track 的收尾补强（前序：`run_chart_plan` 确定性执行 spec、argv resolve spec、ETHO-10 产物真实性不变式 spec）。修的是「工具确定性产物被 LLM 二次封存覆盖」这个 run_chart_plan 引入后新暴露的结构缺口。
> 受保护文件（sync surgical）：`tools/builtins/seal_handoff_tools.py`、`tools/builtins/run_chart_plan_tool.py`、`subagents/builtins/chart_maker.py`、`subagents/handoff_schemas.py`。

---

## ⚠️ 〇、根因校正（取证坐实，守 ETHO-2 教训）

> 本 spec 来自 dogfood handoff `2026-06-24-epm-dogfood-chart-success-askviz-gate-handoff.md` §5 + problem doc `2026-06-24-chart-maker-run-chart-plan-disk-race-false-partial.md`。**两份文档的根因都被本会话取证推翻**，本章如实记录，避免把未坐实假设写进根因。

### 原 problem doc 归因（**未坐实，本 spec 不据此立论**）
> 「`run_chart_plan` 用 `ProcessPoolExecutor` 进程内并行渲染 112 张图，chart-maker 在**进程池尚未全部把 png flush 到磁盘**的窗口期核盘（`ls outputs/*.png`），只看到最后落盘的一批（s27 的 4 张），故自述 `n_rendered=4 / n_failed=108`。根因=核盘时机竞态。」

### 取证铁证（thread `a6e3775c` 一手 trace + 代码层验证，2026-06-25）

1. **代码层证伪"核盘时机竞态"**：`run_chart_plan_tool.py:394-396` 的 `finally` 块**已有** `pool.shutdown(wait=True, cancel_futures=True)`——`_execute_tasks` 返回前所有 worker 进程已等至全部退出。核盘（Step 7，[L224-259](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L224)）在 `_execute_tasks` **返回之后**才执行，且 `results` dict 是逐个 `fut.result(timeout=)` 按**提交顺序**等齐才返回（[L374-385](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L374)）。**核盘时所有绘图 worker 早已退出、文件早已 flush。**"进程池未 flush 就核盘"与代码不符。
2. **handoff 铁证**：`handoff_chart_maker.json`（thread `a6e3775c` workspace 实读）`sealed_by=model`（**不是** `run_plan`），summary 是中文长句「112/112 张图表生成完成：开放臂时间比例柱状图(28)…」——**绝不是** run_chart_plan 工具的机械 summary（`f"{n_rendered}/{n_total} charts rendered (run_chart_plan)"`，[L420](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L420)）。证明 chart-maker **手调** `seal_chart_maker_handoff` 覆盖了工具封存。
3. **trace 铁证（训练数据 jsonl，chart-maker 15 步工具序列）**：
   ```
   1-4: read_file / prep_chart_plan / read_file
   5:   run_chart_plan          ← 工具调了，确定性封存 sealed_by=run_plan 写盘（真相）
   6-13: bash ls / write_file    ← chart-maker 不信工具返回，自己核盘（"n_rendered=4/partial/fix_handoff"叙事即此段 LLM 自述）
   14:  seal_chart_maker_handoff ← 手调 seal，无条件覆盖 → sealed_by=model
   15:  present_files
   ```
   chart-maker 违反 system_prompt step 9「run_chart_plan 内部已 seal handoff，**不要再调 seal_chart_maker_handoff**」（chart_maker.py:63）。
4. **覆盖语义铁证**：`_seal_handoff_to_workspace`（[seal_handoff_tools.py:674-726](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py#L674)）`os.rename(tmp, final)` **无条件覆盖**，不检查磁盘是否已有 `sealed_by=run_plan` 封存。故 chart-maker 第 14 步手调直接覆盖第 5 步的确定性封存。

### 本 spec 据实的立论（真根因，**非"核盘竞态"**）

| # | 真根因 | 依据 |
|---|---|---|
| R1 | **double-seal 覆盖**：`_seal_handoff_to_workspace` 无条件覆盖 → 任何对 chart-maker handoff 的二次 seal（chart-maker 手调 `seal_chart_maker_handoff`，或 `run_chart_plan` 子集调用）都会把先写的确定性封存（`sealed_by=run_plan`）覆盖掉，退回 LLM 自报（`sealed_by=model`）。 | handoff 铁证 + trace 铁证 + 覆盖语义铁证 |
| R2 | **results dict id 多对一覆盖**（顺带发现）：`_execute_tasks` 的 `results[tid]` 用 chart id 做 key（[L377](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L377)），per_subject 112 个 chart 共享 4 个 id（`open_arm_time_ratio_bar`×28 等）→ 112 个结果互相覆盖成最后写入的 4 个。**不丢图**（核盘靠磁盘 `chart_files` 不靠 results），只丢失败 reason（同 id 只留最后一个）。 | 实测 `_execute_tasks` 跑 112 图 `ok=4/112 fail=0`（4=唯一 id 数）|

**与"核盘竞态"的关系**：problem doc 描述的「n_rendered=4 / 想写 fix_handoff.py」是 chart-maker 第 6-13 步**自己 bash ls 的 LLM 自述**，不是 run_chart_plan 工具内部行为。本次 dogfood **零磁盘竞态、零丢图**——112 张 png 全在盘，最终 handoff 计数正确。**唯一的真问题**是确定性封存被 LLM 手调覆盖（R1），使 `sealed_by` 从确定性值退回 LLM 自报值。

### 与其他 chart-maker spec 的关系（正交性核对）

| spec | 根因层 | 与本 spec |
|---|---|---|
| `run_chart_plan-permissionerror-argv-resolve`（已合 #192）| 渲染层（savefig /mnt 不 resolve）| 正交（已修）|
| `run-chart-plan-subset-overwrite-silent-drop`（待实施）| **封存层**（子集调用写残缺 handoff 覆盖全量）| **同族**——本 spec R1 的"封存只允许一次"不变式**同时治它**（见 §三 M1 说明）|
| `intent-gate-signals-frontend-leak`（待实施）| 前端 render 层 | 正交 |
| 本 spec | **封存层**（手调 seal 覆盖工具 seal）+ **计数层**（results key）| — |

> **重要**：本 spec 的 M1（封存只允许一次）实施后，`run-chart-plan-subset-overwrite-silent-drop` spec 的 M1（子集以全 plan 对账）部分**可被吸收**——两者都靠"已封存则拒覆盖"这一个结构门。接手 agent 优先实施本 spec，subset-overwrite spec 的对账部分（M2/M3）仍独立需要。

---

## 一、给实施 agent 的一句话

在 `_seal_handoff_to_workspace`（chart-maker 分支）加「**封存只允许一次**」不变式：磁盘已有 `handoff_chart_maker.json` 且其 `sealed_by == "run_plan"` 时，任何后续 chart-maker seal **raise ValueError**（fail-loud，LangChain 转 error ToolMessage 让 LLM 看到拒绝原因），除非显式传 `force=True`（重跑全量场景）。`run_chart_plan` 工具自身调用走 `force=True`（它是确定性来源，合法覆盖场景）。同步把 `_execute_tasks` 的 results dict key 从 chart id 改成唯一 index（修 R2）。chart-maker system_prompt step 9 强化「禁止手调 seal_chart_maker_handoff」。TDD：先复现「手调 seal 覆盖 run_plan 封存」红，再绿。

---

## 二、取证坐实的可行性（全部跑了代码）

### R1 可修性 —— ✅
- `_seal_handoff_to_workspace`（[L674](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py#L674)）是 chart-maker 所有封存路径的**单一注入点**（spec 2026-06-22 已论证）：`seal_chart_maker_handoff` 工具 + `run_chart_plan` 工具 + harness auto-seal 三条路径都经它。**在此一处加不变式 = 全路径覆盖**。
- 已封存判定：读 `workspace / "handoff_chart_maker.json"`，`json.loads` 取 `sealed_by`。文件不存在/解析失败 → 视为未封存（放行）。这和现有 `_load_chart_files_map`（[L196](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py#L196)）读同一文件的既有模式一致。
- `force=True` 透传：`_seal_handoff_to_workspace` 加 `force: bool = False` 参数；`run_chart_plan_tool.py` 调用处传 `force=True`（L288 `_seal_handoff_to_workspace(...)` 加 kwarg）；`seal_chart_maker_handoff` 工具**不暴露** force（LLM 无法绕过——守"LLM 提议，确定性门定生死"）。

### R2 可修性 —— ✅
- `_execute_tasks`（[L315](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L315)）的 `results: dict[str, tuple[int,str]]` key 改成**唯一**。候选：
  - **方案 a（推荐）**：key 用 `task index`（enumerate 位置），value 带 `tid`。`results[idx] = (tid, rc, err)`。
  - 方案 b：key 用 `output 虚拟路径`（per_subject 唯一，但 output 可能为空 → 不稳）。
  - **选 a**：index 恒唯一、零依赖、与现有 `ordered = list(...)` 枚举天然对齐。Step 7 核盘循环同步改用 index 对齐 `chart_meta`（`chart_meta` 也按 index 建，或改为 `list[dict]`）。
- **注意 chart_files/failed_charts 构造不依赖 results 的 tid 唯一性**（[L229-256](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L229)）：核盘靠 `chart_meta[tid].output` 磁盘 exists。改 results key 后，Step 7 循环改为 `for idx, (tid, (rc, err)) in enumerate_by_task(...)` 即可。**务必保证 112 个 chart 全核盘**（修 R2 前 results 只 4 entry，但核盘循环是 `for tid in results` → 同样只核 4 个！见 §五 风险 1）。

> **🚨 R2 的真实危害（比预想重）**：原写「不丢图」是基于「核盘靠磁盘」的直觉，但代码 Step 7 循环是 `for tid, (rc, err) in results.items()`（[L229](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L229)）——**results 只有 4 个 key 就只核盘 4 次！** 之所以 dogfood 最终 112 张图都在 handoff 里，是因为 chart-maker 手调 seal（R1）自己重新 ls 了全部 112 张。**即 R1 的 LLM 自报反而"救"了 R2 的计数缺口**——若修了 R1（封存只允许一次）而不修 R2，`run_chart_plan` 路径会只核盘 4 张、handoff 只列 4 张。**R1 和 R2 必须同 PR 修，缺一即回归。**

---

## 三、修法（M1/M2，按锚点）

### M1：封存「只允许一次」不变式（治 R1）

**改 `seal_handoff_tools.py:_seal_handoff_to_workspace`**（[L674](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py#L674)）：

```python
def _seal_handoff_to_workspace(
    model_cls: type,
    filename: str,
    payload: dict[str, Any],
    workspace: Path,
    *,
    force: bool = False,   # 新增
) -> str:
    # ... 现有 setdefault / task_context / reconcile 逻辑不变 ...

    # M1：封存「只允许一次」不变式（仅 chart-maker，spec 2026-06-25）
    # 已有确定性封存（run_plan）时，拒绝覆盖——堵 chart-maker 手调 seal 覆盖 run_chart_plan。
    # force=True 仅 run_chart_plan 工具（确定性来源）可传，LLM 工具 seal_chart_maker_handoff 不暴露 force。
    if model_cls is ChartMakerHandoff and not force:
        existing = workspace / filename
        if existing.exists():
            try:
                prev = json.loads(existing.read_text(encoding="utf-8"))
                prev_sealed_by = prev.get("sealed_by")
            except Exception:
                prev_sealed_by = None
            if prev_sealed_by == "run_plan":
                raise ValueError(
                    f"seal_{filename}: 已存在确定性封存 (sealed_by=run_plan)，拒绝覆盖。"
                    f"run_chart_plan 已确定性封存真相，不要再调 seal_chart_maker_handoff。"
                    f"如需重跑，调 run_chart_plan（它带 force=True 合法覆盖）。"
                )

    # ... 现有 Pydantic 校验 + atomic write + manifest 不变 ...
```

**改 `run_chart_plan_tool.py` 调用处**（[L288](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L288)）：传 `force=True`：

```python
_seal_handoff_to_workspace(ChartMakerHandoff, "handoff_chart_maker.json", payload, workspace, force=True)
```

理由：`run_chart_plan` 是确定性来源，重跑全量是合法覆盖场景（用户追加图后重画）。它带 `force=True` 合法覆盖；`seal_chart_maker_handoff`（LLM 手调）不暴露 force → 撞门即被拒。

**不改 `seal_chart_maker_handoff` 工具签名**：它内部调 `_seal_handoff(...)` → `_seal_handoff_to_workspace(...)`（[L740](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py#L740)），`_seal_handoff` 不传 force → 走默认 `force=False` → 撞门拒绝。**LLM 无法绕过**（守"LLM 提议，确定性门定生死"）。

**harness auto-seal 路径**（`_attempt_auto_seal_from_artifacts`，executor.py）：同样不传 force → `force=False`。若 run_chart_plan 已封存，auto-seal 撞门拒绝（这是对的——已有确定性封存就不该 auto-seal 覆盖）。auto-seal 只在 run_chart_plan 没跑过时生效。

### M2：results dict key 唯一化（治 R2）

**改 `run_chart_plan_tool.py:_execute_tasks`**（生产路径 [L365-396](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L365) + 同步 runner 路径 [L346-362](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L346)）：

results 改为按 **task index** 存：

```python
# 生产路径
results: dict[int, tuple[str, int, str]] = {}   # idx -> (tid, rc, err)
...
for idx, (fut, tid) in enumerate(ordered):
    try:
        _tid, rc, err = fut.result(timeout=timeout)
        results[idx] = (tid, rc, err)
        ...
```

**改 Step 5 `chart_meta`**（[L197](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L197)）：从 `dict[str, dict]`（按 chart_id）改为 `list[dict]`（按 task 顺序），或 `dict[int, dict]`（按 index）。**因为同 id 多 chart，dict 按 id 会丢**。

**改 Step 7 核盘循环**（[L229-256](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L229)）：按 index 遍历 results + chart_meta，保证 112 个全核盘：

```python
chart_files: list[str] = []
failed_charts: list[dict[str, str]] = []
for idx in range(n_total):
    tid, rc, err = results.get(idx, ("", 1, "missing result"))
    meta = chart_meta[idx]   # list 形态
    output_virtual = meta.get("output", "")
    ...
```

**改 failures 兜底**（[L250-256](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L250)）：原用 `seen_failed_ids`（按 id 去重，会误并同 id 不同 chart）→ 改按 index 去重，或直接信任 Step 7 循环已覆盖（每个 idx 恰好核一次）。

**同步 runner（测试路径）**：[L349-362](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L349) 改同款 `results[idx] = (tid, rc, err)`，保持两条路径语义一致。

### M3：chart-maker system_prompt step 9 强化（治 R1 的人因面）

**改 `chart_maker.py` step 9**（[L63-67](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py#L63)）：把「不要再调 seal_chart_maker_handoff」从软提示升级为**带后果说明的硬规则**，并明示现在有结构门会拒：

```
9. **据 run_chart_plan 返回值收尾**（run_chart_plan 内部已 seal handoff，**禁止再调 seal_chart_maker_handoff**）。
    - 系统已加「封存只允许一次」结构门：run_chart_plan 写盘后 sealed_by=run_plan，你再调 seal_chart_maker_handoff 会被**确定性拒绝**（报错 "已存在确定性封存，拒绝覆盖"）。
    - run_chart_plan 返回 status=completed → 全部图真落盘，直接进 step 10。
    - run_chart_plan 返回 status=partial/failed → 据 failures 向 lead 汇报即可，handoff 已由工具落盘，**不要自己 bash ls 核盘、不要手调 seal**。
```

> M3 是人因面（prompt），M1 是结构面（门）。按 [[feedback_seal_missing_root_cause_is_react_no_toolcall_exit_gate_not_fallback]]：**结构门定生死，prompt 是辅助**。即使 LLM 仍违反 prompt 试手调 seal，M1 的门会确定性拒绝——本次 dogfood 的覆盖**不会再发生**。M3 的价值是让 LLM 提前知道会被拒、不浪费一次 tool call。

---

## 四、TDD（红→绿，强制）

测试文件：`backend/tests/test_run_chart_plan_seal_once.py`（新）+ 扩 `test_run_chart_plan.py`（R2 用例）。

### T1（R1 红→绿）：手调 seal_chart_maker_handoff 不能覆盖 run_plan 封存
```python
def test_t1_seal_once_rejects_model_overwrite_run_plan(ws_and_outputs, monkeypatch):
    # 1. run_chart_plan 先封存（force=True 合法）→ sealed_by=run_plan
    # 2. 模拟 chart-maker 手调 seal_chart_maker_handoff（force 默认 False）
    # 红：当前 _seal_handoff_to_workspace 无条件覆盖，第 2 步成功覆盖成 sealed_by=model
    # 绿：M1 后第 2 步 raise ValueError "已存在确定性封存，拒绝覆盖"
```

### T2（R1）：run_chart_plan 重跑全量 force=True 仍合法覆盖
```python
def test_t2_run_chart_plan_force_overrides_existing(ws_and_outputs, monkeypatch):
    # 已有 sealed_by=run_plan 封存 → run_chart_plan 再调（force=True）→ 合法覆盖
    # 保证用户追加图重跑场景不被门误伤
```

### T3（R1）：auto-seal 撞已有 run_plan 封存也被拒
```python
def test_t3_auto_seal_rejected_when_run_plan_sealed(...):
    # run_chart_plan 已封存 → harness auto-seal（force=False）→ 拒绝
    # 防止 auto-seal 覆盖确定性封存
```

### T4（R2 红→绿）：per_subject 多 chart 同 id 全核盘
```python
def test_t4_per_subject_same_id_all_reconciled(ws_and_outputs, monkeypatch):
    # 构造 plan：1 个 id（open_arm_time_ratio_bar）× 28 subject，全 rc=0 落盘
    # 红：当前 results dict 按 id 覆盖，核盘循环只核 1 次 → chart_files 只 1 张
    # 绿：M2 后 results 按 index → chart_files 28 张全核出，handoff chart_files=28
```

### T5（R2）：同 id 部分失败 reason 不丢
```python
def test_t5_same_id_partial_failures_all_reasons_kept(...):
    # 28 个同 id，3 个失败（不同 reason）→ failed_charts 应含 3 条
    # 红：当前只留最后 1 条（覆盖）；绿：3 条全保留
```

### T6（回归）：112 图全成功 handoff 自洽（复现 dogfood 正路径）
```python
def test_t6_112_charts_all_success_handoff_consistent(...):
    # 复现 a6e3775c 形态：112 图全 rc=0 落盘 → status=completed,
    # chart_files=112, failed_charts=0, sealed_by=run_plan（不是 model!）
    # 这是 dogfood 应走的正路径，封存只允许一次后 chart-maker 不能再覆盖成 model
```

### T7（导入环）：裸导入两生产入口
改 `seal_handoff_tools.py` 后必须跑（守 CLAUDE.md harness 导入环铁律）：
```bash
PYTHONPATH=. python -c "import app.gateway"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
```

---

## 五、风险与注意事项

1. **🚨 R1/R2 必须同 PR**：见 §二 R2 真实危害。修 R1 不修 R2 → run_chart_plan 只核盘 4 张、handoff 只列 4 张（dogfood 当前靠 R1 的 LLM 自报"救"了 R2，修了 R1 这个救兵就没了）。反之修 R2 不修 R1 → LLM 仍可手调覆盖。**缺一即回归**。
2. **`force=True` 只在 run_chart_plan 工具内部传**，绝不暴露给 `seal_chart_maker_handoff` 工具签名——否则 LLM 学会传 force 绕门（reward hacking）。grep 确认 `seal_chart_maker_handoff` 工具定义无 force 参数。
3. **auto-seal 路径核对**：`executor.py:_attempt_auto_seal_from_artifacts` 调 `_seal_handoff_to_workspace` 时**不传 force**（走默认 False）。若 run_chart_plan 已封存，auto-seal 撞门拒绝——这是对的（auto-seal 是 chart-maker 漏调 run_chart_plan 时的兜底，已封存就不该兜底）。**但要确认 auto-seal 的 ValueError 不会让 executor 判 FAILED 白等重派**（auto-seal 失败应静默跳过，不升级为 subagent 失败）。
4. **改共享 helper 跑全量**：`_seal_handoff_to_workspace` 是 4 个 handoff 共用，加 `force` 参数 + chart-maker 分支门后，跑 `make test` 全量 + grep 所有调用方（code-executor/data-analyst/report-writer 的 seal 工具 + run_metric_plan + auto-seal），确认 `force` 默认值不影响其余 3 个 handoff（它们 model_cls 不是 ChartMakerHandoff，门不触发）。
5. **chart_meta 形态改 list**：M2 把 chart_meta 从 dict 改 list，Step 5/7/8 所有访问点同步改。grep `chart_meta` 全改齐（守 catastrophic forgetting）。
6. **manifest 一致性**：`_seal_handoff_to_workspace` 写 manifest（[L724](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py#L724)）。M1 拒绝覆盖发生在 manifest 写之前（raise 在校验前），manifest 不会被污染。但 `force=True` 覆盖时 manifest 正常更新（sha256 变）。确认 `_update_manifest` 对覆盖场景幂等。
7. **不改 run_metric_plan**：R2 是 run_chart_plan 独有（metric plan 的 results key 机制若有同问题另查，但本 spec 不扩范围——守"单 PR 单根因"）。

---

## 六、实施步骤（建议顺序）

1. **TDD 红**：先写 T1+T4，确认当前代码红（T1 手调覆盖成功=红；T4 同 id 只核 1 张=红）。
2. **M2 先于 M1**：先改 results key（M2）+ chart_meta list + Step 7 循环，跑 T4/T5/T6 绿。**理由**：M2 是纯计数逻辑改动，风险低；先修保证 run_chart_plan 正路径全核盘。
3. **M1**：改 `_seal_handoff_to_workspace` 加 force + 门，改 run_chart_plan 调用处传 force=True。跑 T1/T2/T3 绿。
4. **M3**：改 chart_maker.py step 9 文案。跑 `test_chart_maker_skill.py` / `test_chart_maker_system_prompt_budget.py` 确认 prompt 断言不破。
5. **全量**：`make test` + 裸导入两入口（T7）+ grep 所有 `_seal_handoff_to_workspace` / `chart_meta` 调用方。
6. **dogfood 验收**：重跑 EPM dogfood（核进程加载新代码：`make stop && make dev` + `inspect.getsource` 验运行时模块含 M1 门），验收 handoff `sealed_by=run_plan`（**不是 model**）+ chart_files=112 + 核盘不靠 LLM 自报。

---

## 七、milestone 建议

归入「chart-maker 确定性化 / reward hacking 治理」track。本 spec checkpoint：「EPM dogfood（thread a6e3775c）112/112 画图成功后取证，**推翻 problem doc 的"核盘竞态"归因**（代码 shutdown(wait=True) 已在，竞态不成立）→ 坐实真根因=chart-maker 手调 seal_chart_maker_handoff 覆盖 run_chart_plan 的 run_plan 封存（sealed_by 退回 model）+ 顺带坐实 results dict id 多对一覆盖计数 bug。M1 封存只允许一次结构门 + M2 results key 唯一化 + M3 prompt 强化。**纠正了 dogfood handoff §5 与 problem doc 两处误诊**（"良性竞态"实为"封存被覆盖"，"n_rendered=4 竞态"实为"chart-maker 自核盘 LLM 自述"）。」

**对 dogfood handoff / problem doc 的勘误建议**（接手 agent 可顺手补，或留待文档治理）：
- `2026-06-24-epm-dogfood-chart-success-askviz-gate-handoff.md` §5「chart-maker 读盘竞态误判（良性）」→ 实为「chart-maker 手调 seal 覆盖 run_plan 封存（R1），sealed_by=model 是病非善；n_rendered=4 是 chart-maker 自核盘 LLM 自述非工具竞态」。
- `2026-06-24-chart-maker-run-chart-plan-disk-race-false-partial.md` 全文「核盘时机竞态」→ 根因不成立（shutdown(wait=True) 已在），应标注「已由 2026-06-25 spec 推翻，真根因见 double-seal spec」。
