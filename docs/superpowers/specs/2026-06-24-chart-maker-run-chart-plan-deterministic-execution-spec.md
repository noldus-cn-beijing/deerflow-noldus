# Spec：chart-maker 执行确定性化 —— 新增 `run_chart_plan` 工具（对标 `run_metric_plan`）

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-24
> 代码基线：dev HEAD `f1a98458`
> 性质：🟡 中 · 架构对齐 + reward hacking 结构防御。把 chart-maker 从「计划确定性（prep_chart_plan）但**执行靠 LLM bash 编排**」升级为「计划 + 执行**全确定性**（prep_chart_plan + run_chart_plan）」，对齐 code-executor 的 `run_metric_plan` 范式。
> **方向（用户拍板 2026-06-24）**：给 chart-maker 加 `run_chart_plan` 确定性工具——LLM 调**一次** → 工具内 `ProcessPoolExecutor` 进程内并行画完 plan_charts.json 全部图 → 确定性核磁盘落盘 + 构造 handoff + 返回真实计数。LLM **0 次 bash 画图**。
> 受保护文件（sync surgical）：`subagents/builtins/chart_maker.py`、`tools/tools.py`、`tools/builtins/__init__.py`、`subagents/handoff_schemas.py`、`agents/middlewares/seal_gate_middleware.py`。
> **范式归属**：本 spec 是「[确定性批量执行工具 pattern](2026-06-24-deterministic-batch-execution-tool-pattern.md)」的**第 2 个实例应用**（实例 1 = run_metric_plan）。骨架/红线/铁律见 pattern 文档；本 spec 只写 chart 专属部分 + 集成点。**实施前先读 pattern 文档 §八 checklist。**

---

## ⚠️ 〇、根因校正（取证坐实，守 ETHO-2 教训）

> 本 spec 来自 handoff `2026-06-24-chart-maker-run-chart-plan-batch-execution-handoff.md`。handoff 要求**先取证再写 spec**（守 memory `feedback_etho2_spec_misdiagnosed_infra_bug_already_schema_rejected`）。取证已完成，**推翻了 handoff 的「诱因」前提**，本章如实记录，避免把未坐实假设写进根因。

### handoff 原前提（**未坐实，本 spec 不据此立论**）
> 「112+ 张 per_subject 图撞 loop-detection 熔断（warn=30/hard=50）→ 图没画完 + 伪报 completed」。

### 取证铁证（thread `15c9805e` EPM 28-subject dogfood + 进程内实跑，2026-06-24）
1. **loop-detection freq 按 bash *调用次数*计，不计 bash 内 python 命令数**（[loop_detection_middleware.py:485](../../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py) `freq[freq_key]+=1` 每 tool_call 一次，bash 的 freq_key=`"bash"`）。chart-maker 按 system_prompt step 8 把 ~19 个 `python -m` 塞进**一条** `bash -c "...& ...& wait"`。
2. **实测**：113 图 → chart-maker 只调 **~6 次 bash** → freq["bash"]≈6 **≪ warn30/hard50**，结构上根本够不着熔断线。SSE 全文 **0 个** "Tool frequency hard limit reached"。
3. **产物**：outputs/ **113/113 png 全真存**；handoff status=completed、**sealed_by=model（LLM 诚实自封口）**、failed_charts=0。**无熔断、无伪完成**。
4. ETHO-10 spec 引的「另一个 run：0 png + 57 幻影路径」是**不同 run**，且该 spec line 39 自己标注 loop-detection 熔断是「**强假设需 dogfood 坐实**」——至今无 run 坐实。ETHO-10 真根因是 2.2 门只对账 aggregate、per_subject 漏画无门可拦（另一条线，见关联 spec）。

### 本 spec 据实的立论（真价值，**不挂「治熔断诱因」**）
| # | 真价值 | 依据 |
|---|---|---|
| V1 | **执行确定性化**：消除 LLM bash 编排脆弱——LLM 手拼 `python -m <script> <args 拼接>` 易漏 `--parameters-json`（FewZones 列对齐参数），**dogfood thread 339512dd 真栽过**（bar 图第二批失败靠第三次重试才救活，见 [chart_maker SKILL.md:34](../../../packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md)）。工具直接透传 plan 的 `args` 数组，零重拼。 | SKILL.md:34 注释 + memory `feedback_chart_requires_columns_gate_distinct_from_zone_alias_overrides` |
| V2 | **产物经确定性工具不经 LLM** → `chart_files` 天生是磁盘真相（工具画完即知哪些 png 真落盘）→ **顺带堵 ETHO-10 reward hacking**（LLM 不再碰 chart_files，没法伪报）。与 ETHO-10 spec 协同（见 §七）。 | HarnessX 第 1 病理（验收看真产物）；ETHO-10 spec |
| V3 | **架构对齐 code-executor**：code-executor 早把批量执行收进 `run_metric_plan`（确定性聚合 + 进程内并行 + 消除 subprocess 冷启），chart-maker 是「计划确定性但执行半套」。对齐后两者范式一致，降维护成本。 | run_metric_plan_tool.py（spec S4）|
| V4（潜在风险，顺带消除，**非当前已发生的病**）| 若未来 LLM **退化成逐张 bash**（113 次 bash 调用而非批量），**才会**撞 loop-detection hard50。run_chart_plan 让 bash 次数与图数彻底解耦（图数再多 LLM 也只调一次）→ 该潜在风险一并消除。**但这是预防性收益，不是当前坐实的病。** | 取证 §〇 + loop_detection freq 机制 |

---

## 一、给实施 agent 的一句话

照 [`run_metric_plan_tool.py`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_metric_plan_tool.py) 范本，加 `run_chart_plan` 工具：读 workspace 的 `plan_charts.json`（无需 LLM 传图清单，像 run_metric_plan 读 plan_metrics.json）→ `ProcessPoolExecutor` 进程内并行 `importlib.import_module(chart["script"]).main(chart["args"])` 画全部图（worker initializer 设 `DEERFLOW_PATH_*` env + `MPLBACKEND=Agg`）→ 每个核 `chart["output"]` 的 png 真落盘（磁盘真相）→ 经 `_seal_handoff_to_workspace(ChartMakerHandoff, ...)` 构造 handoff（`sealed_by` 加新枚举值 `run_plan`）→ 返回紧凑结果（rendered/failed 计数，明细落盘不进 context）。chart-maker prompt step 8「LLM 拼 bash 并行画图」→「调一次 run_chart_plan」（像 code-executor step 2）。**装配链三处缺一不可**（§六）。

---

## 二、取证坐实的可行性（全部跑了代码，非推断）

### 范本可抄性（run_metric_plan_tool.py）—— ✅
骨架直接套用（行号为 run_metric_plan_tool.py）：
- `_available_cpus()`(L33) + `MAX_WORKERS`(L64)：进程池并行度 = 实际可用核（尊重 cgroup），照抄。
- `_worker_init(path_env)`(L77)：ProcessPoolExecutor initializer，在**每个 worker 进程内** `os.environ.update(path_env)`（不污染父进程，Gateway 多线程安全）——照抄，**额外加 `os.environ.setdefault("MPLBACKEND", "Agg")`**（见下 matplotlib 坑）。
- `_run_metric_task(script, args, task_id)`(L116)：`importlib.import_module(script).main(args)` 进程内调，吞一切异常逐个记失败不杀池——**run_chart_plan 的 worker 完全同构**（plot 脚本同样 `def main(argv)->int`）。
- `_execute_tasks(..., runner=None, path_env=)`(L467)：per-task 超时 + on_error + `_TASK_RUNNER_OVERRIDE` 测试钩子注入串行 runner（绕 fork/pickle）——照抄。
- `_seal_handoff_to_workspace(...)`(seal_handoff_tools.py:535) + 紧凑返回——照抄结构。

### 绘图脚本进程内可调 —— ✅（进程内实跑 4 张 rc=0 真出 png）
- **26 个 plot 脚本全有 `def main(argv: list[str] | None = None) -> int`**（与 run_metric_plan 期望的 `mod.main(args)` **同签名**）。grep 全覆盖。
- **进程内实跑坐实**（用 thread 15c9805e 真实 plan + DEERFLOW_PATH env）：
  | 脚本 | 类型 | 结果 |
  |---|---|---|
  | `epm.plot_box_open_arm`（aggregate，经 charts.py）| box | rc=0，84519B png ✅ |
  | `epm.plot_open_arm_time_ratio_bar`（per_subject，**裸 import pyplot**）| bar | rc=0，19974B png ✅ |
  | `_common.plot_heatmap`（per_subject，经 charts.py）| heatmap | rc=0，53402B png ✅ |
  | `_common.plot_trajectory`（per_subject，经 charts.py）| trajectory | rc=0，177116B png ✅ |

### plan_charts.json 契约够用 —— ✅
thread 15c9805e 真实 plan_charts.json：**113 条 charts[] 全含** `id`/`script`/`args`/`output`/`output_mode`/`subject_index`（113/113 字段齐全）：
```json
{
  "id": "box_open_arm",
  "script": "ethoinsight.scripts.epm.plot_box_open_arm",   // importlib 可直调
  "output": "/mnt/user-data/outputs/plot_box_open_arm.png", // exists() 核磁盘
  "output_mode": "aggregate",                               // status 派生 + reconcile 2.2 门
  "args": ["--inputs","...","--output","...","--parameters-json","{\"open_arm_zones\": [\"open\"]}"]
                                                            // 完整 argv，直接喂 main()
}
```
`run_chart_plan` 读 plan → 遍历 charts[] → `import_module(c["script"]).main(c["args"])` → 核 `c["output"]` png 落盘。**零新路径逻辑。**

### ⚠️ matplotlib backend 坑（spec 必须处理）—— 已坐实
- `ethoinsight/charts.py:12-14` **顶层已 `matplotlib.use("Agg")`**（box/heatmap/trajectory/timeseries 经它，安全）。
- **但 5 个脚本裸 `import matplotlib.pyplot` 不经 charts.py、不设 Agg**：`epm/plot_open_arm_time_ratio_bar`、`epm/plot_zone_entry_distribution`、`ldb/plot_zone_entry_distribution`、`oft/plot_center_entry_summary`、`oft/plot_center_time_ratio_bar`。
- 本机默认 backend 已是 `agg`（无 `$DISPLAY`），故**当前侥幸不炸**；但进程池 worker 从 Gateway 父进程 fork，**若父进程曾初始化非 Agg backend**（某部署有 `$DISPLAY` / 其他工具先 touch pyplot），fork 的 worker 继承交互式 backend → 非主线程 `savefig` 崩。
- **处方**：worker initializer 设 `os.environ.setdefault("MPLBACKEND", "Agg")`（在任何 `import matplotlib.pyplot` 之前生效，因 worker 是 fresh 子进程入口）。免费保险，对 5 脚本 + 父进程状态无关地确定性安全。

### 路径解析 env 键名（易错点，已坐实）—— ✅
- plot 脚本读 `--inputs`/`--output` 内路径靠 `resolve_sandbox_path`（`_cli.py:81`）查 `DEERFLOW_PATH_*` env。
- **键名含 `MNT`**：`prefix.strip("/")` 保留 `mnt` → `/mnt/user-data/workspace` 的 env key = **`DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE`**（不是 `DEERFLOW_PATH_USER_DATA_WORKSPACE`）。**直接用 `_build_path_env(thread_data)`（sandbox/tools.py:562）生成，别手拼键名。** run_metric_plan 就这么用的。

---

## 三、方案设计（run_chart_plan，对标 run_metric_plan）

### 签名
```python
@tool("run_chart_plan", parse_docstring=True)
def run_chart_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    plan_path: str = "/mnt/user-data/workspace/plan_charts.json",
    only_chart_ids: list[str] | None = None,   # 重跑子集（对标 only_metric_ids）；None=全画
    on_error: str = "continue",                # "continue"=遇失败继续 / "abort"=遇首失停
) -> dict[str, Any]:
```
**红线**：LLM 在工具边界只传「哪些」(`only_chart_ids`) 和「怎么对待」(`on_error`)，**永不传「是什么」(args)**——内容从 plan artifact 读（守 run_metric_plan 同款红线）。

### 内部流程（步骤对齐 run_metric_plan）
1. **resolve workspace** from `thread_data.workspace_path`（缺失 → `_error_result("workspace_missing")`）。
2. **read plan**：`replace_virtual_path(plan_path)` → 读 plan_charts.json。缺失/解析失败 → **seal failed**（`_seal_failed(workspace, ChartMakerHandoff, ...)`）+ error result（对齐 run_metric_plan plan-missing 路径）。
3. **apply `only_chart_ids` selector**：过滤 charts[]（重跑子集用）。过滤后空 → seal failed。
4. **build path_env**：`_build_path_env(thread_data)`（worker resolve /mnt 用 + worker 内设 `MPLBACKEND=Agg`）。
5. **build task list**：遍历 charts[]，每条 `(script, args, chart_id)`；缺 script/args 的 entry 是 plan bug，空 script → worker 立即 ImportError 失败（响亮，对齐 run_metric_plan §3 迁移风险 1）。
6. **execute via ProcessPoolExecutor**：`_execute_tasks(tasks, on_error, timeout, runner=_TASK_RUNNER_OVERRIDE, path_env)`。
   - **per-task timeout**：绘图比 compute 轻，但保守取 `PER_TASK_TIMEOUT_SECONDS = 120`（与 run_metric_plan 同值，不魔法常数）。
7. **核磁盘 + 构造 chart_files / failed_charts**（**这是 run_chart_plan 区别于 run_metric_plan 的核心：直接核 output png 落盘**）：
   - 对每个 chart：`output` 虚拟路径 → `replace_virtual_path` 物理路径 → `exists()`。
   - 落盘 → 进 `chart_files`（虚拟路径，schema 要求 `/mnt/user-data/outputs/` 前缀）。
   - 未落盘（rc≠0 或 rc=0 但 png 不存在=脚本半成功）→ 进 `failed_charts`（`{chart_id, reason: "<rc/stderr 摘要 或 'rendered rc=0 but png missing'>"}`）。
   - `remaining_charts`：plan 的 `charts_budget_remaining[]` 透传（被 chart_budget 截断未画的 per_subject 指纹，对齐现有语义）。
8. **derive status**（纯函数，对齐 `_derive_status`）：
   - 全部 chart 落盘 → `completed`。
   - 部分落盘 → `partial`。
   - **aggregate 图有未落盘**（组间对比 must_have）→ 即使有别的图也降 `partial`（保守；2.2 reconcile 门会进一步把 completed+缺 aggregate 拦成 ValueError，双保险）。
   - 0 落盘 → `failed`。
9. **assemble + seal handoff**（确定性，`sealed_by="run_plan"`）：
   - payload：`{status, paradigm（从 plan.paradigm 或 handoff_code_executor.json）, chart_files, failed_charts, remaining_charts, summary（机械默认）, gate_signals, sealed_by:"run_plan"}`。
   - `_seal_handoff_to_workspace(ChartMakerHandoff, "handoff_chart_maker.json", payload, workspace)` —— **自动过 `_reconcile_chart_maker_payload` 对账门**（单一注入点 seal_handoff_tools.py:562）。因 chart_files 已是磁盘真相，2.2 aggregate 门干净通过。
   - seal schema 校验失败 → `_error_result("seal_failed")`（回归探针）。
10. **return 紧凑结果**：
    ```python
    {"status": ..., "handoff_path": "/mnt/user-data/workspace/handoff_chart_maker.json",
     "n_total": int, "n_rendered": int, "n_failed": int,
     "failures": [{"chart_id","reason"}],  # 仅失败明细摘要
     "gate_signals": {...}}
    ```

### `sealed_by` 枚举扩展（必改 schema）
`ChartMakerHandoff.sealed_by`（handoff_schemas.py:589）当前 `Literal["model", "after_agent_artifacts", "executor_artifacts"]` —— **加 `"run_plan"`**（对齐 CodeExecutorHandoff 的 `sealed_by="run_plan"` 语义：确定性工具封存）。不加会 Pydantic 校验失败。

### gate_signals
复用 chart-maker gate_signals 形态（charts_generated / failed_charts 等）；run_chart_plan 据真实落盘计数确定性构造（不靠 LLM 自报）。

---

## 四、chart-maker prompt 改动（三指令源对齐，守 memory `feedback_subagent_system_prompt_higher_authority_than_skill`）

> ⚠️ 改 chart-maker 行为**必须改 `chart_maker.py` 的 `SubagentConfig.system_prompt`（最高权威，常驻 context）**，光改 SKILL.md = 白改。两处都要改，且对齐。

### 改动 1：`chart_maker.py` system_prompt
- **step 8「执行绘图脚本（LLM 拼 bash 并行）」→「调一次 `run_chart_plan`」**：像 code-executor step 2「调一次 run_metric_plan」。删掉 `bash -c "...& ...& wait"` 并行模板 + 逐个串行重试逻辑 + bash 失败排查语义（这些都收进工具了）。
- **step 9 seal**：run_chart_plan 内部已 seal（`sealed_by="run_plan"`）→ chart-maker **不再自调 seal_chart_maker_handoff**（避免双 seal）。改为「run_chart_plan 返回后，据其 result 输出 [gate_signals] + present_files」。
  - ✅ **SealGate 交互（已取证坐实，非开放问题）**：chart-maker 当前在 `SealGateMiddleware._REQUIRES_SEAL`（seal_gate_middleware.py:L85）。**code-executor 早已不在该集合**——docstring 明写「code-executor is intentionally excluded: run_metric_plan already produces-and-delivers in one tool (structurally cannot miss seal — verified across many dogfoods)」。**run_chart_plan 对 chart-maker 是同构的产出+交付合一**，因此 **§四改动 3 = 把 `"chart-maker"` 从 `_REQUIRES_SEAL` 和 `_RECONSTRUCTABLE` 两个集合移除**（precedented 一行集合编辑，非新机制）。移除后 SealGate after_model 对 chart-maker 走 rule 1「Not a seal-requiring subagent → return None」直接放行，after_agent auto-seal 也不再需要（run_chart_plan 已确定性保证 handoff，同 code-executor 不在 `_RECONSTRUCTABLE`）。
- **`<bash_constraints>`**：bash 不再用于画图。保留 ls/cp/mv/mkdir（文件操作仍可能需要）；移除「绘图脚本 python -m」+ 并行模板说明。或**直接从 chart-maker tools 移除 bash**（见下「工具集」决策）。
- **`<workflow>` step 5 prep_chart_plan 不变**（计划生成仍走它）；新增 step「调 run_chart_plan 执行」。

### 改动 2：`ethoinsight-chart-maker/SKILL.md`
- step 5「for each entry ... bash 跑脚本」→「调一次 run_chart_plan」。
- 「失败硬规范」表的「bash 预算还剩 ≤ 2 次」「某 chart 脚本连续失败 ≥ 2 次」等 bash 语义 → 改为「run_chart_plan 返回 failed_charts/partial 时如何向 lead 报」。
- ⚠️ **注意当前 SKILL.md 是旧串行框架**（step 5「for each entry」），与 system_prompt 的并行批已分歧；本次一并对齐到 run_chart_plan。

### 改动 3：`seal_gate_middleware.py` —— chart-maker 移出 SealGate 集合（precedented，对齐 code-executor）
- **`_REQUIRES_SEAL`（L85）移除 `"chart-maker"`** → `frozenset({"data-analyst", "report-writer"})`。理由：run_chart_plan 产出+交付合一（确定性落 handoff），chart-maker 不再有「漏调 seal」的可能——与 code-executor 用 run_metric_plan 后被移出 `_REQUIRES_SEAL` **完全同构**（docstring L82-84 明载该排除理由）。
- **`_RECONSTRUCTABLE`（L99）移除 `"chart-maker"`** → `frozenset({"report-writer"})`。理由：after_agent auto-seal 是为「L1 reminder-cap 放行后兜底从 plot_*.png 重建」设计的；run_chart_plan 已确定性保证 handoff，无需该兜底（同 code-executor 不在 `_RECONSTRUCTABLE`）。
- ⚠️ **保留 report-writer 在两集合**（它仍走 LLM 写报告 + 自调 seal_report_writer_handoff，未确定性化）。**只动 chart-maker。**
- 受保护文件 sync surgical：seal_gate_middleware.py（含 Noldus 定制集合）。

### 工具集决策（chart_maker.py `tools=[...]`）
- **加** `run_chart_plan`。
- **bash 去留**（二选一，实施时定）：
  - **方案 A（激进，推荐）**：从 chart-maker `tools` 移除 `bash`（画图全走 run_chart_plan，文件操作 run_chart_plan 内部做或不需要）。彻底消除 LLM bash 画图可能 + 简化 guardrail。**风险**：若有边缘场景 chart-maker 需 ls/mv（如检查 outputs），移除 bash 会断。需确认无此依赖。
  - **方案 B（保守）**：保留 bash 但 `<bash_constraints>` 收窄到只 ls/cp/mv/mkdir（移除 python -m 画图）。prompt 强引导「画图只调 run_chart_plan」。
  - 实施时先按 **B** 落地（最小改动、可回退），dogfood 验证 chart-maker 不再用 bash 画图后，后续 PR 可收紧到 A。

---

## 五、测试清单（TDD 红→绿，对齐 run_metric_plan 测试 + memory `feedback_processpoolexecutor_test_runner_injection_and_ssot_parity`）

> ProcessPoolExecutor 测试：注入 `_TASK_RUNNER_OVERRIDE` 同步串行 runner（绕 fork/pickle），mock 出脚本行为（写/不写 png）。

1. **T1 全画成功**：plan 3 charts（1 aggregate + 2 per_subject），runner 全成功写 png → status=completed，chart_files=3，n_rendered=3，handoff sealed_by="run_plan"。
2. **T2 部分失败**：runner 让 1 个 per_subject 不写 png（rc=1）→ status=partial，chart_files=2，failed_charts=1（reason 含机读摘要），n_failed=1。
3. **T3 aggregate 缺失降级**：runner 让唯一 aggregate 不写 png（per_subject 全成功）→ status=partial（aggregate must_have 缺失即使 per_subject 全画也不 completed）；且 `_reconcile_chart_maker_payload` 2.2 门对 completed 才抛——这里 status 已是 partial 不抛，但断言 status 正确降级。
4. **T4 only_chart_ids 子集**：传 `only_chart_ids=["box_open_arm"]` → 只跑该 chart，n_total=1。
5. **T5 plan 缺失**：plan_charts.json 不存在 → seal failed + error_result("plan_missing")，磁盘有 handoff_chart_maker.json（status=failed）。
6. **T6 plan 空 charts[]**：charts=[] → seal failed + error_result("empty_plan")。
7. **T7 rc=0 但 png 缺失**（脚本半成功）：runner 返回 rc=0 但不写 png → 该 chart 进 failed_charts（reason="rendered rc=0 but png missing"），不进 chart_files。
8. **T8 chart_files 磁盘真相 + reconcile 协同**：T1 的 handoff 过 `_reconcile_chart_maker_payload`——因 chart_files 全真存盘（test 用真实 tmp png），2.2 aggregate 门通过、不抛。断言 run_chart_plan 落的 handoff 与 ETHO-10 不变式协同（产物经工具天生过门）。
9. **T9 sealed_by 枚举**：handoff sealed_by="run_plan" 通过 ChartMakerHandoff schema 校验（需先加枚举值，否则红）。
10. **T10 on_error=abort**：首个 chart 失败 → 取消后续，failures 含 aborted 标记。
11. **T11 MPLBACKEND**：worker initializer 设 MPLBACKEND=Agg（单测断言 `_worker_init` 后 `os.environ["MPLBACKEND"]=="Agg"`；或集成测试真画一张 5-脚本之一的 bar 图确认不炸）。
12. **T12 import 环**（#187 + memory `feedback_tool_definition_export_not_equal_registered_in_builtin_tools`）：改完裸导入 `app.gateway` + `make_lead_agent` 0 退出。
13. **T13 装配链可见性**（#187 教训核心）：`get_available_tools()` 返回的工具 name 列表**含 `run_chart_plan`**（不能只测 @tool 定义/导出；#187 就栽在工具有定义+导出+prompt 教但没注册 BUILTIN_TOOLS → get_available_tools 不含 → 工具悬空）。
14. **T14 SealGate 放行**（回归，解法已坐实见 §四改动 3）：chart-maker 移出 `_REQUIRES_SEAL` 后，构造一个「chart-maker 终止于纯文本、未调 seal_chart_maker_handoff」的 state，断言 SealGate after_model **返回 None 放行**（rule 1）。即 run_chart_plan 落 handoff 后 chart-maker 可正常终止，不被卡死。同时断言 data-analyst / report-writer 仍在 `_REQUIRES_SEAL`（不误删）。

---

## 六、装配链（三处缺一不可，守 #187 血泪教训 memory `feedback_tool_definition_export_not_equal_registered_in_builtin_tools`）

> #187 把 fill/finalize 工具有 @tool 定义 + __init__ 导出 + prompt 教 + SealGate 催，**唯独漏注册 tools.py:BUILTIN_TOOLS** → get_available_tools 不含 → data-analyst 拿不到 → 生产 100% FAILED。run_chart_plan **三处全做**：

1. **`tools/builtins/run_chart_plan_tool.py`**：新文件，`@tool("run_chart_plan")` 定义。
2. **`tools/builtins/__init__.py`**：`from .run_chart_plan_tool import run_chart_plan_tool` + 加进 `__all__`（对齐 L8/L35 run_metric_plan_tool 的两处）。
3. **`tools/tools.py`**：
   - import 块加 `run_chart_plan_tool`（对齐 L21）。
   - **`BUILTIN_TOOLS` 列表加 `run_chart_plan_tool`**（对齐 L43，**这处是 #187 漏的**）。
4. **chart_maker.py `tools=[...]`**：加 `"run_chart_plan"`（subagent 工具白名单——BUILTIN_TOOLS 注册让工具**存在**，subagent tools 列表让 chart-maker **拿得到**，两者都要）。

> 注：装配链是「工具能被 chart-maker 调用」的四处。另有两处**非装配链但同样必改**（功能正确性）：① `handoff_schemas.py` 加 `sealed_by="run_plan"` 枚举（§三）；② `seal_gate_middleware.py` 把 chart-maker 移出两集合（§四改动 3）。漏前者 → seal schema 校验失败；漏后者 → chart-maker 被 SealGate 卡死。

**改完验证（裸导入 + 工具可见性双验）**：
```bash
cd packages/agent/backend
PYTHONPATH=. python -c "import app.gateway"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
# 工具可见性（#187 教训）：dump chart-maker executor.tools name 确认含 run_chart_plan
```

---

## 七、与 ETHO-10 spec 的协同（不冲突，正交，建议本线后合）

- **ETHO-10 spec**（`2026-06-24-etho10-chart-maker-product-reality-invariant-spec.md`，别的 agent 在做）：改 `_reconcile_chart_maker_payload` 加 chart_files 磁盘真实性核对（封存核 exists()，幻影路径剔除）。
- **本线**：改 chart-maker 执行路径 + 加 run_chart_plan 工具。
- **协同**：run_chart_plan 落地后，chart_files 天生是磁盘真相（工具画完即知）→ ETHO-10 的「封存核磁盘门」从**主防御**降为**双保险**（产物已不经 LLM，没法伪报）。**两 spec 不冲突**——ETHO-10 改 seal 对账门（防御层），本线改执行路径（消除诱因层）。
- **合并顺序建议**：**ETHO-10 先合**（让磁盘真实性门先在，作为兜底），**本线后合**（run_chart_plan 消除 LLM 伪报可能）。本线的 handoff 经 `_seal_handoff_to_workspace` 自动过 ETHO-10 改造后的门，零冲突。
- ⚠️ 实施时若 ETHO-10 已合：本线 T8 直接验证「run_chart_plan 落的 handoff 过 ETHO-10 真实性核对干净通过」。若 ETHO-10 未合：T8 验证现有 2.2 aggregate 门通过即可。

### 实施期接触点（与正在并行的两个 PR，文件级核实）
> ETHO-10 spec + fill-race spec 正由别的 agent 并行实施。文件级冲突矩阵核实如下：

| 文件 | 本 spec（run_chart_plan）| ETHO-10 | fill-race | 冲突？|
|---|---|---|---|---|
| `seal_handoff_tools.py` | ❌ 不改函数体（只**调用** `_seal_handoff_to_workspace`/`_reconcile_chart_maker_payload`）| ✅ 改 `_reconcile_chart_maker_payload`(426) | ✅ 改 fill×3/finalize(867-986)+加模块级锁 | **无**（三方各改不同区/不改）|
| `handoff_schemas.py` | ✅ 加 `sealed_by` 枚举值(589) | ✅ 可能改 `_completed_requires_core_output`(600)/`_validate_chart_paths`(615) | ❌ | ⚠️ **同类不同行**（接触点 1）|
| `seal_gate_middleware.py` | ✅ 移除 chart-maker 出两集合 | ❌ | ❌ | **无** |
| `chart_maker.py`/`tools.py`/`__init__.py` | ✅ | ❌ | ❌ | **无** |

**接触点 1（`handoff_schemas.py` 的 `ChartMakerHandoff`）**：本 spec 加 `sealed_by` 枚举值 `"run_plan"`（L589），ETHO-10 可能改同类的 `_completed_requires_core_output`/`_validate_chart_paths`（不同行）。git 层大概率不冲突，但**谁后合谁 rebase 时确认另一方改动还在**（catastrophic forgetting 自检：grep `ChartMakerHandoff` 全字段）。

**接触点 2（`_reconcile_chart_maker_payload` 语义依赖，协同非冲突）**：本 spec **不改**该函数，但**经它落 handoff**。ETHO-10 给它加 chart_files 磁盘真实性核对。run_chart_plan 的 chart_files 天生磁盘真相（工具画完即知）→ 过 ETHO-10 强化门**干净通过**（T8 验）。

**接触点 3（`_RECONSTRUCTABLE` vs ETHO-10 的 auto-seal T8）**：本 spec 移除 chart-maker 出 `_RECONSTRUCTABLE`（run_chart_plan 后不需 after_agent auto-seal）。ETHO-10 的 T8 假设「auto-seal-chart-maker 分支(executor.py:398)过同一门」。**若本 spec 先实施**，chart-maker 不再 auto-seal → ETHO-10 的 auto-seal-chart-maker T8 失效。**故本 spec 后合**（见合并顺序），ETHO-10 的 T8 先在；本 spec 合时在 PR 说明「chart-maker 移出 `_RECONSTRUCTABLE`，ETHO-10 auto-seal-chart-maker T8 改标 N/A 或迁到 report-writer」。

**fill-race spec：完全正交无冲突**（它全在 data-analyst 的 fill/finalize 800-990 区 + 模块级锁，本 spec 碰都不碰；chart-maker vs data-analyst 两条线）。

---

## 八、风险与三大病理自检（守 CLAUDE.md）

### 集成点（已取证坐实，precedented，非开放风险）
- **SealGate 放行 —— 解法已坐实**：chart-maker 当前在 `SealGateMiddleware._REQUIRES_SEAL`。**code-executor 早已不在该集合**（docstring 明载：run_metric_plan 产出+交付合一，结构上不会漏 seal）。run_chart_plan 对 chart-maker 同构 → **§四改动 3 把 chart-maker 移出 `_REQUIRES_SEAL` + `_RECONSTRUCTABLE`**，SealGate 走 rule 1 直接放行。**这是 precedented 集合编辑（一行），不是 deadlock 风险。** T14 守回归。
  - 取证依据：`seal_gate_middleware.py:L82-84` docstring「code-executor is intentionally excluded: run_metric_plan already produces-and-delivers in one tool」+ `_REQUIRES_SEAL`(L85) 当前不含 code-executor。
  - **不需要改 SealGate 判据逻辑**（after_model rule 1/2/3 不动）——只改集合成员。最小、可回退、有先例。

### 三大病理
1. **Reward hacking**：本方案正面防御——产物经确定性工具不经 LLM，chart_files 磁盘真相，LLM 无从伪报。**不加 prompt「别伪报」规则**（守 HarnessX Telecom 禁令）。
2. **Catastrophic forgetting**：改 chart_maker.py system_prompt 前 grep 所有 prompt 镜像（SKILL.md + input_contract + output_contract），全量对齐到 run_chart_plan，别只改一处留分歧（当前 system_prompt 并行批 vs SKILL.md 串行已分歧，本次一并修）。改 ChartMakerHandoff schema（加 sealed_by 枚举）前确认不破现有 model/after_agent_artifacts/executor_artifacts 三值 + `_completed_requires_core_output` / `_validate_chart_paths` 校验。
3. **Under-exploration**：本方案是**结构改动**（确定性工具批量执行），不是加 prompt 规则——正解（对齐 seal 漏调最终上 SealGateMiddleware 的教训：结构缺失→上结构，别打地鼠）。

### 其他铁律
- **确定性可测**：worker/tmp 用 pid/tid，不用 `Date.now()`/随机数。
- **绝不 crash**：读盘异常 → warning + 跳过；但「核心图全空 + completed」的响亮拒绝交给 reconcile 门（不在豁免内）。
- **受保护文件 sync surgical**：chart_maker.py / tools.py / __init__.py / handoff_schemas.py / seal_handoff_tools.py。
- **TDD 强制** + 端到端 dogfood 验收（同份 28-subject EPM 复跑：run_chart_plan 一次画完 113 张、handoff chart_files 全真实落盘、sealed_by="run_plan"、chart-maker 0 次 bash 画图、不被 SealGate 卡死、不伪报）。

---

## 九、关键代码锚点（已核实，行号 = dev f1a98458）

- **范本**：`tools/builtins/run_metric_plan_tool.py`（`_available_cpus`33 / `MAX_WORKERS`64 / `_worker_init`77 / `_run_metric_task`116 / `run_metric_plan_tool`145 / `_execute_tasks`467 / `_TASK_RUNNER_OVERRIDE`68 / `_derive_status`551 / `sealed_by="run_plan"`349 / 紧凑返回373）
- **上游**：`tools/builtins/prep_chart_plan_tool.py`（产 plan_charts.json）
- **绘图脚本**：`packages/ethoinsight/ethoinsight/scripts/{_common,<paradigm>}/plot_*.py`（26 个全 `def main(argv)->int`）；`ethoinsight/charts.py:12-14`（`matplotlib.use("Agg")`，box/heatmap/trajectory/timeseries 经它）；**裸 import pyplot 不设 Agg 的 5 脚本**：`epm/plot_open_arm_time_ratio_bar`、`epm/plot_zone_entry_distribution`、`ldb/plot_zone_entry_distribution`、`oft/plot_center_entry_summary`、`oft/plot_center_time_ratio_bar`
- **路径 env**：`sandbox/tools.py:_build_path_env`(562)、`replace_virtual_path`(491)；`ethoinsight/scripts/_cli.py:resolve_sandbox_path`(81)、键名 `DEERFLOW_PATH_MNT_USER_DATA_*`（含 MNT）
- **seal/reconcile**：`tools/builtins/seal_handoff_tools.py:_seal_handoff_to_workspace`(535)、`_reconcile_chart_maker_payload`(426，单一注入点 562)、`seal_chart_maker_handoff`(1031)、`_load_plan_charts`(407)、`_outputs_dir_for`(421)、`_seal_failed` 模式（run_metric_plan_tool.py:616）
- **schema**：`subagents/handoff_schemas.py:ChartMakerHandoff`(559)、`sealed_by` 枚举(589，**加 run_plan**)、`_completed_requires_core_output`(602)、`_validate_chart_paths`(615)
- **chart-maker 现状**：`subagents/builtins/chart_maker.py`（system_prompt step 8 bash 画图 98-107 + 62-72 并行模板 / step 9 seal / tools 166 / SealGate 受管）；`packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md`（旧串行框架 step 5 / 失败硬规范 40-50）
- **loop-detection**（取证锚点）：`agents/middlewares/loop_detection_middleware.py:485`（freq 按 tool type 计）；`subagents/executor.py:973`（warn30/hard50）
- **SealGate**（集成风险）：`agents/middlewares/seal_gate_middleware.py`；`subagents/executor.py:987`（挂 SealGate）
- **装配链**（#187）：`tools/tools.py:BUILTIN_TOOLS`(34，**run_chart_plan 必须加 43 后**)、`tools/builtins/__init__.py`(8/35)
- **executor auto-seal**：`subagents/executor.py:398`（chart-maker glob plot_*.png 已真实，过同一 reconcile 门）
- **取证证据**：thread `15c9805e-f01b-4a4c-9a94-5cd2dd09e4c4`（user `e281f251`，workspace 在 `.deer-flow/users/.../user-data/`）；`/tmp/pw-driver/evidence-2026-06-24/` + `analyze-2026-06-24.py`；进程内实跑验证（4 脚本 rc=0）

---

## milestone 建议
「chart-maker 架构对齐 code-executor / 批量执行确定性化」track：chart-maker 从「计划确定性（prep_chart_plan）但执行靠 LLM bash」升级为「计划 + 执行全确定性（prep_chart_plan + run_chart_plan）」，对齐 code-executor 的 run_metric_plan 范式。checkpoint：「dogfood 取证**推翻**『loop-detection 熔断诱因』前提（thread 15c9805e 画全 113 张、~6 次 bash 未熔断、sealed_by=model）→ 据实改根因（真价值=执行确定性化消除 args 重拼脆弱 + 产物经工具堵 ETHO-10 + 架构对齐）→ run_chart_plan 确定性工具 spec 已写（可行性进程内实跑坐实）」。与 ETHO-10 spec 协同（本线消除诱因层、ETHO-10 守防御层）。
