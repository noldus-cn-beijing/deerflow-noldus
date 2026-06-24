# Pattern：确定性批量执行工具（Deterministic Batch-Execution Tool）

> 类型：**架构范式文档（reusable pattern）**，非一次性 spec。
> 日期：2026-06-24
> 适用层：DeerFlow harness（`packages/agent/backend/packages/harness/deerflow/tools/builtins/`）。
> 一句话：当一个 subagent 需要「跑一批彼此独立的 ethoinsight 脚本」时，**不要让 LLM 在 bash 里编排**（逐条/并行拼 `python -m ...`），而是给它一个**确定性工具**——LLM 调一次 → 工具内 `ProcessPoolExecutor` 进程内并行跑完 + 确定性落盘 handoff + 返回真实计数。LLM **0 次 bash**。
>
> **已落地实例**：`run_metric_plan`（code-executor，2026-06-12 Spec S4）。
> **在写实例**：`run_chart_plan`（chart-maker，[2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md](2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md)）。
> **未来候选**：任何「subagent 批量跑独立脚本」的需求（如批量重算子集、批量导出、批量校验）。

---

## 〇、为什么固化成范式（不只是「又一个工具」）

这是 DeerFlow harness 里**反复出现的范式转换**，不是孤立功能。每次「让 LLM 用 bash 编排批量执行」都会反复踩同一组坑：

1. **执行脆弱**：LLM 手拼 `python -m <script> <args>` 易漏参数（如 run_metric_plan 前身漏 `--parameters-json` 致列对齐丢失；chart-maker dogfood thread 339512dd bar 图第二批失败靠第三次重试才救活）。
2. **bash 调用次数与任务数耦合**：逐张 bash → 任务多时撞 LoopDetectionMiddleware 频率熔断（warn=30/hard=50，按 **tool 调用次数**计）；即使批量也是 LLM 凭运气拼对 `& ... wait`。
3. **产物经 LLM → reward hacking 温床**：LLM 自报「我跑了这些」可能与磁盘真相不符（chart-maker 伪报 chart_files，ETHO-10）。
4. **subprocess 冷启开销**：每条 `python -m` 起一个解释器进程，N 条 = N 次冷启。
5. **聚合/校验靠 LLM 在 bash 间手动串**：脆弱、不确定。

确定性批量执行工具**一刀治这 5 类**：执行确定性化（工具透传 plan 的 args，零重拼）+ bash 次数与任务数解耦（LLM 调一次，任务 4000 个也只调一次）+ 产物经工具不经 LLM（落盘即磁盘真相）+ 进程内 `importlib` 调（无 subprocess 冷启）+ 聚合/校验在工具内确定性跑。

> 对齐 CLAUDE.md「**用确定性结构约束行为，不用 prompt 规则**」（"LLM 提议，确定性门定生死"）+ HarnessX 三病理 under-exploration 反面（结构改动，不是加 prompt 规则）。

---

## 一、范式判据：何时该用这个 pattern（何时不该）

### ✅ 该用（全部满足）
1. 一个 subagent 的核心工作是**跑一批脚本**（不是一次跑一个）。
2. 这批脚本**彼此独立**（无依赖，可并行；输出不同文件、输入不同文件）。
3. 脚本有**统一的进程内入口**：`def main(argv: list[str] | None = None) -> int`（ethoinsight `scripts/**/*.py` 全合规）。
4. **「跑哪些」「怎么对待失败」是 LLM 该决策的，「每条脚本的 args」不是**——args 来自一个 plan artifact（plan_metrics.json / plan_charts.json），LLM 只做指向不传内容（**红线**，见 §三）。
5. 完成度可由**磁盘产物**确定性判定（m_*.json 对账 / output png exists()），不靠 LLM 自述。

### ❌ 不该用
- 脚本间**有依赖**（A 的输出是 B 的输入）→ 需要编排，不是 fan-out。
- 任务是**单个**（跑一个脚本）→ 直接调，无需池。
- 「跑哪些 + 每条的 args」**都该 LLM 现场决策**（高度自由裁量）→ 不是「指向已有 plan」模型，本 pattern 的红线不成立。
- 脚本**没有进程内入口**（只有 `if __name__=="__main__"` 无 `main(argv)`）→ 先抽 `main(argv)`，或退回 subprocess（失去进程内优势，慎用）。

> ⚠️ **不跨范式复用判据**（守 memory `feedback_no_cross_paradigm_reuse_accept_duplication`）：本 pattern 是**结构范式**（怎么写一个 batch 工具），**不是**让 metric 工具和 chart 工具共享业务代码。每个工具的 handoff 组装 / 聚合 / 校验 / 完成度判定**各写各的**（指标专属 vs 图专属），即使结构相似也各存一份。共享的只有「进程池执行骨架」这层纯机械逻辑（见 §四分层）。

---

## 二、骨架模板（照抄结构，业务各填各的）

> 范本：`tools/builtins/run_metric_plan_tool.py`（673 行，dev f1a98458）。新工具照此结构，**业务逻辑各填各的**。

```python
"""run_<x>_plan — 一步确定性跑完 plan_<x>.json 的全部 <脚本>。"""
from __future__ import annotations
import contextlib, logging, os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any
from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT
from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)
PER_TASK_TIMEOUT_SECONDS = 120          # spec 锁定，不魔法常数

# ① 进程池并行度 = 实际可用核（尊重 cgroup，吃满 elastic 配额）
def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))   # Linux: 尊重 cgroup/affinity
    except AttributeError:
        return os.cpu_count() or 4
MAX_WORKERS = _available_cpus()

# ② 测试钩子：注入同步串行 runner 绕 ProcessPoolExecutor fork/pickle（生产恒 None）
_TASK_RUNNER_OVERRIDE = None

# ③ worker initializer — 在【每个 worker 子进程内】设 DEERFLOW_PATH_* env
#    （不在父进程 mutate 全局 os.environ：Gateway 多线程并发跑多 thread subagent，
#     全局 mutate 会让一个 thread 的 workspace 路径泄漏污染其他 thread）
def _worker_init(path_env: dict[str, str]) -> None:
    if path_env:
        os.environ.update(path_env)
    # ⚠️ chart 类工具额外：os.environ.setdefault("MPLBACKEND", "Agg")
    #    （5 个绘图脚本裸 import matplotlib.pyplot 不设 Agg，fork 的 worker 须兑底）

# ④ worker — 必须模块级（供 ProcessPoolExecutor pickle）+ 顶层无 deerflow/ethoinsight import
#    进程内调脚本：importlib.import_module(script).main(args)，无 subprocess 冷启
def _run_task(script: str, args: list[str], task_id: str) -> tuple[str, int, str]:
    import importlib
    try:
        mod = importlib.import_module(script)
        rc = mod.main(args)
        return (task_id, int(rc or 0), "")
    except SystemExit as e:                     # argparse 失败 / sys.exit
        return (task_id, int(e.code) if isinstance(e.code, int) else 1, f"SystemExit({e.code})")
    except BaseException as e:                  # worker 必须吞一切异常，逐个记失败不杀池
        return (task_id, 1, f"{type(e).__name__}: {e}")

@tool("run_<x>_plan", parse_docstring=True)
def run_<x>_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    plan_path: str = "/mnt/user-data/workspace/plan_<x>.json",
    only_<x>_ids: list[str] | None = None,      # 选择器（重跑子集）；None=全跑
    on_error: str = "continue",                 # "continue" / "abort"
) -> dict[str, Any]:
    # 惰性 import（守 harness 顶层 import 闭环纪律，见 §五）
    import json
    from deerflow.sandbox.tools import _build_path_env, replace_virtual_path
    # ... 各工具专属 import（handoff schema / 聚合器 / seal）...

    # Step 1: resolve workspace from thread_data（缺失 → _error_result）
    # Step 2: read plan（replace_virtual_path + 读 JSON；缺失/解析失败 → _seal_failed + error）
    # Step 3: apply only_<x>_ids selector
    # Step 4: path_env = _build_path_env(thread_data)   # worker resolve /mnt 用
    # Step 5: build task list [(script, args, id), ...]（缺 script/args 的 entry 空 script → 响亮失败）
    #         ⚠️ 必须对每条 args 跑 args=[replace_virtual_path(a, thread_data) for a in args]
    #         预解析 argv 里的 /mnt（铁律 11——进程内调脚本无 bash 的命令字符串重写，
    #         不能假设脚本自 resolve；run_chart_plan 漏了致 112/113 崩塌）。
    # Step 6: results, failures = _execute_tasks(tasks, on_error, timeout, runner=_TASK_RUNNER_OVERRIDE, path_env)
    # Step 7【业务专属】: 各填各的——
    #         metric: 聚合 + statistics + L-A/L-B 校验（aggregate_metrics_to_handoff）
    #         chart : 核每个 output png exists() → chart_files（磁盘真相）
    # Step 8: derive status（纯函数：完成度由磁盘产物对账决定，非 task_results 篡改）
    # Step 9: assemble payload + _seal_handoff_to_workspace(<X>Handoff, "handoff_<x>.json", payload, ws)
    #         sealed_by="run_plan"（确定性工具封存标记，对齐枚举）
    # Step 10: return 紧凑结果（明细落盘不进 context）：
    #          {status, handoff_path, n_total, n_completed/n_rendered, n_failed, failures, gate_signals}
```

`_execute_tasks` / `_scoped_path_env` / `_derive_status` / `_seal_failed` / `_error_result` **直接照抄 run_metric_plan_tool.py**（这几个是纯机械逻辑，§四「通用层」）。

---

## 三、红线与铁律（每个实例都必须守）

### 红线 1：LLM 只传「哪些 / 怎么对待」，永不传「是什么（args）」
工具边界 LLM 只传 `only_<x>_ids`（选择器）+ `on_error`（策略）。**每条脚本的 args 从 plan artifact 读**，LLM 只做指向。理由：args 是确定性 resolve（catalog.resolve）的产物，让 LLM 重拼 = 重新引入「漏 `--parameters-json`」类脆弱。

### 红线 2：完成度是 (plan, 磁盘产物) 的纯函数，不是 (plan, LLM 自述)
status 由磁盘对账决定（m_*.json glob / output png exists()），不信 LLM 自报。这是 reward hacking 防御的核心（HarnessX 第 1 病理）。`_derive_status` 透传聚合器/对账算出的 status，不另用 task 成败篡改（唯一例外：n_failed>0 但磁盘却 completed = 脚本半成功，仍报 failed，响亮）。

### 铁律 3：worker 必须模块级 + 顶层无 deerflow/ethoinsight import
ProcessPoolExecutor pickle 要求 worker 函数模块级；顶层 import deerflow/ethoinsight 会闭 harness 导入环（见 §五）。worker 内部惰性 `import importlib` 即可。

### 铁律 4：path_env 经 initializer 在子进程内设，不在父进程 mutate 全局 os.environ
Gateway 多线程并发跑多 thread 的 subagent，父进程全局 `os.environ.update` 会让一个 thread 的 workspace 路径泄漏成全局、污染其他 thread。用 `_worker_init` initializer 关在 worker 子进程里。父进程内跑的子步骤（如 statistics）用 `_scoped_path_env`（with 块内临时设、退出即恢复）。

### 铁律 5：DEERFLOW_PATH_* env 键名含 `MNT`
`_sandbox_env_key_for_prefix` 用 `prefix.strip("/")` 保留 `mnt` → `/mnt/user-data/workspace` 的 env key = `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE`（**不是** `_USER_DATA_`）。**直接用 `_build_path_env(thread_data)` 生成，绝不手拼键名。**

### 铁律 6：matplotlib 子进程必须 Agg（仅绘图类工具）
进程池 worker 从 Gateway 父进程 fork；若父进程曾初始化非 Agg backend，worker 继承交互式 backend → 非主线程 savefig 崩。worker initializer 设 `os.environ.setdefault("MPLBACKEND", "Agg")`。（ethoinsight `charts.py` 顶层已 `matplotlib.use("Agg")`，但有脚本裸 import pyplot 不经它——env 兑底覆盖所有情况。）

### 铁律 7：测试用 `_TASK_RUNNER_OVERRIDE` 注入同步串行 runner
绕 ProcessPoolExecutor 的 fork/pickle（单测里 monkeypatch 不传 fork）。等价性测试要厘清对比两端同层（memory `feedback_processpoolexecutor_test_runner_injection_and_ssot_parity`）。worker/tmp 用 pid/tid 不用 `Date.now()`/随机数（确定性可测）。

### 铁律 8：装配链三处缺一不可（#187 血泪）
`@tool` 定义（新文件）+ `builtins/__init__.py` 导出（+`__all__`）+ **`tools.py:BUILTIN_TOOLS` 注册** + subagent `tools=[...]` 白名单。漏 BUILTIN_TOOLS = 工具悬空，get_available_tools 不含（#187 fill/finalize 就栽这）。改完裸导入 `app.gateway` + `make_lead_agent` 0 退出 + dump subagent executor.tools name 确认含新工具（memory `feedback_tool_definition_export_not_equal_registered_in_builtin_tools`）。

### 铁律 9：SealGate 集合——确定性工具产出+交付合一，移出 `_REQUIRES_SEAL`
若该 subagent 受 `SealGateMiddleware` 管（data-analyst/chart-maker/report-writer），改用 batch 工具后**产出+交付合一**（工具内确定性落 handoff），**应移出 `_REQUIRES_SEAL` + `_RECONSTRUCTABLE`**——对齐 code-executor 用 run_metric_plan 后已被移出（docstring 明载「run_metric_plan already produces-and-delivers in one tool」）。否则 SealGate 会卡 subagent 要求调 seal_<name>_handoff 工具。

### 铁律 10：subagent prompt 三指令源对齐
改 subagent 行为必须改 `SubagentConfig.system_prompt`（最高权威），SKILL.md 同步对齐。光改 SKILL.md = 白改（memory `feedback_subagent_system_prompt_higher_authority_than_skill`）。把「LLM 拼 bash 批量跑」改成「调一次 run_<x>_plan」。

### 铁律 11：argv 路径解析对齐——Step 5 必须预解析 args，不假设脚本自 resolve
bash 调脚本时 `replace_virtual_paths_in_command` 重写命令字符串里的 `/mnt` 字面量 → 脚本 argv 已是真实路径；**进程内 `import_module(script).main(args)` 没这步，argv 原样透传 `/mnt`**。所以 **Step 5 build task list 时必须对每条 args 跑 `replace_virtual_path(arg, thread_data)` 预解析**（`args = [replace_virtual_path(a, thread_data) for a in args]`），让进程池与 bash 在 argv 语义上对齐。
- **不能假设脚本自己 resolve output**：run_metric_plan 的 compute 脚本恰好走 `save_output_json`→`resolve_sandbox_path` 自 resolve（侥幸），run_chart_plan 的 plot 脚本 `fig.savefig(args.output)` / `charts.py:_resolve_output_path` 只 makedirs 不 resolve → **2026-06-24 生产 112/113 图崩塌坐实**（PermissionError: `/mnt/user-data` 宿主机不存在）。
- `replace_virtual_path` 对非路径项（`--input`/`--parameters-json`/JSON 字符串/数字）原样返回（幂等无害），故可对整个 args 数组逐项跑。脚本内部对已是真实路径的输入二次 resolve 也幂等（`resolve_sandbox_path` 对非 `/mnt` 原样返回）。
- **「对齐范本」时必须核实范本能工作的隐性前提（脚本自 resolve）在新链路是否成立，别只看形似**（memory `feedback_align_template_must_verify_hidden_premise_not_just_shape`）。
- 注：Step 7 核磁盘的 output 仍用**原始虚拟路径**（chart_files 要 `/mnt/user-data/outputs/` 前缀过 schema 校验），与 Step 5 预解析的 args（喂 worker 用真实路径）解耦，各取所需。

### 铁律 12：子集调用（only_*_ids）必须以全 plan 为对账分母，handoff 不被子集覆盖
`only_*_ids` 选择器**只决定「跑哪些 task」，不决定「handoff 对账谁」**。**对账/status/n_total 必须始终基于全 plan（扣 skipped），不是本次子集**。否则 LLM 能用 `only_*_ids=[能成功的单个]` 调一次 → 工具写「1/1 completed」**整体覆盖**之前「1/N partial」→ 把失败的 N-1 个静默抹掉伪装成功（reward hacking 温床，2026-06-24 dogfood 坐实：chart-maker 逐个 only_chart_ids 调，成功单张覆盖失败全量）。
- **run_metric_plan 做对了（照抄它）**：Step 8 `aggregate_metrics_to_handoff(plan, ...)` 传**全 plan**（不是 only_metric_ids 过滤后的子集），聚合器 glob 全 plan 期望产物对账磁盘 → 没跑的计 missing → partial。子集调用天然不覆盖。
- **run_chart_plan 抄漏了**：它 Step 7 只遍历本次 `results`（子集）、`n_total=len(tasks)`（子集大小）→ 子集成功覆盖全量。修法=对账遍历全 plan、`n_total=全 plan 数`（spec 2026-06-24-run-chart-plan-subset-overwrite-silent-drop）。
- **配套封存门兜底**：`_reconcile` 加「对账总数不变式」——全 plan 每个 output（用 output 不用 id，因 per_subject id 多对一）必须出现在 chart_files∪failed_charts∪remaining_charts，缺则抛 ValueError（静默丢图绝不放行）。
- **判据**：凡有 `only_*_ids` 子集选择器的 batch 工具，对账分母恒为全 plan，handoff 是「全 plan 磁盘真相快照」而非「本次子集结果」。

---


## 四、共享 vs 专属分层（决定「能不能照抄」）

> ⚠️ 当前决策（2026-06-24 用户拍板）：**只写本 pattern 文档，不抽公共代码模块**。理由：守「overlap 也接受重复」铁律 + 避免过早抽错边界（目前仅 2 个实例）。代码各存一份，共享的是**本文档的范式**。等第 3 个实例真来时再评估是否抽 `batch_script_executor.py` 公共模块。

| 层 | 组件 | 跨实例关系 |
|---|---|---|
| **通用执行骨架**（~180 行，纯机械，可照抄）| `_available_cpus`/`MAX_WORKERS`、`_worker_init`、`_scoped_path_env`、`_run_task`（importlib.main）、`_execute_tasks`（per-task 超时+on_error+runner 注入）、`_TASK_RUNNER_OVERRIDE`、`_seal_failed`、`_error_result` | **结构相同 → 照抄**（各文件复制一份，不 import 共享模块）。这层稳定、与业务无关。|
| **业务专属**（~490 行，各填各的，**绝不跨用**）| handoff payload 组装、聚合（metric 的 aggregate_metrics_to_handoff vs chart 的 png exists() 对账）、校验（metric 的 L-A/L-B vs chart 无）、statistics 子步骤（metric 专属）、旁路拆分（metric 的 outlier sidecar）、gate_signals 构造、status 派生细节 | **各写各的**（守不跨范式复用）。metric 的聚合绝不跑到 chart，反之亦然。|

**判据**：通用层 = 「换成任何脚本批量执行都一字不变」的代码；专属层 = 「依赖这批脚本是『指标』还是『图』」的代码。抽象（若未来抽）只抽通用层，专属层永远各存一份。

---

## 五、harness 导入环铁律（每个实例都要验）

batch 工具放 `tools/builtins/`，**顶层 import deerflow/ethoinsight 可能闭环**（subagents/metric_aggregation、sandbox/tools、seal_handoff_tools 都在环上）。处方：
1. **工具体内所有 deerflow/ethoinsight import 惰性放函数体**（不放模块顶层）。
2. **worker 函数顶层无 deerflow/ethoinsight import**（pickle 友好 + 守环纪律）。
3. 改完**裸导入两生产入口**（backend/ 下，绕 conftest mock）：
   ```bash
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
   两者 0 退出才算过（pytest 全绿是假绿——conftest mock 了 executor 短路了环，见 CLAUDE.md「harness 模块顶层 import 闭环风险」）。

---

## 六、与其他确定性门的协同

batch 工具落盘的 handoff **经 `_seal_handoff_to_workspace` 自动过对账门**（如 chart-maker 的 `_reconcile_chart_maker_payload`）。因产物经工具不经 LLM（chart_files/metrics_summary 天生磁盘真相），过门干净通过——batch 工具是**诱因层**（消除 LLM 伪报可能），对账门是**防御层**（兜底核磁盘真相），两者正交协同。详见 [run_chart_plan spec §七](2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md) 与 [ETHO-10 spec](2026-06-24-etho10-chart-maker-product-reality-invariant-spec.md)。

---

## 七、三大病理自检（写新 batch 工具前过一遍）

1. **Reward hacking**：✅ 本 pattern 正面防御——产物经确定性工具不经 LLM，完成度核磁盘（红线 2）。不加「别伪报」prompt 规则。
2. **Catastrophic forgetting**：改 handoff schema（加 sealed_by="run_plan" 枚举）/ 改 SealGate 集合 / 改 subagent prompt 前 grep 所有消费者，全量跑回归。改共享组件先 grep 调用方。
3. **Under-exploration**：✅ 本 pattern 是结构改动（确定性工具替换 LLM bash 编排），不是加 prompt 规则——正解。但**不是所有「LLM 跑脚本」都该上 batch 工具**：见 §一「不该用」判据（有依赖/单任务/高自由裁量/无 main 入口）。

---

## 八、checklist（实施新 batch 工具时逐项打勾）

- [ ] §一判据全满足（独立脚本 + main(argv) 入口 + args 来自 plan + 完成度可磁盘判定）。
- [ ] 照 §二骨架建 `tools/builtins/run_<x>_plan_tool.py`，通用层照抄 run_metric_plan，专属层各填各的。
- [ ] worker 模块级 + 顶层无 deerflow/ethoinsight import（铁律 3）。
- [ ] **Step 5 对每条 args 跑 `replace_virtual_path` 预解析 argv（铁律 11——不假设脚本自 resolve，run_chart_plan 112/113 崩塌坐实）。**
- [ ] path_env 经 `_build_path_env` + `_worker_init`（铁律 4/5）；绘图类加 MPLBACKEND=Agg（铁律 6）。
- [ ] 红线：LLM 只传 selector + on_error，args 从 plan 读（红线 1）；status 核磁盘（红线 2）。
- [ ] **对账/status/n_total 基于全 plan（扣 skipped），不基于 only_*_ids 子集；handoff 是全 plan 磁盘快照不被子集覆盖（铁律 12——run_metric_plan 传全 plan 给聚合器做对了，run_chart_plan 抄漏致成功子集覆盖失败全量）。**
- [ ] handoff schema 加 `sealed_by="run_plan"` 枚举值。
- [ ] 装配链三处 + subagent tools 白名单（铁律 8）。
- [ ] SealGate 集合移出（若 subagent 受管，铁律 9）。
- [ ] subagent prompt 三指令源对齐（铁律 10）。
- [ ] 测试：`_TASK_RUNNER_OVERRIDE` 注入串行 runner（铁律 7）+ 进程内实跑一张验证 + 装配链可见性测试（dump executor.tools）+ 裸导入两入口 0 退出（§五）。
- [ ] 端到端 dogfood：真实数据复跑，确认 subagent 0 次 bash 跑脚本、handoff 产物磁盘真实、sealed_by="run_plan"、不被 SealGate 卡死、不伪报。

---

## 九、关键锚点
- **范本**：`tools/builtins/run_metric_plan_tool.py`（通用层 `_available_cpus`33/`_worker_init`77/`_scoped_path_env`91/`_run_metric_task`116/`_execute_tasks`467/`_TASK_RUNNER_OVERRIDE`68/`_derive_status`551/`_seal_failed`616/`_error_result`656；专属层 `aggregate_metrics_to_handoff` 调用 329/statistics 245/sidecar 415）
- **来源 spec**：Spec S4（run_metric_plan，2026-06-12），见 handoff `2026-06-15-s4-run-metric-plan-completed-handoff.md`
- **第二实例 spec**：[2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md](2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md)
- **路径 env**：`sandbox/tools.py:_build_path_env`(594)；`ethoinsight/scripts/_cli.py:resolve_sandbox_path`(81)，键名 `DEERFLOW_PATH_MNT_*`
- **装配链**：`tools/tools.py:BUILTIN_TOOLS`(35)、`tools/builtins/__init__.py`
- **SealGate**：`agents/middlewares/seal_gate_middleware.py:_REQUIRES_SEAL`(68)/`_RECONSTRUCTABLE`(85)，code-executor 已排除
- **导入环 guard**：`tests/test_gateway_import_no_cycle.py`
- **相关范式文档**：`2026-05-12-script-per-metric-architecture-design.md`（脚本即指标，本 pattern 的上游——脚本统一 main(argv) 入口就是它定的）、`2026-05-13-metric-catalog-architecture-design.md`（catalog SSOT，plan artifact 的来源）

---

## milestone 建议
归入「chart-maker 架构对齐 code-executor / 批量执行确定性化」track。本文档把 run_metric_plan（实例 1）+ run_chart_plan（实例 2）背后的范式固化成可复用 pattern，未来「subagent 批量跑独立脚本」需求照本文档 checklist 落地。checkpoint：「run_metric_plan + run_chart_plan 提炼出『确定性批量执行工具』范式 → 用户拍板固化为 pattern 文档（不抽公共代码，守 overlap 接受重复）→ 未来第 3 实例照 checklist 写、达 3 实例再评估抽公共模块」。
