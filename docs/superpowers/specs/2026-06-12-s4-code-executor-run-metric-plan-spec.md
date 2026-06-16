# Spec S4: code-executor 执行器重构 —— bash 编排反模式 → run_metric_plan 确定性工具

> 日期：2026-06-12
> 顺序：第 4 份（共 4 份，按序实施）。最大、最core。前三份（S1/S2/S3）都不依赖它、它也不依赖前三份，但放最后——地基稳定后做，验证环境干净。
> 来源：dogfood thread 38be2753/a5b97c00 性能诊断 + 用户/Fable-5 五轮架构讨论
> 实施方式：新开 worktree 基于**已合 S1+S2+S3 的最新 dev**，单 PR（commit 可分多笔）
> 前置：calamine 已合 dev（parse 加速）；ParamValue/seal 三态已合 dev

---

## 0. 背景与问题（实证）

dogfood thread 38be2753 的 code-executor 第二次 run 实测 **18:16:09→18:29:08 ≈13 分钟**。拆解（gateway.log 时间戳）：
- **真正计算只 ~2 分钟**（5 个 metric 批次 #7→#13）。
- 大头是 **LLM 逐 token 吐 28 行 bash 脚本 × 5 批 + thinking** + stats 重试 + cat 聚合 + 手构 handoff。

**根本问题（架构层）**：code-executor 的工具只有裸 `bash`（[code_executor.py:184](packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py)），prompt 教它把结构化 plan_metrics.json（~140 个确定性 task）**逐字翻译成 `python -m <script> <args>` 命令行字符串**、分批 `&` 并行、`cat` 聚合、手构 handoff（[code_executor.py:30-94](packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py)）。

这是**反模式**：
1. **LLM 当人肉命令行编排器**：plan 是确定性施工单（`for task: run(script, args)`，无歧义无需试错），却让 LLM 逐 token 生成 140 行 bash 文本（大半运行时间）。
2. **转录错误面**：LLM 手抄 140 个文件名/引号/转义/参数顺序，整类 bug。
3. **140 次进程冷启**：每个 `python -m` 独立进程，import pandas/ethoinsight ~0.83s × 140 ≈ 110s。
4. **seal 故障门类**：手构 handoff + 手调 seal 是「terminated without handoff」三个独立根因的温床（prompt 矛盾 / metrics 字段三分裂 / list zone 参数被 schema 拒）。

## 0.1 架构决策（用户 + Fable-5 五轮讨论，已锁定）

**保留 code-executor subagent，只换工具粒度（裸 bash → run_metric_plan 确定性工具）。不取消 subagent。**

三条保留理由（缺一不可）：
1. **orchestrator 本质**：上层是动态 orchestrator 不是静态 workflow；取消 subagent 会把动态派遣焊成静态流水线。（注：派遣边界——lead 不持执行类工具——是**政策**选择，非逻辑必然，但本项目守此政策。）
2. **脚本已写好，病在工具粒度**：计算逻辑全在 `ethoinsight.scripts.*`，缺的是把它暴露成 first-party 工具，而非让 LLM 用 bash 字符串够它（守 memory `feedback_subagent_consumption_via_first_party_tool`）。
3. **小模型上下文隔离（最硬）**：目标部署 30B 小上下文模型。subagent 把执行噪音 + 错误处理指令隔离在自己的 context/system_prompt 里，不污染 lead 的有限上下文。Fable：不只 token 隔离，还有**指令隔离**（错误分诊提示词若折进 lead prompt 是常驻每轮的干扰）。

**改造后 code-executor 的形态**（Fable 收尾）：
> 读 plan 路径 → 一次同步 `run_metric_plan`（进程池 + policy 化机制错误 + 确定性落盘 handoff）→ 默认无事可做；仅当结构化失败摘要出现策略解决不了的失败时，行使唯一真职责：**分诊失败语义**，决定覆盖状态或回报 lead 重规划。

happy path 薄到近乎透明——这是「智能在节点内、约束在节点间」在单个 agent 内部的投影：智能只出现在规则失效那一刻。

**否决记录（防后续重提）**：
- ❌ **取消 subagent / 焊进编排图**（Fable 的 B 方案）——违 orchestrator 本质 + 小模型隔离需求。
- ❌ **Celery / 分布式队列**——对「一次调用内同步跑完 140 个短任务」是错配，常驻 broker + 分布式失败面与「减少故障门类」目标相反。用标准库 `ProcessPoolExecutor`。
- ❌ **异步 handle + 轮询**（Q3）——把确定性执行变成「依赖 LLM 记得轮询」的有状态协议，是 seal 故障门类同构复刻；LLM 不擅长安静等待（TodoMiddleware 重唤起故障佐证）。用**同步长调用**。
- ❌ **`run_tasks(tasks[])` 签名**（Q4）——把 args 重新序列化进 tool_call JSON = bash 转录反模式换成 JSON 转录。用 `run_plan(plan_path)` 整 plan + 声明式选择器。

---

## 1. 新工具 `run_metric_plan`（first-party，确定性）

### 1.1 工具签名（吃整 plan + 声明式选择器，不传 content）
```python
@tool("run_metric_plan", parse_docstring=True)
def run_metric_plan_tool(
    runtime: Runtime = None,
    plan_path: str = "/mnt/user-data/workspace/plan_metrics.json",
    only_metric_ids: list[str] | None = None,      # 选择器：只跑这些 metric（重跑子集）
    on_error: str = "continue",                     # policy: "continue" | "abort"
) -> dict[str, Any]:
    """一步跑完 plan_metrics.json 的所有 compute + statistics，确定性聚合并落盘 handoff。

    不需要逐条拼 bash、不需要手构 handoff。工具内 ProcessPoolExecutor 并行跑所有
    compute 脚本（每 worker import 一次）+ 确定性聚合 + 原子落盘 handoff_code_executor.json。
    """
```
**原则（Fable Q4）**：LLM 在工具边界只传「哪些」（`only_metric_ids` 选择器）和「怎么对待」（`on_error` policy），**永远不传「是什么」（content）**——内容（args）从 plan artifact 读，LLM 只做指向。

参考工具模板：[prep_metric_plan_tool.py](packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py)（`@tool` + `Runtime` + 从 `runtime.state["thread_data"]["workspace_path"]` 取 workspace）。

### 1.2 工具内分层（Fable：执行器 + 聚合函数，代码内组合，LLM 不当胶水）
```
run_metric_plan_tool（薄壳：解析参数、组装、返回紧凑结果）
  ├─ _execute_plan(plan, only_ids, on_error) → list[TaskResult]   # 通用执行器，不懂 handoff
  │    └─ ProcessPoolExecutor：每 task 调 importlib.import_module(script).main(args)
  └─ _aggregate_to_handoff(plan, task_results, workspace) → handoff payload  # 确定性聚合
       └─ _seal_handoff_to_workspace(CodeExecutorHandoff, ...) 落盘
```

### 1.3 执行器（ProcessPoolExecutor，进程内调脚本）
- **worker 调用方式**：`importlib.import_module(f"ethoinsight.scripts.{plan_metric.script}").main(plan_metric.args)`——脚本 `main(argv)` 签名统一（调研坐实，[oft/compute_center_time_ratio.py:29](packages/ethoinsight/ethoinsight/scripts/oft/compute_center_time_ratio.py)），可进程内调，无需 subprocess。
- **import 复用**：N worker 各 import pandas/ethoinsight 一次，跨 task 复用 → import 从 140×0.83s 降到 N×0.83s 并行摊销（几秒）。
- **进程池非线程池**：pandas 持 GIL，线程池在 CPython 并行不起来（Fable Q4）。
- **worker 继承 `DEERFLOW_PATH_*` env**（虚拟路径解析靠它，[_cli.py:77](packages/ethoinsight/ethoinsight/scripts/_cli.py) `resolve_sandbox_path`）——ProcessPoolExecutor 默认继承 parent `os.environ`，确认 fork/spawn 模式下 env 可见。
- **per-task 超时**：`ProcessPoolExecutor` 原生无 per-task 超时，用 `future.result(timeout=...)` + 超时则记该 task 失败（不杀整池）。外层加 wall-clock 预算（按 `n_tasks × 单task预算 ÷ workers + slack` 算，非魔法常数）。
- **单 task 异常逐个捕获**进结果、不杀池；`on_error="abort"` 时遇首个失败停止提交后续。
- **结果按 plan 顺序排序**保可复现。
- **max_tasks_per_child**：防 matplotlib/pandas 全局状态污染（compute 不画图，风险低，但设上保险）。

### 1.4 聚合器（复用 auto-seal 逻辑 = Fable「兜底转正」）
**关键：不靠 stdout。** 脚本 `main()` 内 `save_output_json(args.output, payload)` 把完整 payload（含 `parameters_used`）写到 `m_*.json`（[compute 脚本:38](packages/ethoinsight/ethoinsight/scripts/oft/compute_center_time_ratio.py)）——和 `emit_result` print 的是同一 payload。所以聚合器**跑完 task 后直接读 output 文件**，跟现有 auto-seal 一样，全程不碰 stdout。

**复用 [executor.py:_attempt_auto_seal_from_artifacts 的 code-executor 段（:401-559）](packages/agent/backend/packages/harness/deerflow/subagents/executor.py)**：
- 它已实现「读 plan → 枚举期望 vs 实际产物 → 逐 m_*.json 读 metric/value/parameters_used → 聚合 metrics_summary + per_subject（含 _signal_distributions）→ subject↔group 映射 → 完整性判 completed/partial → 构造 payload」。
- **这正是 run_metric_plan 聚合器要做的**。把这段抽成**独立纯函数 `aggregate_metrics_to_handoff(plan, workspace) → payload`**，run_metric_plan 和 auto-seal **都调它**（守 SSOT，消除两份聚合逻辑）。
- **VALIDATION_ERROR**：聚合时对读到的 value 直接调 `validate_metrics`（[validate.py:14](packages/ethoinsight/ethoinsight/validate.py) 纯函数）+ 跑 `validate_catalog`（L-B 范围校验）→ 汇入 `data_quality_warnings`，不靠抓 stdout。

### 1.5 完成度 = 纯函数计算（不是 LLM 判断，Fable Q2）
`(plan, results, artifacts) → status` 是纯函数：
- `completed` = plan 期望产物集与磁盘 m_*.json 对账齐。
- `partial` = 数据驱动统计跳过（n 小，skip_reason 非空）。
- `failed` = 失败比例/关键产物缺失超阈值。
聚合器直接算，写进 payload。

### 1.6 seal 反转：工具确定性落盘（Fable Q2）
- 工具执行成功即调 `_seal_handoff_to_workspace(CodeExecutorHandoff, "handoff_code_executor.json", payload, workspace)`（[seal_handoff_tools.py:354](packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py) 纯函数）**确定性落盘**。
- `sealed_by="run_plan"`（区别于 LLM 手调的 `sealed_by="model"` 和 auto-seal 的 `framework_rebuild`）。
- **LLM 不再必须发 seal**——「忘了做」这个故障类结构性消失。
- 叙事字段（summary）可机械生成默认（`"140/140 tasks completed"`），LLM 增润可选、缺席不致 handoff 无效。

### 1.7 返回给 LLM 的内容（刻意紧凑，Fable Q3）
```python
return {
    "status": status,                    # completed/partial/failed
    "handoff_path": "/mnt/.../handoff_code_executor.json",
    "n_total": 140, "n_completed": 140, "n_failed": 0,
    "failures": [...],                   # 仅失败 task 的 {id, error} 摘要（成功的不回传）
    "gate_signals": {...},               # lead 需要的决策信号
}
```
成功 task 的 140 条明细**落盘不进 context**（烧 token + 诱导 LLM 折腾）。失败给详情供分诊。

### 1.8 进度可观测（Fable Q3：给人不给模型）
长调用（几十秒~几分钟）期间 per-task 进度写**日志/进度文件/流式通道**（供 UI/运维 tail），**不回灌 LLM context**。注意 harness 工具超时上限——确认 run_metric_plan 不撞 subagent timeout（当前 15min，足够）。

---

## 2. code-executor subagent 改造

### 2.1 工具列表（收 bash，加 run_metric_plan）
[code_executor.py:184](packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py)：
```python
# 改前
tools=["bash", "read_file", "write_file", "ls", "str_replace", "seal_code_executor_handoff"]
# 改后
tools=["run_metric_plan", "read_file", "ls", "seal_code_executor_handoff"]
```
- **删 `bash`**（核心：防 LLM 退回手拼命令老路，Fable Q1 + memory `feedback_deny_messages_must_direct`）。
- **删 `write_file`/`str_replace`**（手构 handoff 不再需要）。
- **保 `read_file`/`ls`**（读 plan、验证产物存在）。
- **保 `seal_code_executor_handoff`**：仍保留作 LLM **override 通道**（run_metric_plan 默认落盘后，LLM 在错误分诊时可覆盖 status；不是义务）。

### 2.2 prompt 重写
- **删 `<workflow>` 的逐条拼命令段（:34-48）+ `<bash_constraints>` 段（:109-118）+ 手构 handoff 段（:66-92）**。
- **新 workflow**（正面指令，deepseek 铁律）：
  1. read plan_metrics.json（确认存在；不存在则 seal failed，同现状）。
  2. **调一次 `run_metric_plan`**——它确定性跑完所有计算 + 聚合 + 落盘 handoff。
  3. 读工具返回的紧凑结果。**若全部成功**：直接输出 `[gate_signals]`（从返回读），完成。
  4. **若有失败**（返回 `failures` 非空）：行使分诊职责——读失败摘要判断：plan 层错误（脚本名/路径错）→ 回报 lead 重规划；数据层（个别文件坏）→ 接受 partial；不确定 → 标 partial + 记 errors。必要时用 seal override 覆盖 status。
- **保留**：plan 不存在 / ModuleNotFoundError 的 failed 处理、gate_signals 输出格式、constitution_acknowledged 等。
- **错误分诊段**是 subagent 唯一的真 LLM 职责，写清楚三类失败的处置（plan层/数据层/环境层）。

### 2.3 收 bash 的连带影响
- **guardrail**：`ScriptInvocationOnlyProvider`（[script_invocation_only_provider.py](packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py)）对 code-executor bash 的限制变冗余——但 **chart-maker 仍用 bash 跑 catalog.resolve**，**不删 provider**，只是 code-executor 不再触发它（无 bash 工具）。确认 provider 不会因 code-executor 无 bash 而报错。
- **`run_metric_plan` 不进 bash guardrail**（它是 first-party 工具不是 bash 命令）——确认它不被任何 pre-tool guardrail 误拦。

---

## 3. 迁移风险（Fable 四补丁，写进 spec 让实施者预期）
1. **失败率会"上升"=feature 非回归**：以前 LLM 静默修补的 plan 错误（改路径/调参）现在响亮失败，暴露上游 plan 生成 bug。第一轮 dogfood 别误读成「新工具更脆」。
2. **长调用可观测性**：进度给人不给模型（§1.8），别变黑盒卡 3 分钟。
3. **禁 bash fallback**：prompt 显式说「用 run_metric_plan，不要尝试 bash（已无此工具）」——最干净是直接收走 bash（§2.1 已做）。
4. **聚合从磁盘真实产物读不从 plan 假设推**（§1.4 已遵）：metrics_summary 数据源是 m_*.json 实际产出（实际用了什么参数），不是 plan 声称会发生什么。

---

## 4. 测试（TDD）

### 单元 — 执行器
1. **进程内调脚本**：mock 一个小 plan（2-3 个真实 compute task，用 #125 入库的 EV19 fixture），run_metric_plan 跑完，断言 m_*.json 产出 + 聚合 metrics_summary 正确。
2. **import 复用**：断言 N worker 跑 M task（M>N）只 import 脚本模块 N 次量级（或测耗时显著低于 M 次冷启）。
3. **per-task 超时不杀池**：一个 task 故意挂死，断言其余 task 仍完成、该 task 记失败。
4. **on_error policy**：`abort` 遇首个失败停后续；`continue` 跑完全部。
5. **only_metric_ids 选择器**：只跑子集，断言只产出对应 m_*.json。

### 单元 — 聚合器（抽出的纯函数）
6. **聚合一致性**：同一组 m_*.json，`aggregate_metrics_to_handoff` 与现有 auto-seal 聚合产出**字节一致**（证明抽出复用无行为变化）。这是守「兜底转正、单一聚合逻辑」的关键测试。
7. **list zone 参数透传**：parameters_used 含 `{"open_arm_zones":["open"]}` 全程保全（回归 #125 修的 bug）。
8. **完成度纯函数**：plan 全产出→completed；缺产物→partial；skip_reason→partial。

### 单元 — seal 反转
9. **工具落盘 handoff**：run_metric_plan 成功后 handoff_code_executor.json 存在、schema 合法、`sealed_by="run_plan"`。
10. **LLM 不发 seal 也完整**：模拟 subagent 调 run_metric_plan 后不调 seal，断言 handoff 已落盘（fail-safe 反转）。

### 契约 — prompt
11. prompt 含「调 run_metric_plan」「无 bash」「失败分诊三类」；不含逐条拼命令/bash_constraints（删除验证）。

### 集成 — 端到端
12. 真实 EPM plan（多 subject 多 metric）走 run_metric_plan，handoff 链可被 data-analyst 消费。

### 回归
13. 现有 code-executor / auto-seal 测试全绿（抽聚合纯函数不破坏 auto-seal）。
14. chart-maker bash 路径不受影响（guardrail 保留）。

---

## 5. 验证（端到端）
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. .venv/bin/python -m pytest tests/test_run_metric_plan.py tests/test_auto_seal_from_artifacts.py tests/test_code_executor_workflow.py -q
# 改了 subagents/ + tools/builtins 核心 → 裸导入两生产入口（conftest mock 藏循环导入，铁律）
PYTHONPATH=. .venv/bin/python -c "import app.gateway"
PYTHONPATH=. .venv/bin/python -c "from deerflow.agents import make_lead_agent"
# 全量回归（基线含 test_subagent_executor 环境债，勿归因本次）
PYTHONPATH=. .venv/bin/python -m pytest -q
```
最终：复跑 dogfood EPM 数据，code-executor 调一次 run_metric_plan 即完成（不写 bash），13 分钟 → 预期数分钟内（计算 + import 摊销），handoff 链走通。

---

## 6. 关键文件
- `packages/agent/backend/packages/harness/deerflow/tools/builtins/run_metric_plan_tool.py`（**新建**，工具薄壳 + 执行器 + 调聚合器）
- `packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py`（注册 run_metric_plan）
- `packages/agent/backend/packages/harness/deerflow/subagents/executor.py`（抽 `aggregate_metrics_to_handoff` 纯函数，auto-seal 改调它）
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`（工具列表收 bash + prompt 重写）
- `packages/ethoinsight/ethoinsight/scripts/_cli.py` / 各 compute 脚本（确认 main(argv) 可进程内调，应无需改）
- `packages/agent/backend/tests/test_run_metric_plan.py`（新建）

## 7. 红线（勿违）
- **保留 code-executor subagent**——只换工具粒度，不取消（三条理由：orchestrator/脚本已写好/小模型隔离）。
- 用 **`ProcessPoolExecutor`**，**不用 Celery/异步轮询**（否决记录 §0.1）。
- 工具签名 **`run_plan(plan_path)` 整 plan + 选择器**，**不用 `run_tasks(tasks[])`**（防 JSON 转录反模式）。
- **聚合抽单一纯函数**，run_metric_plan 和 auto-seal 共用（守 SSOT，不造两份聚合逻辑）。
- 聚合**从磁盘 m_*.json 真实产物读**，不从 plan 假设推；**不靠 stdout**（脚本已 save_output_json 落盘）。
- **收走 bash**（防 LLM 退回老路）；prompt 用正面指令（deepseek 铁律）。
- **不删 ScriptInvocationOnlyProvider**（chart-maker 仍用 bash）。
- 进度**给人不给模型**（不回灌 context）。
- 改 subagents/tools 核心后**必跑裸导入两生产入口**（conftest mock 藏循环导入，pytest 假绿，backend/CLAUDE.md 铁律）；新 helper import 放函数体内防闭环。
- list zone 参数透传回归测试（#125 修的 bug 别复发）。
- `test_subagent_executor` 的 ModuleNotFoundError 是纯 dev 环境债，勿归因本次。
