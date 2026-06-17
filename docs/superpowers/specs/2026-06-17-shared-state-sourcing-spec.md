# Spec 4 (P4): 横切 session 状态的统一收口 + parity 回归（元问题：路径传来传去）

> 日期：2026-06-17
> 类型：结构病根治（红线二元层 —— 防止"下一条路径又漏接"）
> 触发源：用户元问题——"路径/列对齐这种 session 级共享变量，不是应该作为全局直接拿吗？为什么传来传去？"
> 状态：待 review → 批准后 worktree 实施。**本 spec 是元收口，建议在 P3 之后实施**（P3 先把 charts 这条具体路径接上，P4 再加"所有路径都接了"的回归网）。
> **遵循工程实践**：[红线二全文](../../refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md#红线二session-级横切状态用共享源--自取禁止传参--各路径单独接线) + [可执行判据](../../refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md#可执行判据-1)。

---

## 0. 元问题（用户原话 2026-06-17）

> "这一个 session 会有大量文件保存，base dir 都差不多。**每一个 tool 应该都有一个 session 级别的"全局"变量**，这样 tool 在保存/读取文件时可以直接拿到，而不是靠 prompt 或自己想怎么拿。"

session 级常量（实验状态 column_aliases/groups/paradigm；物理 base dir / 路径映射）本质是"一个 thread 一份"的共享态，**应该每个 tool 直接从它的 session 注入态拿**，而不是在工具/脚本/CLI 间"传来传去"，更不是靠 LLM 在 prompt 里记得拼。现状是每条消费路径各自接线，每新增一条就多一个"忘了接"的点——所以 dogfood 每走到新路径就发现它漏接（statistics→PR#141、prep→#6、charts→P3）。

## 0.1 机制已存在：`ToolRuntime` + `runtime.state["thread_data"]`

用户要的"tool 的 session 级全局变量"**DeerFlow 已经有**：工具签名带 `runtime: ToolRuntime`，`runtime.state["thread_data"]` 装着这个 session 的所有常量（`workspace_path` / 路径映射等），由 ThreadDataMiddleware 在 session 启动时建好、每次 tool 调用自动注入。

`prep_metric_plan_tool.py:124-130` 是**正范例**：
```python
thread_data = runtime.state.get("thread_data") if runtime.state else None
if not thread_data or not thread_data.get("workspace_path"):
    return _error(..., "thread_data.workspace_path is not set ... 基础设施 bug")  # fail-fast
real_workspace_path = thread_data["workspace_path"]
```
拿不到就 fail-fast 报"基础设施 bug"，不静默兜底。**这就是终态范式**——本 spec 是把所有路径都拉到这个范式上，而不是发明新机制。

问题路径：charts（chart-maker 纯 bash 拼 CLI，根本没 runtime，P3 修）；物理路径解析（散落各处 `resolve_sandbox_path` / `_scoped_path_env` 而非统一从 thread_data 取）。

## 1. 两类横切状态，现状与终态

### B 类：实验状态（column_aliases / groups / paradigm / parameter_overrides）
- **SSOT**：`experiment-context.json`（已是 session 单一源，Gate 1 写）。
- **现状**：metrics 路径工具自读（对）；charts 路径靠 LLM 拼 CLI（错，P3 修）；statistics/validate 各自处理。
- **终态**：所有消费路径走**同一个 reader**（已有 `read_context`），工具内部自取。**禁止任何路径用 CLI 入参传 B 类状态**（传参 = 把状态传递交给调用方记性）。

### A 类：物理路径 / base dir（虚拟 /mnt ↔ 宿主真实路径）
- **SSOT**：`runtime.state["thread_data"]`（含 workspace_path / base dir / 路径映射，session 启动时由 ThreadDataMiddleware 建好）。
- **现状**：`resolve_sandbox_path` / `_scoped_path_env` 散落各处各自调；memory 记≥3 次 dogfood 炸在"某路径忘 resolve / 忘包 env"。
- **终态**：每个 tool 从**自己的 `runtime.state["thread_data"]`** 拿 base dir + 路径映射（用户要的"tool 的 session 级全局变量"），读写 workspace 文件统一走"从 thread_data 取 base + replace_virtual_path"。**禁止用全局可变 env 当共享**——`DEERFLOW_PATH_*` 是进程级全局，并发 subagent 共享进程会串台。session 隔离靠 `runtime.state`（每次 tool 调用绑定自己的 thread_data），天然不串台。

> **A 类的范围警告**：A 类终态（统一文件 I/O 入口）改动面大（碰几乎所有读写 workspace 的工具/脚本）。本 spec **不要求一次做完 A 类**。本 spec 的核心交付是 **B 类收口 + parity 回归网**；A 类只产出"诊断 + 渐进收口清单"（见 §3），实际收口分批做、每批带回归，避免与正在跑的修复撞车。

## 2. 修法 —— B 类收口（本 spec 主交付）

### 2.1 确立唯一 reader
所有消费 experiment-context 的路径统一走 `read_context()`（已存在）。审计当前直接 `json.load(experiment-context.json)` 或自己拼路径读的地方，全部改走 `read_context`。

### 2.2 禁止 CLI 传 B 类状态
- 审计所有工具/CLI 的 `add_argument` 与工具签名，找出"用参数传 column_aliases/groups/paradigm"的口子。
- 这些口子改为**工具内部自读 context**（charts 由 P3 修；其余路径本 spec 收口）。
- CLI 若必须保留 `--column-aliases-file`（向后兼容/测试注入），则**未传时自动从 context 兜底**，不让"未传"退化成"无对齐"。

### 2.3 parity 回归网（防下一条路径漏接 —— 本 spec 的灵魂）
新增一个 parity 测试模块，**枚举所有消费 B 类状态的路径**（metrics / charts / statistics / validate / …），喂同一份 `column_aliases`，断言**每条路径都解析出一致的 zone 对齐**：

```
给定 column_aliases = {open: open_arms, closed: closed_arms}
for path in [resolve_metrics, resolve_charts, statistics_dispatch, validate_catalog]:
    assert path 能看到 open_arm_zones=["open"]  # 每条都接了
```

**这是关键**：下一条新增的消费路径如果漏接 column_aliases，这个测试缺它一项就红，CI 当场抓——**不再靠 dogfood 撞**。这条回归网把"靠人记得接"变成"漏接就红"。

## 3. A 类诊断 + 渐进收口清单（本 spec 不实施，只产出）

实施 agent 产出一份清单文档（`docs/problems/` 下）：
- 列出所有调 `resolve_sandbox_path` / `_scoped_path_env` / 裸 `open(/mnt...)` 的站点。
- 标注每个站点是否在并发路径（串台风险）。
- 给出"统一 I/O 入口"的目标签名 + 分批收口顺序（按风险排序：先收口已出过事的 statistics/validate/进程内脚本）。
- 每批一个独立 spec + 回归，分批合入。

## 4. 测试

- §2.3 的 parity 测试（**核心**，修复前可能部分红——charts 项在 P3 合入前是红的，合入后转绿；其余路径补齐）。
- `test_no_cli_passes_session_state`（守护）：grep 式或 AST 测试，断言工具签名里没有 column_aliases/groups 作为 LLM 可传入参（除注入测试用的兜底口）。
- B 类各路径自读 context 的单测。

## 5. 风险边界
- 本 spec 聚焦 B 类 + parity 网；A 类只诊断不动刀（改动面大，分批）。
- parity 测试要随"新增范式/新增消费路径"持续扩充——它是活的回归网，不是一次性。
- 与 P3 有重叠（charts 自读）：P3 先落地 charts 这条，P4 的 parity 网把它纳入并守住全集。两者按 P3→P4 顺序。

## 6. 为什么这是元收口
P3 修 charts 一条路径，但**不能阻止下一条路径漏接**。P4 的 parity 回归网才是答案：它把"所有消费路径都接了 column_aliases"变成一条 CI 断言。这正面回答用户的元问题——**不是"传来传去"，而是"唯一源 + 自取 + 一条断言保证全员自取"**。
