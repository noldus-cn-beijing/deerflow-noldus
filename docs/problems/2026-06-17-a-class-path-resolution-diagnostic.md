# A 类横切状态（物理路径 / base dir）诊断 + 渐进收口清单

> 日期：2026-06-17
> 类型：诊断清单（spec 2026-06-17-shared-state-sourcing-spec §3 交付物 —— **只产出，不实施**）
> 关联：[spec P4](../superpowers/specs/2026-06-17-shared-state-sourcing-spec.md) §1 A 类 / §3 / §5 风险边界
> 状态：诊断完成；分批收口待各批独立 spec（每批带回归，按风险排序合入）

---

## 0. 为什么有这份清单

spec §1 把横切 session 状态分两类：

- **B 类**（实验状态 column_aliases/groups/paradigm/parameter_overrides）—— P4 已收口（统一 `read_context` reader + parity 回归网，见 `tests/test_column_aliases_parity_net.py`）。
- **A 类**（物理路径 / base dir：虚拟 `/mnt` ↔ 宿主真实路径）—— **本清单对象**。

A 类现状：物理路径解析散落各处（`resolve_sandbox_path` / `_scoped_path_env` / `DEERFLOW_PATH_*` env / `replace_virtual_path`），memory 记录 ≥3 次 dogfood 炸在「某路径忘 resolve / 忘包 env」。spec §3 明确 **A 类改动面大（碰几乎所有读写 workspace 的工具/脚本），本次只诊断不动刀**，实际收口分批做、每批带回归，避免与正在跑的修复撞车。

A 类终态目标（spec §1）：每个 tool 从自己的 `runtime.state["thread_data"]` 拿 base dir + 路径映射（用户要的「tool 的 session 级全局变量」），读写 workspace 文件统一走「从 thread_data 取 base + replace_virtual_path」。**禁止用全局可变 env 当共享**——`DEERFLOW_PATH_*` 是进程级全局，并发 subagent 共享进程会串台（session 隔离靠 `runtime.state`，每次 tool 调用绑定自己的 thread_data，天然不串台）。

---

## 1. 站点全量清单（按"解析机制"分组）

### 1.1 `resolve_sandbox_path`（ethoinsight 库层，scripts/_cli.py:81）

进程内 / CLI 脚本侧的虚拟路径解析。按 `DEERFLOW_PATH_*` env 解析 `/mnt/...` 虚拟路径回宿主真实路径，三级 fallback（env → workspace 兜底 → 原样返回 + WARNING）。

| 调用文件 | 角色 | 并发路径？（串台风险） |
|---|---|---|
| `ethoinsight/scripts/_cli.py` | 定义点 + `read_inputs_json` / `read_groups_json` / `save_output_json` 的统一 I/O 入口 | **高**：所有 metrics/statistics 脚本经此读输入/写输出，ProcessPoolExecutor worker 共享进程 env |
| `ethoinsight/validate_catalog.py` | L-B 校验读 plan 内 `/mnt` 虚拟路径 | 中（进程内调） |
| `ethoinsight/catalog/resolve.py` | resolve 时 I/O（detect/parse） | 中 |
| `ethoinsight/parse/_core.py` | parse_header 读文件 | 中 |
| `harness/subagents/metric_aggregation.py` | 聚合读多 subject 结果 | **高**（并发聚合） |
| `harness/tools/builtins/run_metric_plan_tool.py` | run_metric_plan Step8 validate | 中 |

### 1.2 `_scoped_path_env` / `DEERFLOW_PATH_*` env（进程级全局，串台风险源）

| 调用文件 | 角色 | 串台风险 |
|---|---|---|
| `harness/sandbox/local/local_sandbox.py` | **注入点**：给 sandbox 子进程设 `DEERFLOW_PATH_*` env | 源头（per-sandbox 设置，但进程内共享） |
| `harness/sandbox/tools.py` | `replace_virtual_path` / `replace_virtual_paths_in_command` / `_build_path_env` / `_scoped_path_env` | 高（所有 bash/read/write 经此） |
| `ethoinsight/scripts/_cli.py` | 消费 env 解析路径 | 高 |
| `ethoinsight/catalog/cli.py` | `_resolve_virtual_workspace_dir` 反推 env key | 中 |
| `ethoinsight/parse/_core.py` | 消费 env | 中 |
| `harness/subagents/metric_aggregation.py` | 消费 env | **高** |
| `harness/tools/builtins/run_metric_plan_tool.py` | `_scoped_path_env` 包 validate 调用 | 中 |

### 1.3 `replace_virtual_path`（harness 层，thread_data 显式传入 —— **正范例**）

| 调用文件 | 角色 |
|---|---|
| `harness/tools/builtins/prep_metric_plan_tool.py` | **正范例**：`replace_virtual_path(uf, thread_data)` 显式传 thread_data，不依赖进程 env |
| `harness/tools/builtins/prep_chart_plan_tool.py` | **正范例**：同上 |
| `harness/sandbox/tools.py:486` | 定义点 |

**`replace_virtual_path(path, thread_data)` 是 A 类终态想要的统一入口雏形**——它显式接收 session 级 `thread_data`，天然 session 隔离（不依赖进程全局 env）。问题在于它目前只在 prep_* 工具用，ethoinsight 库层的 `resolve_sandbox_path` 还在走进程 env。

---

## 2. 已出过事的路径（memory 实证，按风险排序优先收口）

来源：memory `feedback_*` 系列记录的 dogfood 故障。

| 路径 | 出过的事 | 根因 | memory |
|---|---|---|---|
| **statistics dispatch** | 列对齐参数没接（PR#141） | statistics 路径 resolve 没接 column_aliases 参数 | `feedback_2026-06-16_statistics_third_layer_file_subject_bridge` |
| **validate_catalog** | `result_file_unreadable` 误报（PR#132） | Step8 validate 没包 `_scoped_path_env` → 路径解析失败 | `feedback_run_metric_plan_step8_validation_not_scoped_path_env` |
| **进程内跑脚本** | 全 FileNotFoundError | 进程内无 mount，compute 脚本裸 `args.input` 不 resolve /mnt | `feedback_run_metric_plan_inprocess_scripts_dont_resolve_sandbox_path` |
| **虚拟路径 env 故障族** | 进程内调 ethoinsight 读 /mnt 静默退化 | resolve_sandbox_path env 不全 | `project_2026-06-16_virtual_path_resolution_env_family_root_fix` |

---

## 3. 统一 I/O 入口的目标签名 + 分批收口顺序

### 目标签名（A 类终态，渐进逼近）

```python
# 统一 workspace 文件 I/O 入口（A 类终态）：base dir + 路径映射都从 session 注入态取。
def open_workspace_file(virtual_path: str, thread_data: ThreadDataState, mode: str = "r"):
    """读/写 workspace 文件：从 thread_data 取 base + replace_virtual_path，绝不依赖进程 env。"""
    real_path = replace_virtual_path(virtual_path, thread_data)
    return open(real_path, mode)
```

核心约束：
- 入口**显式接收 `thread_data`**（不读进程 env）→ session 隔离天然成立，并发 subagent 不串台。
- ethoinsight 库层的 `resolve_sandbox_path`（走进程 env）逐步替换为接收显式 base/path-mapping 的版本，最终库层不依赖 `DEERFLOW_PATH_*` env。
- 保留 `DEERFLOW_PATH_*` env 仅作 sandbox 子进程（bash 工具）的兼容层，**进程内调用一律走显式参数**。

### 分批收口顺序（按风险排序，每批独立 spec + 回归）

1. **批 1（最高风险，已出过事）**：`run_metric_plan_tool.py` + `validate_catalog.py` 的 `_scoped_path_env` 包裹 → 改为显式传 base/path-mapping。回归：`test_run_metric_plan.py` + 进程内脚本不再 FileNotFoundError 的断言。
2. **批 2（高并发串台风险）**：`metric_aggregation.py` 的 env 消费 → 改为显式 thread_data。回归：并发聚合 n-subject 不串台断言（两个 thread 同时聚合，结果互不污染）。
3. **批 3（库层 I/O 入口）**：`scripts/_cli.py` 的 `read_inputs_json` / `read_groups_json` / `save_output_json` 接收显式 base 参数，`resolve_sandbox_path` 退化为兜底。回归：`test_statistics_column_alignment.py` + 进程内路径解析族测试。
4. **批 4（parse 层）**：`parse/_core.py` 的 env 消费 → 显式参数。回归：parse header 不依赖 env。
5. **批 5（CLI 层）**：`catalog/cli.py` 的 `_resolve_virtual_workspace_dir` env 反推 → 显式虚拟路径参数。回归：`test_catalog_resolve_env_var.py`。

---

## 4. 不做什么（风险边界，对齐 spec §5）

- **本清单不改任何 A 类代码**。spec §5 明确 A 类改动面大、分批做、与正在跑的修复隔离。
- **不发明新机制**。`ToolRuntime` + `runtime.state["thread_data"]` + `replace_virtual_path(path, thread_data)` 已是终态范式（prep_* 工具的正范例）；A 类收口是把所有站点拉到这个范式上，不是新建抽象。
- **不删除 `DEERFLOW_PATH_*` env**。它是 sandbox 子进程（bash）的兼容层，删了进程内/容器侧会炸；只在进程内调用改为显式参数。

---

## 5. 与 B 类收口（P4 已完成）的关系

B 类（实验状态）已由 P4 收口：统一 `read_context` reader + parity 回归网（`tests/test_column_aliases_parity_net.py`）+ 守护测试（`tests/test_no_cli_passes_session_state.py`）。A 类（物理路径）结构同构：同样是「散落各处 → 统一从 session 注入态自取 → 一条断言保证全员自取」，但因改动面大分批做。两类的终态范式一致：**唯一源（thread_data / context）+ 自取（不传参）+ 回归网守住全集**。
