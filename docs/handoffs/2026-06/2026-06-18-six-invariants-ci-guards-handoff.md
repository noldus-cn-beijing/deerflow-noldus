# Handoff: 6 个代码不变量 CI 守护落地（2026-06-18）

> 分支：`dev`（已直接 push，commit `4297b155`）
> 状态：实施完成，6 不变量 CI step 本地全绿，已进 dev。
> 缘起：本会话从"这个 repo 的 CI 是干什么的"问答展开 → 诊断出 CI 只守 6 不变量中的 2 个 → 系统性补齐。

## 背景：为什么做这件事

当前 CI（`.github/workflows/backend-blocking-io-tests.yml`）只跑 `pytest tests/blocking_io/`，仅守护 6 个核心代码不变量中的 2 个：

- **#2 依赖声明完整**（CI 已有 `uv sync` + `import ethoinsight` 断言）
- **#4 blocking_io 回归**（CI 已跑 `tests/blocking_io/`）

其余 4 个不变量——**#1 入口无 import 环、#3 harness 不顶层 import ethoinsight、#5 seal/handoff 回归、#6 受保护定制没被 sync 洗掉**——全靠 memory 提醒人遵守，CI 不拦。

**最关键证据**：`tests/test_gateway_import_no_cycle.py`（守护 #1，subprocess 干净导入两生产入口）早已存在且写得对，**但 CI 从不执行它**。这正是"产品起不来"级 bug（import 环让 `make dev` 卡死）踩过 4 次却始终没被 CI 拦住的根本原因——守门员根本没站在门口。见 memory `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`。

## 6 个不变量定义（供后续维护参考）

| # | 不变量 | IDE 为什么抓不到 | 只能靠什么抓 |
|---|---|---|---|
| 1 | 入口无 import 环 | 静态分析不报环 + conftest mock 致 pytest 假绿 | 干净进程裸 import（CI） |
| 2 | 依赖声明完整 | 用本地已装环境，不重读声明 | 干净环境 `uv sync`（CI） |
| 3 | harness 不顶层 import ethoinsight | 本地有 ethoinsight，顶层 import 不报错 | AST 扫顶层 import（CI） |
| 4 | blocking_io 回归 | 数值对错静态判断不了 | 固化断言（CI，原有） |
| 5 | seal/handoff 逻辑 | 逻辑分支错误静态判断不了 | 固化断言（CI，本次新增） |
| 6 | 受保护定制没被洗 | 文件改了 IDE 不知道原来有什么 | 关键串存活校验（CI，本次新增） |

核心洞察：**CI 的价值是覆盖 IDE 的三类盲区**——环境差异类(#2#3)、运行时顺序类(#1)、语义正确性类(#4#5#6)。凡是 memory 里写了 feedback 且机器可检查的违反，都该追问"它进 CI 了吗"。

## 改动（10 files changed, commit 4297b155）

### 阶段 0：修 4 个污染测试（前置依赖，必须最先）
否则阶段 3 的 #5 step 一上线就红。

- `pyproject.toml`：加 `asyncio_mode = "auto"`（auto 模式下 async 测试自动被 pytest-asyncio 跑，无需逐个加 marker）。
- 4 个 `test_async_delegates_to_sync`（`test_inspect_gate_guardrail` / `test_paradigm_identification_gate` / `test_seal_gate_middleware` / `test_guardrail_middleware`）：`def` + `asyncio.get_event_loop().run_until_complete(coro)` → `async def` + `await coro`。

**根因诊断（先诊断再决定）**：这几个测试全量跑红/单跑绿，根因是**测试侧用了过时 asyncio API**（`get_event_loop()` 在已有 running loop 时不创建新 loop），**非性能/被测代码 bug**。诊断方式：临时 `-o asyncio_mode=strict` 复现——strict 下同样红，证明与我的改动无关。

### 阶段 1：#3 扩展 `tests/test_harness_boundary.py`
- 新增 `BANNED_TOP_LEVEL_PREFIXES = ("ethoinsight",)`
- 新增 `_collect_top_level_imports()`：用 `tree.body`（仅模块顶层语句）而非 `ast.walk`，区分"顶层 import（模块加载时触发，破坏 harness 可独立发布）"与"函数体内 lazy import（允许）"。
- 新增 `test_harness_does_not_top_level_import_ethoinsight()`。

### 阶段 2：#6 新建 `tests/test_protected_files_intact.py`
- **SSOT**：从 `scripts/sync-deerflow.sh` regex 解析 `PROTECTED_FILES=( ... )` 数组（不硬编码清单，避免漂移）。
- `test_protected_files_list_is_nonempty_and_parseable()`：守护 SSOT 本身（解析出 ≥20 项）。
- `test_protected_noldus_anchors_survive()`：对 5 个含 Noldus 关键定制的文件校验锚串存活 + 注册表卫生（有 anchor 的文件必须在 PROTECTED 清单里）。
- 锚串映射：`prompt.py`(noldus_order/set_experiment_paradigm/BUILTIN_SUBAGENTS import)、`subagents/builtins/__init__.py`(BUILTIN_SUBAGENTS dict/4 个 CONFIG)、`mcp/tools.py`(4096 截断)、`sandbox/tools.py`(SHARED_PATH_PREFIX//mnt/shared)、`config/paths.py`(SHARED_PATH_PREFIX 定义)。

### 阶段 3：CI workflow 追加 4 个独立 step
`.github/workflows/backend-blocking-io-tests.yml` 在 blocking_io step 后追加：#1（test_gateway_import_no_cycle）、#3（test_harness_boundary）、#5（39 个 seal/handoff 文件精确列举）、#6（test_protected_files_intact）。每个 step 独立命名，失败时一眼看出哪个不变量破了。

### 顺带修 2 个 stale 契约测试（#5 解阻塞）
plan 执行中发现障碍：#5 清单内的 `test_chart_maker_config`（1 处）和 `test_data_analyst_step28_contract`（3 处）单跑就红，期望值与代码当前设计脱节（与 asyncio 无关）。核实为 stale 非 bug：
- `test_chart_maker_config`：`model == "inherit"` 期望过时，chart-maker 有意用 `deepseek-v4-pro-summary`（与 code-executor/report-writer 同组，需 summary 能力）。改为 `model != "inherit"`（不硬编码模型名，锁设计意图）。
- `test_data_analyst_step28_contract`：3 个 prompt 锚句契约测试，prompt 已重写为"产出/交付合一"合并句（seal 重构产物，见 `feedback_seal_missing_root_cause_is_react_no_toolcall_exit_gate_not_fallback`）。锚句对齐当前真实表述，保留每个契约的防退化意图 + 注释说明更新原因。

**注意**：改契约测试期望值要谨慎——保留防退化意图，只把锚句对齐到当前真实表述，绝不能变成掩盖未来真退化的橡皮图章。改动处均加注释说明"为何更新"。

## 验证（全部通过）

- **6 不变量 CI step 全绿**：#2(ethoinsight import OK)、#4(blocking_io 8 passed 1 skipped)、#1(2 passed)、#3(2 passed)、**#5(475 passed)**、#6(2 passed)。
- **污染测试根治**：4 个 `test_async_delegates_to_sync` 一起跑 61 passed，不再互相污染。
- **asyncio_mode=auto 零新增副作用**：全量 4196 passed（改前 4189），失败 7→**3**。差额解释：4 个 asyncio 污染被根治为始终绿 + #6 新增 2 个测试。
- **反向验证**：#3（塞顶层 `import ethoinsight` 变红）、#6（删锚串变红 + 删 PROTECTED 注册表卫生变红）都正确触发。
- **裸导入两生产入口 OK**：`import app.gateway` + `from deerflow.agents import make_lead_agent` 均 0 退出，无 import 环。

## 重要发现：全量污染测试实际有 7 个（memory 只记了 4 个）

memory `feedback_known_full_suite_test_pollution_4_tests` 只记录了 4 个 asyncio 污染。本次诊断坐实全量"确定性失败"实际有 **7 个**，分两类：

1. **4 个 asyncio 污染**（inspect_gate / paradigm_gate / seal_gate / guardrail_middleware 的 `test_async_delegates_to_sync`）→ 本次根治。
2. **4 个 stale 测试**（期望值脱节）：
   - `test_chart_maker_config`（1）+ `test_data_analyst_step28_contract`（3）→ 在 #5 范围，本次修对。
   - `test_local_sandbox_provider_mounts`（3 处失败）→ **不在任何 CI 清单**（sandbox 前缀），是独立的 sandbox 路径格式漂移 stale，**未处理**（不影响 CI，但全量跑仍红）。

**建议下次更新 memory**：把 `feedback_known_full_suite_test_pollution_4_tests` 扩成"7 个已知失败"，区分 asyncio 污染（已根治）vs stale 期望值（待对齐）。若要全量 `pytest tests/` 转绿，还需修 `test_local_sandbox_provider_mounts` 的 3 处 sandbox 路径格式断言。

## 未做 / 边界

- **未建 CI 专项 milestone**。判断：本次是 CI 守护补齐，不是新 feature track，未到 checkpoint 切换/阻塞解除的程度。若后续要把"全量 pytest 转绿"作为独立工作项，那时再建 milestone。
- **未提交别人的在途文件**：`docs/problems/2026-06-17-*.md`、`uv.lock`、2 个 `docs/superpowers/specs/2026-06-18-*.md`、`reports/report for june/` 均非本次改动，留在工作区未动。
- **workflow paths 触发面未扩**：仍只 `packages/agent/backend/**`。#6 读 `scripts/sync-deerflow.sh`（仓库根），若只改 sync 脚本不改 backend，CI 不会触发——边缘情况，改 sync 时手动跑即可。
- **#6 锚串清单是手动维护的**：新增受保护定制时需同步加 ANCHOR_STRINGS。注册表卫生测试会抓"有 anchor 但不在 PROTECTED"的情况，但抓不到"在 PROTECTED 但没 anchor"（后者无害，只是不校验）。

## 后续可选改进

1. 修 `test_local_sandbox_provider_mounts` 的 3 处 stale，让全量 `pytest tests/` 彻底转绿。
2. 扩 workflow `paths` 含 `scripts/sync-deerflow.sh`，让 sync 脚本改动也触发 #6。
3. 加 post-submit nightly workflow 跑全量 + flaky 检测（当前 #5 是精确列举子集，干净但覆盖有限）。
