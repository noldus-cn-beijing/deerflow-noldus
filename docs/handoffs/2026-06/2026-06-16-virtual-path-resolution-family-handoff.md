# 交接：2026-06-16 虚拟路径解析 env 故障族根治（feat/virtual-path-resolution-family）

## 起点
- spec：[docs/superpowers/specs/2026-06-16-virtual-path-resolution-env-family-spec.md](../../superpowers/specs/2026-06-16-virtual-path-resolution-env-family-spec.md)
- 基线：origin/dev `a6a49ef7`（spec 写的是 `868ed712`，但当前 origin/dev 已推进到 a6a49ef7；本次基于 a6a49ef7 建 worktree）
- worktree：`git worktree add -b feat/virtual-path-resolution-family /tmp/wt-virtual-path origin/dev`
- 分支：`feat/virtual-path-resolution-family`，**已 push，待用户建 PR**

## 做了什么（根治，非点修）
收口"ethoinsight 机制 B（`resolve_sandbox_path`）依赖 `DEERFLOW_PATH_*` env 才能解析 `/mnt` 虚拟路径"的整个故障族。#5（run_metric_plan Step8 validation）/ #6a（prep `_derive_group_counts`）是该族两个已点修实例；本 spec 把兜底从"每个出血点各打补丁"上升为"机制 B 本身的能力"，让"进程内调 ethoinsight 读 workspace /mnt 文件"不再依赖调用方记得设 env。

### 改动①（根治）— `resolve_sandbox_path` 加 `workspace_base` 参数
`packages/ethoinsight/ethoinsight/scripts/_cli.py`：
- 新增可选 `workspace_base: str | Path | None = None`。解析优先级：① env（沙箱子进程/worker 内不变）→ ② **env 全扫完后**若仍无 env 且给了 `workspace_base` 且路径是 workspace 前缀，用 base 拼后缀兜底 → ③ 原样返回（fail-safe）。
- **关键设计决策（spec 未显式写、实施时落实）**：`workspace_base` 兜底放在 env 循环**全部结束后**，而非循环内某前缀命中即 return。这保证"workspace env 缺失时退化到 user-data env"的历史语义不变——若在循环内 return 会抢占这条退化路径，违反"env 优先路径字节不变"红线。docstring 写明此约束。
- **debug 可观测信号**（改动③）：无 env 无兜底原样返回时 `logger.debug` 一行（非 warning——正常沙箱外测试合法走这条，不该刷 warning）。给未来排查"读不到 /mnt 文件"留 grep 锚点。
- docstring 把隐性契约显性化：harness 进程内直接调 ethoinsight 读 /mnt 文件时必须传 `workspace_base`（或先 `replace_virtual_path` 预解析），不能依赖 env。

### 改动②（去重）— `_derive_group_counts` 收口到机制 B
`packages/ethoinsight/ethoinsight/catalog/resolve.py:1079`：
- 原来手维护三候选路径解析（物理路径 / `resolve_sandbox_path(groups_file)` / `workspace_dir/basename`）→ 收口为两候选：`resolve_sandbox_path(groups_file, workspace_base=workspace_dir)` + `Path(groups_file)` 物理兜底。
- 不再在 resolve 里维护第二套 workspace 兜底逻辑（消除双存），同一处方收口到机制 B。行为等价（#6a 既有 13 测试 + `test_resolve_self_derives_with_virtual_groups_file_no_env` 仍绿坐实）。

### 不做什么（守 spec §2 边界）
- 不动机制 A（`replace_virtual_path`，harness 层，thread_data 无 env 依赖）。
- 不强行统一 A/B（分属不同包，合并会让 ethoinsight 反向依赖 harness）。
- **不删 #5 的 `_scoped_path_env`、不删 #6a 的 prep 显式派生**——layer③ 双保险保留，本 spec 是双保险的底层兜底。
- 不动 data-analyst #6b prompt。
- `workspace_base` 只对 workspace 前缀兜底，不泛化到 uploads/outputs（uploads 兜底由 prep 已做的 `replace_virtual_path` 在传入前解决）。

## 测试（全 importlib 加载 worktree 源，守 worktree 共享主仓 venv editable 指向主仓的铁律）
- **新增** `packages/ethoinsight/tests/test_resolve_sandbox_path_workspace_base.py`（9 测试）：
  - `TestResolveSandboxPathWorkspaceBase`（7）：env 优先 / workspace_base 无 env 兜底 / 精确前缀 / 只对 workspace 前缀不泛化 uploads / 无 env 无 base passthrough / 真实路径幂等 / 旧调用形态零变化。
  - `TestEnvPathUnchangedOnWorktreeSource`（2 红线守护）：env longest-prefix 优先 / workspace 无 env 退化到 user-data env（且 workspace_base 不抢占 env）。
- **层 B**：复用 #6a `tests/test_stats_gate_self_derive_counts.py`（13 测试），改动②后仍全绿 = 等价证据。
- **回退验证坐实**：临时去掉 workspace_base 兜底分支 → `test_workspace_base_fallback_when_no_env` + `test_workspace_base_fallback_exact_prefix` 红（2 failed），其余 7 绿；恢复后全绿。符合 spec §3 回退锚点要求。
- **ethoinsight 全量**：844 passed, 70 skipped。
- **backend 回归**（#5/#6a/run_metric_plan/sandbox 相关，importlib 加载 worktree 源）：65 passed，无回归。
- **裸导入**：`import ethoinsight` + `resolve_sandbox_path` / `_derive_group_counts` 签名正确。

## 测试环境坑（重要）
- worktree 借主仓 ethoinsight `.venv`（editable `_editable_impl_ethoinsight.pth` 指向主仓源）。**实测**：从 `/home/wangqiuyang`（主仓父目录）cwd 跑，`import ethoinsight` 命中 **worktree 源**；从 `/tmp` cwd 跑命中主仓源——cwd 影响 editable finder 解析（确切机制未深究，行为可复现）。
- 因此既有直接 import 的测试 + importlib 加载的测试，在 `/home/wangqiuyang` cwd 下都命中 worktree 源（真绿）。但为绝对保险，新增的层 A 测试 + 红线守护都用 importlib 显式加载 worktree `_cli.py`（不依赖 cwd 行为）。
- backend `#6a`/`#5` 测试用 importlib 加载 worktree `prep_metric_plan_tool.py`，其内部 `from ethoinsight...import` 链在该 cwd 下命中 worktree ethoinsight（全链路 worktree = 真绿）。
- **lint 基线债**（非本次引入）：`_cli.py` 有 2× F821（`sys` undefined，行 258/261，主仓同款）+ I001（import 排序，主仓同款 3 处）。ruff check 未新增错误类型。

## 下一步（给用户）
1. 建并合并 PR。
2. merge 后建议 dogfood 复跑 EPM（验证 #5/#6a 场景仍正常，且新调用点天然可靠）。
3. 未来任何"harness 进程内调 ethoinsight 读 workspace /mnt 文件"的新代码，调 `resolve_sandbox_path(path, workspace_base=<真实workspace路径>)` 即可（thread_data 里就有），不再是潜伏雷——这是本次根治的核心收益。
