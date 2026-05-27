# 2026-05-26 chart-maker FST 端到端故障 5 根因彻底修复 (PR #44 已合 + PR fix/chart-maker-fst-contract 待合)

## 当前任务目标

修复 2026-05-25 DeerFlow upstream sync 全部 5 PR 合入后的 FST 端到端测试 chart-maker 卡死问题。grill-me skill 反查证据，定位 5 个独立根因：

| # | 根因 | 落点 |
|---|---|---|
| Layer A | LLM 把宿主机绝对路径写进 handoff JSON，下游 chart-maker 透传给 catalog.resolve 后被 sandbox guardrail 拦截 | PR #44 |
| Layer B | `CodeExecutorHandoff` pydantic schema 没运行时强制 raw_files 必须用虚拟路径 | PR #44 |
| Layer C | `ethoinsight-chart-maker/SKILL.md` 工作流第 2 步指引混乱（"从 handoff 抄 raw_files"，但 handoff 可能被污染）| PR #44 + PR-2 |
| B1 | `catalog/resolve.py` 给所有 chart 写死 `--input <single>`，但 box/bar/struggle 系列脚本只接 `--inputs <JSON>` | PR-2 |
| D1 | catalog.resolve 完全无视 user_intent，"箱线图/轨迹图/时序图" 用户偏好无法过滤 | PR-2 |
| D2 | chart-maker 失败时没有硬规范，bash 预算耗光后静默挂掉，handoff 没写 | PR-2 |
| P1 | lead 派遣 chart-maker 时把 ASKVIZ 选项原文当用户硬性意图转发 | PR-2 |

**状态：PR #44 已合入 dev (HEAD `0cdbc37c`)；PR fix/chart-maker-fst-contract 已 push 待用户在 GitHub 建 PR + merge。**

## 当前进展

### ✅ PR #44 (path-pollution-defense) 已合入 dev

**HEAD commit**：`f9ef705b`，merge 在 `0cdbc37c`。

**三层防御依赖 deerflow 现成 infra (满足 CLAUDE.md 第 12 条"复用 deerflow infra")**：

| Layer | 文件 | 关键改动 |
|---|---|---|
| A | `sandbox/tools.py` | `write_file_tool` + `str_replace_tool` 写盘前调用 `mask_local_paths_in_output(content, thread_data)` 反向 mask，宿主路径自动转 `/mnt/user-data/...` |
| B | `subagents/handoff_schemas.py` | 新增 `CodeExecutorInputs` 子 schema + `field_validator` 强制 `raw_files` 每条 `/mnt/user-data/` 开头 |
| B | `middlewares/experiment_context.py` | `read_handoff(thread_data)` 反向 mask + pydantic soft validate，错误记 `_schema_violations` 但不丢弃 dict |
| B | `middlewares/gate_enforcement_middleware.py` | 把 `state["thread_data"]` 透传给 `get_critical_warnings` |
| C | `skills/.../ethoinsight-chart-maker/SKILL.md` | L24 改为"raw_files 单一真源 = plan_metrics.json.inputs.raw_files" |

**测试**：14 + 4 个新测试（`tests/test_handoff_path_pollution_defense.py` / `tests/test_write_file_path_pollution_defense.py`）+ 全量 3016 passed / 19 skipped / 0 failed。

### ✅ PR fix/chart-maker-fst-contract 已 push (待用户建 PR)

**HEAD commit**：`3c68c650`，已 rebase 在 dev `0cdbc37c` 之上。

**5 类改动**：

#### 1. C 方案：21 个 plot 脚本统一 `--inputs JSON` 契约

- `_cli.py` 新增 `resolve_per_subject_input(args)` helper，支持双签名 `--input <single>` (legacy) 或 `--inputs <json>` (uniform)
- 改 7 个 per-subject 脚本接 `--inputs`：fst/tst plot_activity_intensity, epm plot_open_arm_time_ratio_bar/plot_zone_entry_distribution, oft plot_center_time_ratio_bar/plot_center_entry_summary/plot_time_progress
- `_common/plot_*` 三个原本支持双签名，box/bar/struggle 原本只接 --inputs，不动

#### 2. catalog YAML 加字段

- `ChartEntry.output_mode: per_subject` (默认) / `aggregate`
- `ChartEntry.needs_groups: bool` (默认 False)
- 8 个 paradigm catalog 中 box/bar/struggle 类 chart 全部标 `output_mode: aggregate`；box/bar 标 `needs_groups: true`

#### 3. resolve.py 永远发 `--inputs <JSON>`

- `_chart_to_plan` 物化 `inputs_<chart_id>[_s<idx>].json` 到 workspace_dir
- aggregate + needs_groups 时同时物化 `groups_<chart_id>.json`，从 `handoff_code_executor.json.inputs.groups` 派生（接受 `{arena: group}` 或 `{group: [paths]}` 两种 shape）
- args 永远是 `["--inputs", <json>, ("--groups", <groups_json>,) "--output", <png>, ("--paradigm", <p>)]`

#### 4. catalog CLI 加 `--groups-json` 形参 + chart-maker SKILL.md 同步指引

`groups.json` 由 chart-maker 从 `handoff_code_executor.json.inputs.groups` 派生写盘。

#### 5. D1 + D2 + P1

- **D1**: `resolve.py` 新增 `_filter_charts_by_user_intent`，识别"箱线图/柱状图/轨迹图/热力图/时序图/分布图"等中英文关键词，过滤 catalog charts (legacy 兼容：无关键词命中时透传全部)
- **D2**: `ethoinsight-chart-maker/SKILL.md` 新增"失败硬规范"段落，列 5 种触发条件 (bash 预算耗尽 / 脚本连续失败 / resolve 失败 / guardrail 拒绝 / 未预期异常) 的处理路径，强调"任何提前退出都必须先写 handoff_chart_maker.json"
- **P1**: `chart_maker.py` `input_contract` 加用户意图字段填法说明 (具体图种词原样转 / ASKVIZ 默认选项写"未明确指定")

**测试**：
- ethoinsight 384 passed / 51 skipped（更新 catalog v1.1 args 断言为 `--inputs` 契约）
- backend 3016 passed / 19 skipped / 0 failed
- ruff lint 全绿

## 关键上下文

### 仓库与分支

- 仓库根：`/home/wangqiuyang/noldus-insight/`
- 主分支：`dev` (HEAD `0cdbc37c`，含 PR #44)
- 远程：`origin` → `github.com:noldus-cn-beijing/noldus-insight.git`
- PR #44 (path-pollution-defense) 已 merge
- PR fix/chart-maker-fst-contract 已 push，需在 https://github.com/noldus-cn-beijing/noldus-insight/pull/new/fix/chart-maker-fst-contract 手工建 PR
  - PR body 在 `/tmp/PR-chart-maker-fst-contract-description.md`

### worktrees

- `.claude/worktrees/path-pollution-defense` → branch `fix/path-pollution-defense` (PR #44 已 merge，可清理)
- `.claude/worktrees/chart-maker-fst-fix` → branch `fix/chart-maker-fst-contract` (HEAD `3c68c650`，已 push，待建 PR)

### 测试运行（避坑）

⚠️ [[feedback_worktree_uv_editable_install_pitfall]] worktree 内**没有 .venv**。正确方式：

```bash
cd packages/agent/backend
PYTHONPATH=$PWD/packages/harness:$PWD \
DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml \
/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/python -m pytest tests/ --tb=line -q
```

ethoinsight 测试用主仓库 venv：

```bash
cd packages/ethoinsight
/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/python -m pytest tests/ -q
```

### Lint

```bash
/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/ruff check <file>
```

### gh CLI 不可用

环境中 `gh` 不可用。**push 后告诉用户去 GitHub URL 手工创建 PR**，body 从 `/tmp/PR-*-description.md` 复制。

## 关键发现

### deerflow 已经有完美的反向 path mask infra

`sandbox/tools.py:562 mask_local_paths_in_output(output, thread_data)` 原本用于 bash/grep/view_image 等**读出端**自动反向 mask 宿主机路径回虚拟路径，但 **write_file/str_replace 端没用**。PR #44 把它接入写入端，**零新轮子**完成 Layer A 防御。

[[feedback_deny_messages_must_direct]] 印证：skill prompt 已多处明令禁止 `Path.resolve()`，但 LLM 仍然违反。**prompt-only 提醒不可靠，必须结构性强制**。Layer A (写入侧 mask) + Layer B (pydantic field_validator) 双重保险。

### plot 脚本签名脱节是真根因

- `catalog/fst.yaml` 中 `box_immobility`、`bar_immobility`、`struggle_distribution` 三个 chart 的脚本签名都是 `--inputs <JSON>` + 可能 `--groups <JSON>`
- 但 `resolve.py:_chart_to_plan` 给所有 chart 写死 `args = ["--input", raw_file, "--output", output_path]`
- chart-maker 试错 (`--input` → `--inputs <comma>` → `--inputs <json>` → ...) 耗光预算后 handoff 没写

C 方案 (统一签名 `--inputs <JSON>`) 是最根治：脚本侧 + catalog 侧 + resolve 侧 + chart-maker SKILL 侧契约**完全对齐**，未来再加 chart 不会再有"args 形态"玄学。

### user_intent 之前是装饰字段

`resolve.py:resolve_charts` 把 `user_intent` 作为参数收下但**只写到 PlanCharts 字段**，从不参与过滤。D1 改正这一点：识别 6 大类中英文图种关键词 → 过滤 catalog charts (附加式，无关键词命中时透传)。

## 未完成事项（按优先级）

### 🟡 中优先级

1. **用户在 GitHub 建 PR**：
   - URL: https://github.com/noldus-cn-beijing/noldus-insight/pull/new/fix/chart-maker-fst-contract
   - body 从 `/tmp/PR-chart-maker-fst-contract-description.md` 复制

2. **PR 合并后重新跑 FST 端到端验证**：上传两个 FST 大鼠数据文件 (`轨迹-Porsolt forced swim test XT190-Trial 1-Arena 1/2-Subject 1.txt`)，确认：
   - chart-maker 不再卡死
   - bar_immobility / box_immobility (n=3 时) 跑通组间对比
   - activity_intensity 用 `--inputs <JSON>` per-subject 模式跑通
   - `inputs_<chart_id>.json` / `groups_<chart_id>.json` 物化在 workspace
   - handoff_chart_maker.json 在所有失败路径都被写
   - 用户意图"箱线图、轨迹图、时序图"过滤到 box/trajectory/time 命中 chart (FST 范式无 trajectory，预期 fallback 到 _common/trajectory_plot)

3. **清理 worktree**：PR-2 merge 后清理 `.claude/worktrees/path-pollution-defense` 和 `.claude/worktrees/chart-maker-fst-fix`

### 🟢 低优先级 — 后续改进

1. **catalog `output_mode` / `needs_groups` 字段补全 EPM / OFT / LDB / zero_maze 的非 box/bar chart**：当前只标了明显需要 aggregate 的 box/bar/struggle，其他 chart 默认 per_subject。如果发现某些 chart (如 `zone_entry_distribution` 在多 subject 场景) 应当 aggregate，再补字段

2. **chart-maker 失败硬规范的运行时强制**：D2 当前是 prompt-only。可考虑在 `chart_maker.py` 加 post-tool-call middleware，bash 预算 ≤ 2 时自动写 partial handoff

3. **groups.json shape 适配器扩展**：当前接受 `{arena_key: group_name}` (5-26 FST 上游格式) 和 `{group_name: [subject_path]}` (canonical)，未来若上游再加新 shape (如 `{subject_id: group_name}` 倒挂 dict) 需要再扩 `_build_groups_payload`

## 风险与注意事项

### ⚠️ 不要做的事

1. **不要 force-push 到 main/master**
2. **不要 amend 已合入 PR 的 commit**
3. **不要在 worktree 内跑 backend pytest 不设 PYTHONPATH** — 会用主仓库代码 (uv editable install 副作用)
4. **新增 in-graph `create_chat_model` 必须传 `attach_tracing=False`** — 见 `lead_agent/agent.py` 顶部 INVARIANT docstring (来自 2026-05-25 sync)
5. **新增 plot 脚本不要再用 `--input <single>` 单签名** — 用 `make_plot_parser` (天然双签名) + `resolve_per_subject_input(args)` helper

### ⚠️ 易混淆

- `output_mode: per_subject` 不等于 "脚本只读 1 个文件"。**含义是 resolve 把 chart 展开成 N 条 PlanChart，每条 inputs.json 只放 1 个 path**。脚本侧 `resolve_per_subject_input(args)` 读 paths[0]
- `output_mode: aggregate` 不等于 "脚本读全部文件"。**含义是 resolve 收敛成 1 条 PlanChart，inputs.json 放全部 paths**。脚本侧 `read_inputs_json(args.inputs)` 取列表
- `needs_groups: true` 只对 aggregate 有意义。per_subject 即使写了也会被忽略（resolve.py 只在 `output_mode == "aggregate"` 分支处理 groups）

## 下一位 Agent 的第一步建议

### 起始确认

```bash
cd /home/wangqiuyang/noldus-insight
git fetch origin
git log --oneline dev -5
# 看到 0cdbc37c Merge pull request #44 ... 就表示 PR #44 已合
# 之后期待新一个 PR (chart-maker-fst-contract) merge commit 出现
```

### 如果 PR-2 还没 merge

提醒用户去 https://github.com/noldus-cn-beijing/noldus-insight/pull/new/fix/chart-maker-fst-contract 手工建 PR，body 在 `/tmp/PR-chart-maker-fst-contract-description.md`。

### 如果 PR-2 已 merge 但 FST 端到端还有问题

按 grill-me 流程：
1. 找到 thread 实际 workspace (`backend/.deer-flow/users/.../threads/.../user-data/workspace/`)
2. 检查 `handoff_code_executor.json.inputs` 是不是虚拟路径 (PR #44 应该保证)
3. 检查 `plan_charts.json.charts[i].args` 是不是 `["--inputs", <json>, ...]` 形式 (PR-2 应该保证)
4. 检查 `inputs_<chart_id>.json` / `groups_<chart_id>.json` 是否物化 (PR-2 应该写盘)
5. 检查 `handoff_chart_maker.json` 是否写盘 (D2 硬规范应该保证)

按异常发生的层定位 root cause。

### 如果切到完全不同任务

不需要做任何 chart-maker 相关的事，直接进入新任务。

## 关键文件清单

### Plan / Handoff

- 本文件 — 5 根因彻底修复交接
- [/tmp/PR-path-pollution-description.md](/tmp/PR-path-pollution-description.md) — PR #44 description（已 merge）
- [/tmp/PR-chart-maker-fst-contract-description.md](/tmp/PR-chart-maker-fst-contract-description.md) — PR-2 description（待建）

### 项目文档

- [CLAUDE.md](CLAUDE.md) — 仓库说明
- [packages/agent/backend/CLAUDE.md](packages/agent/backend/CLAUDE.md) — backend 架构
- [packages/ethoinsight/ethoinsight/catalog/resolve.py](packages/ethoinsight/ethoinsight/catalog/resolve.py) — 改动核心 (新增 `_filter_charts_by_user_intent` + `_chart_to_plan` 重写)
- [packages/ethoinsight/ethoinsight/catalog/schema.py](packages/ethoinsight/ethoinsight/catalog/schema.py) — `ChartEntry` 新增 `output_mode` + `needs_groups`
- [packages/ethoinsight/ethoinsight/scripts/_cli.py](packages/ethoinsight/ethoinsight/scripts/_cli.py) — 新增 `resolve_per_subject_input` helper
- [packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py) — `CodeExecutorInputs` (PR #44)
- [packages/agent/backend/packages/harness/deerflow/sandbox/tools.py](packages/agent/backend/packages/harness/deerflow/sandbox/tools.py) — `write_file_tool` / `str_replace_tool` 反向 mask (PR #44)

### 相关 memory

- `feedback_deny_messages_must_direct.md` — 拒绝消息必须含明确指令，prompt-only 不够
- `feedback_worktree_uv_editable_install_pitfall.md` — worktree pytest 盲区
- `feedback_grill_handoff_must_be_verified.md` — handoff 不能直接信任
- `project_2026-05-25_deerflow_sync_all_prs_merged.md` — 5-25 sync 完成记录 (本 chart-maker 修复的上游 context)
- `project_2026-05-25_deerflow_sync_cleanup_pr_pending.md` — sync 善后 PR
