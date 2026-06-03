# 2026-06-03 会话交接 — FST seal 卡死真根因双层修复（未 commit）+ 前端现代化 worktree review 通过

> **本 handoff 用途**：交接一次会话，主线是**根治 FST dogfood 反复 "data-analyst terminated without emitting handoff" 卡死**。诊断挖到两层真根因并都已修复（**改动全部未 commit，在主仓库 working tree**），但 **skill 改动需重启 dev 才生效，用户正在重跑 dogfood 验证中**。副线是 review 了前端现代化 worktree（已通过，可建 PR）。
>
> **当前状态**：主仓库 dev == origin/dev == `04770788`，working tree 有 10 个 modified + 2 个 untracked（见 §3）。前端 worktree 分支 `feat/frontend-modernization` HEAD=`52471202`，已 rebase 到 dev、已 push。

---

## 0. 一句话现状

FST seal 卡死的真根因是**两层 "parameters_used 报告了未参与计算的幽灵参数"**：① compute 脚本无脑回显全量参数（已修，dev 已热生效）+ ② code-executor LLM 把空 `{}` 当 bug、改用 plan 的全量 parameters_in_use（已修 skill，**需重启 dev 生效**）。两层都修完后 data-analyst step 2.8 会自然跳过、走到 seal。**改动未 commit，等用户 dogfood 验证通过后提交。**

---

## 1. 任务目标

用户 dogfood 大鼠 FST（Porsolt forced swim test XT190，n=1/组 Drug vs Saline）反复出现：
```
Error: Subagent 'data-analyst' terminated without emitting 'handoff_data_analyst.json'.
```
报错信息**误导**（说"忘了调 seal 工具"），真因是 data-analyst 在 step 2.8 参数审计陷入死循环、烧光 12 turn 预算，永远走不到 seal。目标：根治，让 FST/TST 走通到出报告。

---

## 2. 当前进展（全部已验证，**但未 commit**）

### ✅ 诊断：两层真根因（关键，下一 agent 必读）

**触发器**：真实 EV19 FST 导出文件**没有 Activity 列**，只有 EthoVision 自带的 `mobility_state_immobile` 列。所以 immobility 走 **mobility_state 路径**（EthoVision 自己判的），pendulum（10个）+ velocity（2个）共 12 个参数**一个都没参与计算**。pendulum 算法吃的是逐帧 Activity，这份数据根本没有。

**Bug ①（compute 层，已修，dev 已热生效）**：6 个 compute 脚本（FST+TST × immobility_time/latency/bout_count）把 catalog 注入的全 12 参数无脑回显进 `parameters_used`，signal_distribution 还误报 velocity。
- 修复：`_common.py` 新增 `resolve_immobile_with_path`（暴露实际 path）+ `filter_parameters_for_path`（按路径裁剪：mobility_state→空/仅路径无关参数；pendulum→仅 pendulum_*；velocity→仅 velocity_*）；`_resolve_immobile_series` 变薄包装零波及。`_signal_distribution.py` 新增 `resolve_immobility_metadata`，signal_key 与实际路径同源（mobility_state 不报分布）。6 脚本统一调用。
- **顺手修了一个 latent bug**：fallback 底层函数（`detect_pendulum`/`_resolve_immobile_from_velocity`）只接受自己那组具名参数，转发前必须按前缀裁剪 kwargs 否则 TypeError（真实数据走 mobility_state 路径才一直没触发）。
- 实证：真实文件跑修复后 compute → `parameters_used={}`、无 signal_distribution。
- **dev 已热生效**：本轮 dogfood 的 `m_immobility_time_s0.json` = `parameters_used:{}`（因为 ethoinsight 是 import 的库代码，会热加载）。

**Bug ②（code-executor 聚合层，已修 skill，⚠️ 需重启 dev）**：compute 输出 `m_*.json` 已是 `{}`，但 **code-executor LLM 聚合 handoff 时把空 `{}` 当 bug**，改用 `plan_metrics.json` 的 `parameters_in_use`（lead 读数据前 resolve 的全 12 个）→ handoff metrics_summary 又报全量幽灵参数 → step 2.8 照旧死循环。（transcript 实证：LLM 自己写 "empty parameters_used might be a script behavior quirk. I'll go with the plan's parameters_in_use"）
- 修复：`code_executor.py` skill workflow step 3/5 **正面**声明「`[result]` 是 parameters_used 唯一真相源，空 `{}` 是正确且有意义的（表示该路径没用可调参数），plan 的 parameters_in_use 只是"打算用"非"实际用"」（正面措辞，不用"禁止/不要"，遵守 CLAUDE.md §6）。
- **system_prompt 在 agent 创建时构建，必须重启 dev 才生效**。

**Bug ⓿（已于上轮 ab6c3470 修复，本轮仅补回归确认）**：`ParameterAuditFinding` 归一化（`used_value=None`→""、`observed_distribution={"note":"文字"}`→{}）。已有 31 个测试覆盖（`test_parameter_audit_schema.py`），全过。

### ✅ 测试 + 回归
- 新增 `packages/ethoinsight/tests/test_immobility_resolution_path.py`（三路径 × 参数裁剪 + signal_key + 真实 Porsolt 形态回归）。
- `test_code_executor_workflow.py` 加 `test_parameters_used_passthrough_from_compute_result_not_plan`（锁死 skill 契约措辞）。
- **ethoinsight 全量 553 passed / 0 failed**；**后端全量 3603 passed / 2 failed**。
- ⚠️ 那 2 个失败（`test_inspect_gate_guardrail.py::...test_async_delegates_to_sync` / `test_paradigm_identification_gate.py::...test_async_delegates_to_sync`）**是 pre-existing、与本改动无关**——已用 `git stash` 我的改动在 clean dev `04770788` 上复现同样 2 个失败（`2 failed, 3603 passed`，数量完全一致）。是 event-loop 测试隔离问题（`asyncio.get_event_loop()` 在全量跑时被前序测试污染）。**不是回归。**

### ✅ 前端现代化 worktree review（副线，已通过）
- 分支 `feat/frontend-modernization` HEAD=`52471202`（infra commit `7fa7086e` + bug fix `52471202`），已 rebase 到 dev `04770788`、已 push origin。
- review 结论：**mergeable，可建 PR**。硬约束守住（workflow_mode/flywheel 37 处不变、mode 两态未动、Noldus 定制零丢失）、typecheck exit 0、rebase 干净（#80/#81 没碰前端文件，真·0 conflict）、historyMessageRunIds 重置 bug 已修（hooks.ts:1005）。
- 非阻塞待办：分页 infra（`loadMoreHistory`/`hasMoreHistory`）建好但 3 个 page 未接 UI——建议 PR 描述里注明"分页 UI 接线待后续"。
- PR 链接：https://github.com/noldus-cn-beijing/noldus-insight/pull/new/feat/frontend-modernization

---

## 3. 改动清单（**全部未 commit**，主仓库 working tree）

```
M packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py   ← Bug② skill 修复
M packages/agent/backend/tests/test_code_executor_workflow.py                            ← Bug② 回归测试
M packages/ethoinsight/ethoinsight/metrics/_common.py                                    ← Bug① 核心：path 暴露+参数裁剪
M packages/ethoinsight/ethoinsight/scripts/_signal_distribution.py                       ← Bug① resolve_immobility_metadata
M packages/ethoinsight/ethoinsight/scripts/fst/compute_immobility_bout_count.py          ← Bug① (×6 脚本)
M packages/ethoinsight/ethoinsight/scripts/fst/compute_immobility_latency.py
M packages/ethoinsight/ethoinsight/scripts/fst/compute_immobility_time.py
M packages/ethoinsight/ethoinsight/scripts/tst/compute_immobility_bout_count.py
M packages/ethoinsight/ethoinsight/scripts/tst/compute_immobility_latency.py
M packages/ethoinsight/ethoinsight/scripts/tst/compute_immobility_time.py
?? packages/ethoinsight/tests/test_immobility_resolution_path.py                         ← Bug① 新测试
?? docs/handoffs/2026-06/2026-06-02-seal-robustness-phase2-complete-handoff.md           ← 非本会话产物，pre-existing untracked
```

建议提交时拆两个 commit：
1. `fix(ethoinsight): parameters_used 只报实际 resolution path 的参数 + signal_key 对齐 (FST+TST)` — 8 个 ethoinsight 文件 + 新测试
2. `fix(code-executor): parameters_used 以 compute [result] 为唯一真相源,空{}是正确结果` — code_executor.py + 其测试

⚠️ **别 commit** `docs/handoffs/.../2026-06-02-seal-robustness-phase2-complete-handoff.md`（不是本会话产物，pre-existing untracked，归属上一会话）。本 handoff 文件本身可以 commit。

---

## 4. 关键发现（避免重蹈覆辙）

1. **"terminated without emitting handoff" 报错是误导**——真因往往是 subagent 在某步（这里 step 2.8）烧光 turn 预算走不到 seal，不是"忘了调"。加"提醒调 seal"无效（5-29~6-02 至少 6 个 thread 复发，多次加提醒均失败）。
2. **parameters_used 必须如实反映"实际用到了什么"**——报告未参与计算的参数 = 元数据撒谎，把 LLM 带进 step 2.8 死循环。这是 memory `feedback_parameters_used_must_reflect_actual_resolution_path` 锁定的核心教训。
3. **幽灵参数有两个源，必须都堵**：compute 脚本回显（已修）+ LLM 从 plan 回退（已修 skill）。只修一层会被另一层绕过。
4. **dev 热加载边界**：ethoinsight 库代码（import）改动 `make dev` 进程会热生效；**subagent skill 的 system_prompt 在 agent 创建时构建，改 skill 必须重启 dev**。本轮卡死部分就是因为 skill 旧文本还在跑。
5. **接 review/grill 必须现场核实**（memory `feedback_grill_handoff_must_be_verified`）：本会话靠 `git stash` + clean dev 复现，证明 2 个后端失败是 pre-existing 非回归；靠读真实 handoff/m_*.json 证明 compute 修复已生效而 plan 仍灌全量。
6. **harness seal-resume 不要强制 tool_choice**：代码注释（executor.py:712）明文记载探针证明强制 tool_choice 会产空 args，会把"无 handoff"换成"空 handoff"（Sprint5.5 认为更糟）。本会话据此**未动 harness**。

---

## 5. 未完成事项（按优先级）

### 5.1 🔴 高：等用户 dogfood 验证 → 重启 dev 是前提
- **用户正在重跑 FST dogfood**。验证前必须 `cd packages/agent && make stop && make dev`（让 Bug② skill 修复生效）。
- 验证点：data-analyst 是否跳过 step 2.8（因 parameters_used={}）→ 走到 `seal_data_analyst_handoff` → 出报告。
- 若仍卡：先查新 thread 的 `handoff_code_executor.json` 的 `metrics_summary[*].parameters_used` 是否已变 `{}`（证明 Bug② 修复生效）；若仍是全 12 个，说明重启没生效或 LLM 仍在回退。

### 5.2 🟡 中：验证通过后 commit 两批改动到 dev（见 §3 拆分建议）
- commit 前**跑全量确认**（改了 ethoinsight 共享 helper `_resolve_immobile_series`，memory `feedback_pr_merge_must_run_full_suite_on_shared_logic`）：
  - ethoinsight：`cd packages/ethoinsight && .venv/bin/python -m pytest tests/ -q`（预期 553 passed）
  - 后端：见 §8 命令（预期 3603 passed + 2 pre-existing failed，**不是回归**）

### 5.3 🟡 中：前端现代化建 PR
- worktree 已就绪。建 PR（链接见 §2.✅前端），描述里注明"分页 UI 接线待后续"。

### 5.4 🟢 低：seal 卡死的"无 tool_call"模式（仍 live，本会话未碰）
- 本会话只治了"step 2.8 幽灵参数"这一条死循环路径。日志里还有另一种模式（trace b7566a33/cf80346f）：data-analyst 整轮无 tool_call、seal-resume 也救不回。本轮幽灵参数修复消除了最主要诱因，但若未来在别的 step 仍出现"无 tool_call"卡死，需另案处理（不要走强制 tool_choice，见 §4.6）。

---

## 6. 建议接手路径（第一步）

1. **先读 memory**：`feedback_parameters_used_must_reflect_actual_resolution_path`（本会话核心）、`feedback_grill_handoff_must_be_verified`、`feedback_pr_merge_must_run_full_suite_on_shared_logic`、`feedback_skill_describing_tool_output_enables_hallucination`。
2. **问用户 dogfood 结果**：走通了吗？走通 → 跑全量 → commit（§5.2）。没走通 → 按 §5.1 查 handoff 的 parameters_used。
3. 别重复诊断——两层根因已确证，改动已就位，缺的只是"重启 dev + 验证 + commit"。

---

## 7. 风险与注意事项

- ⚠️ **改 skill 必须重启 dev**（system_prompt 构建时机），否则验证假阴性。
- ⚠️ **后端全量那 2 个 `test_async_delegates_to_sync` 失败是 pre-existing**，不是本改动引入（已 stash 复现证明）。别误判为回归而回滚。
- ❌ **别 commit** `.env.wecom`（密钥）/ `2026-06-02-seal-robustness-phase2-complete-handoff.md`（非本会话）。
- ❌ **别给 seal-resume 加强制 tool_choice**（探针证明产空 args）。
- ❌ **别改 data-analyst step 2.8 的"跳过空 parameters_used"逻辑**——它本就正确，根治在上游不灌幽灵参数。
- ⚠️ 前端 worktree 别把 mode 扩成上游四态、别删 workflow_mode（产品特性）。

---

## 8. 关键路径/命令速查

- 主仓库 = dev：`/home/wangqiuyang/noldus-insight`（HEAD = origin/dev = `04770788`）
- 前端 worktree：`/home/wangqiuyang/noldus-insight/.claude/worktrees/frontend-modernization`（HEAD=`52471202`）
- ethoinsight 全量：`cd packages/ethoinsight && .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`
- 后端全量：`cd packages/agent/backend && DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`
- 前端 typecheck：`cd packages/agent/frontend && npx tsc --noEmit`
- 重启 dev：`cd packages/agent && make stop && make dev`
- 单指标 compute 验证（真实文件，确认 parameters_used 裁剪）：
  ```
  cd packages/ethoinsight
  F="<某个真实 FST txt/xlsx::sheet 路径>"
  .venv/bin/python -m ethoinsight.scripts.fst.compute_immobility_time --input "$F" --output /tmp/o.json --parameters-json '{...全12参数...}'
  # 期望 parameters_used={}（mobility_state 路径）、无 signal_distribution
  ```
- 真实 dogfood handoff 位置：`packages/agent/backend/.deer-flow/users/<uid>/threads/<tid>/user-data/workspace/handoff_code_executor.json`

## milestone 建议
本会话让 **seal/handoff 鲁棒性** track 到达 checkpoint：定位并修复了 FST/TST dogfood 反复卡死的**两层真根因**（compute 幽灵参数 + code-executor plan 回退），加双层回归测试，且前端现代化 track 完成 review→rebase→可合并。建议待用户 dogfood 验证 + commit 后，更新 milestone：FST/TST parameters_used 路径化（消除 step 2.8 死循环）+ 前端上游现代化 infra 嫁接完成。
