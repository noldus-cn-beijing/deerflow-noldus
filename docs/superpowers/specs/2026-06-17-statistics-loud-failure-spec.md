# Spec 2 (P2): statistics 崩溃被静默吞成空结果 —— 亮可机读降级信号（P7 熔断器的信号源）

> 日期：2026-06-17
> 类型：结构病根治（红线一：降级不静默）
> 触发源：2026-06-17 EPM dogfood，gateway.log 实证。
> 状态：待 review → 批准后 worktree 实施。
> **遵循工程实践**：[红线一](../../refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md#红线一失败必须响亮数据层失败必须-fail-loud-触发自愈hitl禁止静默降级)。
> **与 P7 的关系（重要）**：本 spec 是 **P7 降级熔断器的信号源（L2）**。P2 负责"statistics 崩了就亮 `statistics_status=crashed` 信号、不再静默 statistics={}"；P7 负责"监听这个信号 → 有限自救 → HITL 确认 → 重试"。**P2 不再主张"直接 fail-loud 报错摆烂"**（那太硬）——它只保证降级不静默、信号可机读，怎么处理降级交给 P7 熔断器。建议 P2+P7 一起实施（信号+熔断同 PR），或 P2 先落信号、P7 再落熔断。

---

## 0. 症状（gateway.log 实证，thread 9409c66f）

```
11:20:28 [run_metric_plan] statistics 失败 (rc=1): ZeroDivisionError: float division by zero
11:20:28 [run_metric_plan] done: status=completed n_total=140 n_completed=140 n_failed=0
```

statistics 脚本崩溃（rc=1）后，**同一秒** run_metric_plan 报 `status=completed`、`n_failed=0`、handoff `statistics={}`。下游 data-analyst 收到空 statistics，按 fast-fail 规则 1a 理解成"本次无推断统计"→ 走描述性手算螺旋。

**这是"为什么 statistics={} 反复出现还查不到原因"的结构机制**：statistics 有≥3 个独立成因（B=stats gate 误 skip、C=env 没包、A=ZeroDivision），全部汇到 `statistics={}` 这一个出口，且都伪装成 completed。每修一个成因就宣称"修好 statistics 空了"，但出口还在，下个成因继续撞。

## 1. 根因：失败信息在 run_metric_plan Step7/Step8 边界丢失

`run_metric_plan_tool.py` Step7（约 line 247-257）：

```python
_tid, src_rc, src_err = _stats_runner(stats_script, stats_argv, "statistics")
if src_rc == 0:
    statistics_payload = json.loads(...)   # 成功才填
else:
    logger.warning("statistics 失败 (rc=%d): %s", src_rc, src_err)   # 崩了只 warning
    failures.append({"id": "statistics", "error": src_err[:500]})    # 记一笔
# ... statistics_payload 保持 None
```

两个缺陷：
1. **statistics 崩溃没改变顶层状态**：`status` 仍由 compute 的 140/140 决定为 `completed`，`n_failed=0`（statistics 那笔 failure 没算进 n_failed，n_failed 只数 compute 任务）。
2. **崩溃（statistics_payload=None）与"本就无统计"（单组分析合理 skip）在 handoff 里字节相同**：两者 handoff 都是 `statistics={}`。下游无法区分。

下游 data-analyst 的 fast-fail 规则 1a 把"statistics 为空"一律当"本次无推断统计"，于是把"统计崩了应当报警重试"误当"正常走描述性"。

## 2. 修法（照红线一的正模式 1/2/3）

### 2.1 正模式 1：子步骤失败改变顶层状态 / 亮可机读信号

statistics 脚本非零退出（且 plan 本应跑 statistics，即 statistics 段未被合理 skip）时：

- **gate_signals 亮一个新信号 `statistics_status`**，三态可机读：
  - `"ok"`：statistics 跑成功，payload 非空。
  - `"crashed"`：statistics 脚本非零退出（崩溃）——**带 error 摘要**。
  - `"absent_by_design"`：plan.statistics 段本就被合理 skip（单组 / n 不足，skip_reason 已知）。
- **`statistics_status == "crashed"` 时，run_metric_plan 的 `status` 不再无条件 completed**：建议降为 `status="completed_with_errors"`（若无此态则 `partial`），并把 statistics 崩溃计入一个明确字段（如 `n_failed_steps` 含 statistics，或单列 `statistics_error`）。具体状态枚举实施时对齐现有 CodeExecutorHandoff schema 的 status 取值（completed/partial/failed/...），**关键是 status 必须可让下游区分"statistics 崩了"**。

> 实施第一步：核实 CodeExecutorHandoff 的 status 枚举有哪些值，选一个能表达"compute 全成功但 statistics 崩了"的态；若没有合适的，加 `completed_with_errors`（三态对齐，参考之前 partial 补齐的做法）。

### 2.2 正模式 2：空容器 ≠ 完成（对账）

`statistics={}` 落 handoff 前，对账：plan 是否声明了 statistics 段？
- plan 有 statistics 段 + payload 为 None → 这是**崩溃**（`statistics_status="crashed"`），不是"无统计"。
- plan 无 statistics 段（或有明确 skip_reason）→ 这是**设计内缺席**（`statistics_status="absent_by_design"`）。

### 2.3 下游对"数据崩溃"必须 fail-loud + 自愈/HITL，不降级出残废报告（用户 2026-06-17 立场）

**用户明确推翻了"statistics 崩了就走描述性 partial"这条降级。** 数据层崩溃不是"本次没统计、凑合给个描述性"，而是"数据计算出错了，得修"。data-analyst 收到 `statistics_status="crashed"` 时：

- **不走描述性手算路径**（那是把数据崩溃藏成一份残废报告）。
- 而是 **fail-loud**：handoff `status=failed`（或 partial 但 method_warnings 头条赫然标"⚠️ 统计层崩溃，本次结果不可用"），key_findings 第一条就是"统计计算崩溃：<error 摘要>，需修复后重跑或检查数据"。
- 让 lead 有机会**自愈或 HITL**：lead 读到 data-analyst 的 failed/统计崩溃 → 可以重新派 code-executor（如果是偶发）、或 `ask_clarification` 问用户"数据里某指标全为 0 / 异常，是否符合预期？是不是数据有问题？"

对比 `statistics_status="absent_by_design"`（单组/n 不足，本就无推断统计）：**这种才走描述性 partial**（合理缺席，不是崩溃）。两者由 §2.1 的信号区分。

> 关键区别：`crashed` → fail-loud + 触发修复/问用户；`absent_by_design` → 正常描述性 partial。现状把两者都当后者处理，把数据崩溃糊弄成"仅描述性报告"。

> ⚠️ 这处 prompt 改动是承重的下游消费——真正的结构修复在 §2.1（harness 亮 `statistics_status` 信号）。信号不亮时 prompt 怎么写都没用。但本 spec 与 P1 的关系要讲清：**P1 让"用正确数据不该崩"成立（修 ZeroDivision）；P2 让"万一还崩，响亮报错触发自愈，而非降级糊弄"。** 两者都要——P1 是"本不该崩"，P2 是"崩了也不许藏"。

## 3. 测试（红→绿，照红线一可执行判据）

`packages/agent/backend/tests/`（run_metric_plan 测试用 ProcessPoolExecutor 注入 runner 的既有模式，见 memory `feedback_processpoolexecutor_test_runner_injection`）：

- `test_statistics_crash_sets_crashed_signal`（**修复前必红**）：注入一个 statistics runner 返回 rc=1 → 断言 `gate_signals.statistics_status == "crashed"` 且 status 非裸 completed（可区分）。当前代码 status=completed/n_failed=0 → 红。
- `test_statistics_skipped_by_design_sets_absent`：plan.statistics 段有 skip_reason → 断言 `statistics_status == "absent_by_design"`，status 正常 completed。
- `test_statistics_ok_sets_ok`：statistics 成功 → `statistics_status == "ok"`，payload 非空。
- **端到端语义**：`test_crashed_distinguishable_from_absent` — 两种情况的 handoff 不再字节相同（crashed 带 error，absent 带 skip_reason）。
- data-analyst prompt 改动无单测可咬，靠 dogfood 复跑验证（crashed 时不再走手算螺旋而是报"统计崩溃"）。

## 4. 风险边界

- 不改 statistics 计算逻辑本身（那是 P1）；本 spec 只改"崩溃如何被上报"。正交。
- status 枚举改动要全链核对：gateway 读 handoff status 的地方、前端展示、seal schema 校验——别引入未知 status 值导致 schema 拒绝（参考 memory `feedback_dataanalyst_reportwriter_handoff_status_missing_partial`）。
- `absent_by_design` 的判据（plan 是否合理 skip statistics）要复用 resolve 已有的 skip_reason，不另造判断。

## 5. 为什么这是根治而非打地鼠

P1（ZeroDivision）修后，下一个让 statistics 崩的成因（成因 D）出现时：**有了 statistics_status="crashed" 信号 + 回归测试，它会响亮失败而非静默空**。用户/CI 立刻看见"statistics 崩了:<错误>"，而不是又一次 `statistics={}` 伪装 completed、靠 dogfood 撞。**这一层才终结"statistics={} 反复出现"这个元问题。**
