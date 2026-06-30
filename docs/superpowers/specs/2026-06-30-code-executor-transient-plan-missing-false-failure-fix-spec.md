# 设计 spec：消除 code-executor「plan_metrics.json 缺失」瞬时假失败（已自愈，根因待坐实）（2026-06-30）

> 来源：2026-06-30 EPM 28 文件端到端 dogfood 实测。诊断走 `diagnose` skill，**磁盘证据坐实**
> （thread `8827351d-e2b1-4292-b69e-a3bde14b5fb0`）。本 spec 含一个**待坐实的开放问题**（见 §三），
> 实施 agent 须先复现/定位再修，不可凭推测改。交别的 agent 实施。

---

## 一、现象 + 已坐实事实

用户看到 code-executor 首次派发返回「**失败：plan_metrics.json 缺失**」（已 seal status=failed），
随后**分析其实全程跑通**——磁盘真相（失败 thread workspace）：
- `plan_metrics.json` ✅ 存在（mtime 16:43:07，147KB，有效）
- `handoff_code_executor.json` ✅ `status: completed`，**28 subject**，含 stats/assessment
- `groups.json` ✅ key 是真实上传文件路径（自动填的分组实际有效，未导致报错）
- `handoff_data_analyst.json`（16:45:48）+ `handoff_chart_maker.json`（16:48:11）全 completed

**结论**：你看到的「失败」是 **code-executor 第一次派发落在 plan_metrics.json 写盘完成之前**，
它正确 seal failed，lead 随即重跑 prep + 重派，**第二趟全程成功**。

**这是 degradation（降级自愈）不是 elimination（消除）**（守 `feedback_code_has_fix_not_equal_bug_eliminated`）：
现象「第一趟假失败吓用户」仍会发生，只是自愈、数据不丢。值得修成「不发生」。

**已排除**：
- ❌ 非 #251/#252 引入（#252 纯前端；#251 只加 stage narration 观测，`wrap_tool_call` 干净
  `return result`、异常透传，不碰派发时序）。用户「两 PR 后出现」是巧合，此 ordering 是 pre-existing。
- ❌ 非路径/线程隔离不一致（plan 写在正确的 per-user workspace，code-executor 读得到——最终 completed）。
- ❌ 非 groups 校验失败（groups.json key 是合法文件路径）。

## 二、已有结构门现状（关键背景）

`guardrails/path_sequence_provider.py:95-110` **已有**派发前置门：派 code-executor 前检查
`plan_metrics.json` 存在且非空，不存在则 **deny + 提示先调 prep_metric_plan**。该门**已注册启用**
（`agents/lead_agent/agent.py:479-486`，gated on `guardrails_cfg.enabled`）。

## 三、⚠️ 开放问题（实施 agent 必须先坐实，再定修法）

**既然 path_sequence guard 已在派发前检查 plan_metrics.json 存在，为什么第一趟还会让 code-executor
跑起来并 seal「plan 缺失」？** 两种待验证机制：

- **H-A（TOCTOU 窗口）**：guard 检查 plan 存在的时刻通过了，但 code-executor 真正 ls 的时刻更早/
  文件尚未 flush 落盘——guard 与 code-executor 自检之间有时序窗口。
- **H-B（guard 未拦该次派发）**：lead 在 prep_metric_plan 返回前就发了 code-executor 的 task tool_call，
  guard 评估时 plan 确实不在 → **本该 deny**，但 deny 没拦住（provider 评估路径/fail_closed 配置/
  bridge 时序问题），code-executor 仍被派出并自行 seal failed。

**实施第一步 = 复现 + 定位（diagnose Phase 1 反馈环）**：
- 拿失败 thread 的 SSE 事件流 / checkpoint 时间线，确认 prep_metric_plan 返回时刻 vs code-executor
  task 派发时刻 vs path_sequence guard 评估结果（deny 了没）。
- 或构造集成测试：lead→prep→派 code-executor 链，注入「prep 写盘延迟」，断言 guard 是否拦住。

## 四、修复方向（坐实根因后择一/组合）

- **若 H-A（TOCTOU）**：guard 检查从「文件存在」加强为「文件存在且 prep_metric_plan 已返回 status=ok
  信号」（用 state 标志位而非纯磁盘探测），消除写盘-检查窗口。
- **若 H-B（guard 没拦）**：修 guard 评估/注册/fail_closed，确保 plan 不存在时 code-executor 派发被
  确定性 deny（lead 收到 deny → 先补 prep）。
- **通用兜底**：lead prompt 不是修法（守 HarnessX 别加 reminder）；终止条件/门是结构层。

## 五、验收（TDD + 防 vacuous）

1. **复现测**：构造「plan 未落盘即派 code-executor」→ 断言 path_sequence guard **deny**（不让 code-executor
   跑起来 seal failed）。先让此测复现假失败（红），再修，再绿。
2. **正向不回归**：plan 已落盘 → code-executor 正常派发跑通（28-subject 路径不破）。
3. **防 vacuous**：去掉修复（guard 加强逻辑）→ 复现测应变红。
4. **裸导入两生产入口**（改了 guard / 派发链）：`import app.gateway` + `make_lead_agent` 0 退出。
5. `make test` + `make lint` 绿；现有 `test_path_sequence_*.py` 全绿不回归。

## 六、不做什么

- ❌ 不在没坐实 H-A/H-B 前凭推测改 guard（diagnose 铁律：先有复现反馈环）。
- ❌ 不加 prompt reminder 修时序（结构层问题用门）。
- ❌ 不改 #251/#252（已排除，与本 bug 无关）。

## 七、关联

- 守 memory：`feedback_code_has_fix_not_equal_bug_eliminated`（判消除 vs 降级）、
  `feedback_seal_missing_root_cause_is_react_no_toolcall_exit_gate_not_fallback`（结构门家族）、
  `feedback_run_metric_plan_inprocess_scripts_dont_resolve_sandbox_path`（路径坑家族，本次已排除但同域）。
- 现有：`path_sequence_provider.py:95-110`（已有门）、`agent.py:479-486`（注册点）、
  `prep_metric_plan_tool.py:319`（plan 写盘点）、`stage_narration_middleware.py`（#251 新增，已排除）。
- 证据 thread：`8827351d-e2b1-4292-b69e-a3bde14b5fb0`。
