# 交接：chart-maker 批量执行 = run_chart_plan 确定性工具（诱因层治本，待取证 + 写 spec）

> **会话日期**：2026-06-24
> **一句话**：chart-maker 现在让 LLM 自己 bash 逐张/分批画图（plan 确定性但执行靠 LLM），112+ 张 per_subject 图撞 loop-detection 熔断（warn=30/hard=50）→ 图没画完 + 还伪报 completed。**用户拍板治本方向 = 给 chart-maker 加一个 `run_chart_plan` 确定性工具（对标 code-executor 的 `run_metric_plan`）**：LLM 调一次 → 工具内 ProcessPoolExecutor 并行画完 plan_charts.json 全部图 → 确定性落盘 + 返回真实计数。LLM **0 次 bash 画图**。
> **为什么交接**：上下文出错/快满。方向已定、范本已找到、待取证项已列清，待接手 agent **先取证（守 ETHO-2 误诊教训，不凭推断写 spec）再写 spec**。
> **代码基线**：dev HEAD `f1a98458`。

---

## 〇、给接手 agent 的两步走（用户明确：先取证，再写 spec）

1. **先取证**（§三）：坐实 chart-maker 这次 dogfood 实际怎么调 bash 的（逐个/分批/重试放大哪种）+ run_metric_plan 范本可抄性 + 绘图脚本能否进程池并行。**不跑代码不凭推断写根因**（memory `feedback_etho2_spec_misdiagnosed_infra_bug_already_schema_rejected`：ETHO-2 就是没跑代码把"schema 已拒"误诊成"registry 不自洽"）。
2. **再写 spec**：`run_chart_plan` 确定性工具，照 `run_metric_plan_tool.py` 范本。

---

## 一、背景：这条线怎么来的（2026-06-24 EPM 28-subject dogfood）

用户跑 EPM dogfood（数据 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`，28 xlsx），暴露一串问题，已修/已写 spec 的：
- **#187 漏注册 BUILTIN_TOOLS**（data-analyst fill/finalize 工具没注册 → 生产 100% FAILED）→ 已修 `6a4e4bf9`（已 push）。
- **ETHO-10 chart 伪完成**：outputs/ 0 png，但 handoff `status=completed` + chart_files 列 113 个不存在路径。dogfood 坐实根因 = plan 113 张图（1 aggregate + **112 per_subject**），2.2 对账门只盯那 1 个 aggregate → per_subject 全漏画也放行。→ spec 已写 `docs/superpowers/specs/2026-06-24-etho10-chart-maker-product-reality-invariant-spec.md`（封存核磁盘真实性门 + 增量补做）。
- **#187 fill 并发竞态**：fill 读-改-写无锁 + 固定 tmp → 同消息多 fill 抢同一 tmp → handoff 损坏。→ spec 已写 `docs/superpowers/specs/2026-06-24-fill-handoff-concurrent-write-race-spec.md`。

**本 handoff 是第 4 条线 = ETHO-10 的「诱因层」**：图为什么一开始就没画完。上面 ETHO-10 spec 治的是「没画完却伪报 completed」（诚信层）；本线治「让图一开始就画完」（性能层）。两层正交。

---

## 二、根因方向（待取证坐实，但代码已核实大半）

### 已核实的代码事实
1. **chart-maker 的 LoopDetectionMiddleware 阈值 = warn=30 / hard=50**（`subagents/executor.py:973` 给所有 subagent 统一传，非默认 3/5）。
2. **chart-maker 执行绘图靠 LLM 调 bash**（`subagents/builtins/chart_maker.py` step 8）：prompt 教它把 charts[] 的 `python -m ethoinsight.scripts.* <args>` 拼成 `bash -c "...&...&...wait"` 并行跑，失败的「逐个串行重试一次拿 stderr」。
3. **bash 频率熔断按 tool *type* 计数**（`loop_detection_middleware.py:485`，per-tool override 优先，否则用全局 warn/hard）。bash 调用 ≥50 次 → hard stop → partial strip 掉 bash 的 tool_call → 「被中断未返回」。
4. **对比 code-executor 不撞**：它有 `run_metric_plan` 确定性工具（`run_metric_plan_tool.py`），LLM 调**一次**，工具内 ProcessPoolExecutor 并行跑 140 指标 + 落盘，LLM **0 次 bash**。chart-maker 是「计划确定性（prep_chart_plan）但执行靠 LLM bash」的半套。

### 病根（方向已明，机制待取证）
**chart-maker 把「批量执行」留给了 LLM 调 bash**，112+ 张图无论怎么分批/并行都容易撞 50 次硬上限；且执行经 LLM → 还能伪报完成（ETHO-10）。**code-executor 早把批量执行收进确定性工具（run_metric_plan），chart-maker 没有。**

### ⚠️ 待取证确认的关键点（决定 spec 细节，别跳过）
- **chart-maker 这次实际调了几次 bash、怎么调的**？三种可能、修法不同：
  - 可能 A：一条并行 bash 画一批，但失败多 → 逐个串行重试（prompt step 8）→ 重试次数飙到 >50。
  - 可能 B：LLM 没按 prompt 用一条 bash，逐个调 → 112 次。
  - 可能 C：LLM 想并行却写成 N 条独立 bash → 超 50。
  - **run_chart_plan 对三者都治本（LLM 0 次 bash），但取证能确认"真是 bash 熔断"而非别的原因致图没画**（守 ETHO-2 教训）。
- 取证手段：dump 这次 dogfood 的 chart-maker thread 的 AI messages，看 bash tool_call 次数 + gateway.log 找 `Tool frequency hard limit reached`。或独立复现 chart-maker（手段 c，见 §五铁律）。

---

## 三、取证清单（接手第一步，先做这些）

1. **run_metric_plan 范本可抄性**：读 `tools/builtins/run_metric_plan_tool.py`（已核实结构：`ProcessPoolExecutor`(L17) + 测试钩子注入 runner 绕 fork/pickle(L66-67，`_RUNNER` monkeypatch 串行化) + worker 模块级 + `_pool_initializer` 设 `DEERFLOW_PATH_*` env(L78) + 原子落盘 `sealed_by="run_plan"`(L155) + 紧凑返回(L163)）。确认 run_chart_plan 能照抄这套骨架。
2. **绘图脚本能否进程池并行**：
   - 脚本在 `packages/ethoinsight/ethoinsight/scripts/`（`_common/plot_trajectory.py` / `plot_timeseries.py` / `plot_heatmap.py` + 各范式 `<paradigm>/plot_*.py`）。
   - **核 matplotlib backend**：进程池并行画图必须用非交互后端（`matplotlib.use("Agg")`），否则子进程炸。grep 脚本/公共模块是否已设 Agg；没设则 run_chart_plan 的 worker initializer 要设 `MPLBACKEND=Agg` env 或脚本入口设。
   - 核脚本入口签名：run_metric_plan 是「进程内调脚本函数」（无 subprocess 冷启）；绘图脚本若只有 `if __name__=="__main__"` argparse 入口，要么抽出可调函数、要么 worker 里 `runpy`/`subprocess` 调——确认哪种可行（优先抽函数进程内调，对齐 run_metric_plan）。
3. **plan_charts.json 契约**：`prep_chart_plan_tool.py` 已产 plan_charts.json（charts[] 每条含 script/args/output/output_mode）。run_chart_plan 读它、遍历 charts[] 进程池并行执行、核每个 output png 落盘、构造 chart-maker handoff。确认 charts[] 字段够 run_chart_plan 用（script 模块路径 + args 数组 + output 虚拟路径）。
4. **chart-maker bash trace 取证**（坐实诱因，守 ETHO-2）：dump 本次 dogfood chart-maker thread，看 bash 实际调用次数/方式 + 是否 `Tool frequency hard limit reached`。确认 run_chart_plan 真能消除它（而非图没画是别的原因）。

---

## 四、spec 修法方向（取证后据实写，这里给骨架）

**加 `run_chart_plan` 确定性工具（对标 run_metric_plan）**：
- 签名：`run_chart_plan_tool(runtime)` —— 读 workspace 的 plan_charts.json，无需 LLM 传图清单（像 run_metric_plan 读 plan_metrics.json）。
- 内部：`ProcessPoolExecutor` 并行执行 charts[] 全部绘图脚本 → 每个核 output png 落盘 → 构造 `handoff_chart_maker.json`（`sealed_by="run_chart_plan"`，对标 `run_plan`）→ 返回紧凑结果（chart_count / rendered / failed / failed_charts 明细落盘不进 context）。
- **chart_files 天生是磁盘真相**（工具画完即知哪些 png 真存在）→ **顺带堵 ETHO-10 reward hacking**（LLM 不再碰 chart_files，没法伪报）。与 ETHO-10 spec 协同：ETHO-10 的核磁盘门从「防 LLM 伪报」降级为「防御性双保险」。
- **chart-maker prompt 改**：step 8「LLM 拼 bash 并行画图」→「调一次 run_chart_plan」（像 code-executor step 2「调一次 run_metric_plan」）。从 chart-maker 工具集移除/弱化 bash 画图职责。守三指令源（system_prompt 最高权威 + SKILL.md）。
- **loop-detection 阈值不用动**（bash 次数和图数解耦后根本不触发熔断）——这比「放宽阈值到 150/300」高明：图数再多（4000 张）LLM 也只调一次。
- **测试钩子**：照 run_metric_plan 的 `_RUNNER` monkeypatch（注入同步串行 runner 绕 ProcessPoolExecutor fork/pickle，spec §4 单测用，memory `feedback_processpoolexecutor_test_runner_injection_and_ssot_parity`）。

**三问题一刀治**：① 诱因（不撞熔断，113 张一次画完）② ETHO-10（产物经确定性工具不经 LLM，不伪报）③ 架构一致（chart-maker 对齐 code-executor）。

---

## 五、铁律（接手必守）

- **取证优先，不凭推断写根因**（memory `feedback_etho2_spec_misdiagnosed_infra_bug_already_schema_rejected`）。
- **看 subagent 内部行为必须独立复现**（memory `feedback_diagnose_subagent_behavior_replay_subagent_not_dump_lead`）：chart-maker bash trace 不能只 dump lead thread。手段 c：SubagentExecutor + 真实 workspace（set_current_user contextvar + acquire(THREAD_ID)）+ 真实 task prompt + 同 model/max_turns 跑一次 dump ai_messages；破环先 `import app.gateway`。
- **import 环铁律**：加 run_chart_plan 到 `tools/builtins/` + 注册 `tools.py:BUILTIN_TOOLS`（⚠️ #187 就栽在漏注册 BUILTIN_TOOLS，memory `feedback_tool_definition_export_not_equal_registered_in_builtin_tools`：@tool 定义 + __init__ 导出 + **tools.py:BUILTIN_TOOLS 注册** 三处缺一不可）。改完裸导入 `app.gateway` + `make_lead_agent` 0 退出。
- **ProcessPoolExecutor 测试**：注入 runner 串行化（不传 fork），等价性测试厘清对比两端同层（memory `feedback_processpoolexecutor_test_runner_injection_and_ssot_parity`）。
- **matplotlib 子进程**：必须 Agg backend，否则并行画图炸。
- **确定性可测**：worker/tmp 用 pid/tid，不用 `Date.now()`/随机数。
- **守 HarnessX 三病理**：本方案是结构（确定性工具批量执行），不加 prompt 规则（不加「别伪报」「多画几张」提醒）。reward hacking 防御靠「产物经确定性工具不经 LLM」。
- **三指令源**：改 chart-maker 行为先 grep `subagents/builtins/chart_maker.py` 的 SubagentConfig.system_prompt（最高权威）+ SKILL.md。
- 受保护文件 sync surgical：chart_maker.py / tools.py / executor.py。
- **TDD 强制** + 端到端 dogfood 验收（同份 28-subject EPM 复跑，run_chart_plan 一次画完 113 张、handoff chart_files 全真实落盘、不熔断、不伪报）。

---

## 六、关键代码锚点（已核实）
- **范本**：`tools/builtins/run_metric_plan_tool.py`（`ProcessPoolExecutor`(L17)、`_RUNNER` 测试钩子(L66)、`_pool_initializer`(L78)、`run_metric_plan_tool`(L145)、`sealed_by="run_plan"`(L155)、紧凑返回(L163)）
- **上游**：`tools/builtins/prep_chart_plan_tool.py`（产 plan_charts.json，charts[] 含 script/args/output/output_mode）
- **绘图脚本**：`packages/ethoinsight/ethoinsight/scripts/_common/plot_*.py` + `<paradigm>/plot_*.py`（核 matplotlib backend）
- **chart-maker 现状**：`subagents/builtins/chart_maker.py`（step 5 prep_chart_plan / step 8 LLM bash 画图 / `<bash_constraints>` L98「bash 预算最多 ~10 次」——这条注定撞 112 张图）
- **loop-detection**：`agents/middlewares/loop_detection_middleware.py`（`_TOOL_FREQ_SEMANTIC_OVERRIDES`(L196)、freq 判定(L485)）；`subagents/executor.py:973`（给 subagent 传 warn=30/hard=50）
- **工具注册**：`tools/tools.py`（BUILTIN_TOOLS，run_chart_plan 必须加这里——#187 教训）
- **ETHO-10 spec**（协同）：`docs/superpowers/specs/2026-06-24-etho10-chart-maker-product-reality-invariant-spec.md`
- **dogfood 数据**：`/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`
- **playwright 脚手架**：`/tmp/pw-driver/`（login.js 站点 localhost:2026 + 凭据；run-dogfood.cjs / answer-and-capture.cjs）

---

## 七、与其他三条线的关系（别冲突）
- **ETHO-10 spec**（已写，别的 agent 在做）：本线 run_chart_plan 落地后，ETHO-10 的「封存核磁盘门」从主防御降为双保险（产物已不经 LLM）。两 spec 不冲突——ETHO-10 改 seal 对账门，本线改 chart-maker 执行路径 + 加工具，正交。建议本线后合（让 ETHO-10 门先在，本线再消除诱因）。
- **#187 fill 竞态 spec**（已写，别的 agent 在做）：与本线完全无关（data-analyst vs chart-maker）。
- **#187 漏注册修复** `6a4e4bf9`：已 push，本线注册 run_chart_plan 时复用同一教训（grep BUILTIN_TOOLS）。

---

## ✅ 后续：取证完成 + spec 已写（2026-06-24 接手 agent 回写）

接手 agent 按 §〇 两步走（先取证再写 spec）执行完毕。**取证推翻了本 handoff 的「诱因」前提**（守 ETHO-2 教训，如实记录）：

- **「112+ 图撞 loop-detection 熔断致图没画」未坐实**：loop-detection freq 按 **bash 调用次数**计（非 bash 内 python 命令数，`loop_detection_middleware.py:485`）。chart-maker 按 system_prompt step 8 把 ~19 个 `python -m` 塞进一条 `bash -c "...& wait"`，113 图 → **~6 次 bash ≪ warn30/hard50**，结构上不熔断。
- **thread `15c9805e` dogfood 实测**：chart-maker **画全 113/113 png**、SSE 0 个 "hard limit reached"、`sealed_by=model`（诚实自封口）、failed_charts=0。**无熔断、无伪完成。**（与用户同日 dogfood 交付结论一致。）
- ETHO-10 spec 引的「另一 run：0 png + 57 幻影」是**不同 run**，且该 spec line 39 自标 loop-detection 熔断为「强假设需坐实」——至今无 run 坐实。ETHO-10 真根因是 2.2 门只对账 aggregate（另一条线）。

**run_chart_plan 仍写（架构对齐 + 堵 ETHO-10 + 消除 args 重拼脆弱），但价值论证据实改**：不挂「治熔断诱因」，真价值 = ① 执行确定性化（消除 LLM bash 编排 + args 重拼漏 `--parameters-json` 风险，thread 339512dd 真栽过）② 产物经工具不经 LLM（chart_files 天生磁盘真相，堵 ETHO-10 reward hacking）③ 架构对齐 code-executor。熔断列为「潜在风险（LLM 退化逐张 bash 才触发），顺带消除」。

**可行性已进程内实跑坐实**（4 脚本 rc=0 真出 png）。**SealGate 集成解法已坐实**：code-executor 用 run_metric_plan 后已被移出 `_REQUIRES_SEAL`，chart-maker 同构 → 移出两集合即可（precedented，非 deadlock 风险）。

**spec**：[`docs/superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md`](../../superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md)（可直接交付实施 agent）。
**memory**：`feedback_chart_maker_loop_detection_burnout_premise_refuted_by_forensics`。

---

## milestone 建议
「chart-maker 架构对齐 code-executor / 批量执行确定性化」track：chart-maker 从「计划确定性（prep_chart_plan）但执行靠 LLM bash」升级为「计划 + 执行全确定性（prep_chart_plan + run_chart_plan）」，对齐 code-executor 的 run_metric_plan 范式。~~一刀治诱因（不撞熔断）~~ **诱因（熔断）经取证未坐实** + ETHO-10（不伪报）+ 架构一致。checkpoint：「~~dogfood 坐实 chart-maker bash 熔断致图没画~~ **取证推翻熔断诱因前提（thread 15c9805e 画全 113 张、~6 次 bash 未熔断、sealed_by=model）** → 用户拍板批量执行确定性化（架构对齐 + 堵 ETHO-10）→ run_chart_plan 确定性工具 → **取证完成 + spec 已写（可行性进程内实跑坐实、SealGate 解法 precedented）**」。
