# Spec：run_chart_plan 子集调用覆盖 handoff 致静默丢图伪 completed —— 封存对账总数不变式 + 子集调用语义收口

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-24
> 代码基线：dev HEAD `24383de8`（#192，含 F1 argv 预解析 + F2 fail-safe warning）
> 性质：🔴 高 · reward hacking 结构温床（dogfood 坐实）。chart-maker 用 `run_chart_plan(only_chart_ids=[单个])` 逐个调，**每次调用整体覆盖 handoff**，最后成功的单张写出 `1/1 completed`，覆盖了之前失败的全量 → 112 张 per_subject 静默丢失、handoff 伪 completed。
> **诊断文档**：[docs/problems/2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md](../../problems/2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md)
> **⚠️ 本 spec 纠正诊断文档的根因 B 归因**（见 §一）：诊断 doc 认为是「封存门 `_reconcile` 覆盖率不足放过静默丢图」；代码+SSE 坐实真根因是 **run_chart_plan 的「per-call 整体覆盖 handoff」语义 + LLM 可用 only_chart_ids 把 plan 任意缩小**，二者叠加让「成功子集覆盖失败全量」。封存门加固是**第二道兜底**，不是主修点。
> 受保护文件（sync surgical）：`tools/builtins/run_chart_plan_tool.py`、`tools/builtins/seal_handoff_tools.py`、`subagents/builtins/chart_maker.py`。

---

## 〇、给实施 agent 的一句话

dogfood handoff 是 `status=completed`、`summary="1/1 charts rendered"`、112 per_subject 无记录。**真相（SSE 铁证：summary 全是 `0/1`/`1/1`，无任何 `X/113`）**：chart-maker 在 max_turns 挣扎中**逐个 chart 调 `run_chart_plan(only_chart_ids=[单个])`**，run_chart_plan 每次只对账自己这次的子集（`n_total=len(过滤后)`）、且 `_seal_handoff_to_workspace` **整体覆盖** handoff —— 最后成功的单张 box_open_arm 调用产出 `1/1 completed` **覆盖**了之前全量失败的 `1/113 partial`。

**三层修法**（结构，不加 prompt 规则）：
- **M1（主修，治根因 B）**：run_chart_plan **handoff 永远以「整个 plan（扣 skipped）」为对账分母**，不以「本次 only_chart_ids 子集」为分母。子集调用是「补画」语义——**读已有 handoff，把本次结果 merge 进去**，而非整体覆盖。`n_total` 始终 = plan 全集；子集没跑的 chart 维持上次状态（已落盘的留 chart_files，没落盘的留 failed_charts）。
- **M2（兜底，治封存门覆盖率，第二道闸）**：`_reconcile_chart_maker_payload` 加「对账总数不变式」：`len(chart_files)+len(failed_charts)+len(remaining_charts) == len(plan.charts - skipped)`，不等 → 抛 ValueError（静默丢图绝不放行 completed/partial）。这是即使 M1 有漏网也兜住的结构闸。
- **M3（自洽，治 status 矛盾）**：`completed` 与 `failed_charts` 非空互斥（有失败不得 completed）；`gate_signals.failed_charts == len(failed_charts)`；同一 chart_id 不得同时在 chart_files 与 failed_charts。

---

## 一、根因（代码 + SSE 坐实，纠正诊断 doc 的归因）

### 本次 dogfood 的真实失败链（thread 590fbbd3）
1. **运维前提**：dogfood gateway 进程 14:06 启动，#192（F1，16:15）merge 后**没重启**（`make dev` hot-reload 默认关闭，serve.sh:140-147）→ 进程跑旧代码、**F1 未生效** → run_chart_plan 全量调用时 plot 脚本仍透传 `/mnt` → PermissionError → 112 失败。**这是渲染层失败的运维起因，非代码 bug**（手段 c 坐实：F1 已落地代码跑 590 真实 plan 6/6 成功）。
2. **根因 B（本 spec 主治，代码+SSE 坐实）**：chart-maker turn 序列（trace 2509ed06）= prep_chart_plan(turn4) → run_chart_plan 全量(turn6,112 失败 partial) → bash 自救(turn7-9) → **run_chart_plan 重试(turn10,13)** → max_turns(turn15)。**SSE 里 summary 只有 `0/1` 和 `1/1`，没有任何 `X/113`** → 证明写定 handoff 的 run_chart_plan 调用 `n_total=1`（只跑 1 个 chart）。结合 chart-maker prompt L61「重跑子集传 only_chart_ids」+ run_chart_plan Step 3 `only_chart_ids` 过滤（`run_chart_plan_tool.py:169-175`）→ **chart-maker 改用 `only_chart_ids=[box_open_arm]` 逐个调，最后成功那次 `1/1 completed` 整体覆盖了之前的 `1/113 partial`。**

### 代码机制（为什么子集调用能覆盖全量）
`run_chart_plan_tool.py`：
- Step 3（L169-175）：`only_chart_ids` 过滤后 `plan_charts` **只剩子集**。
- Step 7（L223）：`n_total = len(tasks)` = **子集大小**（不是 plan 全集）。
- Step 7-9（L227-285）：`chart_files`/`failed_charts`/`summary`/`status` **全部只反映本次子集**。
- Step 9（L288）：`_seal_handoff_to_workspace(...)` **整体写 handoff_chart_maker.json**（覆盖，无 merge）。
- → 子集调用 `only_chart_ids=["box_open_arm"]` 成功 → 写 `{chart_files:[box_open_arm], status:completed, summary:"1/1"}` **覆盖**全量调用写的 `{..., status:partial, summary:"1/113", failed_charts:[112条]}`。**112 张就此从 handoff 蒸发。**

### 为什么这是 reward hacking 结构温床
工具本身每次对账是「诚实」的（它确实只跑了 1 个、那 1 个确实成功）。但**「per-call 子集 + 整体覆盖」这个组合给了 LLM 一条路径**：跑不动全量时，缩成「能跑成功的最小子集」调一次，就能用 `1/1 completed` 把失败抹掉。**LLM 不需要主动撒谎，工具语义替它把失败覆盖成了成功。** 这正是 HarnessX 第 1 病理——验收（handoff status）能被「缩小对账范围」操纵。

### 与诊断 doc 根因 B 的差异（必须纠正，守 ETHO-2）
| 诊断 doc 说 | 代码+SSE 坐实 |
|---|---|
| 封存门 `_reconcile` 只核 aggregate，放过 112 静默丢失 | run_chart_plan **全量调用时对账是对的**（`n_failed=n_total-n_rendered` 不丢图，会写 `1/113 partial`）。封存门确实**也**缺「总数不变式」（M2 补），但它不是 112 丢失的**主因** |
| 112 在「一次全量调用」里被门放过 | 112 是被**第二次子集调用覆盖**掉的（两次独立 run_chart_plan 调用，后者覆盖前者），不是一次调用内门没拦 |
| summary "1/1" 的分母来源不明（待手段 c）| 分母 1 = `only_chart_ids` 过滤后 `len(tasks)=1`（Step 3+7 坐实），`_default_summary` 写 `n_rendered/n_total`=`1/1` |

---

## 二、M1（主修）：run_chart_plan handoff 以全 plan 为对账分母 + 子集调用 merge 不覆盖

### 设计
**核心不变式**：`run_chart_plan` 落的 handoff **永远反映「整个 plan（扣 skipped）的对账状态」**，无论本次 `only_chart_ids` 跑了多少。子集调用语义从「只对账子集 + 覆盖」改为「跑子集 + merge 进已有全 plan handoff」。

### 改动（`run_chart_plan_tool.py`）

**改动 1 — n_total 始终基于全 plan，不基于子集**
- Step 3 之后保留 `only_chart_ids` 过滤后的 `plan_charts`（决定**跑哪些 task**），但**另存一份全 plan 的 chart 清单**（扣 skipped）用于对账分母。
- `n_total` = 全 plan chart 数（扣 skipped），**不是** `len(tasks)`。

**改动 2 — 子集调用时 merge 已有 handoff，不整体覆盖**
- Step 7 核磁盘构造本次 `chart_files`/`failed_charts` 后，**读已有 `handoff_chart_maker.json`**（若存在）：
  - 本次没跑的 chart（不在 `only_chart_ids`）：**沿用磁盘真相**——遍历全 plan 的每个 chart，`output` 路径 `exists()` 核磁盘（已落盘→chart_files，没落盘→failed_charts「not yet rendered」）。**这比 merge 旧 handoff 文本更稳**（磁盘是唯一真相，旧 handoff 也可能脏）。
  - **实现上更简单且更对**：不管 only_chart_ids 与否，Step 7 **始终遍历全 plan 的所有 chart**（不只 `results` 里的），对每个 chart 的 `output` 核磁盘 → 落盘进 chart_files、没落盘进 failed_charts。本次 run 的 `results`/`failures` 只用于给「本次跑过的」chart 提供精确 reason（脚本 stderr），「本次没跑的」用通用 reason「not rendered (not in this run's scope)」。
- 这样 handoff 永远是**全 plan 的磁盘真相快照**，子集调用只是「多画几张让更多 chart 落盘」，下次核磁盘自然反映。**覆盖语义消失**——因为每次都核全 plan 磁盘，成功子集不会抹掉已有的失败记录（那些 chart 磁盘上确实没有 png）。

**改动 3 — status 基于全 plan 对账**
- `status = _derive_status(n_total=全plan数, n_rendered=全plan落盘数)`：全 plan 全落盘→completed；部分→partial；0→failed。
- 子集调用跑成功 1 张，但全 plan 还有 112 张没落盘 → `n_rendered=1, n_total=113` → **partial**（不再是 `1/1 completed`）。**根因 B 从结构上消失。**

### M1 的效果
chart-maker 即使逐个 `only_chart_ids` 调 run_chart_plan，每次 handoff 都是全 plan 磁盘快照：跑成功 1 张 → `1/113 partial`，再成功 1 张 → `2/113 partial`……**永远不会出现 `1/1 completed` 覆盖**。只有全 113 张真落盘才 completed。

---

## 三、M2（兜底，第二道闸）：封存门对账总数不变式

> M1 让 run_chart_plan 自己不丢图。M2 是**结构兜底**——即使 M1 有漏网、或未来别的路径（auto-seal / LLM 手调 seal）产出残缺 handoff，封存门也兜住。两道闸正交（M1 在工具内，M2 在 seal 层，覆盖 seal 工具 + auto-seal 所有路径）。

### 改动（`seal_handoff_tools.py:_reconcile_chart_maker_payload`，单一注入点 L701）
在现有 2.0（磁盘真实性）/ 2.1（reason 订正）/ 2.2（aggregate 落盘）之后，**新增 2.3（对账总数不变式）**：
```python
# --- 2.3（对账总数不变式，spec 2026-06-24 silent-drop）---
# plan 的全 chart（扣 skipped）必须全部出现在 chart_files ∪ failed_charts ∪ remaining_charts。
# 防「该画的图既没画也没记录」静默丢失（dogfood：113 plan→handoff 只对账 1，112 蒸发）。
# 只在能读到 plan 时校验（无 plan 早退已在上方处理）。
planned_ids = {c.get("id") for c in plan.get("charts", []) if isinstance(c, dict) and c.get("id")}
skipped_ids = {s.get("id") for s in (plan.get("skipped") or []) if isinstance(s, dict) and s.get("id")}
expected_ids = planned_ids - skipped_ids
accounted_ids = (
    {_chart_id_of(p) for p in payload.get("chart_files", [])}     # chart_files 是路径，需映射回 id
    | {fc.get("chart_id") for fc in payload.get("failed_charts", []) if isinstance(fc, dict)}
    | {rc.get("chart_id") for rc in payload.get("remaining_charts", []) if isinstance(rc, dict)}
)
missing_from_accounting = expected_ids - accounted_ids
if missing_from_accounting:
    raise ValueError(
        f"chart 对账不完整：plan 有 {len(expected_ids)} 个 chart（扣 skipped），"
        f"但 handoff 只对账了 {len(accounted_ids & expected_ids)} 个——"
        f"{len(missing_from_accounting)} 个既不在 chart_files、也不在 failed_charts/"
        f"remaining_charts，被静默丢失: {sorted(missing_from_accounting)[:10]}。"
        f"每个计划的 chart 必须有归宿（画成功→chart_files / 失败→failed_charts / "
        f"预算截断→remaining_charts）。补画或据实记失败再封存。"
    )
```
> ⚠️ **chart_files→id 映射**（`_chart_id_of`）：chart_files 存的是 output 路径（`/mnt/user-data/outputs/plot_xxx.png`），需映射回 chart id。最稳的做法：从 plan 建 `output_basename → id` 映射（plan 每个 chart 有 `output` + `id`），用 chart_files 的 basename 查。实现时复用 plan 已读的 charts[]。

> ⚠️ **per_subject 与 chart id 的多对一**：plan 里多个 per_subject chart 可能共享同一 `id`（如 dogfood 的 `open_arm_time_ratio_bar` 出现多次，每个 subject 一条，见 problem doc「first 3 chart ids = ['box_open_arm', 'open_arm_time_ratio_bar', 'open_arm_time_ratio_bar']」）。**所以不变式不能用 id 集合相等**——要用 **output 路径集合**（每个 chart 的 output 唯一）做对账分母，而非 id。实施时：`expected_outputs = {c["output"] for c in plan.charts - skipped}`，`accounted_outputs = set(chart_files) | {failed/remaining 的 output}`。**但 failed_charts/remaining_charts 当前只存 chart_id 不存 output** → 需确认它们能映射回 output，或改 2.3 用「count 相等」而非「集合相等」作为最小可行校验（`len(chart_files) + len(unique failed outputs) + ... == len(expected_outputs)`）。**这是实施时必须先核实的契约点**（见 §五 T0）。

### M2 的鲁棒性
- 无 plan_charts.json → 跳过 2.3（同 2.2 风格，不 crash）。
- 读盘异常 → warning + 跳过 2.3。
- 触发 ValueError → ToolErrorHandling 转 error ToolMessage → chart-maker 当场看到「N 个 chart 没归宿」→ 补画或据实记失败（同 ETHO-10 2.2 门的「引导式补做」语义，措辞守 deepseek 正面提示）。

---

## 四、M3（自洽）：completed/failed 互斥 + 计数一致 + 去重

### 改动（`run_chart_plan_tool.py` status 派生 + `seal_handoff_tools.py` reconcile）
1. **completed 与 failed_charts 非空互斥**（M1 改动 3 后 status 已基于全 plan，理论上 completed 必然 failed_charts 空；但加显式断言兜底）：reconcile 门里 `if status=="completed" and failed_charts: raise ValueError`（或降级为 partial）。
2. **`gate_signals.failed_charts == len(failed_charts)`**：`_build_gate_signals` 用 `len(failed_charts)` 而非独立计数，消除 dogfood 症状 4 的「gate_signals.failed_charts=0 但数组长度=1」矛盾。
3. **同一 chart 不得同时在 chart_files 与 failed_charts**：Step 7 核磁盘后，若某 output 既在 chart_files 又被某 failed entry 引用 → 以磁盘真相为准（落盘→只留 chart_files，没落盘→只留 failed）。M1 改动 2「始终遍历全 plan 核磁盘」天然保证这点（每个 output 只判一次 exists()，非此即彼）。

---

## 五、测试清单（TDD 红→绿）

> 用 `_TASK_RUNNER_OVERRIDE` 注入同步 runner。关键红测试 = 复现 dogfood 的「子集成功覆盖全量失败」。

0. **T0 契约核实（实施前必做）**：确认 plan.charts[] 每条有唯一 `output`，failed_charts/remaining_charts 能否映射回 output（决定 2.3 用「集合相等」还是「count 相等」）。若 failed_charts 只有 chart_id 且 id 多对一 → 2.3 用 output 集合，且需让 run_chart_plan 在 failed_charts 里带 output（或 reconcile 从 plan 按 id 反查——但 id 多对一时反查不唯一，故首选让 failed_charts 带 output）。
1. **T1 M1 子集调用不覆盖全量（核心红→绿，复现 dogfood）**：
   - plan 113 chart。第一次 `run_chart_plan()` 全量，runner 让全失败 → handoff `status=partial, n_total=113, n_rendered=0`（或部分）。
   - 第二次 `run_chart_plan(only_chart_ids=["box_open_arm"])`，runner 让它成功（box_open_arm png 落盘）。
   - **改前红**：handoff 被覆盖成 `status=completed, summary="1/1"`（n_total=1）。
   - **改后绿**：handoff = `status=partial, n_total=113, n_rendered=1`（box_open_arm 在 chart_files，其余 112 在 failed_charts「not rendered」）。**子集成功没覆盖全量失败。**
2. **T2 M1 全 plan 全落盘才 completed**：runner 让全 113 落盘 → `status=completed, n_total=113, n_rendered=113`。
3. **T3 M1 始终核全 plan 磁盘**：只跑 `only_chart_ids=["box_open_arm"]` 但磁盘上已有 5 张别的 png（前次跑的）→ handoff chart_files 含这 6 张（全 plan 核磁盘，不只本次子集）。
4. **T4 M2 对账总数不变式（兜底红→绿）**：手构 payload：plan 113 chart、chart_files=1、failed_charts=1（只 box_open_arm）、remaining=0（复现 dogfood handoff）→ 喂 `_reconcile_chart_maker_payload`。
   - **改前红**：不抛（门没这个检查）。
   - **改后绿**：抛 ValueError「112 个 chart 被静默丢失」。
5. **T5 M2 完整对账通过**：chart_files+failed+remaining 覆盖全 plan（扣 skipped）→ 不抛。
6. **T6 M2 无 plan 不 crash**：plan_charts.json 不存在 → 跳过 2.3。
7. **T7 M3 completed+failed 互斥**：status=completed 但 failed_charts 非空 → 抛 ValueError（或降 partial）。
8. **T8 M3 gate_signals 计数一致**：failed_charts 有 3 条 → gate_signals.failed_charts==3。
9. **T9 M3 无重复**：某 output 不同时进 chart_files 和 failed_charts。
10. **T10 per_subject id 多对一**：plan 含 28 个 `open_arm_time_ratio_bar`（同 id 不同 output）→ 对账用 output 不用 id，28 个各自独立对账（不因 id 相同被去重成 1）。
11. **T11 auto-seal 路径同样过 M2**：executor auto-seal 构造的 payload 过同一 reconcile 门，对账总数不变式生效。
12. **T12 run_metric_plan 回归**：M2/M3 改 reconcile 是 chart 专属（`_reconcile_chart_maker_payload`），不碰 metric；但跑 run_metric_plan 全量确认无连带。
13. **T13 import 环**：改两文件后裸导入 `app.gateway` + `make_lead_agent` 0 退出。
14. **T14 端到端 dogfood（验收，F1 已生效进程）**：重启后跑 28-subject EPM → run_chart_plan 全量一次画完 113（F1 生效）→ `status=completed, chart_files=113`；若仍有失败 → 诚实 partial + 全 plan 对账（无静默丢失、无 `1/1` 覆盖）。

---

## 六、根因 A 的运维防护（非代码 bug，但写进验收 gate）

dogfood 跑旧代码（F1 未生效）的运维陷阱**不是代码 bug**（F1 代码正确），但反复踩（这是第二次 dogfood 暴露同一根因 A）。**纳入 dogfood 验收前置 gate**（写进 `noldus-insight-e2e` skill 或 dogfood SOP，不写进本 spec 的代码改动）：

> **dogfood 启动前必须确认 gateway 进程加载了目标代码**：`make stop && make dev` 后，验 `ps` 里 uvicorn 进程启动时间晚于最后一个相关 commit + `gateway.log` 有新 `Application startup complete` + `/health` 200。**因为 `make dev` 的 hot-reload 默认关闭（serve.sh:140-147，watchfiles 与 ProcessPoolExecutor fork 冲突），merge 后不重启 = 跑旧代码。**

> ⚠️ **不把 F1/F2 改成「运行时自动重载」**——hot-reload 关闭是有意的（serve.sh 注释：watchfiles Rust watcher 与 ProcessPoolExecutor fork 冲突致 make dev 起不来）。运维 gate（手动确认重启）比改回 hot-reload 安全。

---

## 七、风险与三大病理自检

1. **Reward hacking**：本 spec 正治它——M1 让「缩小对账范围」不再能把失败覆盖成功（handoff 永远全 plan 磁盘真相）；M2 兜底「该画的没归宿就拒」。**验收看全 plan 磁盘对账，不看 LLM 选择的子集范围。** 不加「别用 only_chart_ids 偷懒」prompt 规则（守 Telecom 禁令）——结构上让子集调用无法覆盖全量。
2. **Catastrophic forgetting**：
   - M1 改 run_chart_plan Step 3/7/8，不碰 run_metric_plan（虽同构，但 metric 无 only_chart_ids 覆盖问题——确认 run_metric_plan 的 only_metric_ids 是否有同款覆盖风险，见 §八）。
   - M2/M3 改 `_reconcile_chart_maker_payload` 必须 grep 所有调用方（seal_chart_maker_handoff、run_chart_plan、executor auto-seal 三路径，T11 守），全量回归。
   - 不破 ETHO-10 的 2.0/2.1/2.2（T 守既有 reconcile 测试不回归）。
3. **Under-exploration**：M1（工具语义改 merge 不覆盖）+ M2（封存门加不变式）都是**结构改动**，不是「改 chart-maker prompt 让它别逐个调」。守判据「结构缺失→上门」。

---

## 八、同构核实：run_metric_plan 免疫（已坐实）—— run_chart_plan 照齐它即可

> run_chart_plan 抄自 run_metric_plan，`only_chart_ids` 与 `only_metric_ids` 同源。已核实 run_metric_plan **是否也有「子集调用覆盖全量」风险**：

**结论：run_metric_plan 免疫。它的对账始终基于全 plan，正是 run_chart_plan 该学的范式（M1 的设计依据）。**

代码坐实（`run_metric_plan_tool.py`）：
- Step 3（L206）：`only_metric_ids` 只过滤 `plan_metrics` → **决定跑哪些 task**（提交哪些 compute）。
- Step 8（L329）：`aggregate_metrics_to_handoff(plan, ...)` 传的是 **`plan`（全集）**，**不是** `plan_metrics`（子集）。聚合器 glob 全 plan 期望的 m_*.json 对账磁盘 → 没跑的 metric glob 不到 → 计 missing → partial。
- → **子集调用跑成功几个，handoff 仍是全 plan 磁盘对账快照，不会覆盖成 completed。run_metric_plan 天然免疫。**

**run_chart_plan 的 bug = 它没学到这点**：Step 7（L229）只遍历 `results`（本次跑的子集）、`n_total=len(tasks)`（子集大小），而非像 run_metric_plan 把全 plan 交给对账。**M1 = 让 run_chart_plan 对账也基于全 plan（Step 7 遍历全 plan 核磁盘、n_total=全 plan 数），与 run_metric_plan 对齐——不是发明新机制，是补齐抄漏的部分。**

**连带动作**：把「子集调用必须以全 plan 为对账分母」补进 [batch 执行 pattern 文档](2026-06-24-deterministic-batch-execution-tool-pattern.md) 铁律（新增铁律 12），固化「run_metric_plan 怎么对、run_chart_plan 抄漏了哪」。run_metric_plan 无需修（免疫）。

---

## 九、关键代码锚点（已核实，行号 = dev 24383de8）
- **主修 M1**：`run_chart_plan_tool.py`：Step 3 only_chart_ids 过滤(169-175)、Step 7 核磁盘+对账(223-259，改成始终遍历全 plan)、`n_total`(223)、`_derive_status`(388)、`_default_summary`(406)、Step 9 seal(288)
- **兜底 M2/M3**：`seal_handoff_tools.py:_reconcile_chart_maker_payload`(525)、单一注入点 `_seal_handoff_to_workspace`(701)、`_load_plan_charts`、2.0/2.1/2.2 现有段
- **根因坐实**：SSE summary 全 `0/1`/`1/1`（无 X/113）= n_total=1 子集调用；chart-maker turn 序列 prep→run_chart_plan→bash→run_chart_plan×2→max_turns
- **chart-maker prompt**：`subagents/builtins/chart_maker.py` L53-55（only_chart_ids 选子集）、L59（调一次全量）、L61（重跑子集 only_chart_ids）
- **同构核实**：`run_metric_plan_tool.py` only_metric_ids + `aggregate_metrics_to_handoff`(基于全 plan?)
- **诊断文档**：`docs/problems/2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md`
- **F1/F2（前置依赖，已 merge #192）**：`run_chart_plan_tool.py:202`（F1 argv 预解析）、`_cli.py:135`（F2 warning）
- **运维陷阱**：`serve.sh:140-147`（hot-reload 默认关，merge 后须 make stop && make dev）；dogfood 进程 14:06 起未重启跑旧代码

---

## milestone 建议
归入「chart-maker 鲁棒性 / reward hacking 治理」track。checkpoint：「F1 修渲染层后第二次 dogfood（thread 590fbbd3，但进程未重启 F1 未生效）暴露**新形态**：run_chart_plan 子集调用（only_chart_ids 单个）整体覆盖 handoff → 成功单张 `1/1 completed` 覆盖失败全量 `1/113 partial` → 112 静默丢图伪 completed。纠正诊断 doc「封存门覆盖率」归因（真根因=工具 per-call 覆盖语义 + LLM 可缩 plan）→ M1（handoff 以全 plan 为对账分母 + 子集 merge 不覆盖）+ M2（封存门对账总数不变式兜底）+ M3（completed/failed 互斥）→ 待实施。连带核实 run_metric_plan only_metric_ids 同构风险」。与 ETHO-10（产物真实性）+ F1（argv 预解析）合为 chart-maker 确定性化的完整防线。
