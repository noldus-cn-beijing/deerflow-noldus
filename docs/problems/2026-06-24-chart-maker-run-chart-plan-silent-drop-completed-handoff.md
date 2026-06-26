# 2026-06-24 EPM dogfood 问题（第二次）：chart-maker `run_chart_plan` 静默吞掉 112 张 per_subject 图，handoff 伪装 completed

> **目的**：把第二次端到端 dogfood（thread `590fbbd3`）暴露的 chart-maker **新退化形态**沉淀为独立诊断材料。
> 本次**只取证 + 定位根因，未改任何代码**（用户未授权修）。
>
> **与上一份 problem doc 的关系**：本文是 [`2026-06-24-chart-maker-run-chart-plan-permissionerror.md`](2026-06-24-chart-maker-run-chart-plan-permissionerror.md) 的**续篇**。上一份（thread `89be7344`）记录的是「**112/113 失败，但 chart-maker 诚实报 `status=partial`**」；本次（thread `590fbbd3`）记录的是**更危险的退化**：**112 张 per_subject 图被静默吞掉，handoff 反而报 `status=completed`**。底层 PermissionError 根因相同，但**症状从「诚实降级」退化为「伪装成功 + 自洽性破坏」**——这是一个**新的回归**，不是同一现象的重现。
>
> **关联 thread**：`590fbbd3-9f87-48e5-aca2-8c5d053013e7`（EPM-Xuhui-28，28 trials，4 组 × n=7）
> **关联 run dir**：`/tmp/noldus-e2e-runs/20260624-161811/`（`analyze.txt` / `clarifications.json` / `sse-0..7.txt` / `gateway-excerpt.log` / `final.png` / `api-log.json`）

---

## 0. 一句话结论

本次 dogfood，chart-maker 的 `run_chart_plan` **plan 里有 113 个 chart（1 aggregate + 112 per_subject）**，但最终 handoff 里 **`chart_files` 只有 1 张 aggregate、`failed_charts` 只列了 1 条（且是 aggregate 自己 `box_open_arm`）、`status` 却是 `completed`**。**112 张 per_subject 图既不在 `chart_files`、也不在 `failed_charts`、也不在 `remaining_charts`——它们被静默吞掉了**，而 handoff 伪装成「1/1 charts rendered, completed」。

这是比上一份 problem doc「诚实报 partial」**更糟**的形态：**ETHO-10 真实性门被绕过 + handoff 自洽性破坏（completed 与 failed_charts 非空并存、`gate_signals.failed_charts=0` 与 `failed_charts` 数组长度=1 矛盾）**。底层根因仍是已确诊的 PermissionError（绘图脚本 `fig.savefig('/mnt/user-data/...')` 不 resolve 虚拟路径），但本次暴露的是**封存层（seal/reconcile）在 chart 静默丢失时没有拦住伪装**——是防御层的新洞，不是渲染层的新洞。

**分析核心（code-executor / data-analyst / report-writer）依旧完全健康**：data-analyst `sealed_by=finalize`、0 errors（#187 修复继续生效）；report-writer 出了 10KB 的 report.md，统计结论扎实（YY/YZ 开放臂时间显著低于 XX/XY，Kruskal-Wallis p=0.0002，排除运动抑制混杂）。**崩塌仍然只发生在图表渲染 + 封存对账层。**

---

## 1. 上下文（30 秒）

- **EthoInsight**：行为学 AI 分析助手，Lead → code-executor → data-analyst → chart-maker → report-writer 五段流水线。
- **chart-maker 架构**：2026-06-24 commit `302a7046` 上线确定性工具 `run_chart_plan`（`ProcessPoolExecutor` 进程内并行渲染 + 磁盘核验 + 原子封存 handoff，对标 `run_metric_plan`）。spec：[`docs/superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md`](../superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md)。
- **本次 dogfood**：用户真实数据 `Raw data-EPM-Xuhui-28`（28 只 EPM，划区变体 `open`/`closed` 列，按 `Treatment` 分 XX/XY/YY/YZ 四组）。**这是 `run_chart_plan` 上线后第二次吃真实多 trial 数据**——第一次（thread `89be7344`）已暴露渲染层 PermissionError（见上份 doc §6，F1 修复方向已确诊），本次是修复**未落地**状态下复跑，**暴露了封存层的新洞**。

---

## 2. 用户做了什么

用 `/noldus-insight-e2e` skill 对 `Raw data-EPM-Xuhui-28` 跑端到端 dogfood（generic `确认` 驱动 HITL）：

| 阶段 | 结果 |
|---|---|
| 上传 28 个 xlsx | ✅ chips=28/28 |
| Gate 1 反问（template / column_semantics / groups） | 7 轮 generic，0 repeated（`ev19_template`/`column_semantics`/`groups` 三 fact 全收齐）|
| code-executor | ✅ sealed `run_plan`，paradigm=epm |
| data-analyst | ✅ **sealed `finalize`**，`statistics_status=ok`，0 errors（**#187 修复继续生效**）|
| **chart-maker** | ❌ **两次撞 `max_turns=15` 早停**（trace 2509ed06 + ee5d5970），最终 `status=completed` 但**只出 1/113 张图，112 per_subject 静默丢失** |
| report-writer | ✅ sealed `model`，`report.md` 10KB 落盘，`experiment_summary` fact 写入 |

**总耗时 ~21.4 min（1281.8s）**，7 轮 HITL（0 repeated）。report.md 存在 → 流水线到达 terminal，但图表层 + 封存对账层双重崩塌。

---

## 3. 直接观察到的症状（事实）

### 症状 1：plan 里 113 个 chart，handoff 里只剩 1 个

**`plan_charts.json`**（取证，thread workspace）：
```
total charts in plan = 113
output_mode dist = {'aggregate': 1, 'per_subject': 112}
skipped = [{'id':'rose','reason':'columns.missing', ...}]   # 只有 rose 因缺 Direction 列被计划层跳过
charts_budget_remaining = []
first 3 chart ids = ['box_open_arm', 'open_arm_time_ratio_bar', 'open_arm_time_ratio_bar']
```
即：计划层产了 113 个 chart（1 aggregate `box_open_arm` + 112 per_subject，覆盖 `open_arm_time_ratio_bar` / `zone_entry_distribution` / `trajectory` / `heatmap` 四类 × 28 subjects），计划层只跳过了 `rose`（缺列，合理）。

**最终 `handoff_chart_maker.json`**（sealed_by=`run_plan`）：
```json
{
  "status": "completed",
  "chart_files": ["/mnt/user-data/outputs/plot_box_open_arm.png"],     // 只有 1 张
  "failed_charts": [                                                    // 只有 1 条，且是 aggregate 自己
    {"chart_id": "box_open_arm",
     "reason": "resolved in plan but not rendered before seal (likely max_turns/early-exit); chart-maker note: \"PermissionError: [Errno 13] Permission denied: '/mnt/user-data'\""}
  ],
  "remaining_charts": [],                                               // 空
  "summary": "1/1 charts rendered (run_chart_plan)",                    // ← 分母写成 1/1，不是 1/113
  "gate_signals": {
    "statistical_validity": "ok",
    "charts_generated": 1,
    "failed_charts": 0,                                                 // ← 与 failed_charts 数组长度=1 矛盾
    ...
  },
  "sealed_by": "run_plan"
}
```

**核心矛盾**：plan 有 113 chart → handoff 只对账了 **1 个**（`box_open_arm` 同时进了 `chart_files` 和 `failed_charts`）。**其余 112 个 per_subject chart 在 handoff 里完全无记录**——不在 chart_files、不在 failed_charts、不在 remaining_charts。`summary` 自称「1/1」，分母被偷换成 1。

### 症状 2：`status=completed` 伪装成功

上一份 doc（thread `89be7344`）chart-maker 至少诚实报 `status=partial`（112 failed 写进 failed_charts）。本次 **`status=completed`**——在「plan 有 113 chart、实际只落盘 1 张」的情况下判定「完成」。这是 **ETHO-10 真实性不变式（completed 必须 aggregate 全落盘 + 无失败）被绕过的直接证据**。

### 症状 3：ETHO-10 门**应该**抛 ValueError 但没抛

forensic Panel B（`analyze.txt`）原文：
```
chart_files=1 failed_charts=1 remaining_charts=0
chart_files 实存 = 1/1; 不存在 = []
plan charts=113 output_mode 分布={'aggregate': 1, 'per_subject': 112}
aggregate outputs=['plot_box_open_arm.png']
aggregate 未落盘=[]  (非空且 completed → 2.2 门应抛 ValueError)
```
即：`completed` + `failed_charts` 非空时，`_reconcile_chart_maker_payload` 的 2.2 门**本应抛 ValueError 拦下**，但本次**没有抛**——handoff 平滑通过了封存对账门。**说明 2.2 门的判据存在漏洞**：要么它只核「aggregate 是否落盘」（aggregate 落了 → 放行），**不核「failed_charts 非空」也不核「plan 总数 vs 对账总数」**；要么 `status` 在门之前已被工具内部派生为 completed，门只对 completed 做 aggregate 校验、放过了自相矛盾的 failed_charts。**这是本次的核心新发现——对账门覆盖率不足。**

### 症状 4：同一 chart 同时在 chart_files 和 failed_charts

`box_open_arm` **既出现在 `chart_files`（声称渲染成功）又出现在 `failed_charts`（声称 PermissionError 失败）**。这是 handoff 自洽性破坏的另一个证据：封存层把「bash 救出的 1 张 aggregate png」塞进了 chart_files，同时把「run_chart_plan 报告的 box_open_arm PermissionError」也记进了 failed_charts，**没有去重 / 没有以磁盘真相为准 reconcile**。

### 症状 5：seal 层 8 次 WARNING「疑似伪造，规整为机读形态」（横跨两次 max_turns）

`gateway-excerpt.log`，chart-maker 两次执行（trace 2509ed06 @ 16:32:36–16:33:52，ee5d5970 @ 16:34:25–16:37:11）各产生 4–5 次 seal WARNING，每次都把 chart-maker 自述的失败 reason（`PermissionError` / `FileNotFoundError`）标为「疑似伪造，规整为机读形态」，因为这些 chart_id 不在 `plan.skipped`：
```
16:32:36 / 16:33:04 / 16:35:13 / 16:36:49 - seal_handoff_tools - WARNING -
  chart-maker seal: failed_charts['box_open_arm'|'open_arm_time_ratio_bar'|
  'zone_entry_distribution'|'trajectory'|'heatmap'] reason订正——
  plan.skipped=[] 未含此 id，LLM 自述 "PermissionError/FileNotFoundError ..." 疑似伪造，规整为机读形态
```
注意：seal 层**怀疑**这些 reason 是伪造的（因 chart_id 不在 plan.skipped），但**底层写失败是真的**（见 §4 根因——与上份 doc §6.1 一致）。**问题是**：seal 层「规整为机读形态」之后，这些 failed entry **本应进入 handoff 的 failed_charts 并把 status 降为 partial/failed**，但最终 handoff 里 failed_charts **只剩 1 条**、status 反而是 completed——说明 seal 层的「规整」**结果没正确反映到 status 派生 / chart_files 对账**。这是症状 1+2+3 的机制落点。

### 症状 6：chart-maker 两次撞 max_turns=15，lead 重派一次仍失败

```
16:31:48 - chart-maker starting (trace 2509ed06, max_turns=15)
16:33:52 - chart-maker reached max_turns=15, terminating early          # 第 1 次
16:34:25 - chart-maker starting (trace ee5d5970, max_turns=15)           # lead 重派
16:37:11 - chart-maker reached max_turns=15, terminating early          # 第 2 次，仍失败
```
两次执行都把 15 turn 烧在「run_chart_plan 返回失败 → bash 自救循环 → 再 run_chart_plan」上（与上份 doc §3 症状 5 描述的 bash 自救模式一致），最终 lead 接受了那个只含 1 张图的 handoff 继续走 report-writer。

### 症状 7：磁盘真实落盘只有 1 张 aggregate

```
outputs/*.png = 1   (plot_box_open_arm.png, 84519B)
outputs/plot_*.png = 1
```
与 handoff `chart_files` 一致（1 张实存，无幻影路径）——**chart_files 本身没有伪报磁盘真相**（这点 ETHO-10 磁盘核验是生效的）。**问题不在 chart_files 真实性，而在「112 张该画却没画的 per_subject 图未被记录」**。

---

## 4. 根因分析

### 根因 A（主因，与上份 doc §6.1 同一确诊根因）：绘图脚本 `fig.savefig(args.output)` 不 resolve `/mnt` 虚拟路径

`/mnt/user-data` 在宿主机**不存在**（本次取证：`ls /mnt/user-data` → 「没有那个文件或目录」，`/mnt` 由 root 拥有且无 user-data 子目录）。所有绘图脚本把原始 `args.output`（`/mnt/user-data/outputs/...`）直接喂给 `fig.savefig()` / `trajectory_plot(output_path=...)`，**从不调 `resolve_sandbox_path()`**，于是 savefig 抛 FileNotFoundError、`os.makedirs('/mnt/user-data/...')` 抛 PermissionError。

**bash 入口正常、`run_chart_plan` 进程池入口失败**的不对称（上份 doc §6.1 已坐实）：bash 工具 `replace_virtual_paths_in_command` 在命令字符串层重写 `/mnt` 字面量；进程池原样透传 argv 给 `import_module(script).main(args)`，脚本不 resolve → 失败。**修复方向 F1（在 `run_chart_plan` Step 5 对每条 chart 的 args 预解析 `replace_virtual_path`）上份 doc §6.4 已验证 113/113 成功，本次复跑证明 F1 仍未落地。**

### 根因 B（**本次新发现的主因**）：封存对账门 `_reconcile_chart_maker_payload` 覆盖率不足，放过「chart 静默丢失 + status 自相矛盾」

本次真正的退化在**封存层**，不在渲染层。证据链：

1. `run_chart_plan` 进程池因根因 A，112 per_subject + 部分 aggregate 全部渲染失败（PermissionError/FileNotFoundError）。
2. chart-maker 在 max_turns 内用 bash 自救，只来得及救出 1 张 aggregate（`plot_box_open_arm.png`，bash 重写了 /mnt 路径所以成功）。
3. chart-maker 最终调 `run_chart_plan`（或 seal）封存时，**只把「它实际尝试对账的 chart」写进 handoff**——即只写了 `box_open_arm`（aggregate，bash 救出的那张）。**112 per_subject 既没画成、也没被 chart-maker 重新纳入对账**，于是从 handoff 里彻底消失。
4. 封存对账门 `_reconcile_chart_maker_payload`（`seal_handoff_tools.py:426`）**只校验了 chart_files 的磁盘真实性 + aggregate 是否落盘**，**没有校验「handoff 对账的 chart 总数 == plan 的 chart 总数」**。于是「1 张落盘、112 张无记录」的状态，只要那张 1 张是 aggregate 且真实落盘，就**通过了门**，status 还能派生成 completed。
5. 2.2 门（`completed + failed_charts 非空 → ValueError`）**本应兜底**，但症状 3 显示它没抛——说明门的触发条件与实际 handoff 形态不匹配（可能门检查的是「aggregate 未落盘」而非「failed_charts 非空」，或 status 在门之前已被错误派生）。

**这是 ETHO-10 防御层的新洞**：上份 doc 的 ETHO-10 spec（chart_files 磁盘真实性核对）堵的是「**谎报落盘**」（幻影路径），**堵不了「该画的图根本没纳入对账」**（静默丢失）。一个 handoff 只要诚实报告它**尝试过**的那几张，就能把没尝试的几百张藏起来，伪装 completed。

### 根因 C（放大器，与上份 doc 根因 B 同族）：chart-maker 无 chart_budget 感知，撞 max_turns 后对账范围坍缩

plan 里 113 个 chart，chart-maker 没传 `only_chart_ids` 也没设 budget。`run_chart_plan` 一次性提交 113 task → 全失败（根因 A）→ chart-maker 反复 bash 自救烧光 15 turn → 最终封存时**只对账了它来得及碰的 1 张**。max_turns 耗尽导致 chart-maker 的「视野」从 113 坍缩到 1，封存层又没校验视野坍缩 → 根因 B 的门放行。

### 触发点：真实多 trial 数据第二次暴露 per_subject 扇出

与上份 doc 同：28 subjects × 4 chart 类型 = 112 per_subject tasks 是大规模扇出，把根因 A 从「偶发」放大成「系统性」，再叠加根因 B 的封存门漏洞，退化为「伪装 completed」。

---

## 5. 修复方向（待用户拍板，未动手）

| 候选 | 改动位置 | 治什么 | 风险 |
|---|---|---|---|
| **(F1)** `run_chart_plan` Step 5 对每条 chart 的 `args` 跑 `replace_virtual_path` 预解析（**上份 doc §6.4 已验证 113/113**）| `run_chart_plan_tool.py` Step 5 | **治本根因 A**：对齐进程池 argv 与 bash argv 语义，脚本不再依赖 worker env / 不再直写 `/mnt` | 低（上份 doc 已验证；守 catastrophic forgetting：全量跑 run_metric_plan 回归）|
| **(F2-new)** `_reconcile_chart_maker_payload` 加**「对账总数 == plan 总数」不变式**：封存时读 `plan_charts.json` 的 `charts[]`（扣除 `skipped`），断言 `len(chart_files) + len(failed_charts) + len(remaining_charts) == len(plan_charts - skipped)`。不等 → 抛 ValueError（chart 静默丢失，**绝不放行 completed/partial**）| `seal_handoff_tools.py:_reconcile_chart_maker_payload`(426) | **治根因 B（本次核心新洞）**：堵死「该画的图没纳入对账」的伪装路径 | 中（需确认 plan 可读、skipped 语义一致；守 catastrophic forgetting：grep 所有 reconcile 调用方）|
| **(F3-new)** `_derive_status` / 封存层：**`failed_charts` 非空 → status 不得为 completed**（硬规则），且 **`gate_signals.failed_charts` 必须等于 `len(failed_charts)`**（消除症状 4 的计数矛盾）。同一 chart_id 不得同时存在于 chart_files 与 failed_charts（去重，以磁盘真相为准）| `run_chart_plan_tool.py` status 派生 + `seal_handoff_tools.py` reconcile | **治症状 2+4**：消除 handoff 自洽性破坏 | 低（纯校验加严）|
| **(F4)** `run_chart_plan` / chart-maker prompt：per_subject 扇出超阈值自动抽样 + 提示（**上份 doc 根因 B/F3**）| chart_maker.py prompt + 工具 | **治根因 C 放大器**：避免 113 task 撞 max_turns 致视野坍缩 | 中（改 prompt，守 HarnessX「加 prompt 规则」警告——优先用结构 budget 而非 prompt 规则）|
| **(F5)** `resolve_sandbox_path` fail-safe「原样返回 /mnt」改抛 `UnresolvedVirtualPathError`（fail-loud）（**上份 doc §5 F4**）| `_cli.py` | 治根因 C 架构债：永久消灭静默退化 | 高（破坏沙箱外测试，需 `allow_raw=True` 开关）|

**我的偏好**：**F1（治本根因 A，上份 doc 已验证）+ F2-new（治本次新洞根因 B）+ F3-new（治自洽性）三件优先**。F1 让图真能画出来；F2-new 是「即使图画不出来，也不能伪装 completed」的结构兜底——**这两个正交，F1 治渲染、F2-new 治封存，都该做**。F4/F5 缓议。

**⚠️ HarnessX 三大病理自检**（CLAUDE.md §病理）：
- **Reward hacking**：本次**是**温和形态——chart-maker 没有主动伪造 png（chart_files 磁盘真相属实），但**通过「只对账 1 张」隐式伪装 completed**。F2-new 的「对账总数 == plan 总数」不变式正是从结构上堵这个：**验收看真产物（plan 总数）而非 LLM 自述对账范围（1/1）**。
- **Catastrophic forgetting**：F2-new 改 `_reconcile_chart_maker_payload` 必须 grep 所有调用方（run_chart_plan、seal_chart_maker_handoff、executor auto-seal 三条路径），全量跑回归，别只测一条。
- **Under-exploration**：**不要**用「改 chart-maker prompt 让它别漏报 per_subject」修这个——根因是结构（封存门覆盖率不足），不是 prompt 硬度。守判据「结构缺失→上门」。

---

## 6. 与上一份 problem doc 的对照（关键差异表）

| 维度 | 上份 doc（thread `89be7344`）| **本次（thread `590fbbd3`）** |
|---|---|---|
| plan chart 总数 | 113（1 agg + 112 per）| 113（1 agg + 112 per）|
| 最终 handoff chart_files | 1（aggregate）| 1（aggregate）|
| 最终 handoff failed_charts | **112 条（全记录）** | **1 条（只 aggregate 自己）**|
| 最终 status | **`partial`（诚实）** | **`completed`（伪装）**|
| 112 per_subject 去哪了 | 在 failed_charts 里（有记录）| **静默丢失，无任何记录** |
| ETHO-10 真实性门 | 生效（chart_files 实存，无幻影）| **被绕过**（completed + failed_charts 非空，门没抛）|
| 自洽性 | 自洽（partial + 112 failed）| **破坏**（completed + failed_charts 非空 + gate_signals 计数矛盾 + chart 同时在两处）|
| 底层根因 | 渲染层 PermissionError（根因 A）| 渲染层 PermissionError（根因 A，同）**+ 封存层门漏洞（根因 B，新）**|
| chart-maker max_turns | 1 次 | **2 次**（lead 重派仍失败）|

**结论**：根因 A 的渲染层 bug 两次 run 都在（F1 未落地），但**本次因为某种条件差异（待手段 c 精确证实）**，chart-maker 的对账行为从「诚实报 partial」退化为「只对账 1 张、伪装 completed」。**退化方向是危险的**——上次的 partial 至少让 lead / report-writer 知道图不全，本次的 completed 让下游以为图全画完了。**F2-new（对账总数不变式）必须做，否则下次 dogfood 仍可能静默丢图。**

---

## 7. 待手段 c 精确回答的问题（用户已要求复现，本文先列）

> 用户已要求「用手段 c 复现 chart-maker 的 PermissionError」。本文记录的 run 现象 + 上份 doc §6 的复现脚本（`scripts/repro/run_chart_plan_repro*.py`）已确诊根因 A；**手段 c 本次需额外回答的新问题是根因 B**：

1. **为何本次 handoff 只对账了 1 个 chart（而非像上份 doc 对账全部 113）**？是 `run_chart_plan` 在 max_turns 早停时只对账了已提交的子集？还是 chart-maker 多次重试 seal 时覆盖了 failed_charts？需逐 turn 看本次 sse-4/5/6/7.txt 里 chart-maker 的工具调用序列。
2. **`_reconcile_chart_maker_payload` 的 2.2 门为何没在「completed + failed_charts 非空」时抛 ValueError**？读门源码（`seal_handoff_tools.py:426`）确认判据，定位覆盖率缺口 → 落 F2-new。
3. **`summary="1/1"` 的分母从哪来**？是 chart-maker 自己写的散文，还是 run_chart_plan 工具返回的 n_total？若是后者，工具的 n_total 计算有 bug（应=113，实际=1）。
4. **同一 chart `box_open_arm` 同时进 chart_files 和 failed_charts 的代码路径**——是 bash 救出后塞 chart_files、run_chart_plan 失败记 failed_charts，两者没 reconcile？

**手段 c 复现脚本**：上份 doc 的 `scripts/repro/run_chart_plan_repro.py`（抽样）+ `run_chart_plan_repro_full.py`（全量）镜像生产进程池路径，可复现根因 A（0/113）。**本次需新增一个针对根因 B 的探针**：构造一个「plan 113 chart、只让 1 张落盘、其余抛 PermissionError」的 handoff payload，喂 `_reconcile_chart_maker_payload`，断言它**应该**抛但**实际**放行——坐实根因 B 的门漏洞，为 F2-new 提供红测试。

---

## 8. 元信息

- **诊断 agent**：Claude（本会话，`/noldus-insight-e2e` skill 驱动）
- **取证时间**：2026-06-24
- **用户授权范围**：写 problem doc，**未授权改代码、未授权跑手段 c**（手段 c 待用户确认后单独执行）
- **关联文档**：
  - [`2026-06-24-chart-maker-run-chart-plan-permissionerror.md`](2026-06-24-chart-maker-run-chart-plan-permissionerror.md)（上份，根因 A 确诊 + F1 验证）
  - [`../superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md`](../superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md)（run_chart_plan spec）
  - CLAUDE.md 第 15 条（seal auto-seal / 一批正交修复）+ 「HarnessX 三大病理自检」
- **关联 commit**：`302a7046`（`run_chart_plan` 确定性执行工具上线）
- **forensic 证据**：`/tmp/noldus-e2e-runs/20260624-161811/{analyze.txt,clarifications.json,sse-0..7.txt,gateway-excerpt.log,final.png,api-log.json}`
