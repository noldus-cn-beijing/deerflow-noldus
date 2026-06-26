# 2026-06-24 chart-maker `run_chart_plan` 核盘「竞态」误判 partial（本次良性；**2026-06-26 取证更正：工具无竞态，根因是子代理自身 mid-flight ls**）

> **目的**：把 EPM dogfood（thread `a6e3775c`）中 chart-maker **过程中自述**「n_rendered=4 / n_failed=108 / failures=[]」、但**磁盘 112/112 + 最终 handoff 正确**的现象沉淀。
> **2026-06-26 更正**：原假设「`run_chart_plan` 进程池未 flush 就核盘」**经取证证伪**——工具自首版 `302a7046` 即 `fut.result()`+`shutdown(wait=True)` 等所有 worker 退出才核盘，核盘计数从来正确。真实症状是 chart-maker **子代理自己**在思考里 mid-flight `ls outputs/*.png`（工具返回前），非工具核盘报错。详见 §3、§5。
> 本次现象**只取证 + 定位**；**2026-06-26 已 TDD 落地一个防回归结构守卫**（`_assert_reconcile_consistent` + `error_code=reconcile_inconsistent`，见 §5）。
>
> **与同日另两份 chart-maker problem doc 的关系**（三者形态不同，正交）：
> - [permissionerror](2026-06-24-chart-maker-run-chart-plan-permissionerror.md)（thread `89be7344`）：渲染层 `fig.savefig('/mnt/...')` 不 resolve 虚拟路径 → **112/113 真失败**，chart-maker 诚实报 partial。
> - [silent-drop-completed](2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md)（thread `590fbbd3`）：同 PermissionError 根因，但**封存层吞掉 112 per_subject + 伪装 completed** → 自洽性破坏（新回归）。
> - **本文**（thread `a6e3775c`）：渲染层**全成功 112/112**，工具核盘也正确；**误判只发生在 chart-maker 子代理自己的思考里（mid-flight ls），最终 handoff 正确**。**良性。**
>
> **关联 thread**：`a6e3775c-5a4e-4641-877b-ab6c5cd469e2`（EPM-Xuhui-28，112 charts = 4 图种 × 28 subject）
> **关联 handoff**：[2026-06-24-epm-dogfood-chart-success-askviz-gate](../handoffs/2026-06/2026-06-24-epm-dogfood-chart-success-askviz-gate-handoff.md) §5

---

## 0. 一句话结论

> **⚠️ 本节原结论已被 2026-06-26 取证更正，保留原文用删除线标注，更正见其后。**

~~`run_chart_plan` 用 `ProcessPoolExecutor` 进程内并行渲染 112 张图。chart-maker 在**进程池尚未全部把 png flush 到磁盘**的窗口期核盘（`ls outputs/*.png`），只看到最后落盘的一批（s27 的 4 张），于是 trace 里自述 `status=partial, n_rendered=4, n_failed=108, failures=[]`，甚至动了「写 `fix_handoff.py` 改盘」的念头。但**最终它克制住了、走了正确的 `seal_chart_maker_handoff`（`sealed_by=model`）**，最终 handoff 是 `chart_files=112 / failed=0 / completed` 的真相，112 张 png 全在盘。**本次零危害**，但核盘时机竞态是真实缺口——下次 chart-maker 若不克制（真改盘 / 据 `n_failed=108` 谎报），就从良性退化成 [silent-drop-completed] 那种自洽性破坏。~~

**更正（2026-06-26）**：`run_chart_plan` **本身没有核盘竞态**。`_execute_tasks` 对每个 future 调 `fut.result(timeout=)` 阻塞到 worker 的 `main()`（含 `savefig`）返回，再在 `finally` 里 `pool.shutdown(wait=True, cancel_futures=True)`；核盘（`Path(output_real).exists()`）发生在此之后，png 必已落盘关闭。`git log -S "pool.shutdown(wait=True"` 证明该 wait 在首版 `302a7046`（即本文引用的 commit）就有。chart-maker 自述的 `n_rendered=4` 是**子代理自己**在思考里 mid-flight `ls outputs/*.png`（工具返回前）看到的瞬时值，**不是工具报的**——工具的确定性计数从来正确（最终 handoff `chart_files=112 / completed` 即铁证）。本次良性的本质不是「子代理克制住改盘」，而是「工具压根没报错，子代理的瞬时误读未污染最终产物」。

---

## 1. 上下文（30 秒）

- **`run_chart_plan` 架构**：2026-06-24 commit `302a7046` 上线的确定性画图工具，`ProcessPoolExecutor` 进程内并行跑所有绘图脚本，**逐个核 output png 真落盘（磁盘真相）**，原子封存 handoff。对标 `run_metric_plan`。spec：[`docs/superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md`](../superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md)。
- **本次数据**：EPM-Xuhui-28，`viz_choice=yes`（用户手回 "A. 是，画图"），catalog 命中 4 图种 × 28 subject = 112 charts，用户意图模糊→全画。
- **核盘时机（更正后）**：`_execute_tasks` 对每个 future 调 `fut.result()` + `finally` `shutdown(wait=True, cancel_futures=True)`，**保证核盘时所有 worker 已退出、png 已落盘**。原本怀疑的「并行 + OS buffer flush 窗口期」对工具不成立（in-process `savefig` 在 worker 进程内同步写完关闭后 `main()` 才返回）。窗口期只存在于 chart-maker **子代理自己的** `ls`（不走工具核盘逻辑）。

---

## 2. 直接观察到的症状（事实）

### 症状 1：chart-maker 过程中自述 partial（错误）

chart-maker trace（执行 `run_chart_plan` 后的思考，逐字摘要）：
> *"The run_chart_plan returned status=partial with n_rendered=4, n_failed=108, but failures is empty... ls shows all 112 png files exist on disk... The tool miscounted — it may have only tracked the last batch (s27)."*

> *"Let me update the handoff to reflect reality... cat > /tmp/fix_handoff.py..."*（动了改盘念头）

### 症状 2：磁盘 + 最终 handoff 是真相（正确）

```
outputs/*.png 总数 = 112
  plot_open_arm_time_ratio_bar: 28
  plot_zone_entry_distribution: 28
  plot_trajectory: 28
  plot_heatmap: 28

handoff_chart_maker.json:
  status=completed, sealed_by=model
  chart_files=112（全列出，s0..s27 × 4 图种）
  failed_charts=0
  summary="112/112 张图表生成完成"
  gate_signals.charts_generated=112, failed_charts=0
```

### 症状 3：report.md 正确嵌入代表图

`outputs/report.md` 第 62-71 行：`{{img:plot_trajectory_s3/s7/s11/s18.png}}` + `{{img:plot_open_arm_time_ratio_bar_s3/s7/s11/s18.png}}`（Trial 4/8/12/19——离群/代表 subject）。**图真在、report 真链。**

---

## 3. 根因（2026-06-26 取证更正）

> **⚠️ 本节原诊断错误，已更正。原文保留删除线供对照。**

~~`run_chart_plan` 的核盘逻辑在 `ProcessPoolExecutor` 提交任务后、**所有 worker 进程退出 + OS 把文件 buffer flush 到磁盘之前**就执行了 `os.path.exists` / `ls` 核验。窗口期内只有先完成的批次（这里是最后提交但最先/最快 flush 的 s27 那 4 张图种）可见，故 `n_rendered=4`。等 chart-maker seal 阶段（已过窗口期）再核，112 全在，handoff 正确。~~

~~**为什么 `failures=[]` 却 `n_failed=108`**：核盘逻辑把「盘上没看到」计为 failed（n_failed=108），但因为根本没拿到失败原因（不是脚本 stderr，是「核盘时文件还没 flush」），所以 `failures` 详情数组为空——这个 `n_failed != len(failures)` 的**自洽性裂缝**正是 [silent-drop-completed] 那份 doc 警告的同款指纹（`gate_signals.failed_charts=0` 与 `failed_charts` 数组矛盾）。**本次没爆发是因为 chart-maker 最终走了 seal 重核，但裂缝客观存在。**~~

**更正后的真根因**：误判**不在工具**，在 chart-maker **子代理自己的思考**。

1. **工具核盘正确**：`run_chart_plan` 的核盘在 `fut.result()`（阻塞到 worker `main()` 返回）+ `shutdown(wait=True)` 之后执行，112 张 png 此时全已落盘关闭，工具确定性数出 `n_rendered=112 / completed`（最终 handoff 即此值）。
2. **`n_rendered=4` 是子代理瞬时误读**：chart-maker 在工具**返回前**（或并行思考时）自己跑 `ls outputs/*.png`，撞上 png 还在陆续落盘的瞬间，只看到当时已落盘的一批（s27 的 4 张），于是 trace 里写 `n_rendered=4`。这个数**从未进过任何 handoff 字段**，纯属子代理的中间推理噪声。
3. **`n_failed != len(failures)` 的自洽性裂缝在工具里不存在**：工具的 `n_failed = n_total - n_rendered`（减法导出），`failed_charts` 与 `chart_files` 在**同一个核盘循环**里互斥 append，故 `n_rendered + n_failed == n_total` 且 `n_failed == len(failed_charts)` **结构恒成立**。doc 原先担心的裂缝是 [silent-drop-completed] 那条**独立构造 failures 数组**路径的指纹，本工具的 M2/R2 修复（按 task index 核盘）早已堵死。

---

## 4. 为什么本次良性 / 下次可能恶性

- **良性**：chart-maker 在 trace 里**自己察觉了矛盾**（"failures empty but n_failed=108... ls shows 112 on disk"），克制住 `fix_handoff.py` 改盘，改走 `seal_chart_maker_handoff` 重核，handoff 落真相。
- **恶性路径（假设）**：若 chart-maker 下次没那么「较真」——
  - 路径 A：据误判的 `n_failed=108` 直接向 lead 谎报 partial → lead 可能重派 chart-maker 浪费算力，或报告里写「108 张图失败」误导用户。
  - 路径 B：真执行 `fix_handoff.py` 把 handoff 改成「4 张」（按 n_rendered）→ **人为制造丢图**，退化为 [silent-drop-completed]。
  - 两条路径都源于**核盘时机竞态**这个同一缺口。

---

## 5. 修复方向 → 已落地（2026-06-26，TDD）

原提议三项中，**#1 自始即在、#2 已落地、#3 因工具无真实竞态而调整为不变式回归**：

1. ~~**进程池同步等待**：`ProcessPoolExecutor.shutdown(wait=True)`~~ —— **already present since 302a7046（no-op）**。`_execute_tasks` 的 `finally` 早有 `pool.shutdown(wait=True, cancel_futures=True)`，且每个 future 都 `fut.result()` 过。无需新增。
2. **核盘自洽性校验** —— **✅ 已落地**。新增纯函数 `_assert_reconcile_consistent(n_total, chart_files, failed_charts)`（`run_chart_plan_tool.py`，`_derive_status` 旁），核盘算完 `n_rendered/n_failed` 后、`_derive_status` 前调用：`len(chart_files) + len(failed_charts) != n_total` 即 **fail-loud**（`logger.error` + `_seal_failed` 落 failed handoff + 返 `error_code="reconcile_inconsistent"`，chart-maker 决策树已能 parse）。**不重核不 sleep**（无真实竞态可重试）。仅校验这一条等式即钉住全部不变式（`n_failed == len(failures)` 等皆其代数推论）。注意 `n_failed != len(failures)` 在本工具结构上已不可能（同循环互斥 append）——本守卫纯防**未来重构**（如重新按 chart_id keying / 独立计 n_failed）静默重引入 [silent-drop-completed] 那类裂缝。
3. ~~**回归测试构造「写文件延迟」场景**~~ → 调整为**不变式回归**（测试注入同步 runner，绕开进程池，无法复现真实多进程竞态；故回归靶应是**不变式**而非假竞态）。**✅ 已落地**：`tests/test_run_chart_plan.py`
   - `TestReconcileGuardFires`：守卫纯单元（mismatch→raise / match→pass）+ 工具级（monkeypatch 守卫抛错 → `error_code=reconcile_inconsistent`、无 artifacts、磁盘 failed handoff）。
   - `TestReconcileInvariantHolds`：all-success-112 / partial / rc0-no-png / abort 四路径，从 result dict + 封存 handoff 双向断言 `n_rendered+n_failed==n_total`、`n_failed==len(failures)==len(handoff.failed_charts)`、`gate_signals.failed_charts==errors_count==n_failed`。证守卫不 false-fire + 锁属性。

**验收**：全量 `make test` = 4690 passed / 0 failed；两裸导入入口（`app.gateway` / `make_lead_agent`）0 退出；ruff clean。

**关于子代理 mid-flight ls（真实症状层）**：**不改 prompt**（守 CLAUDE.md 三大病理自检 §3/§6.6——反复误判不靠加提醒规则修，且子代理本次已正确克制走 seal、未污染产物）。结构正解是让确定性工具的自洽性 bulletproof（上述守卫）+ 修正本 doc。

---

## 6. 与其它 chart-maker problem 的横向对照

| 维度 | permissionerror (`89be7344`) | silent-drop (`590fbbd3`) | **本文 (`a6e3775c`)** |
|---|---|---|---|
| 渲染层 | ❌ 112/113 真失败（PermissionError） | ❌ 同 | ✅ 112/112 成功 |
| 核盘/封存 | 诚实报 partial | 吞 112 + 伪装 completed | **工具核盘正确**，子代理思考里瞬时误读 |
| 自洽性 | ok | 破坏（n_failed≠len(failures)） | **工具结构恒自洽**（同循环互斥 append） |
| 危害 | 高（真丢图） | 高（伪装 + 丢图） | **零（良性）** |
| 缺口层 | 渲染（savefig 路径） | 封存对账 | **无工具缺陷**；表象在子代理 mid-flight ls |
| 根因 | 虚拟路径不 resolve | 同 + 封存不核磁盘真相 | 子代理自己 `ls` 撞落盘瞬间（非工具竞态） |

三者**根因正交**：渲染路径、封存对账、（本文）子代理自身 ls 时序。**注意：本文原列为「核盘时机竞态」已证伪**——工具核盘无竞态。silent-drop 的封存裂缝与本文无共享缺口（本文工具侧无裂缝），本文落地的 `_assert_reconcile_consistent` 守卫纯防未来回归。

---

## 7. 取证

- thread `a6e3775c` workspace：`packages/agent/backend/.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/a6e3775c-5a4e-4641-877b-ab6c5cd469e2/user-data/`
- chart-maker trace（自述 partial + fix_handoff 念头）：浏览器 thread `a6e3775c` 的 chart-maker 子任务思考块（用户已贴）
- `outputs/plot_*.png` × 112（磁盘真相）
- `workspace/handoff_chart_maker.json`（`sealed_by=model, chart_files=112`）
- `outputs/report.md` 第 57-71 行（图表段 + `{{img:}}` 引用）
