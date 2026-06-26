# 2026-06-24 EPM dogfood 问题：chart-maker `run_chart_plan` 进程池渲染 PermissionError，112/113 图未落盘

> **目的**：把一次端到端 dogfood 暴露的 chart-maker 渲染崩塌沉淀成独立诊断材料。
> 本次只取证 + 定位根因，**未改任何代码**（用户未授权修）。修复方向在 §5 列出待用户拍板。
>
> **关联 thread**：`89be7344-17fc-4fa1-a1ef-ce469d761c2a`（EPM-Xuhui-28，28 trials）
> **关联 run dir**：`/tmp/noldus-e2e-runs/20260624-140825/`（`analyze.txt` / `clarifications.json` / `sse-0..6.txt` / `gateway-excerpt.log` / `final.png`）
> **手段 c 复现脚本**：`scripts/repro/run_chart_plan_repro.py`（抽样）+ `run_chart_plan_repro_full.py`（全量）。**复现结果见 §6——根因已确诊，§4 的 env 假说被证伪**。

---

## 0. 一句话结论

chart-maker 走确定性工具 `run_chart_plan`（`ProcessPoolExecutor` 进程内并行渲染）时，**112/113 张图全部失败**，失败原因混杂 `FileNotFoundError: .../plot_*_s27.png` 与 `PermissionError: [Errno 13] Permission denied: '/mnt/user-data'`，最终 `max_turns=15` 早停、`sealed_by=run_plan` 封了 `status=partial` 的 handoff。**分析核心（code-executor / data-analyst / report-writer）完全健康**——崩塌只发生在图表渲染层，且是**确定性结构 bug，不是 LLM 抖动**。同一批脚本用 `bash python -m ethoinsight.scripts.*` 直接调**能正常出图**（`plot_box_open_arm.png` + `plot_open_arm_time_ratio_bar_s0.png` 两张落盘证据）。

---

## 1. 上下文（30 秒）

- **EthoInsight**：行为学 AI 分析助手，`~/noldus-insight/`。Lead → code-executor → data-analyst → chart-maker → report-writer 五段流水线。
- **chart-maker 新架构**（2026-06-24 落地，commit `302a7046`）：用确定性 first-party 工具 `run_chart_plan` 替换 LLM bash 编排，对标 `run_metric_plan`。工具内 `ProcessPoolExecutor` 进程内调绘图脚本 + 磁盘核验 + 原子封存 handoff。spec：`docs/superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution.md`（未核实路径，待查）。
- **本次 dogfood**：用户真实数据 `Raw data-EPM-Xuhui-28`（28 只小鼠 EPM，划了区变体，`open`/`closed` 区归属列，按 `Treatment` 分 XX/XY/YY/YZ 四组）。这是 `run_chart_plan` 上线后第一次吃真实多 trial 数据。
- **EVM19/EVM21/EVM22 提示**：CLAUDE.md 第 10 条记录的「EV19 模板识别地基」与本次无关；本次是 chart 渲染层，不是模板识别层。

---

## 2. 用户做了什么

用 `/noldus-insight-e2e` skill 对 `Raw data-EPM-Xuhui-28` 跑端到端 dogfood（generic `确认` 驱动 HITL）：

| 阶段 | 结果 |
|---|---|
| 上传 28 个 xlsx | ✅ chips=28/28 |
| Gate 1 反问（模板 A/B + 分组含义 + 列语义） | 6 轮 generic `确认`，lead 最终默认 A（PlusMaze-AllZones）+ 4 组 |
| code-executor | ✅ sealed `run_plan`，28 subjects 全部 `m_*.json` 落盘 + `stats.json` |
| data-analyst | ✅ **sealed `finalize`**，`statistics_status=ok`，0 errors（**#187 修复生效**） |
| **chart-maker** | ❌ **`max_turns=15` 早停**，`sealed_by=run_plan` 封 `status=partial`，1/113 图落盘 |
| report-writer | ✅ sealed `model`，`report.md` 10.8 KB 落盘，`experiment_summary` fact 写入 |

**总耗时 ~24 min（1356s）**，6 轮 HITL（0 repeated）。report.md 存在 → 流水线到达 terminal，但图表层崩塌。

---

## 3. 直接观察到的症状（事实）

### 症状 1：chart-maker `max_turns=15` 早停
`gateway-excerpt.log`：
```
2026-06-24 14:27:59 - deerflow.subagents.executor - WARNING - [trace=246ec925] Subagent chart-maker reached max_turns=15 AI messages, terminating early
```

### 症状 2：`run_chart_plan` 工具返回 112/113 失败
chart-maker handoff（`handoff_chart_maker.json`，sealed_by=`run_plan`）：
```json
{
  "status": "partial",
  "summary": "1/113 charts rendered, partial (run_chart_plan)",
  "chart_files": ["/mnt/user-data/outputs/plot_box_open_arm.png"],
  "failed_charts": [ ... 5 entries ... ],
  "gate_signals": { "charts_generated": 1, "failed_charts": 112, "statistical_validity": "warning" },
  "sealed_by": "run_plan"
}
```
plan_charts.json：**113 charts** = 1 aggregate（`box_open_arm`）+ 112 per_subject（4 chart 类型 × 28 subjects）。

### 症状 3：失败原因混杂两类错误
`failed_charts[].reason`（经 seal 层「规整为机读形态」后）：
- `open_arm_time_ratio_bar` / `zone_entry_distribution` → `FileNotFoundError: .../plot_*_s27.png`
- `trajectory` / `heatmap` / `box_open_arm` → `PermissionError: [Errno 13] Permission denied: '/mnt/user-data'`

### 症状 4：seal 层 5 次 WARNING「疑似伪造，规整为机读形态」
```
2026-06-24 14:26:07 / 14:28:52 - deerflow.tools.builtins.seal_handoff_tools - WARNING -
  chart-maker seal: failed_charts['...'] reason订正——plan.skipped=[] 未含此 id，
  LLM 自述 "FileNotFoundError/PermissionError ..." 疑似伪造，规整为机读形态
```
**这是 seal 层在 chart-maker 自述失败原因时做的防御性规整**——把 LLM 的散文式失败叙述改成机读形态。注意：seal 层**怀疑**这些 reason 是伪造的（因为对应 chart_id 不在 `plan.skipped`），但**底层写失败是真的**（见 §4 根因）。

### 症状 5：同一脚本，bash 调用能出图，进程池调用不能
chart-maker 在 `max_turns` 耗尽前的自救（clarif round 6 的 SSE 追踪原话）：
> "`run_chart_plan` 工具有进程级权限问题，但直接 `python -m ethoinsight.scripts.*` 可正常工作。让我用 bash 直接读取 plan 并通过循环脚本批量绘图"

落盘证据：`outputs/plot_box_open_arm.png`（aggregate，bash 出）+ `outputs/plot_open_arm_time_ratio_bar_s0.png`（per_subject，bash 只来得及出 s0）。

### 症状 6：ETHO-10 伪完成洞**没**被触发 ✅
forensic Panel B：`chart_files 实存 = 1/1`，`aggregate 未落盘=[]`。chart-maker **诚实报告 partial**（112 failed），没有谎称 completed。这是 `run_chart_plan` 磁盘真相核验的正确行为——**不是 reward hacking**，是 graceful degradation。

---

## 4. 根因分析

### 根因 A（主因，**已确诊**）：绘图脚本 `fig.savefig(args.output)` 不调 `resolve_sandbox_path`，进程池透传原始 `/mnt` 路径

> ⚠️ §4.1 下面的「env 不对称」假说是手段 c **之前**的推测，**已被 §6 证伪**（env 设对了、worker 也读到了）。真正根因见下文「确诊」段 + §6。保留假说段作为排查路径记录。

**确诊根因**（手段 c 全量重放 0/113 + F1 验证 113/113，见 §6）：所有绘图脚本把原始 `args.output`（`/mnt/user-data/outputs/...`）直接喂给 `fig.savefig()` / `trajectory_plot(output_path=...)`，**从不调 `resolve_sandbox_path()`**。`/mnt/user-data` 在宿主机不存在（已确认 `ls /mnt/user-data` → 无此目录），所以 `savefig` 抛 `FileNotFoundError`、`os.makedirs('/mnt/user-data/...')` 抛 `PermissionError`。

**bash vs 进程池的不对称**（确诊）：

| 入口 | `/mnt/user-data` 虚拟路径如何变真实路径 |
|---|---|
| `bash python -m ethoinsight.scripts.* --output /mnt/user-data/outputs/x.png` | bash 工具 `replace_virtual_paths_in_command`（`sandbox/tools.py`）**重写命令字符串里的 /mnt 字面量** → 脚本 argv 已是真实路径 → `savefig(真实路径)` ✅ |
| `run_chart_plan` 进程池 `import_module(script).main(args)` | argv 原样透传 `/mnt/...` → 脚本 `fig.savefig('/mnt/...')` ❌（脚本不 resolve，`/mnt` 不存在）|

对照 `run_metric_plan`（同构兄弟，正常）：compute 脚本经 `_cli.py:save_output_json:157` 显式 `resolve_sandbox_path(path)` 写 JSON → 进程池正常。**chart 链漏了这个对称收口**——`_cli.py` 的 I/O 边界 resolve 只补到 `save_output_json`/`read_inputs_json`/`read_groups_json`，**漏了 plot 的 `fig.savefig`**（plot 脚本不走 `save_output_json`，直接 matplotlib）。

**F1 修复**（已验证 113/113，见 §6.4）：在 `run_chart_plan` Step 5 对每条 chart 的 `args` 整体跑 `replace_virtual_path(arg, thread_data)` 预解析——把 bash 的 argv 重写能力提到工具内，进程池与 bash 在 argv 语义上确定对齐。

---

### 根因 A 旧假说（手段 c 之前的推测，**已证伪，仅留排查记录**）

~~**机制对比**（旧假说：worker env 没设上）~~

| 入口 | `/mnt/user-data` 虚拟路径如何变真实路径 |
|---|---|
| `bash python -m ethoinsight.scripts.* --output /mnt/user-data/outputs/x.png` | bash 工具 `replace_virtual_paths_in_command`（`sandbox/tools.py`）**重写命令字符串里的 /mnt 字面量** → 进程拿到的 argv 已是真实物理路径 |
| `run_chart_plan` 进程池 `import_module(script).main(args)` | argv 原样透传 `/mnt/...` → 脚本靠 `resolve_sandbox_path()`（`_cli.py:81`）读 `DEERFLOW_PATH_*` env 解析；**env 由 `_worker_init` 在 worker 进程内 `os.environ.update(path_env)` 设置**（`run_chart_plan_tool.py:64-79`） |

**不对称点**：bash 路径解析在**命令字符串层**（对脚本透明，脚本看到的 argv 已是真实路径）；进程池路径解析在**脚本进程内**（脚本必须主动调 `resolve_sandbox_path`，且依赖 `DEERFLOW_PATH_*` env 已被 worker initializer 设上）。

**崩塌链**（待 §4.3 手段 c 精确证实）：
1. `run_chart_plan` 建 `ProcessPoolExecutor(initializer=_worker_init, initargs=(path_env,))`，`path_env = _build_path_env(thread_data)`。
2. worker fork 后 `_worker_init` 设 `DEERFLOW_PATH_*` + `MPLBACKEND=Agg`。
3. per_subject 脚本（如 `plot_open_arm_time_ratio_bar`）经 `resolve_per_subject_input` → `read_inputs_json(args.inputs)` → `resolve_sandbox_path(item)`。
4. **若某条路径既无对应 env 又无 workspace_base 兜底** → `resolve_sandbox_path` fail-safe 原样返回 `/mnt/...` → 下游 `Path('/mnt/user-data/...').open()` 抛 `PermissionError: '/mnt/user-data'`（宿主机根下无此目录，且 `/mnt` 可能无写权）或 `FileNotFoundError`。
5. aggregate 脚本 `plot_box_open_arm` **成功**——它的 `--inputs` / `--groups` / `--output` 在 plan 生成时可能已被 `replace_virtual_path` 预解析成真实路径，或其 env 链恰好命中；per_subject 的 `inputs_*_s{i}.json`（62 字节小文件，路径字符串在 JSON 内部）**不经 bash 重写**，全靠 worker env，于是全军覆没。

**为什么 `plot_open_arm_time_ratio_bar_s0` 唯一落盘？** 手段 c 待证伪的两个候选：(a) s0 是首个提交的 task，worker env 在首个 task 时恰好可用、后续 task env 竞态丢失；(b) s0 的 inputs JSON 路径恰好命中某前缀 env。**倾向 (b) 更可能**——`_worker_init` 是 initializer（每个 worker 启动时调一次，非 per-task），env 不应有 per-task 竞态；但 s0 与 s1..s27 的 inputs JSON 结构完全一致（都是 62 字节），为何独 s0 成功仍是谜，**这是手段 c 必须回答的核心问题**。

### 根因 B（次因，触发点）：`run_chart_plan` 工具默认画全部 113 图，无 chart_budget，撞 max_turns

plan_charts.json 里 `charts_budget_remaining` 字段存在但 chart-maker 这次没传 `only_chart_ids` 也没设 budget，`run_chart_plan` 一次性提交 113 个 task。即使渲染层正常，113 图 × per-task 120s 超时也逼近 chart-maker 的 `max_turns=15` 上限。**但本次 max_turns 不是被渲染耗时吃掉的**——是 chart-maker 在 `run_chart_plan` 返回 partial 后，反复尝试 bash 自救（5/15→15/15 的 turn 追踪全是 bash/read_file/run_chart_plan 循环）烧光 15 turn。所以根因 B 是**放大器**不是**起爆器**。

### 根因 C（架构债，与 2026-05-14 根因 B 同族）：JSON 内部路径不经 bash 重写

这与 `docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md` 根因 B 是**同一族 bug**：bash 命令字符串里的 `/mnt` 会被重写，但**文件内容 / JSON 内部 / argv 透传**的 `/mnt` 不会。2026-05-14 的修法是 `_cli.py:resolve_sandbox_path` + `DEERFLOW_PATH_*` env 兜底；本次证明**这个兜底在 `ProcessPoolExecutor` worker 里不可靠**——worker 的 env 依赖 `_worker_init` 正确设上，而 `plot_*_s0` 之外全部失败说明 env 链在某处断了。

### 触发点：真实多 trial 数据首次暴露 per_subject 扇出

`run_chart_plan` 上线后，之前测的可能是单 subject / 少 chart 场景；本次 28 subjects × 4 chart 类型 = 112 per_subject tasks 是**首次大规模 per_subject 扇出**，把根因 A 的 env 不对称从「偶发」放大成「系统性 112/113 失败」。

---

## 4.3 手段 c 复现（确定性取证）

**目标**：绕开 LLM，用与 `run_chart_plan` 完全相同的进程池路径，对 thread 89be7344 的真实 `plan_charts.json` + inputs 跑渲染，回答三个问题：
1. **`PermissionError` 的精确 traceback 栈**——是 `resolve_sandbox_path` 原样返回 `/mnt` 还是 worker env 没设上？
2. **为何 s0 成功、s1..s27 失败**——env 竞态？路径前缀差异？
3. **bash 路径与进程池路径的 argv 差异**——同一脚本同一 inputs，为何一个出一个不出？

复现脚本：`scripts/repro/run_chart_plan_repro.py`（手段 c 产物）。用 thread 89be7344 的 workspace（`.../threads/89be7344.../user-data/workspace/`）作真实输入，构造 `thread_data` 喂 `_build_path_env` + `_worker_init` + `_run_chart_task`，逐 task 打印 `(rc, error, traceback)`。**复现结果见本文档 §6（手段 c 跑完后回填）**。

---

## 5. 修复方向（待用户拍板，未动手）

| 候选 | 改动位置 | 治什么 | 风险 |
|---|---|---|---|
| **(F1)** `run_chart_plan` 在提交 task 前，对所有 chart 的 `args`（含 `--input`/`--inputs`/`--output`）跑一遍 `replace_virtual_paths_in_command` 预解析成真实物理路径，再透传给 worker | `run_chart_plan_tool.py` Step 5（build task list） | **治本根因 A**：让进程池 argv 与 bash argv 语义对齐，脚本不再依赖 worker env | 低——`replace_virtual_path` 已是成熟函数；但需确认所有绘图脚本的 `--output` 在 argv 层解析后，`emit_result`/`save_output_json` 不会二次 resolve 出错 |
| **(F2)** `_worker_init` 改 `os.environ` 为 `os.environ` **副本** + 诊断日志：worker 内启动时 `logger.debug` 打印 `DEERFLOW_PATH_*` 是否设上 + `resolve_sandbox_path` 失败时升级为 WARNING 带原始路径 | `_cli.py` + `run_chart_plan_tool.py` | **治标 + 可观测**：至少让 s1..s27 失败时有响亮日志，不再「疑似伪造」说不清 | 极低——纯加日志 |
| **(F3)** chart-maker prompt / `run_chart_plan` 工具：per_subject 扇出超阈值（如 >40）时自动 `only_chart_ids` 抽样 + 提示用户「per_subject 图过多，建议聚合」 | chart_maker.py prompt + 工具 | **治根因 B 放大器**：避免 113 task 撞 max_turns | 中——改 prompt 可能影响正常场景（守 CLAUDE.md HarnessX「加 prompt 规则」警告） |
| **(F4)** 把 `resolve_sandbox_path` 的 fail-safe「原样返回 /mnt」改成**抛 `UnresolvedVirtualPathError`**（fail-loud），逼调用方显式处理 | `_cli.py` | **治根因 C 架构债**：永久消灭「静默退化到 /mnt」 | 高——可能破坏沙箱外测试（pytest 直接跑脚本依赖 fail-safe），需配 `allow_raw=True` 开关 |

**我的偏好**：**F1（治本）+ F2（可观测）优先**。F1 直接对齐 argv 语义，是从结构上根治（守 CLAUDE.md「用确定性结构约束行为」）；F2 给未来同类问题留 grep 锚点。F3/F4 风险/收益比差，缓议。

**⚠️ HarnessX 自检**（CLAUDE.md §病理）：
- **Reward hacking**：本次**不是**——chart-maker 诚实报 partial，seal 层磁盘核验生效。但 seal 层「疑似伪造」WARNING 的措辞会误导排查者以为 LLM 撒谎，实际是底层真失败——**F2 的可观测日志能消除这个误判**。
- **Catastrophic forgetting**：F1 改 `run_chart_plan` Step 5 必须全量跑 `run_metric_plan` 回归（同构兄弟）+ 现有 chart-maker 测试，确认 argv 预解析不破坏 metric 链。
- **Under-exploration**：**不要**用「改 chart-maker prompt 让它更耐心」修这个——根因是结构（env/argv 不对称），不是 prompt 硬度。守判据「结构缺失→上门」。

---

## 6. 手段 c 复现结果（已完成 2026-06-24）

复现脚本：`scripts/repro/run_chart_plan_repro.py`（抽样诊断）+ `scripts/repro/run_chart_plan_repro_full.py`（全量 113 task）。两者均用 thread 89be7344 的真实 `plan_charts.json` + inputs，镜像生产 `run_chart_plan` 的 `ProcessPoolExecutor(initializer=_worker_init, initargs=(path_env,))` + `_run_chart_task` 路径（backend venv python 跑，`ethoinsight` 经 `_editable_impl_ethoinsight.pth` 可导入——与生产同款）。

### 6.1 精确根因（确诊，推翻了 §4 根因 A 的 env 假说）

**`run_chart_plan` 进程池路径本身没问题，env 也设对了。真正的 bug 在绘图脚本侧：所有 plot 脚本都把原始 `args.output`（`/mnt/user-data/outputs/...` 虚拟路径）直接喂给 `fig.savefig()` / `trajectory_plot(output_path=...)`，从不调 `resolve_sandbox_path()` 解析。**

证据链（全 grep 确认）：

| 绘图脚本 | output 处理 | 是否 resolve |
|---|---|---|
| `epm/plot_open_arm_time_ratio_bar.py:54` | `fig.savefig(args.output, dpi=150)` | ❌ 原样 |
| `epm/plot_zone_entry_distribution.py:58` | `fig.savefig(args.output, dpi=150)` | ❌ 原样 |
| `_common/plot_trajectory.py:41` | `trajectory_plot(df, output_path=args.output)` | ❌ 原样透传 |
| `_common/plot_heatmap.py:36` | `heatmap_plot(df, output_path=args.output)` | ❌ 原样透传 |
| `charts.py:_resolve_output_path` (box_plot/trajectory_plot/heatmap_plot 内部) | `os.makedirs(os.path.dirname(output_path), exist_ok=True)` 后返回原 path | ❌ 只 mkdir 不 resolve |

对照：**`run_metric_plan` 之所以正常**，是因为所有 compute 脚本经 `_cli.py:save_output_json`（`:146`）写 JSON，而 `save_output_json` **显式调 `resolve_sandbox_path(path)`**（`:157`）把 `/mnt/...` 解析成真实路径。**chart 链没有这个对称收口**——这是 `run_chart_plan` 上线时漏掉的镜像点（`_cli.py` 文件头注释也承认「与 save_output_json 在同一 I/O 边界对称」是 2026-06-15 才补的，但当时只补了 read_inputs_json/read_groups_json，**漏了 plot 的 fig.savefig**）。

### 6.2 全量重放：0/113（复现生产）

`run_chart_plan_repro_full.py` 把 `path_env` 的 OUTPUTS 指向干净真实目录，`args` 原样透传（含 `/mnt/user-data/outputs` 虚拟 output），跑全 113 task：
```
=== RESULT: 0/113 succeeded, 113 failed ===
distinct failure error types:
   57  PermissionError       # box_plot/trajectory/heatmap 的 os.makedirs('/mnt/user-data/...') 抛
   56  FileNotFoundError     # fig.savefig('/mnt/user-data/outputs/...') 抛
PNGs actually on disk: 0
```
**完美复现**。错误类型分布与生产 handoff 的 `failed_charts` reason 完全吻合（PermissionError + FileNotFoundError 两类）。

### 6.3 s0 异常真相（解开 §4 谜团）

生产磁盘上 `plot_box_open_arm.png` + `plot_open_arm_time_ratio_bar_s0.png` **不是 `run_chart_plan` 出的**——是 chart-maker 在 `run_chart_plan` 返回 0/113 后，用 **bash 自救循环**（`python -m ethoinsight.scripts.*`，bash 经 `replace_virtual_paths_in_command` 重写 `/mnt` → 真实路径）出的。bash 只来得及出 aggregate（1 张）+ per_subject s0（1 张）就撞 `max_turns=15`。证据：clarif round 6 的 SSE 原话「`run_chart_plan` 有进程级权限问题，但直接 `python -m ethoinsight.scripts.*` 可正常工作。让我用 bash…批量绘图」。

**所以 s0 没有任何特殊性**——它只是 bash 自救循环跑到的第一个 per_subject task。谜团解开。

### 6.4 F1 修复验证：113/113 ✅

对每个 chart 的 `args` 在 task 提交前跑 `replace_virtual_path(arg, thread_data)` 预解析（镜像 bash 重写语义），其余原样：
```
=== F1 RESULT (pre-resolved argv): 113/113 succeeded ===
failures: 0
PNGs on disk: 113
```
**F1（在 `run_chart_plan` Step 5 预解析 argv）从结构上根治，0 改绘图脚本、0 改 prompt。**

### 6.5 修复方案定稿（更新 §5）

**F1 优先级升为「必做、治本、零风险」**：在 `run_chart_plan_tool.py` Step 5（build task list，`:189-201`）对每条 chart 的 `args` 整体跑 `replace_virtual_path`：
```python
# F1: 预解析 argv 里的 /mnt 虚拟路径 → 真实物理路径（对齐 bash 语义）
# 根因：plot 脚本 fig.savefig(args.output) 不调 resolve_sandbox_path，
# 进程池透传原始 /mnt 路径 → PermissionError/FileNotFoundError（见 problem doc 2026-06-24）。
args = [replace_virtual_path(a, thread_data) for a in args]
```
这把「argv 层预解析」从 bash 专属能力提到 `run_chart_plan` 工具内，让进程池与 bash 两条路径在 argv 语义上**确定对齐**——守 CLAUDE.md「用确定性结构约束行为，不用 prompt 规则」。

**F2（可观测）配套**：`resolve_sandbox_path` 收到未解析 `/mnt` 路径时升级为 WARNING 带 chart_id（而非现在的 debug 静默），给未来同类问题留响亮 grep 锚点——也消除本次 seal 层「疑似伪造」措辞对排查者的误导（§5 HarnessX 自检）。

---

## 7. 元信息

- **诊断 agent**：Claude（本会话）
- **取证时间**：2026-06-24
- **用户授权范围**：写 problem doc + 手段 c 复现，**未授权改代码**
- **关联文档**：
  - `docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md`（根因 C 同族先例）
  - CLAUDE.md 第 15 条（seal auto-seal / n=1 / 判读语言对齐一批合入）
  - CLAUDE.md 「HarnessX 三大病理自检」
- **关联 commit**：`302a7046`（`run_chart_plan` 确定性执行工具上线）
- **forensic 证据**：`/tmp/noldus-e2e-runs/20260624-140825/{analyze.txt,clarifications.json,sse-0..6.txt,gateway-excerpt.log,final.png}`
