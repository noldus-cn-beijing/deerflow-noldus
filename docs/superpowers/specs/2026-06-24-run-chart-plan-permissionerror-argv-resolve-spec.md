# Spec：run_chart_plan 进程池透传 /mnt 虚拟路径致渲染崩塌 —— F1 argv 预解析（治本）+ F2 可观测（防再发）

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-24
> 代码基线：dev（含 `302a7046` run_chart_plan 上线）
> 性质：🔴 高 · 确定性结构 bug，生产坐实 112/113 图渲染失败（chart-maker 降级 partial）。**已手段 c 全量复现（0/113）+ F1 修复验证（113/113）**，根因确诊、修法已验证。本 spec 把已验证的修法固化成可实施改动。
> **根因诊断文档（必读）**：[docs/problems/2026-06-24-chart-maker-run-chart-plan-permissionerror.md](../../problems/2026-06-24-chart-maker-run-chart-plan-permissionerror.md)（§6 手段 c 复现结果 = 根因铁证）。
> 受保护文件（sync surgical）：`tools/builtins/run_chart_plan_tool.py`（本仓新增定制）、`packages/ethoinsight/ethoinsight/scripts/_cli.py`（F2，本仓定制）。

---

## 〇、给实施 agent 的一句话

`run_chart_plan` 进程池把绘图脚本的 `--output /mnt/user-data/outputs/x.png` **原样透传**给 worker；**plot 脚本 `fig.savefig(args.output)` 不调 `resolve_sandbox_path`**（与 compute 脚本走 `save_output_json`→自 resolve 不同），`/mnt/user-data` 宿主机不存在 → `PermissionError`/`FileNotFoundError` → 112/113 图全废。**F1（治本）**：在 `run_chart_plan_tool.py` Step 5 build task list 时，对每条 chart 的 `args` 逐项跑 `replace_virtual_path(arg, thread_data)` 预解析（把 bash 的 argv 重写能力提到工具内，进程池与 bash 语义对齐）。**F2（防再发）**：`resolve_sandbox_path` fail-safe 原样返回 `/mnt` 时把 debug 日志升级为 warning（给未来同类静默退化留响亮 grep 锚点 + 消除 seal 层「疑似伪造」对排查者的误导）。**已 repro 验证 F1 = 113/113。**

---

## 一、根因（诊断文档 §6 已确诊，此处摘要 + 代码核实）

### 不对称：compute 脚本自 resolve，plot 脚本不自 resolve
| 链路 | output 写法 | 是否自 resolve | 进程池结果 |
|---|---|---|---|
| **compute（run_metric_plan）** | `save_output_json(args.output, payload)`（`_cli.py:28` 等） | ✅ `save_output_json` 内部 `resolve_sandbox_path(path)`（`_cli.py:157`） | 正常 |
| **plot（run_chart_plan）** | `fig.savefig(args.output)`（`epm/plot_open_arm_time_ratio_bar.py:54` 等）/ `charts.py:_resolve_output_path`（`:48-50` 只 `makedirs` 不 resolve） | ❌ 不 resolve | **崩**（`/mnt/user-data` 宿主机不存在 → makedirs PermissionError / savefig FileNotFoundError） |

### bash vs 进程池的入口不对称（为什么 bash 出图、进程池不出图）
| 入口 | `/mnt` 虚拟路径如何变真实路径 |
|---|---|
| `bash python -m ethoinsight.scripts.* --output /mnt/...` | bash 工具 `replace_virtual_paths_in_command`（`sandbox/tools.py`）**重写命令字符串里的 /mnt 字面量** → 脚本 argv 已是真实路径 → savefig(真实路径) ✅ |
| `run_chart_plan` 进程池 `import_module(script).main(args)` | argv **原样透传** `/mnt/...` → plot 脚本不 resolve → savefig('/mnt/...') ❌ |

### 当前代码的「自相矛盾」（bug 的精确位置）
`run_chart_plan_tool.py`：
- **Step 5（L191）**：`args = list(c.get("args", []) or [])` —— **原样透传**（喂 worker 的是 `/mnt` 路径）。
- **Step 7（L224）**：`output_real = replace_virtual_path(output_virtual, thread_data)` —— **核磁盘时却 resolve 了**。
- → **核的路径（真实）≠ 画的路径（/mnt）** → 画到不存在的 `/mnt` 失败、核真实路径发现没文件 → 正确报 failed。**Step 7 resolve 对了，Step 5 漏了。**

> ⚠️ 诊断文档 §4 最初有「worker env 没设上」的假说，**已被 §6 手段 c 证伪**（env 设对了、worker 也读到了；真因是 plot 脚本根本不读 env 解析 output，直接 savefig 原始路径）。F1 不依赖 env——直接在 argv 层预解析，与 env 机制正交。

---

## 二、F1：argv 预解析（治本，已 repro 验证 113/113）

### 改动位置
`tools/builtins/run_chart_plan_tool.py` Step 5 build task list（L183-201），在 `args = list(c.get("args", []) or [])`（L191）**之后、`tasks.append` 之前**，对 args 逐项预解析。

### 改动内容
```python
    # ---- Step 5: build task list ----
    # 绘图脚本的 args（plan_charts.json 的 entry.args）已含完整 argv（--input --output
    # --parameters-json ...）。
    #
    # F1（spec 2026-06-24-run-chart-plan-permissionerror）：argv 里的 /mnt 虚拟路径必须
    # 在喂 worker 前预解析成真实物理路径。根因：plot 脚本 fig.savefig(args.output) 不调
    # resolve_sandbox_path（与 compute 脚本走 save_output_json→自 resolve 不同），进程池
    # 透传原始 /mnt → PermissionError/FileNotFoundError（112/113 图全废，见 problem doc）。
    # bash 路径靠 replace_virtual_paths_in_command 重写命令字符串；进程池没有这步——F1 把
    # 等价的 argv 重写提到工具内，让两条路径语义对齐（守「用确定性结构约束行为」）。
    # replace_virtual_path 对非 /mnt 项（--input / --parameters-json / JSON 字符串 / 数字）
    # 原样返回（幂等无害），故可对整个 args 数组逐项跑。
    tasks: list[tuple[str, list[str], str]] = []  # (script, args, task_id)
    chart_meta: dict[str, dict[str, Any]] = {}  # task_id -> {output, output_mode, chart_id}
    for c in plan_charts:
        script = c.get("script", "")
        args = list(c.get("args", []) or [])
        # F1: 预解析 argv 里的 /mnt 虚拟路径 → 真实物理路径（对齐 bash 重写语义）。
        args = [replace_virtual_path(a, thread_data) for a in args]
        chart_id = c.get("id", c.get("output", ""))
        output = c.get("output", "")
        output_mode = c.get("output_mode", "per_subject")
        if not script or not args:
            logger.warning("[run_chart_plan] entry 缺 script/args，跳过: id=%s", chart_id)
            tasks.append(("", [], chart_id))
        else:
            tasks.append((script, args, chart_id))
        chart_meta[chart_id] = {"output": output, "output_mode": output_mode}
```

> 注：`chart_meta[chart_id]["output"]` **仍存原始虚拟路径**（`c.get("output")`），Step 7 核磁盘时再 `replace_virtual_path` 解析——这样 chart_files 里存的是虚拟路径（ChartMakerHandoff schema 要求 `/mnt/user-data/outputs/` 前缀，`_validate_chart_paths` 校验），**与 F1 预解析的 args（喂 worker 用真实路径）解耦**。两处各取所需，不冲突。

### F1 安全性核实（已逐项确认，零风险）
1. **`replace_virtual_path` 对非路径项幂等**：函数体（`sandbox/tools.py:493-526`）只对 `path == virtual_base` 或 `path.startswith(f"{virtual_base}/")` 替换，否则 fall through `return path`。`--input`/`--output`/`--parameters-json`/`{"open_arm_zones": ["open"]}`/`--dpi`/`150` 这些非虚拟路径项**原样返回**。✅
2. **脚本内部不会二次 resolve 出错**：plot 脚本读 `--inputs` 走 `read_inputs_json`→`resolve_sandbox_path`，后者对**已是真实路径**（不以 `/mnt/` 开头）**原样返回**（`_cli.py:107` `if not p.startswith("/mnt/"): return Path(p)`，幂等）。所以 F1 预解析后脚本内部 resolve 是无操作，不冲突。✅
3. **thread_data 为 None 兜底**：`replace_virtual_path(a, None)` 直接返回原 `a`（`sandbox/tools.py:508`）。F1 不会因 thread_data 缺失 crash（虽然 Step 1 已保证 workspace_path 存在）。✅

### repro 验证（诊断文档 §6.4，已完成）
```
=== F1 RESULT (pre-resolved argv): 113/113 succeeded ===
failures: 0 ; PNGs on disk: 113
```
对比改前 `=== RESULT: 0/113 succeeded ===`（57 PermissionError + 56 FileNotFoundError）。

---

## 三、F2：fail-safe 可观测（防再发 + 消除误判）

### 背景（为什么要 F2）
1. **本次排查被误导**：seal 层对 chart-maker 自述失败原因做「疑似伪造，规整为机读形态」WARNING（`_reconcile_chart_maker_payload`）——因 chart_id 不在 `plan.skipped`。但**底层渲染是真失败**，不是 LLM 伪造。排查者看到「疑似伪造」WARNING 容易误判成 reward hacking，实际是路径 bug。
2. **`resolve_sandbox_path` fail-safe 静默**：当前收到 `/mnt` 路径却无 env 无兜底时，原样返回 + **debug 日志**（`_cli.py:135`）。debug 在生产日志级别下不可见 → 静默退化无痕。

### 改动位置
`packages/ethoinsight/ethoinsight/scripts/_cli.py:135` 的 `logger.debug(...)`。

### 改动内容
把 fail-safe 原样返回的 debug 升级为 warning，带原始路径（响亮 grep 锚点）：
```python
    # 都没有 → 原样（fail-safe：非沙箱环境/测试直接运行）
    # F2（spec 2026-06-24-run-chart-plan-permissionerror）：收到 /mnt 路径却既无 env 又无
    # workspace_base 兜底 = 虚拟路径未解析，下游用它做文件 IO 必炸（PermissionError/
    # FileNotFoundError）。升级为 WARNING（原 debug 在生产日志级别不可见 → 静默退化无痕，
    # 本次 chart 渲染崩塌排查被「疑似伪造」误导正因此处无响亮信号）。给未来同类问题留 grep
    # 锚点。沙箱外测试直接跑脚本也会走这条（合法），但 WARNING 不致命、只留痕。
    logger.warning(
        "resolve_sandbox_path: 虚拟路径 %s 未解析（无 DEERFLOW_PATH_* env 且无 workspace_base "
        "兜底）→ 原样返回；若下游用它做文件 IO 将抛 PermissionError/FileNotFoundError",
        p,
    )
    return Path(p)
```

> ⚠️ **F2 与 F1 的关系**：F1 修复后，正常路径下 plot 脚本 argv 已是真实路径、不再走到 `resolve_sandbox_path` 的 `/mnt` 分支，所以 F2 的 warning **正常情况下不会触发**。F2 是**防御性可观测**——万一未来又有「虚拟路径漏解析」的新路径（如新增脚本/新 I/O 边界），warning 会响亮报警，而非像本次一样静默退化 + 被「疑似伪造」误导排查方向。

> ⚠️ **F2 的潜在噪音权衡**：沙箱外单测直接跑 ethoinsight 脚本（pytest）会走 fail-safe 分支 → 刷 WARNING。**可接受**（warning 不致命，测试日志里偶现一行不影响结果）。若实施时发现某高频测试路径被刷屏，可加一个 `_warned_paths` set 去重（同一路径只 warn 一次），但**不要因为怕噪音退回 debug**——静默正是本次被误导的根源。

---

## 四、为什么不做 F3/F4（守范围，避免过度工程）

| 候选 | 为什么不做 |
|---|---|
| **F3**（per_subject 扇出超阈值自动抽样 + 提示用户）| 治的是「放大器」根因 B（113 task 撞 max_turns），**不是起爆器**。且改 prompt 加规则违反 HarnessX Telecom 禁令（CLAUDE.md §病理 under-exploration）。F1 修复后 113 图真能画完，max_turns 压力本身缓解（不再反复 bash 自救烧 turn）。**缓议，本 spec 不含。** |
| **F4**（`resolve_sandbox_path` fail-safe 改 fail-loud 抛异常）| 高风险——会破坏所有沙箱外测试（pytest 直接跑脚本依赖 fail-safe 原样返回），需配 `allow_raw=True` 开关，改动面大。F2（升 warning）已达「可观测」目的且零破坏。**不做。** |

> **根因 B（max_turns 放大器）是否需要单独处理？** 诊断 §4 根因 B 明确「是放大器不是起爆器」——本次 max_turns 被 chart-maker **反复 bash 自救**烧光（run_chart_plan 返回 partial 后挣扎）。**F1 修复后 run_chart_plan 一次画完 113 张返回 completed，chart-maker 不再需要 bash 自救 → max_turns 压力自然消失。** 故 F1 顺带缓解根因 B，无需单独改 max_turns/budget。若 F1 上线后 dogfood 仍见 max_turns 早停（理论上不应），再单开 spec 议 chart_budget 默认值。

---

## 五、测试清单（TDD 红→绿）

> 测试用 `_TASK_RUNNER_OVERRIDE` 注入同步 runner（绕 ProcessPoolExecutor fork/pickle，memory `feedback_processpoolexecutor_test_runner_injection_and_ssot_parity`）。但 F1 的核心是**argv 预解析**，可在 runner 注入层断言 worker 收到的 args 已是真实路径。

1. **T1 F1 argv 预解析（核心红→绿）**：构造 plan，charts[].args 含 `--output /mnt/user-data/outputs/x.png`。注入 runner 捕获实际收到的 args。
   - **改前红**：runner 收到的 args 含 `/mnt/user-data/outputs/x.png`（未解析）。
   - **改后绿**：runner 收到的 args 含真实物理路径（`<workspace_parent>/outputs/x.png`），无 `/mnt` 前缀。
2. **T2 F1 非路径项不被破坏**：args 含 `--parameters-json`, `{"open_arm_zones": ["open"]}`, `--dpi`, `150`。断言这些项预解析后**原样不变**（只路径项被重写）。
3. **T3 F1 多 subject 全解析**：plan 含 28 个 per_subject chart（args 各含不同 `/mnt/.../inputs_*_s{i}.json` + `/mnt/.../plot_*_s{i}.png`）。断言全部 args 的 `/mnt` 项都被解析（不是只首个）——复现本次 112/113 场景的正向验证。
4. **T4 F1 thread_data=None 兜底**：thread_data 缺 workspace_path（理论上 Step 1 已拦，但守 robustness）→ replace_virtual_path 返回原 args，不 crash。
5. **T5 chart_files 仍存虚拟路径**：F1 改 args（worker 用真实路径），但 chart_meta["output"] 仍存虚拟路径 → Step 7 核磁盘后 chart_files 里是 `/mnt/user-data/outputs/` 前缀（过 ChartMakerHandoff `_validate_chart_paths`）。断言 F1 不破坏 chart_files 的虚拟路径契约。
6. **T6 F2 fail-safe warning**：`resolve_sandbox_path("/mnt/user-data/workspace/x.json")` 无 env 无 workspace_base → 断言记 WARNING（用 caplog 捕获），含原始路径字符串。
7. **T7 F2 真实路径不 warn**：`resolve_sandbox_path("/real/path/x.json")`（非 /mnt）→ 原样返回，**不** warn。
8. **T8 run_metric_plan 回归（catastrophic forgetting 自检）**：F1 只改 run_chart_plan，不碰 run_metric_plan；但 F2 改 `_cli.py:resolve_sandbox_path` 是**两工具共用**——跑 run_metric_plan 既有测试全绿，确认 F2 升 warning 不破坏 compute 链（compute 脚本走 save_output_json 正常 resolve，不触发 fail-safe warning）。
9. **T9 import 环**：改 run_chart_plan_tool.py + _cli.py 后裸导入 `app.gateway` + `make_lead_agent` 0 退出。
10. **T10 端到端 dogfood（验收）**：同份 28-subject EPM 复跑——run_chart_plan **一次画完 113 张**、chart-maker handoff `status=completed`、`chart_files=113` 全真实落盘、`sealed_by=run_plan`、**不再 PermissionError/FileNotFoundError**、chart-maker 不 bash 自救、不撞 max_turns。

---

## 六、验收（确定性 gate 序列）
1. T1（F1 核心）改前红（runner 收到 `/mnt` args）、改后绿（真实路径）——这是「复现 dogfood 根因」的红验证。
2. T2-T7 全绿。
3. **回归（seesaw）**：run_metric_plan 全量（F2 共用 `_cli.py`）+ run_chart_plan 既有测试 + chart-maker seal 邻域（`_reconcile_chart_maker_payload` 不碰但确认不回归）+ backend 全量（守已知污染基线）。
4. import 环：T9 两入口 0 退出。
5. **端到端 dogfood**：T10 同份 28-subject EPM，113/113 落盘、completed、不报错。**这是真验收**（诊断 §6 手段 c 已离线验证 113/113，dogfood 是在线确认）。

---

## 七、风险与三大病理自检

1. **Reward hacking**：本次**不是** reward hacking——chart-maker 诚实报 partial（磁盘核验生效，ETHO-10 不变式正确工作）。但 seal 层「疑似伪造」WARNING 措辞误导排查者以为 LLM 撒谎。**F2 的响亮 warning 消除这个误判**（让底层真失败有痕，不再被「疑似伪造」盖过）。F1/F2 都不削弱 ETHO-10 磁盘真相核验（chart_files 仍核 exists()）。
2. **Catastrophic forgetting**：
   - F1 只改 run_chart_plan Step 5，不碰 run_metric_plan（同构兄弟）。
   - **F2 改 `_cli.py:resolve_sandbox_path` 是 compute + plot 脚本共用**——必须跑 run_metric_plan 全量回归（T8），确认升 warning 不破坏 compute 链。
   - chart_files 虚拟路径契约（`_validate_chart_paths` 前缀校验）：F1 预解析的是**喂 worker 的 args**，chart_meta["output"] 仍存虚拟路径（T5 守），不破坏。
3. **Under-exploration**：F1 是**结构修**（argv 层预解析对齐 bash/进程池语义），**不是改 prompt 让 chart-maker 更耐心**——守判据「结构缺失→上门」。诊断 §5 HarnessX 自检明确「不要用改 prompt 修这个」。F3（改 prompt 抽样）正因此缓议。

### 与本 spec 同源的教训（写进诊断 + 本 spec，避免再犯）
- **run_chart_plan 原 spec 的遗漏**：原 spec（`2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md` §三 Step 5）写「args 直接透传，对齐 run_metric_plan」——**漏核实「直接透传」的前提是脚本自 resolve output**。compute 脚本自 resolve（save_output_json），plot 脚本不自 resolve（fig.savefig）。**「对齐范本」时必须核实范本能工作的隐性前提在新链路是否成立**，不能只看「形似」。
- **实跑取证假绿教训**：原 spec 的进程内实跑（4 张 rc=0）**把 `--output` 改写成 `/tmp/...` 真实路径**（为不污染真实 outputs），无意中做了 F1 预解析，**恰好掩盖了 bug**。**实跑取证必须用与生产完全相同的输入（args 原样 /mnt + 真实 thread_data 的 path_env），不能改写关键输入**——改写 = 没测真实路径。诊断 agent 用手段 c（args 原样透传）做对了才抓到。（同族：memory `feedback_pr115_stage1_equivalence_baseline_is_hollow_error_string` 假绿教训。）

---

## 八、关键代码锚点（已核实，行号 = dev 含 302a7046）
- **F1 改点**：`tools/builtins/run_chart_plan_tool.py:191`（Step 5 `args = list(...)` 后加预解析行）；Step 7 核磁盘 `replace_virtual_path`(L224，已对，不动)；`replace_virtual_path` import(L141)
- **F2 改点**：`packages/ethoinsight/ethoinsight/scripts/_cli.py:135`（fail-safe `logger.debug` → `logger.warning`）
- **根因证据**：`epm/plot_open_arm_time_ratio_bar.py:54`（`fig.savefig(args.output)` 不 resolve）；`charts.py:_resolve_output_path`(48-50，只 makedirs)；对照 `_cli.py:save_output_json`(146)→`resolve_sandbox_path`(157)（compute 自 resolve）
- **安全性核实**：`sandbox/tools.py:replace_virtual_path`(493-526，非虚拟路径项原样返回幂等)；`_cli.py:resolve_sandbox_path`(107，非 /mnt 原样返回幂等)
- **诊断文档**：`docs/problems/2026-06-24-chart-maker-run-chart-plan-permissionerror.md`（§6 手段 c 复现 0/113→F1 113/113）
- **repro 脚本**：`scripts/repro/run_chart_plan_repro.py`（抽样）+ `run_chart_plan_repro_full.py`（全量，可复用作 F1 回归验证）
- **原 spec**：`docs/superpowers/specs/2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md`（§三 Step 5 遗漏点）
- **pattern 文档**：`docs/superpowers/specs/2026-06-24-deterministic-batch-execution-tool-pattern.md`（§三铁律 5 路径解析——本 bug 暴露该铁律对 plot 链不够，下方建议补一条）

---

## 九、pattern 文档需补的一条铁律（实施时一并更新）

本 bug 暴露「确定性批量执行工具 pattern」文档缺一条：**进程内调脚本时，脚本的 output/input 路径解析责任必须明确**。

补进 pattern 文档（`2026-06-24-deterministic-batch-execution-tool-pattern.md`）§三铁律，新增：

> **铁律 11（argv 路径解析对齐）**：bash 调脚本时 `replace_virtual_paths_in_command` 重写命令字符串里的 `/mnt` 字面量，进程内调没有这步。**batch 工具必须在 Step 5 build task list 时对每条 args 跑 `replace_virtual_path` 预解析**，让进程池与 bash 在 argv 语义上对齐。**不能假设脚本自己 resolve**——run_metric_plan 的 compute 脚本恰好走 `save_output_json`→自 resolve（侥幸），run_chart_plan 的 plot 脚本 `fig.savefig(args.output)` 不 resolve（2026-06-24 生产 112/113 崩塌坐实）。「对齐范本」时必须核实范本能工作的隐性前提（脚本自 resolve）在新链路是否成立，别只看形似。

---

## milestone 建议
归入「chart-maker 架构对齐 code-executor / 批量执行确定性化」track。checkpoint：「run_chart_plan 上线（302a7046）首次吃真实 28-subject 数据 → 暴露 argv 透传 /mnt 致 plot 脚本 savefig 崩塌（112/113，因 plot 不自 resolve、compute 自 resolve 的不对称）→ 手段 c 复现 0/113 + F1 验证 113/113 → F1（argv 预解析对齐 bash）+ F2（fail-safe 升 warning 可观测）spec → 待实施 + dogfood 验收」。同时补 pattern 文档铁律 11（argv 路径解析对齐），固化「batch 工具进程内调脚本必须预解析 argv，不假设脚本自 resolve」。
