# LangGraph / DeerFlow Agent Harness 工程化最佳实践

> 日期：2026-06-17
> 来源：2026-06-16/17 EPM dogfood（thread `65e841fc` / `9409c66f`）连续暴露 6 个独立故障后的根因综合。
> 定位：本文是 **infra 层 agent 开发的工程宪法**——所有改 harness（middleware / tool / subagent / catalog / 路径解析）的 spec 和实施都应遵守。与 memory `reference_agent_harness_7_dimensions`（7 维度知识体系）衔接：本文是那 7 维度在本仓库的**可执行红线**。
> 适用对象：写 spec 的 agent、实施 spec 的 agent、review 的 agent。

---

## 0. 为什么需要这份文档

EPM dogfood 反复出现"修了又复发"的问题。用户追问"为什么总是重复"。综合本地 gateway.log + 代码核实，6 个表面不同的故障（除零 / 图不出 / data-analyst 慢 / 报告卡死 / 列对齐丢 / 路径传来传去）收敛到 **4 条结构性病根**。本文把这 4 条病根转成 4 组红线，每条都给"反模式实例（本仓库真实代码）→ 正模式→ 可执行判据"。

核心论点：**这些 bug 不是偶然，是同一批结构缺陷在不同路径上的复发。修实例（打地鼠）不收敛；修结构（守红线）才收敛。**

---

## 红线一：失败必须响亮；数据层失败必须 fail-loud 触发自愈/HITL，禁止静默降级

### 病根
失败信息在每一层边界丢失，最后只剩一个空容器，下游无法区分"本次本就无此物"和"产出此物时崩了"。**更深一层**：数据计算崩溃常被当成"环境/边界问题"容忍降级——但**用正确的数据本就不该崩**。数据层一旦崩，正确的反应不是降级出一份残废结果，而是**响亮报错，让 agent 有机会自己修复，或提醒用户"是不是数据有问题"**。

### 产品定位决定纪律（用户 2026-06-17）
**这是实验数据分析 agent，将来是完整的实验数据 agent。对实验研究员，数据计算的可复现性和准确性是底线，不是"尽量"，是"必须保证"。** 一个会静默给出错误/降级统计的数据 agent 对科研有害——研究员会拿残废结果写论文。所以：
- **数据计算（数值）必须可复现、可验证**。这是代码不变量。
- **判读的自然语言表述每次跑不一样，无所谓**（LLM 措辞本就非确定，不强求）。可复现的是数值，不是叙述。

### 用户立场：降级可以，静默不可以；降级要通知 + 让模型有限自救 + HITL 兜底
1. **"用正确数据就不该出问题"**：数据计算崩溃（如 ZeroDivision）是代码对真实数据合法值（0/空/负）做了非法假设。修复标准是"代码本就该正确处理这个合法值"。
2. **降级允许，但禁止静默**：数据层出问题时——
   - **(a) 必须通知用户**：降级了什么、为什么、影响什么。绝不静默吞成"空结果 + completed"。
   - **(b) 尽量让模型自己 figure out**：给模型一次有限的自救机会（重试 / 换参数 / 重新派 subagent）。
   - **(c) figure out 要有时间/轮次上限**：不能让模型无限重试或手算螺旋（这次 data-analyst 的病）。超限即升级到 HITL。
   - **(d) HITL 兜底（推荐机制）**：用 **middleware 监听降级信号 → 拦住（停在终止前）→ 走 `ask_clarification` 和用户确认细节（是不是数据问题？是否接受降级？参数怎么调？）→ 确认后再重试**。复刻 `SealGateMiddleware`（after_model 拦截 + jump_to）+ `ClarificationMiddleware`（中断问用户）现成范式。详见 P7 spec。

### 反模式（本仓库真实代码，gateway.log 实证）
`run_metric_plan_tool.py` Step7：statistics 脚本 `rc=1`（ZeroDivisionError）→ 只 `logger.warning` + `statistics_payload` 保持 `None` → handoff `statistics={}`，且 **同一秒报 `status=completed n_failed=0`**：

```
11:20:28 statistics 失败 (rc=1): ZeroDivisionError
11:20:28 done: status=completed n_failed=0      ← 崩溃没进 n_failed，伪装成功
```

下游 data-analyst 看到 `statistics={}` → 走描述性手算螺旋，**产出一份"仅描述性、未做推断检验"的残废报告**。对数据分析 agent 而言这是最坏结果：用户拿到看似完成、实则统计崩溃的报告。**降级了（统计→描述性）却没通知、没自救、没 HITL——三宗罪全占。**

同类：catalog charts 路径列对齐缺失 → resolve 静默 skip box/bar/rose → chart-maker 以为"这些图本就不该有"。

### 正模式
1. **数据层降级必须经"降级熔断器"**：不静默继续，由 middleware 拦住 → 通知用户 + 有限自救 + HITL 确认 → 再重试（P7）。
2. **空容器 ≠ 完成**。完成态判据：期望产出集 ∩ 磁盘实际产出 == 期望集，对账后才封 completed。
3. **降级必须可被下游机读区分**："统计崩了" vs "单组本就无统计" 用不同信号/status，下游据此决定熔断还是正常 partial。
4. **数据正确性是代码不变量**。合法值（0/空/负/单元素）由代码正确处理 + 常驻回归断言。

### 可执行判据（review/CI 必查）
- 改任何"跑子进程/子脚本并聚合结果"的工具：写"子步骤非零退出"的测试，断言**亮降级信号 + 顶层状态可区分**（不是只 log，更不是降级成空 + completed）。
- grep 该工具体内所有 `except` / `rc != 0` / `if x is not None` 分支：每个吞错点问"这是降级吗？降级了通知用户没有？下游能区分崩溃 vs 设计内缺席吗？"
- 数据计算纯函数：合法边界值（0/空/负）必须有正确处理 + 回归断言。

---

## 红线二：session 级横切状态用"共享源 + 自取"，禁止"传参 + 各路径单独接线"

### 病根
session 级常量（实验状态 column_aliases/groups/paradigm、物理路径 workspace/DEERFLOW_PATH_*）本该是单一来源，却退化成每条消费路径各自接线。每新增一条路径就多一个"忘了接"的点，且无机制保证"所有路径都接了"。

### 反模式
- **实验状态**：`experiment-context.json` 已是 session 单一来源，但 charts 路径靠 LLM 在 bash 里拼 `--column-aliases-file`，**LLM 一忘传，resolve 看不到对齐 → 有用的图全跳**。而 metrics 路径（`prep_metric_plan_tool.py:213`）做对了——工具内部自己 `read_context()` 拿 column_aliases，不靠 lead 传。同一个状态，两条路径两种拿法，错的那条靠"LLM 记性"。
- **物理路径**：`resolve_sandbox_path` / `_scoped_path_env` 散落在每个工具、每个脚本各自调。memory 记录至少 3 次 dogfood 炸在"某条路径忘 resolve / 忘包 env"（run_metric_plan Step8、validate_catalog、进程内脚本）。

### 正模式
**核心机制（用户 2026-06-17 定）：每个 tool 都有一个 session 级"全局"注入态，存/读文件时直接从它拿，不靠 prompt、不靠 LLM 拼参数、不靠自己想怎么拿。**

DeerFlow 已有这个机制：**`ToolRuntime` + `runtime.state["thread_data"]`**（含 `workspace_path` 等 session 常量，由 ThreadDataMiddleware 在 session 启动时建好）。`prep_metric_plan_tool.py:124-130` 是正范例——工具签名带 `runtime: ToolRuntime`，进函数第一件事 `thread_data = runtime.state["thread_data"]` 拿 `workspace_path`，**缺失即 fail-fast 报"基础设施 bug"**。这正是用户要的"tool 的 session 级全局变量"。

1. **B 类实验状态（column_aliases/groups/paradigm）**：唯一源是 `experiment-context.json`，所有工具走同一 reader `read_context(workspace_path)`，而 `workspace_path` 从 `runtime.state["thread_data"]` 拿。**整条链不经过 LLM**：tool 自己有 runtime → 拿 workspace → read_context → 拿 aliases。禁止把 B 类状态做成 CLI 入参让 LLM 拼。
2. **A 类物理路径（虚拟 /mnt ↔ 宿主真实路径）**：base dir / 路径映射也在 `runtime.state["thread_data"]`。工具读写 workspace 文件统一走"从 thread_data 取 base + replace_virtual_path"，而非各自 `resolve_sandbox_path` / 各自包 `_scoped_path_env`。**不要用全局可变 env 当共享**——`DEERFLOW_PATH_*` 是进程级全局，并发 subagent 共享进程会串台（`_scoped_path_env` 自承"并发线程仍可能瞥见"）。session 隔离靠 `runtime.state`（每个 tool 调用绑定自己的 thread_data），不靠全局变量。
3. **新增工具 = 接 `runtime: ToolRuntime` 拿 session 态**，不是新开 CLI 参数、不是让 prompt 教 LLM 传。纯 bash 调脚本的路径（如 chart-maker 手拼 catalog.resolve）是退化——应改为工具内部用 runtime 拿状态后再调（见 P3）。

### 可执行判据
- 加横切能力（列对齐这类）时，**写一条"所有已知消费路径都接了"的 parity 回归测试**：枚举 metrics/charts/statistics/validate 各路径，喂同一份 column_aliases，断言每条都生效。下一条漏接的路径 CI 就红，不再靠 dogfood 撞。
- 任何新工具读取 session 常量：**必须带 `runtime: ToolRuntime` 从 `runtime.state["thread_data"]` 拿**；禁止新增 CLI 入参传 session 常量、禁止让 prompt 教 LLM 拼。review 时 grep `add_argument` + 工具签名，看有没有又开了"让 LLM 传状态"的口子。
- 工具拿 session 态缺失时 **fail-fast 报"基础设施 bug"**（如 prep_metric_plan_tool.py:125-128），不静默用默认值兜底（默认值兜底会退化成 /mnt 当真实路径那类 bug）。

---

## 红线三：测试 fixture 必须含真实数据的边界值（0 / 空串 / 非标准列名），禁止理想化合成数据

### 病根
真实行为学数据里"某指标为 0"（动物不进开放臂）、"subject 名为空串"、"列名非标准（open/closed 而非 in_zone_*）"是**常态**；而合成 fixture 总是干净、有序、非零。于是 dogfood（真实数据）永远比单测（合成数据）先发现 bug，单测一次次假设边界不存在。

### 反模式（本仓库真实）
- ZeroDivision：`compute_outlier_diagnostics` 的 `grp_median/value`，测试用 `[1,...,100]`（无 0 值）→ 绿；真实 Trial 19 `open_arm_time_ratio=0.0` → `grp_median/0` → 崩。**测了"极大值离群"，漏了"零值离群"，而零值在 EPM 里极常见。**
- memory 已记同类：合成 fixture 藏空串 subject、EV19 列名 head-1 取不到。

### 正模式
1. **每个处理数值的纯函数，测试必须覆盖 `0`、负、`NaN`、单元素、全相同**这几类边界——这些是行为学真实数据的常客。
2. **测试必须用真实数据：`/home/wangqiuyang/DemoData`**（用户 2026-06-17 定）。EPM 真实数据在 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`（28 个 xlsx，含触发 ZeroDivision 的 Trial 19 = open_arm_time_ratio 0.0）；OFT 在 `Raw data-OFT-Xuhui-34`；FST/TST/o迷宫等见 `DemoData/real_data/`。**不许造理想化合成 fixture 代替**——本次 ZeroDivision 假绿正因用了 `[1,...,100]` 而非真实含 0 数据。纯函数单测可用真实数据的数值子集，端到端验收直接喂 DemoData 真实文件。
3. **"这个值会不会是 0/空"是写除法/索引/取首元素前的必答题**。

### 可执行判据
- review 任何 `a / b`、`arr[0]`、`max(.../x)`：问"x 能否为 0 / arr 能否为空 / 首元素能否是脏值"。答不出就补边界测试。
- 测试数据若全是正整数/无缺失/合成 → 视为 fixture 过于理想，**要求改用 `/home/wangqiuyang/DemoData` 真实数据或其数值子集**。
- 新范式/新指标的验收：必须跑过对应的 DemoData 真实数据集，不是只跑合成单测。

---

## 红线四：通用安全机制必须区分"工具语义"，禁止对正常长流程一刀切误伤

### 病根
通用防护（loop-detection、超时、并发限制）用"调用次数/频率"一刀切，不区分工具语义，对正常的长 E2E 流程误判为异常，且误伤无辜。

### 反模式（本仓库真实代码）
`loop_detection_middleware.py` 的 per-tool 频率硬上限（默认 5）：`write_todos` 是**记账工具**，长 E2E（code→data→chart→report，每阶段更新 todo）调 5 次完全正常，却被判死循环 → FORCED STOP → **剥光同一条消息里所有 tool_call，连无辜的 `task(report-writer)` 一起剥** → 报告永远派不出去。两个缺陷叠加：(a) 不该把记账工具计入与 bash 同一个致命计数；(b) 剥离不该殃及同消息里语义无关的 tool_call。

### 正模式
1. **安全阈值按工具语义分级**。`loop_detection` 已支持 `tool_freq_overrides`（per-tool 阈值）——记账类（write_todos）/ 编排类（task）应有显著更高或豁免的阈值，与"可能死循环"的 bash/read_file 区分。
2. **熔断的副作用要最小化**。strip tool_calls 时应只 strip"重复那一个工具"的 call，保留同消息里语义无关的 call（尤其推进流程的 task/seal）。一刀切剥光 = 把"防死循环"变成"杀进度"。
3. **熔断信息要可操作**。FORCED STOP 的 message 要告诉模型"是哪个工具超限、其余 call 是否保留、下一步该做什么"，而非笼统"全剥了，自己看着办"。

### 可执行判据
- 改/加任何"按次数/频率熔断"的 middleware：必须有"正常长流程不被误伤"的测试（喂一条合法的 code→data→chart→report 序列，断言不触发熔断）。
- review 熔断逻辑：问"哪些工具的高频是正常的？"——记账、编排、分页读取通常正常，应豁免或高阈值。

---

## 通用纪律（贯穿四条红线）

1. **修结构不修实例**：发现某路径漏接/某边界漏处理时，先问"还有几条同类路径/同类边界？"，写覆盖全集的回归，而非只补这一处。
2. **真实数据先行**：新功能验收以真实 dogfood 数据为准，合成数据只用于补边界。
3. **响亮优于沉默**：宁可 status=failed 让用户看见，不要静默空结果伪装 completed——后者会反复复发还查不到。
4. **共享源优于传参**：session 常量谁要谁去唯一源取，不靠 LLM 记得传、不靠每路径各自接线。
5. **改 harness 核心后裸导入两生产入口**（CLAUDE.md 导入环铁律），别被 conftest mock 的假绿骗过。
6. **拆 red/green commit**：bug 修复先写"修复前必红"的测试（用真实边界数据），证明测试真咬住 bug，再实施。

---

## 与本轮 spec 的对应关系

| 病根 / 红线 | 本轮 spec | 状态 |
|---|---|---|
| 红线三（边界值，真实数据）| `2026-06-17-outlier-diagnostics-zerodivision-spec.md`（P1）| 已写 |
| 红线一（降级不静默，信号源）| `2026-06-17-statistics-loud-failure-spec.md`（P2，P7 的信号源）| 已写 |
| 红线一核心（降级熔断器 + HITL）| `2026-06-17-data-degradation-circuit-breaker-spec.md`（P7）| 已写 |
| 红线二（tool 用 runtime 拿 session 态）| `2026-06-17-charts-column-alignment-self-read-spec.md`（P3）+ `2026-06-17-shared-state-sourcing-spec.md`（P4，元收口）| 已写 |
| 红线一变体（限错维度）| `2026-06-17-chart-budget-by-type-spec.md`（P5）| 已写 |
| 红线四（机制误伤）| `2026-06-17-loop-detection-tool-semantics-spec.md`（P6）| 已写 |

**实施顺序建议**：P1（修 ZeroDivision，解当前 dogfood 阻塞）→ P6（loop-detection，解报告卡死）→ P3（charts 自读）→ P5（图预算，依赖 P3）→ P2+P7（信号源+熔断器，一起）→ P4（元收口 parity 网，最后织网）。P1/P6 是当前 dogfood 的两个硬阻塞，优先。
