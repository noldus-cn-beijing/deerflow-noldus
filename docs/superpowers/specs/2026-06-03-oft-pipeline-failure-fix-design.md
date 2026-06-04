# 2026-06-03 OFT dogfood 全链路失败修复设计（实施 spec · 三 PR 可分发版）

> **本 spec 用途**：把 2026-06-03 旷场（OFT）dogfood 全链路失败的修复，拆成 **3 个相互独立、可分别交给独立 agent 执行** 的 PR。每个 PR 段落自包含（背景 + 改动点 + 验收 + 回归红线），可整段交给一个 agent。本 spec 经一次完整 grill（15+ 轮）逐项现场核实定稿，**已推翻问题文档 / 前任 handoff 的多处假设**（见 §0）。
>
> **配套问题文档（原始证据）**：[../../problems/2026-06-03-oft-dogfood-pipeline-failure-problem-statement.md](../../problems/2026-06-03-oft-dogfood-pipeline-failure-problem-statement.md)
> **失败 thread**：`packages/agent/backend/.deer-flow/users/cd95effa-.../threads/752980d6-.../`
> **当前基线**：`HEAD == origin/dev == dc933029`

---

## 0. 现场核实结论（grill 推翻的假设 —— 所有执行 agent 必读）

本 spec 经 grill 逐项核实，**纠正了问题文档/前任 handoff 的 4 处错误判定**：

1. **`in_zone=1` = 中心区（非前任所说 in_zone=0）**。几何实测：`in_zone=1` 时 distance_to_wall=14.3cm、距 arena 中心半径=10、占时 10.07%（经典焦虑样中心回避）；`in_zone=0` 贴墙。**别采信前任 handoff 的语义判定。**
2. **"方向 A：catalog 改 `in_zone*` + 脚本默认裸 in_zone=center" 被行为学专家 SSOT 否决**。专家原话（`docs/review-packages/2026-05-12-feedback.md`）：「不知道哪一列代表什么分析区，**要问，不要猜。虽然这次猜中了，但是不要猜。**」`metrics/oft.py:_find_center_zone_column` 刻意拒裸 in_zone 当 center 是**对的，不许放宽**。
3. **专家追加裁定（本会话）**：EV19 导出**极少**出现单匿名分析区；真遇到 → **反问用户确认这个区是什么**，用户明确则算、**用户不知道则停止让用户弄明白再来**，绝不猜。
4. **🔴 越权防御被严重低估**：问题文档说"guardrail 守住底线、venv 干净"——**核实证伪**。实跑 `ScriptInvocationOnlyProvider`：`cp 进 .venv`/`mkdir 进 site-packages`/`mkdir 伪造模块` 全部 **ALLOW**（file-ops 白名单只看首词不校验路径）；且 local sandbox `execute_command` 用 `subprocess.run(shell=True)` 直接落宿主机、无路径围栏、PATH 注入真实 venv。**venv 这次干净是侥幸（命令自身失败/被打断），不是防御。这是真实沙箱逃逸面。** 另：**chart-maker 也有 bash 且完全不受白名单约束**（白名单只 gate code-executor）。

### 同一地基缺陷波及 3 个范式（不止 OFT）
真实 demo data 实测：**OFT / LDB（明暗箱）/ zero_maze（O迷宫）的导出都是单匿名 `in_zone`**（catalog 分别要 `in_zone_center_*`/`in_zone_light*`/`in_zone_open*`，匹配 0 列）。EPM 命名良好（`in_zone_open_arms_center`），不受影响。**PR-1 的"匿名区检测 + 缺列处理" 必须按范式无关设计**，否则 LDB/O迷宫 下次 dogfood 原样复现。

---

## 1. 三 PR 总览 + 并行执行铁律

| PR | 根因 | 文件域 | 优先级 |
|---|---|---|---|
| **PR-3**（建议先合） | 缺陷 3 越权（真实逃逸面） | `guardrails/script_invocation_only_provider.py` + `subagents/builtins/code_executor.py` | 🔴 安全 |
| **PR-1** | 缺陷 1 地基（zone 语义 + 缺列） | `ethoinsight/catalog/{oft.yaml,resolve.py}` + `metrics/oft.py` + `tools/builtins/prep_metric_plan_tool.py` | 🔴 治本 |
| **PR-2** | 缺陷 2/4 编排硬约束 | `guardrails/path_sequence_provider.py` + `agents/lead_agent/prompt.py` | 🔴 纵深 |

### ⚠️ 分发给独立 agent / 并行 worktree 的铁律（每个 agent 必读）
1. **三 PR 源文件零重叠**（已核实），可并行无合并冲突。
2. **最后合入 dev 的那个 PR，必须在"三改齐备的合并态"跑一次全量回归 + 一次完整 OFT dogfood**——不能三个各自绿就直接合（血泪：memory `feedback_pr_merge_must_run_full_suite_on_shared_logic`，PR #66 只跑新测试致 3 个共享 fixture 静默 fail）。
3. **PR-1 改 `resolve.py`（跨范式共享 helper）**：合并前必跑 ethoinsight 全量 + 对 5 范式 demo data 各跑一次 prep（确认 EPM/FST/TST/LDB/zero_maze 的 default metric 仍可算、无误降级）。
4. **PR-2/PR-3 改受保护文件**（`prompt.py`/`code_executor.py`/guardrails）：surgical 编辑，保留所有 Noldus 定制；**改后必重启 dev**（system_prompt 在 agent 创建时构建）。
5. **deepseek 正面措辞铁律**（CLAUDE.md §6）：prompt 改动一律正面描述想要的行为，**禁用"禁止 X/不要 X"**（反向激活）。硬约束交给 harness gate，不靠 prompt 写禁令。
6. 端到端"OFT dogfood 跑通"需三 PR **齐备**才能验（PR-1 能算 + PR-2 plan-gate + PR-3 不越权）。各 PR 自身只验单元 + 自己那截。

---

## 2. PR-1：地基修复（OFT 匿名区状态机 + 缺列硬失败 + center_zone 注入）

### 2.1 背景（执行 agent 必读）
真实 EV19 OFT/LDB/zero_maze 导出只有单个**匿名** `in_zone`(0/1) 列（源列 `分析区中`，`utils.py:53-55` 归一化）。catalog OFT 5 个 default metric 要 `in_zone_center_*`（匹配 0 列）→ `resolve.py:193` 第一个 default metric 缺列即 `raise ResolveError(columns_missing)` → plan 从未生成 → 全链路塌方。

**正解（专家 SSOT）**：检测到单匿名区 → lead 反问"这个区是不是中心区" → 用户明确"是" → 走 override 算；用户"不是/不知道" → 停止、告知数据问题。**系统永不猜 zone 语义。**

### 2.2 OFT 匿名区状态机（权威定义）
```
prep_metric_plan 检测到裸 in_zone 列存在、但命名 zone 列缺失（in_zone_center_* 匹配 0 列）
        ↓ 返回 error_code="zone_unnamed" + hint
lead 读 hint → ask_clarification（开放式，问角色不问值）：
        "数据里有一个未命名的分析区(in_zone)。它代表旷场的中心区吗？"
        ↓
   ┌────────┼────────────────┐
"是中心区"   "不是中心区"        "不知道"
   ↓          └──────合并──────┐    ↓
lead 写 parameter_overrides     停止分析，告知用户：
  = {"center_zone": "in_zone"}  "数据缺少明确的中心区分析区，OFT 核心指标无法计算。
   ↓                             请在 EthoVision 中确认/命名区域后重新导出，再来分析。"
重调 prep_metric_plan            （不猜、不用无关区代算）
   ↓
resolve 检测到 override 含 center_zone → center metric 列检查放行 → 生成 plan
   （plan 里 center metric 的 args 带 --parameters-json {"center_zone":"in_zone"}）
   ↓ code-executor 照 plan 跑 → compute 收到 center_zone → 算出真值（center_time_ratio≈0.10）
```

### 2.3 改动点（4 文件 + 测试）

#### (a) `ethoinsight/metrics/oft.py` —— 补齐 center_zone 参数透传 🔴 必做（否则 plan 注入会 TypeError）
**已核实的坑**：5 个 center metric 里 `compute_center_time` / `compute_center_distance` **签名无 `center_zone`**，但脚本同样 `compute_xxx(df, **parameters)` 调用。一旦 plan 给全部 5 个注入 `center_zone`，这两个会 `TypeError: unexpected keyword argument 'center_zone'`。
- 给 `compute_center_time(df)` → `compute_center_time(df, center_zone="in_zone_center")`，内部 `compute_center_time_ratio(df)` 改为 `compute_center_time_ratio(df, center_zone=center_zone)` 透传。
- 给 `compute_center_distance(df)` → `compute_center_distance(df, center_zone="in_zone_center")`，内部 `_find_center_zone_column(df)` 改为 `_find_center_zone_column(df, hint=center_zone)` 透传。
- `_find_center_zone_column` **行为保持不变**（默认 hint `in_zone_center` 仍拒裸 in_zone）；只有当显式传 `center_zone="in_zone"`（用户声明后）时，其 `if hint in df.columns: return hint` 分支才返回裸 in_zone 列。这逐字符满足专家"不猜"（默认拒，用户声明才用）。
- ⚠️ 注意 EthoVision 约定 `分析区中`(in_zone) 恒 `1=在区内`，compute 全系列已写死 1=在区内（`series.mean()` / `== 1`），**值语义不歧义，无需声明哪个值**——只声明"哪一列是中心区"。

#### (b) `ethoinsight/catalog/oft.yaml` —— 声明 center_zone 为 catalog 参数
- 给 5 个 center default_metric 加一个 `center_zone` 参数声明（走 `paradigm_parameters` 或 `metric.parameters`，schema 支持 `ParamSpec`，见 `schema.py:37,72`），default 值 `in_zone_center`。
- **为什么必须声明**（已核实）：`_compute_parameters_in_use`(`resolve.py:801-803`) 是 **replace-only**（`if pname in result`）——override 只能替换已声明的参数。不声明 `center_zone`，用户的 `parameter_overrides={"center_zone":"in_zone"}` 会被**静默丢弃**，注入失败。
- `requires_columns` **保持 `in_zone_center_*` 不动**（grill 定论：不改契约。契约正确地拒绝不规范数据；触发 zone_unnamed 反问的逻辑在 prep/resolve 层，不靠放宽契约）。
- 接受 `center_zone` 进 `parameters_in_use` → 会出现在 data-analyst 假设面板。**这是想要的**：中心区语义是用户声明的，正该被暴露质疑（呼应 S7 假设面板 + memory `feedback_parameters_used_must_reflect_actual_resolution_path`——它是真参与计算的参数，本就该报）。

#### (c) `ethoinsight/catalog/resolve.py` —— 区分两类失败 + default 缺列硬失败文案
现状 `resolve.py:192-206`：default metric 缺列即 `raise columns_missing`。改为**在 raise 前先判类型**：
- **类型 1 = 匿名区**：missing pattern 是 `in_zone` 带后缀形态（如 `in_zone_center_*`）、且 `available_columns` 里**存在裸 `in_zone`** → 抛新 error code **`zone_unnamed`**（details 带 found_column=`in_zone`、paradigm）。判定纯结构化（"裸 in_zone 存在 + 命名 zone pattern 落空"），范式无关，OFT/LDB/zero_maze 通用（已核实 EPM 不误伤）。
  - **例外**：若 override 已含 `center_zone`（用户已声明）→ 不再抛 zone_unnamed，正常按声明的列放行生成 plan。
- **类型 2 = 真缺列**：missing pattern 对应的列**根本不存在**（裸 in_zone 也没有，或非 zone 列如 distance_moved 缺失）→ 保持 `raise columns_missing`，但 message 改成用户能懂的"数据有问题"基调（grill 定论：default 真缺列 = 实验没做对，该失败告诉用户，**不降级**）。
- **optional metric 缺列**：保持现状（`resolve.py:236-245` 已 skip + 标 skipped，不动）。
- **不引入 default metric 的部分 plan 降级**（grill 定论：降级只适用 optional，已实现）。

#### (d) `tools/builtins/prep_metric_plan_tool.py` —— zone_unnamed error_code + hint 翻译
- `_ERROR_HINTS`（行 24-58）新增 `zone_unnamed` 条目（用户话术，给 lead 转达）：
  > "检测到一个未命名的分析区(in_zone)。请用 ask_clarification 反问用户该区域代表什么（如旷场的中心区）。用户明确是中心区 → 写 parameter_overrides={\"center_zone\":\"in_zone\"} 后重调 prep_metric_plan；用户不确定/非中心区 → 告知其数据缺明确的中心区，请在 EthoVision 确认/命名区域后重新导出再分析，不要勉强计算。"
- `columns_missing` hint（行 38-41）改强硬：
  > "数据缺少 [指标] 必需的列。这通常意味着实验录制或导出设置不完整。请向用户说明数据缺列、建议检查实验设计与导出配置后重新提供数据，不要在缺列情况下勉强分析。"
- 职责分层（grill 定论 A）：**resolve 报技术事实（哪个 metric/缺哪些 pattern/有哪些列）；prep hint 层翻译成用户话术；lead 转达**。ethoinsight 库不承担"对终端用户说话"。

### 2.4 PR-1 验收
- 真实 `旷场_小鼠_三点`：`prep_metric_plan(oft)` 返回 `zone_unnamed`（非 columns_missing）→ 模拟 override `{"center_zone":"in_zone"}` 重调 → plan 生成（含 5 center metric，args 带 `--parameters-json`）→ 5 个 center compute 全部跑出真值（`center_time_ratio≈0.10`，**无 None、无 TypeError**）。
- LDB/zero_maze demo data：同样触发 `zone_unnamed`（验范式无关）。
- EPM/FST/TST demo data：default metric 全可算，无 zone_unnamed 误判、无降级误吞。
- 真缺列场景（人工删 distance_moved）：抛 `columns_missing` + 友好文案。
- **全量回归（共享逻辑铁律）**：`cd packages/ethoinsight && .venv/bin/python -m pytest tests/ -q`（基线 553 passed）。
- 新增测试：`test_oft_zone_unnamed_detection.py`（裸 in_zone → zone_unnamed；有 override → 放行）、`test_oft_center_zone_override.py`（5 个 center metric 收 center_zone=in_zone 出真值、含 center_time/center_distance 不 TypeError）、`test_resolve_failure_classification.py`（匿名区 vs 真缺列分流）。

---

## 3. PR-2：编排硬约束（plan-gate 覆盖 3 subagent + lead prep 失败引导）

### 3.1 背景 + 核实（执行 agent 必读）
失败 thread：prep 失败后 lead **绕过 plan 硬派 code-executor**，第二轮还**授权 code-executor 手写脚本绕过预装脚本**。
**核实 PathSequenceProvider 为何没拦**：`PATHS`（`path_registry.py`）把路径建模为 dispatch/ask step 序列，`prep_metric_plan` 和 `plan_metrics.json` 存在性**根本不是 Step**；且 code-executor 是所有 E2E 路径的**首个** dispatch step（`target_idx=0`），provider 只校验**前序** dispatch（空循环）→ 永远放行。**plan 就绪从来不是被建模的 gate。**
**核实下游也读 plan**：`chart_maker.py:35` 读 plan 拿 raw_files；`report_writer.py:166-179` 从 plan 取展示字段。CHART/REPORT 是独立 intent（不经 code-executor），其触发前提是"已有 handoff"（`prompt.py:299`），但这是 **prompt 约定非 harness 保证**——lead 判错就裸奔。

### 3.2 改动点（2 文件 + 测试）

#### (a) `guardrails/path_sequence_provider.py` —— 增 plan 就绪前置 gate（覆盖 3 个读 plan 的 subagent）
- 触发条件按 **subagent_type** 判（不按 intent，天然只作用于需 plan 的派遣，零误伤 QA/CLARIFY）：
  - 派 **code-executor** 时：校验 `workspace/plan_metrics.json` 存在且非空（至少 1 个可算 metric）。缺失 → deny。
  - 派 **chart-maker / report-writer** 时：校验 `workspace/handoff_code_executor.json` 存在且非空（它们真正依赖的施工产物；handoff 在则 plan 必在）。缺失 → deny。
- deny 文案含明确指令（铁律 `feedback_deny_messages_must_direct`）：
  > "code-executor 需要 plan_metrics.json 作为施工单。请先调 prep_metric_plan(paradigm=...) 生成。若 prep 返回 zone_unnamed/columns_missing，先 ask_clarification 与用户澄清再 prep，不要在无 plan 时派遣 code-executor。"
- 实现：在现有 provider 增独立的"plan 前置"检查段（不动 `PATHS` 数据结构，最小改动）。复用 `_lead_workspace` contextvar（已有 PathSequenceBridge 提供）。
- ⚠️ fail-open 保持现状语义（workspace 不可用时 allow），但 plan 缺失是明确信号、必 deny（非 fail-open）。

#### (b) `agents/lead_agent/prompt.py`（受保护，surgical）—— prep 失败正面引导
**核实空隙**：现有 `prompt.py:255` "任何 subagent 失败 → 必须 ask_clarification" 只覆盖 **subagent 失败**，`prep_metric_plan` 是 **lead 自己的 tool**，**字面没覆盖**——这是空隙。
正面措辞补**两条**（grill 定论：prompt 只管"该做什么"，"不许绕 plan/不许手写脚本"交给 harness gate 物理兜底，prompt 不写禁令）：
1. "`prep_metric_plan` 返回 status=error 时，下一步是读它的 hint 字段、用 ask_clarification 把问题转达用户：error_code=`zone_unnamed` 按未命名区流程反问该区角色；error_code=`columns_missing` 告知用户数据缺列需检查实验/导出。plan 未成功生成时，分析流程在此暂停等待用户。"
2. "ethoinsight 范式脚本是唯一计算途径。某范式脚本暂缺时，向用户报明'该范式 v0.1 未实现'并停止——由 ethoinsight 库补脚本解决。"

### 3.3 PR-2 验收
- 构造"plan 缺失"：lead 试派 code-executor → deny + 明确指令；试派 chart-maker（无 handoff）→ deny。
- 失败 thread 回放：lead 不再绕 plan 硬派、不再授权手写脚本（prep 失败 → ask_clarification）。
- 不误伤：QA_FACT/CLARIFY/CHART(有 handoff)/REPORT(有 handoff) 路径正常放行。
- **全量后端回归**（基线见 §5）+ 重启 dev 验 prompt 生效。
- 新增 `test_path_sequence_plan_precondition.py`（code-executor 缺 plan deny / chart-maker 缺 handoff deny / 有 plan 放行 / QA 路径不受影响）。

---

## 4. PR-3：越权防御（真实沙箱逃逸面，建议先合）

### 4.1 背景 + 核实（执行 agent 必读 —— 这是真实安全洞，非文案优化）
**已实跑核实**（对着 `ScriptInvocationOnlyProvider.evaluate` 逐条）：
| 命令 | 现判定 |
|---|---|
| `python -c ...` | DENY ✅ |
| `PYTHONPATH=... python -m ...` | DENY ✅ |
| `cp /tmp/x.py .../.venv/.../site-packages/...` | **ALLOW** 🔴 |
| `mkdir -p .../.venv/.../site-packages/...` | **ALLOW** 🔴 |
| `mkdir -p .../workspace/ethoinsight/scripts/open_field`（伪造模块） | **ALLOW** 🔴 |

根因：`_ALLOWED_FILE_OPS` 正则 `^\s*(mkdir|cp|mv|ls|cat|grep|head|tail)(\s|$)` **只匹配首词、不校验路径**。且 local sandbox `execute_command`（`local_sandbox.py:362`）`subprocess.run(shell=True)` 直接落宿主机、无 cwd 围栏、PATH 注入真实 venv → **cp/mkdir 进 venv 会真执行**。venv 这次干净是侥幸。
另核实：**chart-maker 也有 bash**（`tools=["bash",...]`）且 `ScriptInvocationOnly` Gate 2 **只 gate code-executor**（`if "code-executor" not in agent_id: return allow`）→ chart-maker 的 bash 能跑**任意命令**（含 cp 进 venv / python -c）。data-analyst/report-writer/knowledge-assistant 已 disallow bash（✅）。lead 无 bash（✅）。

### 4.2 改动点（2 文件 + 测试）

#### (a) `guardrails/script_invocation_only_provider.py` —— 扩 chart-maker + file-ops 路径校验
1. **bash 白名单 gate 范围扩到 chart-maker**：Gate 2 现 `if "code-executor" not in agent_id: allow`，改为 gate **code-executor + chart-maker**（两者都只该跑 `python -m ethoinsight.scripts.*` + 受限 file-ops）。
   - chart-maker 合法 bash 是 `python -m ethoinsight.scripts.<p>.plot_*`——已落在 `_ALLOWED_PYTHON_PATTERN`（`ethoinsight.scripts.\w+.\w+`）内，扩 gate 不误伤其正常绘图。
2. **file-ops 增路径校验**（堵 cp/mkdir/mv 进 venv）：`cp/mkdir/mv` 的目标路径必须落在 `/mnt/user-data/`（thread 沙箱内）；指向 `.venv`/`site-packages`/绝对宿主机路径/`/mnt/skills`(只读) 的 → DENY。
   - deny 文案：「脚本由 ethoinsight 库维护，不可写入已安装包路径或沙箱外路径。缺脚本请在 handoff 标 status=failed 让 lead 处理。」
   - 实现注意：解析命令的目标路径要稳健（处理 `mkdir -p`、`cp src dst`、引号/空格路径）；从严——无法确定目标在沙箱内就 DENY。

#### (b) `subagents/builtins/code_executor.py`（受保护，surgical）—— 正面引导禁猜名
正面措辞补（grill 定论 + deepseek 铁律）：
1. "脚本名只能来自 plan_metrics.json 的 `script` 字段，逐字照抄使用。"
2. "若 plan_metrics.json 不存在，本步骤立即在 handoff 标 status=failed（error 写明 plan 缺失），由 lead 去补 plan——这是正确的完成方式。"
3. "若某脚本 `python -m` 报 ModuleNotFoundError，记 critical warning 并 seal failed。脚本由 ethoinsight 库维护，不在沙箱内补建。"
- 现有 `code_executor.py:75` "不存在的脚本立即停手、不要重试不同名字" 已有类似意图，本 PR 强化为"无 plan→seal failed 是唯一 recovery"。

### 4.3 PR-3 验收
- 实跑 guardrail：`cp 进 .venv`/`mkdir 进 site-packages`/`mkdir 伪造模块` → 全部 DENY（修复前 ALLOW）；正常 `python -m ethoinsight.scripts.oft.compute_*` / `mkdir /mnt/user-data/workspace/x` → ALLOW。
- chart-maker 跑 `python -c` / `cp 进 venv` → DENY；跑 `python -m ethoinsight.scripts.oft.plot_*` → ALLOW。
- code-executor 无 plan 场景：seal failed（不猜名、不伪造模块、不碰 venv）。
- **全量后端回归**（基线见 §5）+ 重启 dev。
- 新增 `test_script_invocation_path_hardening.py`（file-ops 路径校验 + chart-maker gate + 正常命令放行）。

### 4.4 单列（不在本 PR）
- **sandbox 层路径围栏**（`local_sandbox.py:execute_command` 加 cwd/路径白名单，对所有 bash 生效）——纵深防御，防"未来给新 subagent 加 bash 忘上白名单"。属受保护核心 + 上游 deerflow 演进区，**单开 issue**，不在 bug-fix PR 顺手改核心（本仓库 sync 哲学）。

---

## 5. 公共：回归基线 + 命令速查

- **ethoinsight 全量**（PR-1 必跑）：`cd packages/ethoinsight && .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`（基线 **553 passed**）
- **后端全量**（PR-2/PR-3 必跑）：`cd packages/agent/backend && DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/ -q -p no:cacheprovider -o addopts=""`（基线 **3612 passed + 2 pre-existing fail**：`test_inspect_gate_guardrail.py::...::test_async_delegates_to_sync` / `test_paradigm_identification_gate.py::...::test_async_delegates_to_sync`，单跑各自 passed，**非回归**）
- **5 范式 demo data**（PR-1 验范式无关）：`/home/wangqiuyang/DemoData/newdemodata/{旷场_小鼠_三点,高架十字迷宫_小鼠_三点,明暗箱,O迷宫,强迫游泳_大鼠}`
- **重启 dev**：`cd packages/agent && make stop && make dev`
- **dogfood handoff 位置**：`packages/agent/backend/.deer-flow/users/<uid>/threads/<tid>/user-data/workspace/handoff_*.json`
- **合并态终验**（最后合的 PR 必做）：三改齐备后跑两套全量 + 一次完整 OFT dogfood（上传 `旷场_小鼠_三点` → lead 反问中心区 → 答"是" → 走完 code/data/chart/report，center_time_ratio≈0.10 落盘）。

## 6. 善后（失败 thread）
失败 thread `752980d6` workspace 现存 `handoff_code_executor.json`(failed)/`experiment-context.json`/`groups.json`/`.lineage/`，未见问题文档提的伪 `open_field` 包（已清或在别处沙箱位置）。backend venv 已核实干净。留作回归回放素材，无需清理。

## 7. 关键路径速查
- catalog：`packages/ethoinsight/ethoinsight/catalog/{oft.yaml,resolve.py,schema.py}`（缺列逻辑 resolve.py:192-206、`_compute_parameters_in_use` 750-805 replace-only、`_missing_columns` 449）
- 列归一化（zone 语义源头）：`packages/ethoinsight/ethoinsight/utils.py:49-82`（`分析区中`→`in_zone`）
- OFT metrics：`packages/ethoinsight/ethoinsight/metrics/oft.py:17-209`（`_find_center_zone_column` 拒裸 in_zone；center_time/center_distance 缺 center_zone 参数）
- compute 脚本：`packages/ethoinsight/ethoinsight/scripts/oft/compute_center_*.py`（均 `parse_parameters` + `**parameters`）
- override rail：`tools/builtins/prep_metric_plan_tool.py:194-198,253`（context→resolve overrides）
- 编排 guardrail：`guardrails/path_sequence_provider.py` + `path_registry.py`（PATHS）
- 脚本白名单 guardrail：`guardrails/script_invocation_only_provider.py`
- sandbox bash：`sandbox/local/local_sandbox.py:323-379`（execute_command，shell=True 无围栏）
- 受保护 prompt：`agents/lead_agent/prompt.py`、`subagents/builtins/code_executor.py`
- SSOT（zone 语义铁律）：`docs/review-packages/2026-05-12-feedback.md` §OFT + Q2
- 真实 demo data：`/home/wangqiuyang/DemoData/newdemodata/`
