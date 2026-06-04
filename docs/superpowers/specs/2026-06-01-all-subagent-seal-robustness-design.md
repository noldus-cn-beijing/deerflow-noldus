# 2026-06-01 设计 spec — 系统性根治全部 subagent 的 seal handoff 故障

**类型**：可实施版（已对 dev HEAD `f678e8c1` + langgraph.log 双重核验；实施前 `git pull` 复核行号）
**对应**：2026-06-01 E2E dogfood 中 code-executor seal 报错重试 + data-analyst seal 两次卡死（langgraph.log thread `9af3ba6d` 实证）
**估期**：~2-3 天（分两阶段：止血 prompt + B2 数据通路）
**前置**：无硬前置

> **目标（用户 2026-06-01 锁定）**：**所有 subagent 的 seal 功能都没问题**。不是打补丁，是系统性根治。路径 B（data-analyst 补数据通路，真正实现参数审计，不是诚实跳过）。

---

## 0. 根因总结（git + log 双证据钉死，实施前必读）

**seal 机制本身没错。是从 Sprint 0 起 seal 被加上越来越严的 schema 校验 + 越来越重的前置步骤（Sprint 3 step 2.8），而 prompt 从未同步教 subagent 满足这些要求。** 后来加的 seal-resume 兜底（5.7/5.8）是创可贴，没拆病根。

### 两个故障，两个机制（langgraph.log thread 9af3ba6d 实证）

| | code-executor | data-analyst |
|---|---|---|
| 现象 | seal **被调用**，schema 校验失败（5→2→过，10:05/10:06 两次 ValidationError） | seal **根本没调用**（forgot to call），seal-resume 补轮 2 次都没救回（10:13/10:18） |
| 直接触发 | `data_quality_warnings`: `metric=None` / `code='insufficient_sample'`（下划线）/ `evidence='n_per_group=1,...'`（字符串） | step 2.8 参数审计陷入"拿 per_subject 输出指标比对需原始分布的参数"死循环，烧光 max_turns=12 |
| 引入 | Sprint 0 `aed27729`（seal schema 化，加 DOT/dict/str 强校验）+ Sprint 1 `00741752` | Sprint 3 `f5aa5b05`（step 2.8） |

### 全局盘点（4 个 subagent prompt 全部 0 处教 schema 字段格式）

| seal | 会致失败的严约束 | prompt 教了吗 |
|---|---|---|
| code_executor | data_quality_warnings(code DOT / metric str / evidence dict) + raw_files 虚拟路径 | ❌ |
| data_analyst | DataQualityWarning 透传 + ParameterAuditFinding(mismatch_kind/observed_distribution dict) + **step 2.8 卡死** | ❌ |
| chart_maker | chart_files 必须 `/mnt/user-data/outputs/` 前缀 | ❌ |
| report_writer | report_path 等 | ❌ |

---

## 1. 总体方案（三层，全 subagent 覆盖）

### 层 1 — prompt 教格式（治本，4 个 subagent 全覆盖）

每个 subagent 的 prompt 加一段"seal 字段格式速查"，教它**第一次就把 schema 严约束满足**。这是"教会 subagent 如何调用 seal"，对 **code-executor / chart-maker / report-writer 三个"调得到 seal 但格式可能错"的故障是完整治本**。

### 层 2 — seal 工具容错归一化（加固，共享 helper）

`_seal_handoff` 在 Pydantic 校验前对**明显无歧义笔误**保守归一化（下划线→DOT、null→"all"、str evidence→包裹）。这是安全网：即使 prompt 教了 LLM 偶尔仍飘，工具兜住。**只修无歧义笔误，不猜语义**（`insufficient_sample` 这种纯语义错不强改，靠层 1 预防）。

### 层 3 — data-analyst step 2.8 补数据通路（路径 B，真正做对参数审计）

**这是 data-analyst 卡死的真治本。** step 2.8 卡死的根因是"要比对的数据（velocity/periodicity 原始分布）不在 handoff 里"。`_pendulum.py` 计算时这些分布是**现成中间产物**（逐帧 `periodicity`/`activity`，见 `metrics/_pendulum.py:140-144`），只是没写进 handoff。

**做法**：让 code-executor 把参数审计需要的原始分布统计量（p10/p90/median）写进 handoff，data-analyst step 2.8 就有真数据可比 → 参数审计**真正能做**，不再是无米之炊、不再卡死。

---

## 2. 详细设计

### 2.1 层 1 — 4 个 prompt 的 seal 字段格式速查

**通用原则**：每个 prompt 在"封存 handoff"那步前，加一段紧凑的"字段格式速查"，覆盖该 seal 的全部严约束 + 一个完整合法示例。用 deepseek 正面提示（CLAUDE.md §6）。**字段约束的权威源是 handoff_schemas.py 的 validator，prompt 是教学副本**（[[feedback_single_source_of_truth]]：prompt 注释标注"约束权威源见 schema"，schema 改了 prompt 要同步）。

**code_executor.py** —— 加 `data_quality_warnings` + `raw_files` 格式：
```
data_quality_warnings 每条字段（首次写对，避免 seal 校验失败）：
- code: <前缀>.<名称> DOT 格式，前缀仅 SAMPLE/MOTOR/SIGNAL/METHOD。例 SAMPLE.TOO_SMALL。不用下划线。
- metric: 字符串；广泛适用填 "all"，不填 null。
- evidence: dict，如 {"n_per_group":1}；无证据填 {}。不是字符串。
- severity: critical|warning|info ; blocks_downstream: bool
完整示例(n_per_group<2)：{"severity":"critical","code":"SAMPLE.TOO_SMALL","metric":"all",
  "message":"每组仅 n=1，无法组间统计检验","evidence":{"n_per_group":1,"required":2},"blocks_downstream":true}
raw_files: 直接从 plan_metrics.json 抄虚拟路径(/mnt/user-data/...)，不要 Path.resolve()/realpath。
```

**data_analyst.py** —— 加 DataQualityWarning 透传 + ParameterAuditFinding 格式（透传的 warning 上游已合规，重点教 ParameterAuditFinding）：
```
parameter_audit_findings 每条：parameter(str) / metric(str) / severity(critical|warning|info) /
  used_value / observed_distribution(dict，如 {"p10":5,"p90":30,"median":12,"n_subjects":6}) /
  mismatch_kind(仅 threshold_too_high/threshold_too_low/window_too_wide/window_too_narrow/category_mismatch 五选一) /
  suggestion(str) / blocks_downstream(bool)
```

**chart_maker.py** —— 加 chart_files 格式：
```
chart_files 每条必须是 /mnt/user-data/outputs/ 开头的虚拟路径。无图时 chart_files=[] 且 failed_charts 写原因。
```

**report_writer.py** —— 已在 report-chart-404 修复里教过路径；补 seal 字段（report_path 等）速查即可。

### 2.2 层 2 — `_seal_handoff` 归一化（`seal_handoff_tools.py:193` 校验前）

仅对 `data_quality_warnings` 字段（按 key 存在触发，不影响 chart/report handoff）：
1. code 下划线→DOT：仅当首段（按 `_` 切）∈ {SAMPLE,MOTOR,SIGNAL,METHOD}（如 `SAMPLE_TOO_SMALL`→`SAMPLE.TOO_SMALL`）。纯语义错（`insufficient_sample`）不改，留 validator 报错。
2. metric=None/缺失→"all"
3. evidence=str→`{"note":<原值>}`；非 dict 非 str→`{}`
- 前缀集合**从 handoff_schemas import**，不硬编码复制。
- **🔴 共享 helper 改动**：grep `_seal_handoff` 全部 4 个调用方，确认归一化按 key 触发、不误伤（[[feedback_pr_merge_must_run_full_suite_on_shared_logic]]）。

### 2.3 层 3 — data-analyst step 2.8 补数据通路（路径 B 核心）

**B-1. code-executor 产出原始分布** —— 在算 pendulum/velocity 类 metric 时，把逐帧中间量的分布统计量写进 handoff：
- `metrics/_pendulum.py` 已逐帧产 `periodicity`/`activity`（`:140-144`）；compute 脚本聚合 metric 时，**额外计算这些逐帧量的 {p10,p90,median,max,n_frames} 并随 metric 输出**。
- velocity 类同理（compute 脚本内有逐帧 velocity）。
- 写入位置：handoff 的 `per_subject[subject][metric]` 下加一个 `signal_distribution` 字段（dict），或 metrics_summary 的 MetricStat 加字段。**位置需在实施时定，并更新 handoff_schemas（schema 是 SSOT）**。

**B-2. data-analyst step 2.8 改为用真分布比对**：
- 从 handoff 取 `signal_distribution`（B-1 写入的 periodicity/activity/velocity 分布）→ 对 pendulum/velocity 参数做真正的 p90×3 / p10÷3 比对 → 产出有数据支撑的 ParameterAuditFinding。
- **保留降级出口**（修法已部分在 `91ed188e`）：若 signal_distribution 缺失（老 handoff / 该 metric 无此分布）→ 记 info 跳过，不卡死。**这次明确覆盖"分布字段不存在"这个 case**（上一轮 `91ed188e` 漏了它，只覆盖了 n<2 和文档缺失）。
- **轮次硬约束保留**：审计至多 2-3 轮，seal 必达。

> **B 的本质**：把 Sprint 3 step 2.8 从"无米之炊"变成"有米可炊"。不是诚实跳过（那是止血 B1），是真正补上它本该有的输入数据。

> **存储位置定稿（2026-06-02，关闭原 L100 开放项）**：`_signal_distributions` 挂在 **`per_subject[subject]["_signal_distributions"][metric]`**（嵌套命名空间键），不挂 MetricStat、不用 `metric__signal` 兄弟后缀。理由：① per-subject 是分布的天然数据粒度（逐帧序列每只动物一条），step 2.8 审计按 subject 比例判 severity，与 per_subject 粒度对齐，挂 MetricStat（group 聚合粒度）会丢 subject 边界；② step 2.8 读取已锚定 per_subject（"从 per_subject 收集各 subject 值"），同树零分叉；③ 用单一 `_` 前缀命名空间键而非 `metric__signal` 兄弟后缀——遍历 metric 标量的老代码只需跳过 `_` 前缀键（一条规则），不必按后缀逐个过滤，避免把分布块误当标量 metric。per_subject 类型已是 `dict[str, dict[str, Any]]`，`Any` 容得下，schema 只需更新 Field 描述（SSOT）。

### 2.4 领域边界（不变铁律）

- 参数审计**只警告不调参**（Sprint 3 铁律）
- **工程不编 pendulum/velocity 领域阈值数值**（issue #63 归同事）。B 补的是"数据分布"（统计事实），p90×3 是"统计背离判据"（保守默认），**不是领域阈值**。
- 不放宽 schema 契约（下游依赖），只教 + 容错 + 补数据。

---

## 3. 实施阶段（建议分两 PR）

### 阶段 1（止血，纯 prompt + 工具，可立即解锁 dogfood）✅ 已合 dev（PR #?? `e7628c46`）
- 层 1（4 prompt 教格式）+ 层 2（工具归一化）
- data-analyst step 2.8 **先加"分布字段缺失即整类 info 跳过"出口**（覆盖上轮 `91ed188e` 漏的 case）→ 立刻不卡死
- ⚠️ **阶段 1 的遗漏（2026-06-02 dogfood 暴露，见阶段 1.5）**：教了"跳过时记 info finding"，但**没教那个退化 finding 的 `used_value` / `observed_distribution` 字段怎么填才合法** → seal 仍崩。

### 阶段 1.5（🔴 紧急补丁 — 跳过出口的 finding 形状修复）
> **触发**：2026-06-02 FST dogfood（thread `81051535`），data-analyst 三次派遣，前两次 "terminated without emitting"，**第三次真调了 seal 但 schema 校验失败**（`seal_data_analyst_handoff` ValidationError，langgraph.log 03:24:45 / 03:25:12 实证）。

**根因（ValidationError 明细钉死）**：data-analyst 走 step 2.8"分布缺失→info 跳过"出口时，产出的 `ParameterAuditFinding` 形状违反 schema：
```
parameter_audit_findings.0.used_value         Input should be float/int/str, got None
parameter_audit_findings.0.observed_distribution.note  Input should be number, got 'per_subject 仅含标量...无法计算 median 等百分位判据'
```
- `used_value: float|int|str`（schema 非空）← 跳过场景 deepseek 填了 `None`
- `observed_distribution: dict[str,float|int]`（schema 全数字）← deepseek 把**说明文字**塞进了 `{"note": "..."}`（那句文字正是阶段 1 step 2.8 出口的 suggestion 话术变体）

**prompt 内部矛盾（data_analyst.py:118-131）**：L122-123"前置条件"出口说"整类记**一条** finding"但**没说 used_value/observed_distribution 填什么**；而 L128-131 的 a 段 n<2 出口反而教对了（"observed_distribution 只填 n_subjects" = 数字合法）。两个出口粒度不一致 + 前者字段无值可填 → deepseek 拿文字硬塞数字字段。

**修法 A+B（双层，与阶段 1 为 DataQualityWarning 做的归一化同构）**：
- **A（prompt 治本，data_analyst.py:118-124 step 2.8 出口）**：明确教退化 finding 的合法字段：
  - `used_value`：填**该 metric 的某个真实参数值**（从 parameters_used 取任一；若整类跳过就按"每参数一条"而非"整类一条"，与 L128 a 段粒度对齐）
  - `observed_distribution`：填 `{}`（空 dict 合法）**或** `{"n_subjects": N}`（纯数字），**绝不放说明文字**
  - 说明文字一律放 `suggestion`（str 字段，本就是放文字的地方）
  - 统一粒度：跳过也走"遍历每参数各记一条"（L126 a 段），删除 L122 那条无参数的"整类一条"出口
- **B（schema 归一化兜底，handoff_schemas.py `ParameterAuditFinding` 加 model_validator(mode="before")）**：
  - `used_value is None → ""`（或从 metric 推占位）
  - `observed_distribution` 里**非数字值剔除**（`{"note": str}` → `{}`），防 deepseek 偶发再塞文字
  - 与 `DataQualityWarning._normalize_llm_typeros` 同构、同位置、同测试风格
- **不放宽契约**：A 把信息填进对的字段（合法值本就存在）；B 只剔无歧义错填，不猜语义。

**此阶段后 dogfood 才真正跑通**：data-analyst 跳过 pendulum 审计 + seal 一次过。

### 阶段 2（路径 B 真正做对，单独 PR）
- 层 3 B-1（code-executor 产分布 `_signal_distributions`）+ B-2（data-analyst 用真分布审计）
- 存储位置定稿（2026-06-02）：`per_subject[subject]["_signal_distributions"][metric] = {p10,p90,median,max,n_frames,signal_key}`（嵌套命名空间键，不用 `metric__signal` 兄弟后缀；理由见 §2.3 下方补注）
- **B-1 数据流前置缺口（2026-06-02 深挖发现，spec 原 L98 假设过乐观）**：逐帧 periodicity/velocity **并非现成**——`pendulum_immobility_series`（`_pendulum.py`）只漏出 immobile one-hot、`_resolve_immobile_from_velocity`（`_common.py:264`）逐帧 velocity 当场判完即弃。B-1 必须**先**让这两个底层函数吐出逐帧浮点序列（新增 `pendulum_periodicity_series()` 纯函数零 break；`_resolve_immobile_from_velocity` 改返回签名波及 2 fallback 调用方 + 1 单测），**再**在 `dispatcher.py`（per_subject 真正的确定性组装点，非 LLM 拼）算分布。详见 `2026-06-02` 深挖结论。
- handoff_schemas 更新 per_subject Field 描述说明 `_signal_distributions` 约定（per_subject 类型已是 `dict[str,Any]`，容得下）
- 参数审计对 FST/TST 真正产出有数据支撑的 finding（n≥2 时）
- **阶段 2 完成后，阶段 1.5 的"跳过出口"只在真正无分布的边界场景触发**（n=1 等），不再是常态

---

## 4. 验收

### 阶段 1
- [ ] 4 个 subagent prompt 各有 seal 字段格式速查 + 完整示例
- [ ] `_seal_handoff` 归一化 data_quality_warnings 三类笔误；只该字段、按 key 触发、不误伤 chart/report
- [ ] 归一化前缀集合从 schema import
- [ ] data-analyst step 2.8 覆盖"signal_distribution 缺失→整类 info 跳过"
- [ ] **🔴 复现验证（核心，langgraph.log 对照）**：用 FST n=1 数据（thread 9af3ba6d 同款）跑 → code-executor seal **一次过无重试**；data-analyst **不卡死、正常调 seal 产 handoff**
- [ ] 单测：下划线 code 归一化通过 / 纯语义错仍报错 / metric=null→"all" / evidence=str→包裹 / chart/report handoff 不受影响
- [ ] 全量 make test 不退化（改共享 helper + 4 prompt，必跑全量）

### 阶段 1.5（跳过出口 finding 形状修复）
- [ ] step 2.8 出口（data_analyst.py:118-124）教退化 finding 合法字段：`used_value` 填真实参数值 / `observed_distribution` 填 `{}` 或纯数字 / 说明文字放 `suggestion`；统一为"每参数一条"粒度，删 L122 无参数的"整类一条"
- [ ] `ParameterAuditFinding` 加 model_validator(mode="before")：`used_value=None→` 占位 / `observed_distribution` 剔非数字值；与 `DataQualityWarning._normalize_llm_typeros` 同构
- [ ] **🔴 复现验证（核心）**：用本次 dogfood 同款 FST 数据（每组 n=1，Drug vs Saline，thread `81051535`）跑 → data-analyst seal **一次过**（无 ValidationError、无 terminated without emitting）；跳过出口产出的 finding 通过 schema
- [ ] 单测：退化 finding（used_value=None / observed_distribution 含 note 文字）经归一化后通过 ParameterAuditFinding 校验；正常 finding 不受影响
- [ ] 全量 make test 不退化（改 ParameterAuditFinding validator + data_analyst prompt，必跑全量 [[feedback_pr_merge_must_run_full_suite_on_shared_logic]]）

### 阶段 2
- [ ] code-executor handoff per_subject 含 signal_distribution（periodicity/activity/velocity 的 p10/p90/median）
- [ ] handoff_schemas 加该字段（schema SSOT）
- [ ] data-analyst step 2.8 用真分布产出有数据支撑的 ParameterAuditFinding（n≥2 时）
- [ ] 端到端：FST 数据参数审计**真正产出 finding**（不再全跳过），且不卡死
- [ ] 全量 make test + ethoinsight pytest 不退化

---

## 5. 与其他工作的关系
- **覆盖并取代**两个旧方向：原 `2026-06-01-seal-quality-warnings-format-fix-design.md`（只 code-executor）+ 已合入的 `91ed188e`（data-analyst pendulum，降级出口写窄了，本 spec 阶段1补全）
- **不回退** Sprint 3 step 2.8（用户明确）：B 是把它做对
- seal-resume 兜底（5.7/5.8）**保留不动**：它是纵深防御最后一环，本 spec 治的是病根，兜底仍作为意外保险
- 受保护文件（4 个 subagent prompt + seal_handoff_tools）：surgical + 全量测试

## 6. 不在范围
- ❌ 回退 Sprint 3 / 删 step 2.8
- ❌ 放宽 schema 契约（只教+容错+补数据）
- ❌ 工程编领域阈值（issue #63）
- ❌ 删 seal-resume 兜底
- ❌ 改 langgraph/executor 的 max_turns 机制（病根在前置步骤无米之炊，不在 turn 数）

---

## 7. 实施细节（可直接照着写，2026-06-02 补）

> 实施前 `git pull` 复核行号；所有行号基于 dev HEAD `f678e8c1` + 阶段 1（`e7628c46`）已合。

### 7.1 阶段 1.5 — 修法 A（prompt，`subagents/builtins/data_analyst.py:118-124`）

**问题**：当前 L122-123 的"前置条件"出口说"整类记一条 finding"，但①粒度与 L126 a 段（每参数一条）不一致；②没说 `used_value`/`observed_distribution` 填什么 → deepseek 填 `None` + `{"note":文字}` 崩。

**改法**：删掉 L118-124 那条"整类一条"的前置出口，**统一并入 L126 a 段的"每参数一条"粒度**，并在 a 段把"分布缺失"和"n<2"两种降级统一教对字段。改后 a 段（示意）：

```
   a. **遍历每个有 parameters_used 的 metric 的每个参数**：
      从 per_subject 收集该 metric 各 subject 的标量值。
      **降级判定（任一成立即记一条 info finding 并继续下一个参数，不纠结）**：
      - per_subject 缺该 metric 条目，或跨 subject 标量值 < 2（无法算 p10/p90）
      - 当前范式文档无该参数的领域判据
      **降级 finding 的字段必须这样填（否则 seal 校验失败）**：
      - parameter: 当前参数名（真实，从 parameters_used 取）
      - metric: 当前 metric 名（真实）
      - used_value: **该参数的真实值**（从 parameters_used[参数名] 取，绝不填 None）
      - observed_distribution: 填 `{}` 或纯数字如 `{"n_subjects": 1}`，**绝不放说明文字**
      - mismatch_kind: 用最接近的合法值（如 threshold_too_high）；降级场景无真实 mismatch，
        但 schema 要求五选一，填最接近的
      - severity: "info"
      - suggestion: **说明文字放这里**（str 字段），如"per_subject 仅含标量值、样本不足，
        无法计算 p10/p90 百分位判据，参数审计待上游（阶段 2）补逐帧分布后执行"
      - blocks_downstream: false
      若判据可用且 n_subjects ≥ 2 → 正常算分布统计量做比对（原 L132 逻辑不变）。
```

**关键**：合法值本就存在（used_value 在 parameters_used、suggestion 是 str 字段）。这是"填进对的字段"，不是放宽契约。删 L118-124 独立前置出口，避免"整类一条无参数"的退化形态。

### 7.2 阶段 1.5 — 修法 B（schema 归一化，`subagents/handoff_schemas.py` `ParameterAuditFinding`）

照抄 `DataQualityWarning._normalize_llm_typeros` 范式（同位置 mode="before"、同 `ConfigDict(extra="allow")`、同测试风格）。加在 `ParameterAuditFinding` 现有 `@field_validator("parameter")`（`:53`）**之前**（model_validator(mode="before") 本就先于 field_validator 跑）：

```python
@model_validator(mode="before")
@classmethod
def _normalize_audit_finding(cls, values: dict[str, Any] | Any) -> Any:
    """跳过/降级场景 LLM 易填错的字段做保守归一化，校验前。只剔无歧义错填，不猜语义。"""
    if not isinstance(values, dict):
        return values
    # 1. used_value=None → ""（schema 要求 float|int|str 非空；降级场景 prompt 应填真实值，
    #    此处兜底 deepseek 偶发漏填）
    if values.get("used_value") is None:
        values["used_value"] = ""
    # 2. observed_distribution 剔非数字值（schema 要求 dict[str, float|int]）：
    #    {"note": "文字"} / 含 str 值 → 剔除该键；全剔光则 {}
    od = values.get("observed_distribution")
    if isinstance(od, dict):
        values["observed_distribution"] = {
            k: v for k, v in od.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    elif od is not None and not isinstance(od, dict):
        values["observed_distribution"] = {}
    return values
```

⚠️ `ParameterAuditFinding` 当前**没有** `model_config = ConfigDict(extra="allow")` —— 加 model_validator 时一并加上（与 DataQualityWarning 一致），否则 mode="before" 注入的中间态可能被 extra="forbid" 默认行为卡。**实施时确认** `from pydantic import model_validator` 已 import（DataQualityWarning 那次已加）。

**单测**（`tests/test_parameter_audit_schema.py` 扩展）：
- `used_value=None` → 归一化为 `""` 后通过
- `observed_distribution={"note": "文字"}` → 归一化为 `{}` 后通过
- `observed_distribution={"p90": 12.0, "note": "x"}` → 归一化为 `{"p90": 12.0}`
- 正常 finding（used_value=30.0, observed_distribution={"p90":10}）不受影响
- 幂等：已合规 finding 二次归一化不变

### 7.3 阶段 2 — B-1 底层函数吐逐帧序列

**periodicity（`metrics/_pendulum.py`，零 break）**：`detect_pendulum` 返回的 dict 已含 `periodicity`（现成）。新增纯函数，不改 `pendulum_immobility_series`（`-> np.ndarray`）签名：
```python
def pendulum_periodicity_series(activity, dt=0.04, **pendulum_kwargs) -> np.ndarray:
    """逐帧 periodicity 序列 [0,1]，与 activity 等长。供分布统计量计算。"""
    results = detect_pendulum(activity, dt, **pendulum_kwargs)
    return np.array([r["periodicity"] for r in results], dtype=float)
```
调用方：0 个现有调用方受影响（新增函数）。

**velocity（`metrics/_common.py:226 _resolve_immobile_from_velocity`，改返回签名）**：当前 `:272 return series, 1`。velocity 在循环内 `:264` 算完即弃 → 改为循环内收集 `velocity_arr.append(velocity)`，返回 `(series, 1, np.array(velocity_arr))`。
- ⚠️ **波及 2 个 fallback 调用方**（`_common.py:182/188 _resolve_immobile_series` 内）+ 1 单测（`test_metrics_parameter_passing.py`）。改返回签名必 grep 全调用方（[[feedback_pr_merge_must_run_full_suite_on_shared_logic]]）。可考虑用可选返回（默认不返回 velocity，新参数 `return_signal=True` 时返回）降低波及面。

### 7.4 阶段 2 — B-1 分布组装点 + B-2

**组装点 = `metrics/dispatcher.py`（per_subject 真正的确定性组装处，非 LLM 拼）**：`:119-152` 循环 `per_subject[name] = m` 处。在算 immobility 类 metric 时（FST/TST，`:144-151`），额外调 `pendulum_periodicity_series`/velocity 序列 → 算 `{p10,p90,median,max,n_frames}` → 写 `m.setdefault("_signal_distributions", {})[metric] = {...}`。
- ⚠️ **真实流水线是 catalog 路径**（code-executor 逐个 bash 跑 `scripts/fst/compute_*.py`），`dispatcher.py` 是否在该路径上**实施前必须再确认**（2026-06-02 勘探发现 dispatcher 被 plot/run_groupwise_stats 调，compute_* 脚本各自算标量）。若 compute_* 不走 dispatcher，则分布要在 `compute_immobility_time_fst` 等（`metrics/fst.py`）算并随 payload 输出，code-executor 收集时透传进 per_subject。**这是阶段 2 启动第一步要钉死的**。

**B-2（data_analyst.py step 2.8）**：从 `per_subject[subject]["_signal_distributions"][metric]` 取真分布 → 对 pendulum/velocity 参数做 p90×3 / p10÷3 比对 → 产出有数据的 ParameterAuditFinding（n≥2）。`_signal_distributions` 缺失（边界场景）仍走 7.1 的降级出口（阶段 1.5 已修对形状）。

### 7.5 实施顺序（spec 内部）
1. **阶段 1.5 先做**（7.1 + 7.2 + 单测）→ 单独 PR，解锁 dogfood。复现验收用 thread `81051535` 同款 FST n=1 数据。
2. **阶段 2 后做**（7.3 + 7.4 + B-2）→ 单独 PR，从阶段 1.5 合入后的 dev 开 worktree。第一步先钉死"分布在 dispatcher 还是 compute 脚本算"。
