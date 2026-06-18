# Spec：code-executor handoff 瘦身 —— 拆除 50K 截断线下的死重量

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-18
> 性质：结构性根治（非 prompt 补丁）。多 subject 场景下 `handoff_code_executor.json` 体积越过 sandbox `read_file` 的 50K 截断线，导致 data-analyst 读不到尾部的 `gate_signals` / `data_quality_warnings`，反复盲试行号读取、烧光 `max_turns=12` 预算而无法 fast-fail。
> 关联：
> - 前序结论（保留 task_context 但标 TODO「待真实损害场景出现后重评估」）：[`2026-06-04-task-context-reeval-and-pending-items-fix-spec.md`](2026-06-04-task-context-reeval-and-pending-items-fix-spec.md)、[`2026-06-04-task-context-writeside-spec.md`](2026-06-04-task-context-writeside-spec.md)
> - 拓扑事实：memory `feedback_task_context_framework_mismatches_ethoinsight_topology`（task_context 框架不适配 EthoInsight lead 中转拓扑）
> - 根因纪律：memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`（根因未隔离前别叠加兜底——不该靠 prompt 教 data-analyst「换个方式读尾部」）
> - SSOT 铁律：memory `feedback_single_source_of_truth`
> - 全量回归铁律：memory `feedback_pr_merge_must_run_full_suite_on_shared_logic`、`feedback_known_full_suite_test_pollution_4_tests`

---

## 〇、给实施 agent 的一句话

`handoff_code_executor.json` 在真实 28-subject EPM dogfood 里达到 **85K 字符**，其中 **44K（52%）对任何消费者都没用或属对账后即弃的副本**：`task_context`（22K，全 4 个 handoff schema 都注入、但代码注释自承「下游不消费」）、`output_files` 的 140 条中间产物路径列表（8.9K，聚合器靠 glob 磁盘而非读它）、以及这些路径在 `task_context.verify_commands`/`file_changes` 里的二次复制。data-analyst 真正消费的核心字段约 41K——但 `gate_signals` / `data_quality_warnings` 这两个 fast-fail 必读字段恰好排在文件**尾部**，被 50K 截断线斩在外面。

**做四件事**：① 把 `task_context` 从 seal 的无条件注入改为「仅当 schema 声明需要时注入」，并从 4 个 ethoinsight handoff schema 移除该字段（拆为旁路 lineage，不进主 handoff）；② `output_files` 在主 handoff 内瘦成计数 + 旁路文件引用；③ `statistics.outlier_diagnostics`（28K）拆到旁路 `handoff_code_executor_outliers.json`，主 handoff 只留摘要 + 引用，data-analyst 需要时单独读旁路；④ 删掉 data-analyst / chart-maker / report-writer SKILL 里那条**用禁用 `bash` 拼 bundle** 的指引（三个 subagent 都禁了 bash，这指引从来跑不通，且只会把文件越拼越大）。

**核心约束**：不破坏任何现有 handoff 读取方；不改 50K 配置上限（治体积不治阈值，避免「响亮故障换哑故障」）；红→绿坐实（先证明改前主文件 >50K 且尾部字段读不到，改后 <50K 且一次读全）。

---

## 一、根因（逐字节实证，dogfood thread `3bcbee10`）

真实文件 `packages/agent/backend/.deer-flow/users/<uid>/threads/3bcbee10-.../user-data/workspace/handoff_code_executor.json` = **84877 字符**，顶层字段拆账：

| 字段 | 大小 | 占比 | 消费方 | 性质 |
|---|---|---|---|---|
| `statistics.outlier_diagnostics` | 28K | 33% | data-analyst（逐条映射 outlier_findings） | **有用但放错层**——把主文件顶过 50K |
| `task_context` | 22K | 26% | **无**（schema docstring 自承「下游不消费」） | **死重量**——`file_changes`(8K)+`verify_commands`(12.5K) 全是 output_files 路径的副本+模板命令 |
| `output_files.metrics` | 8.9K | 10% | 聚合器（但靠 glob 磁盘，不读此列表）；下游 subagent 不读 | **对账后即弃**——140 条 `m_*.json` 路径 |
| `per_subject` | 6.9K | 8% | data-analyst | 核心，精简（28×5 标量） |
| `statistics.comparisons` | 3.3K | 4% | data-analyst | 核心 |
| `metrics_summary` | 3.1K | 4% | data-analyst / report-writer | 核心 |
| `gate_signals` | 0.4K | <1% | **data-analyst fast-fail（必读）** | **在尾部，被 50K 斩断** |
| `data_quality_warnings` | <0.1K | <1% | **data-analyst fast-fail（必读）** | **在尾部，被 50K 斩断** |

三个事实叠加构成结构性故障：

1. **sandbox `read_file` 在 50K 字符硬截断**（[`config.yaml:59`](../../../packages/agent/config.yaml) `read_file_output_max_chars: 50000`，[`sandbox/tools.py:1683`](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py)）。文件 85K → 只能看前 ~50K。
2. **seal 用 `model_dump_json(indent=2)` 写多行 JSON**（[`seal_handoff_tools.py:386`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py)），尾部的 `gate_signals` / `data_quality_warnings` 排在 50K 之后。
3. **data-analyst 的 fast-fail（[`data_analyst.py:135-145`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py) step 2.1）要求读 `gate_signals` / `data_quality_warnings`**——它不知道精确行号，盲试 `start_line=870/890` 命中空或重复 head，在 `max_turns=12` 里把预算耗在 I/O 上，最终无法 seal。这正是 dogfood trace 里「让我换个方式读尾部」长串挣扎的根因。

**`task_context` 的死重量是已知问题**：[`2026-06-04-task-context-reeval`](2026-06-04-task-context-reeval-and-pending-items-fix-spec.md) §二已判四字段「价值低/为零+bug」，但处置是「留 TODO 不强行做」，TODO 写明「待 v1.0 直接接力拓扑或真实损害场景出现后重评估」。**本 spec 即该 TODO 触发的续集**——损害场景已出现（22K 把主文件顶过截断线）。

---

## 二、设计原则

### 2.1 治体积，不治阈值

**不调高 `read_file_output_max_chars`。** 50K 是全局保护（防 LLM 读巨文件爆 context）。调高它 = 把「响亮故障」（读不到尾部、空转）换成「哑故障」（每次读 handoff 都吃满 context、下游全部变慢且更易触发 summarization）——违反 memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`。根因是文件胖，就让文件瘦。

### 2.2 旁路文件承载「大但偶尔需要」的细节，主 handoff 承载「小但必读」的核心

判据：

- **每次都被 fast-fail / 核心解读读的小字段** → 留主 handoff（status / gate_signals / data_quality_warnings / metrics_summary / per_subject / statistics.comparisons / statistics.summary）。
- **大、且只在特定分支才需要的细节** → 拆旁路文件，主 handoff 留摘要 + 引用路径（outlier_diagnostics）。
- **对账完即弃、无下游消费方的** → 主 handoff 留计数 + 旁路引用（output_files 的 140 条路径），或直接移除（task_context）。

旁路文件命名约定（与主 handoff 同目录 `/mnt/user-data/workspace/`）：
- `handoff_code_executor_outliers.json` —— 完整 `outlier_diagnostics` 数组。
- `handoff_code_executor_outputs.json` —— 完整 `output_files`（140 条产物路径），audit/lineage 用。

### 2.3 不破坏任何现有读取方（SSOT + 向前兼容）

- `task_context` 字段从 4 个 ethoinsight handoff schema 移除时，`_build_task_context` 的注入逻辑改为「仅当 model 仍声明该字段时注入」——避免 Pydantic `extra="forbid"` 报错，也避免给已移除字段的 schema 塞死数据。**注意**：4 个 handoff 类目前都 `extra="allow"`，需确认移除字段后旧 handoff 文件（带 task_context）仍能被新 schema parse（`extra="allow"` 下多余字段被吞，OK）。
- `outlier_diagnostics` 拆旁路后，data-analyst SKILL 的 step 2.7b（[`data_analyst.py:176-187`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py)）必须同步改为「从旁路文件读 outlier_diagnostics」——否则它会在主 handoff 里找不到该字段而降级到「定性兜底」（line 186），丢失反事实精度。这是 SSOT 的下游同步（memory `feedback_ssot_skill_deployment_distinction`：改数据位置必同步 prompt 指引）。

### 2.4 删除从来跑不通的 bash bundle 指引（契约断裂，非性能优化）

data-analyst / chart-maker / report-writer 三个 subagent 的 SKILL 都有一段「`bash cat handoff... > bundle.txt` 再一次 read」的加速指引（[`data_analyst.py:128`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py) / [`chart_maker.py:36`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py) / [`report_writer.py:210`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py)）。但三者的 `disallowed_tools` 都禁了 `bash`（data_analyst.py:380 / report_writer.py:302 / chart_maker 同理）。**这指引从未能执行**——是又一处 SKILL↔工具契约断裂（同 memory `feedback_code_executor_skill_writefile_contradicts_seal_tool`）。且 bundle 把多文件拼一起只会更大、更易越 50K。删除该段，改为「主 handoff 已瘦身到单次 read_file 可读全，逐个 read_file 即可；outlier 细节按需读旁路文件」。

---

## 三、改动清单（按文件）

### 3.1 `seal_handoff_tools.py` —— task_context 注入条件化

[`_seal_handoff_to_workspace`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py) line 370-372 当前：

```python
# 1.5. 自动组装 task_context（确定性，subagent 无感知）。
payload.setdefault("task_context", _build_task_context(payload))
```

改为：仅当目标 schema 声明了 `task_context` 字段时才注入。

```python
# 1.5. 自动组装 task_context —— 仅当目标 schema 仍声明该字段时注入。
# ethoinsight 4 个 handoff 已移除该字段（拆为旁路 lineage，spec 2026-06-18）；
# 通用 handoff schema 若仍有该字段则保持组装，向前兼容。
if "task_context" in getattr(model_cls, "model_fields", {}):
    payload.setdefault("task_context", _build_task_context(payload))
```

`_build_task_context` 函数本身**保留**（通用 handoff 或未来 v1.0 直接接力拓扑可能复用），只是 ethoinsight 路径不再触发它。

### 3.2 `handoff_schemas.py` —— 从 4 个 ethoinsight handoff 移除 task_context + outlier 旁路引用

- 删除 `CodeExecutorHandoff`（line 491）/ `ChartMakerHandoff`（558）/ `DataAnalystHandoff`（663）/ `ReportWriterHandoff`（718）的 `task_context: TaskContext | None` 字段。
- `TaskContext` 类**保留定义**（docstring 已说明 v1.0 复用价值），只是 ethoinsight handoff 不再引用它。
- `CodeExecutorHandoff` 新增两个旁路引用字段（小，留主 handoff）：
  ```python
  outlier_diagnostics_ref: str | None = Field(
      default=None,
      description="完整 outlier_diagnostics 拆到旁路文件的虚拟路径（体积控制，spec 2026-06-18）。"
                  "data-analyst 需逐条映射 outlier_findings 时从此路径读。",
  )
  outlier_diagnostics_count: int = Field(
      default=0, description="离群诊断条目数（摘要，避免下游为拿计数而读旁路文件）。",
  )
  output_files_ref: str | None = Field(
      default=None,
      description="完整 output_files（产物路径表）拆到旁路文件的虚拟路径（audit/lineage 用）。",
  )
  output_files_count: int = Field(
      default=0, description="产物文件数（摘要，取代主 handoff 内嵌完整路径列表）。",
  )
  ```
- `output_files` 字段保留但**改为只读摘要语义**：主 handoff 内 `output_files` 仍存在以兼容旧读取方，但 §3.3 写入时只放计数级摘要（如 `{"metrics_count": 140}`），完整列表进旁路。**或**（更干净，二选一，实施 agent 按测试反馈定）：直接清空 `output_files` 为 `{}`，完整列表只在旁路。**默认选后者**（清空），因为已确认无下游 subagent 消费 `output_files`（§一表格）；聚合器 status 对账在 `run_metric_plan` 内部、aggregate 返回时已完成，不依赖 handoff 里的这份列表。

### 3.3 `run_metric_plan_tool.py` —— 写主 handoff 前拆旁路

[`run_metric_plan_tool.py`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_metric_plan_tool.py) line 304-324 组装 payload 处，在 seal 前：

1. **拆 outlier_diagnostics**：若 `statistics_payload` 含 `outlier_diagnostics` 且非空：
   - 把该数组写到 `workspace / "handoff_code_executor_outliers.json"`（原子写，同 seal 模式）。
   - 主 handoff 的 `statistics` 里**移除** `outlier_diagnostics`（或置为 `[]`，保留 key 兼容），改设 `payload["outlier_diagnostics_ref"] = "/mnt/user-data/workspace/handoff_code_executor_outliers.json"` + `payload["outlier_diagnostics_count"] = len(...)`。
   - **保留** `statistics.summary` / `statistics.comparisons`（小、核心，留主 handoff）。
2. **拆 output_files**：把 `agg["output_files"]` 完整写到 `workspace / "handoff_code_executor_outputs.json"`，主 handoff `payload["output_files"] = {}`（或计数摘要），设 `payload["output_files_ref"]` + `payload["output_files_count"]`。
3. **task_context**：§3.1 改动后此处无需动作（schema 不再有该字段 → seal 不注入）。

旁路写入用与 [`_seal_handoff_to_workspace`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py) 同样的原子 tmp+rename + chmod 0o644 模式（抽一个小 helper `_write_sidecar_json(workspace, filename, data)` 或直接内联）。旁路文件**不进 manifest hash 链**（它们是主 handoff 的附属，主 handoff 的 hash 已覆盖引用路径；若要严格也可加，非本 spec 必须）。

> 注意 `run_metric_plan` 还有降级路径（line 509-549，`output_files: {"metrics": []}`）—— 那条路径 output_files 本就空，无需拆；但 §3.1 的 task_context 移除对它同样生效（走同一 seal）。

### 3.4 `data_analyst.py` SKILL —— 同步 outlier 读取位置 + 删 bash bundle 指引

- **删除 step 2（line 126-134）的 bash bundle 段**，改为：
  ```
  2. **读上下文**：主 handoff 已瘦身，单次 read_file 即可读全：
     read_file /mnt/user-data/workspace/handoff_code_executor.json
     read_file /mnt/user-data/workspace/plan_metrics.json
     （逐个 read_file；禁止用 bash —— 本 subagent 无 bash 工具，且主 handoff 已无需拼接）
  ```
- **改 step 2.7b（line 176-187）的 outlier 读取**：
  ```
  - outlier_diagnostics 已拆到旁路文件（体积控制）。从主 handoff 读
    outlier_diagnostics_ref 和 outlier_diagnostics_count：
    · 若 count == 0：无离群，outlier_findings = []，跳过本段。
    · 若 count > 0：read_file <outlier_diagnostics_ref> 拿完整数组，
      逐条映射进 outlier_findings（subject/value/deviation/counterfactual 原样引用，不重算）。
    · 若 ref 缺失或读取失败（老数据/降级路径）：走 line 186 的定性兜底。
  ```
- **改 fast-fail step 2.1（line 135-145）**：明确「gate_signals / data_quality_warnings 现在稳定位于瘦身后的主 handoff 内，单次 read_file 可达——不要再尝试 start_line/end_line 盲读尾部」。

### 3.5 `chart_maker.py` / `report_writer.py` SKILL —— 删 bash bundle 指引

同 §3.4 第一点：删除各自的 `bash cat ... bundle` 段（chart_maker.py:35-40 / report_writer.py:209-214），改为「逐个 read_file；无 bash 工具」。chart-maker / report-writer 不消费 outlier_diagnostics，无需加旁路读取指引。

---

## 四、测试（红→绿坐实，TDD 强制）

新建 `packages/agent/backend/tests/test_code_executor_handoff_slimming.py`：

### 4.1 红线测试：主 handoff 体积 + 尾部字段可达性

1. **`test_main_handoff_under_read_limit_with_28_subjects`**：构造一个 28-subject × 5-metric 的 plan + 模拟 m_*.json + 含 65 条 outlier_diagnostics 的 statistics，跑 `run_metric_plan`（或直接调 payload 组装 + 拆旁路逻辑），断言 `handoff_code_executor.json` 字节数 **< 50000**。
   - **坐实红**：在改动前（或临时注释掉拆旁路逻辑）跑此测试，断言 fail（>50K）—— 证明测试真咬。用 git stash / 临时改一行的方式记录改前值（应 ~85K）。
2. **`test_gate_signals_and_dq_warnings_in_first_50k`**：读主 handoff 原始文本，截取前 50000 字符，断言 `"gate_signals"` 和 `"data_quality_warnings"` 两个 key 都出现在截断窗口内（即 data-analyst 单次 read 必能看到）。

### 4.2 旁路文件契约

3. **`test_outliers_sidecar_written_and_referenced`**：断言 `handoff_code_executor_outliers.json` 存在、是合法 JSON、内容 == 原 outlier_diagnostics 数组；主 handoff 的 `outlier_diagnostics_ref` 指向它、`outlier_diagnostics_count` == 数组长度；主 handoff 的 `statistics` 内不再内嵌完整 outlier_diagnostics（[] 或缺失）。
4. **`test_outputs_sidecar_written_and_referenced`**：断言 `handoff_code_executor_outputs.json` 存在、含完整 140 条路径；主 handoff `output_files_count` == 140；主 handoff `output_files` 为 {} 或计数摘要。
5. **`test_no_outliers_no_sidecar`**：outlier_diagnostics 为空时，不写旁路文件，`outlier_diagnostics_count == 0`，`outlier_diagnostics_ref is None`。

### 4.3 task_context 移除 + 不破坏读取方

6. **`test_task_context_not_injected_for_ethoinsight_handoffs`**：seal 一个 CodeExecutorHandoff payload，断言产出 JSON **不含** `task_context` 键。
7. **`test_task_context_still_injected_for_schemas_that_declare_it`**：若有通用/测试 handoff 类仍声明 task_context（或临时造一个），断言注入仍生效（§3.1 的条件化未误伤通用路径）。
8. **`test_old_handoff_with_task_context_still_parses`**：用一份带 `task_context` 的旧 handoff dict 实例化新 `CodeExecutorHandoff`，断言不抛（`extra="allow"` 吞掉多余字段）—— 向前兼容。

### 4.4 SKILL 契约（静态断言，防回归）

9. **`test_subagent_skills_have_no_bash_bundle_for_banned_bash`**：对 data_analyst / chart_maker / report_writer 三个 CONFIG，断言：若 `"bash" in disallowed_tools`，则 system_prompt 中**不含** `da_context_bundle`/`bash cat /mnt/user-data/workspace/handoff` 之类用 bash 拼 bundle 的串。这是契约断裂的永久守护（同类问题 P5 已踩过）。

### 4.5 全量回归（铁律）

- 改 `seal_handoff_tools.py` / `handoff_schemas.py` / `run_metric_plan_tool.py` 属共享逻辑，**合前必跑全量** `make test`（memory `feedback_pr_merge_must_run_full_suite_on_shared_logic`），并 grep 所有 `task_context` / `output_files` / `outlier_diagnostics` 读取方确认无遗漏破坏。
- 区分已知污染（memory `feedback_known_full_suite_test_pollution_4_tests`：`test_data_analyst_step28_contract` 等在干净 dev 上即红）——用父 commit detached worktree 全量对比坐实非本改动引入。
- **裸导入两生产入口**（改了 subagents/tools/builtins 核心，CLAUDE.md 闭环铁律）：
  ```bash
  cd packages/agent/backend
  PYTHONPATH=. python -c "import app.gateway"
  PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
  ```

---

## 五、验收标准

1. 28-subject EPM 真实数据复跑（`/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28` 或 dogfood 等价集）：主 `handoff_code_executor.json` < 50K，`gate_signals` / `data_quality_warnings` 在首 50K 内。
2. data-analyst 单次 read_file 读全主 handoff，fast-fail 检查不再盲试 start_line/end_line 尾部读取；有离群时从旁路文件读到完整 outlier_diagnostics 并产出 counterfactual（精度不降级）。
3. `task_context` 不再出现在任何 ethoinsight handoff；旧带 task_context 的 handoff 仍可被新 schema parse。
4. 三个 subagent SKILL 不再含禁用 bash 的 bundle 指引。
5. 全量测试绿（扣除已知污染）；两生产入口裸导入 0 退出。

---

## 六、风险与注意事项

1. **`output_files` 清空 vs 计数摘要的二选一**：默认清空（§3.2/§3.3），但若全量回归暴露出某个我们没 grep 到的读取方（如某测试 fixture 断言 `output_files.metrics` 非空），退回「计数摘要」方案并把该读取方改读旁路文件。**实施时先 grep 全仓 `output_files` 的 `.get`/`["metrics"]` 访问点再定。**
2. **旁路文件的虚拟路径 vs host 路径**：旁路文件写在 host-side `workspace`，但主 handoff 里的 `_ref` 字段必须是**虚拟路径** `/mnt/user-data/workspace/...`（data-analyst 在 sandbox 内 read_file 用虚拟路径）。遵守 `handoff_schemas.py` 的 `_VIRTUAL_USER_DATA_PREFIX` 约束（line 40-54），别填 host 路径。
3. **manifest hash**：主 handoff 仍走 seal 的 manifest（hash 覆盖 _ref 路径字符串）；旁路文件本身不强制进 manifest。若后续 handoff 校验器（`test_handoff_content_validation` 等）要求引用文件存在性校验，再补——非本 spec 必须，但实施时确认现有校验器不会因主 handoff 缺 outlier_diagnostics 内嵌数据而报「残缺」（memory `feedback_handoff_content_validator_rejects_failed_status_infinite_redispatch` 同源风险）。
4. **降级路径（line 509-549）**：那条路径 statistics 缺失、output_files 空，拆旁路逻辑要 `if ... and non-empty` 守护，别给空数据写空旁路文件造成下游困惑。
5. **chart-maker 也读主 handoff**：确认 chart-maker 不依赖被拆走的 outlier_diagnostics / output_files（已确认它走 prep_chart_plan 工具自读 context，不读这两者）——但全量回归仍要覆盖 chart-maker handoff 测试。
6. **本 spec 不碰 50K 配置**：若未来字段自然增长再次逼近 50K，是新的瘦身工单，不是调高阈值的理由（§2.1）。

---

## milestone 建议

本 spec 属「harness 鲁棒性 / dogfood 根因治理」track（见 [`2026-06-18-9spec-dogfood-rootcause-batch-complete-handoff.md`](../../handoffs/2026-06/2026-06-18-9spec-dogfood-rootcause-batch-complete-handoff.md)）的延续——9-spec 批次之后由真实 dogfood 复跑暴露的第 10 个结构根因（handoff 体积越截断线）。合入后更新该 milestone：记录「task_context 死重量在 2026-06-04 reeval 时标 TODO，2026-06-18 由 50K 截断损害场景触发拆除」这条 TODO→闭环的链路，作为「价值存疑字段留 TODO、待真实损害再处置」纪律的正例。
