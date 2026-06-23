# Spec：ETHO-3 残留缺口 —— code-executor 分诊侧读 133K plan_metrics 截断，改走已有 metadata sidecar

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-23
> 性质：◐ 残留缺口收尾 · 小改动。report-writer 侧 #169 已用 `_metric_metadata.json`（去重 ~5 条）消除 133K plan_metrics 截断；但 **code-executor 分诊失败场景**（用 read_file 查证失败细节）若读 133K plan_metrics 仍可能触发截断。
> **诚实标注**：这是**理论可复现路径，未在 2026-06-23 dump 的 4 个生产 thread 复现**（那 4 个 thread 全 `success`，分诊失败场景未触发——分诊只在失败时发生）。属防御性收尾，优先级低于仍活跃复现的 ETHO-1/7/9。
> 关联：
> - 来源：`~/ETHOINSIGHT_BUGS.md` ETHO-3；根因对照 `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md`
> - report-writer 侧已修：#169（`_metric_metadata.json` sidecar，`report_writer.py:181-200` 指向它、禁读 plan_metrics）。
> - **框架**：DeerFlow（LangGraph）原生——复用 #169 已建的 sidecar，只改 code-executor prompt 指向它，零新机制。
> - **HarnessX 背书**：Digester（10M token 轨迹压成结构化 per-task 摘要）= sidecar「大产物去重压成结构化旁路」的同款思路（memory `reference_harnessx_report_and_etho_spec_application`）——这是通用正确模式，非 ad-hoc 补丁。
> - 受保护文件：`subagents/builtins/code_executor.py` sync surgical。

---

## 〇、给实施 agent 的一句话

`code_executor.py` 分诊段（约 L59）写「分诊时用 read_file / ls 查证失败细节」。若失败分诊时 lead/subagent 去 read 133K 的 plan_metrics.json（28×5=140 条 subject 重复），触发 read_file 截断，啃不到尾部。**report-writer 已经有解药**：#169 建的 `_metric_metadata.json`（去重 ~5 条，`report_writer.py:181` 指向它）。**修法：code-executor 分诊段也指向 metadata sidecar，分诊查"指标定义/单位/期望集"走它，不读 plan_metrics 全文。** 纯 prompt 指向改动，复用现成 sidecar。

---

## 一、根因

### 1.1 现象（残留）

code-executor / report-writer 读 `plan_metrics.json`（~133K）反复截断、来回重读、陷入循环。

### 1.2 report-writer 侧已消除（#169）

`plan_metrics.json` 按 subject 重复（140 条 ≈133K），触发 read_file 截断。#169 生成 `_metric_metadata.json`（去重 ~5 条），report-writer prompt（`report_writer.py:181-200`）明确指向它、禁读 plan_metrics。

### 1.3 残留：code-executor 分诊侧未覆盖

- **happy path 不截断**：code-executor 走 `run_metric_plan` 工具内部读 plan、不透传 LLM。
- **分诊失败场景**（`code_executor.py` 分诊段「用 read_file/ls 查证失败细节」）：若读 133K plan_metrics 查证，**仍触发截断**。metadata sidecar 没覆盖这一侧的指引。

### 1.4 为什么诚实标"未在生产复现"

prod 4 thread 全 `success`，分诊失败场景（只在 run_metric_plan 失败时发生）未触发。这是**代码路径真实存在、但本批未踩到**的缺口。修它是防御性收尾，不是救火（守 memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`：根因路径真实但未隔离复现时，诚实标注、低优先）。

---

## 二、设计

### 2.1 修法：code-executor 分诊段指向 metadata sidecar

改 `code_executor.py` 分诊段：分诊查证"指标定义/单位/期望集/算了哪些"时，**读 `_metric_metadata.json`（去重 ~5 条），不读 plan_metrics.json 全文**。仅当分诊确需看某 subject 的具体落盘产物时，read 那个 `m_*.json`（单条小文件），而非整个 plan_metrics。

正面措辞示例（接现有分诊段）：
```
分诊查证指标定义/单位/期望集时，读 _metric_metadata.json（去重元数据，~5 条）。
查某个 subject 的具体产物时，读那个 m_<id>_s<subject>.json 单文件。
plan_metrics.json 是施工单（按 subject 重复、体积大），不用于分诊查证。
```

### 2.2 备选（仅当 sidecar 不够）：read_file 行范围

若分诊确需看 plan_metrics 某段，用 read_file 的行范围/offset 能力（若 `sandbox/tools.py` 的 read_file 支持 offset/limit；实施前先确认该能力存在——grep `def read_file` 的签名）。**优先 sidecar，offset 是 fallback**。

### 2.3 不改的东西

- **不改** `_metric_metadata.json` 生成逻辑（#169 已建，复用）。
- **不改** `run_metric_plan` happy path（本就不截断）。
- **不改** read_file 截断常量（不为这一处调全局截断）。

---

## 三、改动清单（change manifest）

### 3.1 `subagents/builtins/code_executor.py` —— 分诊段指向 metadata sidecar
- **编辑**：分诊段加「查证走 _metric_metadata.json / 单 m_*.json，不读 plan_metrics 全文」。
- **预期改善**：分诊失败场景不再因读 133K plan_metrics 截断、来回重读。
- **可能回归**：分诊确需 plan_metrics 某字段而 metadata 没有 → 保留「看单 m_*.json」出口。
- **病理**：无（prompt 指向改动）。

### 3.2 不改
- metadata 生成、run_metric_plan、read_file 截断常量。

---

## 四、测试（prompt 契约）

> ⚠️ prompt 契约用 importlib 读被测源。

测试文件：`tests/test_code_executor_triage_metadata.py`（新增）。

1. **test_triage_points_to_metadata_sidecar**（红→绿）：渲染 code-executor prompt/system_prompt，断言分诊段含「_metric_metadata.json」+「不读 plan_metrics 全文」语义。改前红、改后绿。
2. **test_triage_keeps_single_subject_file_exit**（守边界）：断言保留「查单 subject 读 m_*.json」出口。
3. **test_metadata_sidecar_generation_unchanged**（守 scope）：#169 的 sidecar 生成逻辑未被本 spec 改动。

> ⚠️ code-executor 行为改动**三指令源**（memory `feedback_subagent_system_prompt_higher_authority_than_skill`）：分诊指引可能在 `code_executor.py` 的 `SubagentConfig.system_prompt`（常驻 context，最高权威）、SKILL.md、lead 派遣 prompt 多处。改前 grep `builtins/code_executor.py` 确认改的是 system_prompt（最高权威源），否则白改。

---

## 五、验收标准

1. manifest 完整（§三）。
2. 红→绿：测试 1 改前红改后绿；2/3 绿。
3. **改对了指令源**：确认改的是 `code_executor.py` 的 system_prompt（最高权威），非只改 SKILL.md。
4. import 环：裸导入 `app.gateway` / `make_lead_agent` 0 退出（code_executor.py 是 subagent 核心）。
5. 回归：code-executor 邻域测试绿。
6. scope：未改 metadata 生成 / run_metric_plan / read_file 截断常量。

---

## 六、风险与注意事项（三大病理自检）

1. **reward hacking**：不适用（prompt 指向）。
2. **catastrophic forgetting**：改分诊段会不会丢失现有分诊的其他正确指引（plan 层/数据层/环境层错误三分类）？→ 只在分诊段加「查证走 metadata」，不动三分类逻辑；diff 守邻段。
3. **under-exploration**：这条**正确选 prompt 而非新 sidecar**——report-writer 的 sidecar（#169）已存在，复用即可，不重造（HarnessX Digester 思路已落地，code-executor 只是接上）。
4. **优先级诚实**：未在生产复现，低优先；先合活跃复现的 ETHO-1/7/9，本条收尾时合。
5. **三指令源**：改 system_prompt 非只 SKILL.md（见 §四警告）。
6. **受保护文件 sync surgical**。

---

## milestone 建议
- 「subagent 派遣/生命周期 infra 加固」track：ETHO-3 残留（分诊侧 metadata）是 #169 metadata sidecar 的覆盖面收尾；与 ETHO-1（seal）同属 subagent 执行鲁棒性。低优先，活跃 bug 后收尾。
