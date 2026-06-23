# Spec：展示元数据瘦身旁路文件 —— 修 report-writer / data-analyst 啃 133K plan_metrics.json 撞 turn 超时

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-22
> 性质：结构性根治（memory 第 4 类 seal 根因复发，对象从 data-analyst 换成 report-writer）。report-writer / data-analyst 的 prompt 指示它们 read 整个 `plan_metrics.json` 找展示/判读元数据，但该文件把每个 metric 的元数据**按 subject 重复 N 遍**（EPM 28 subject × 5 metric = 140 条，133K）。thinking 模型在「按 metric id 在 metrics[] 数组中匹配」「估算行号分段读」上反复烧 token，单个 model turn 撑爆 50K thinking、撞 wall-clock 超时、turn 没结束 → SealGate（after_model 门）结构性救不了 → seal 漏调 → lead 重派。**数据本身没问题——元数据 5 条够用，被重复成 140 条埋在 133K 里。**
> 关联：
> - 治理模式来源：`docs/superpowers/specs/2026-06-18-data-analyst-thinking-overload-spec.md`（**那份 spec 只删了参数审计，没修展示元数据读取**——`data_analyst.py:249` 的同构隐患仍在，本 spec 一并根治）
> - 调研来源：`docs/problems/2026-06-22-thread-3a41e483-four-issues-investigation.md` A 段
> - memory 第 4 类根因：`feedback_seal_fourth_root_cause_thinking_overload_turn_timeout`（判别铁律：seal 漏调先分「收尾漏 call（SealGate 能拦）/ turn 内超时（结构救不了）」）
> - SSOT：memory `feedback_single_source_of_truth`（展示元数据只存一份）、`feedback_skill_describing_tool_output_enables_hallucination`（prompt 用正面指令）

---

## 〇、给实施 agent 的一句话

`report_writer.py:176-192` 和 `data_analyst.py:249-262` 都命令 subagent **read 整个 `plan_metrics.json`**、「按 metric id 在 `metrics[]` 数组中匹配」取 `display_name_zh/unit_zh/one_liner`（report-writer）和 `direction_for_anxiety/statistical_default`（data-analyst）。但 `plan_metrics.json` 的 `metrics[]` 是**每 subject 一条**（`plan_metrics_to_dict`，[resolve.py:1391](../../../packages/ethoinsight/ethoinsight/catalog/resolve.py#L1391)），同一 metric id 重复 28 遍、夹在施工字段（`script/input/output/args/parameters_in_use/subject_index`）之间。thinking 模型要从中捞出 5 条去重元数据，反复估算「这 metric 在文件第几行」「133K 太大一次读不完」——turn 撑爆。

治本：① **生成去重的展示/判读元数据旁路文件 `_metric_metadata.json`**（按 metric id 去重，5 条而非 140 条，几 KB）——`plan_metrics.json` 仍是 code-executor 的施工 SSOT（不动），元数据旁路是它的去重投影（守 SSOT：元数据只一份，在 catalog；旁路是只读投影）；② **report-writer / data-analyst prompt 改读旁路文件**，删「在 metrics[] 数组中匹配」「分段读」诱导，加铁律「展示/判读元数据从旁路文件取，禁止 read plan_metrics.json」。

---

## 一、根因（逐字节实证，dogfood thread `3a41e483`）

### 1.1 现象

[49] / 前端 1736 行：`Error: Subagent 'report-writer' terminated without emitting 'handoff_report_writer.json'`，lead 自动重试一次才成功（[51]）。

### 1.2 thinking 过载实证（前端 1359-1735 行 report-writer 完整 thinking）

report-writer 在**单个 model turn 的 thinking 里**反复读 133K 的 `plan_metrics.json` 找 5 个 metric 的 `display_name_zh`：

```
前端 1378: 读到 open_arm_time_ratio: 开放臂时间比例 ✅
前端 1380: 读到 open_arm_time: 开放臂时间 ✅
前端 1382-1383: open_arm_entry_count / open_arm_entry_ratio → "I need to find the display name"
前端 1500-1730: 反复估算"这 metric 在文件第几行"（"each entry is probably ~950 chars"、"140 entries"、"let me read line 900"、"133K chars too large to read at once"）...
前端 1736: terminated without emitting handoff（=turn 超时/撑爆）
```

### 1.3 关键矛盾：元数据第一次跑就全拿到了

5 个 metric 的 `display_name_zh` 在前端 1562-1566 行 report-writer 自己**已经列出来了**——它根本不需要再去啃 133K 原始 plan。是 prompt 的「指标展示元数据查询」段（[report_writer.py:176-192](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py#L176-L192)）命令它这么做：

```
每个指标的中文展示字段已下沉到 plan_metrics.json,从那里取:
read_file:
    /mnt/user-data/workspace/plan_metrics.json
按 metric id 在 `metrics[]` 数组中匹配,读取以下字段:
- display_name_zh / unit_zh / one_liner
多 subject 场景下同一 metric id 会出现多次(subject_index 区分),展示字段在所有
同 id 行上一致,取首个即可。
```

「按 metric id 在 metrics[] 数组中匹配」「同一 metric id 会出现多次取首个」——**这两句就是诱导 thinking 过载的指令**：它逼模型在 140 条里扫匹配、判断哪条是「首个」、估算行号。对 thinking 模型，这是典型的「遍历 N 条」机械动作（同 data-analyst 参数审计决策树，见 06-18 spec §1.4）。

### 1.4 data-analyst 同构隐患仍在

`data_analyst.py:249-262`（[实证](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py#L249-L262)）：

```
每个指标的判读字段已由 lead 在派遣前 resolve 到 plan_metrics.json,从那里取:
read_file: /mnt/user-data/workspace/plan_metrics.json
按 metric id 在 `metrics[]` 数组中匹配,读取以下字段:
- direction_for_anxiety / statistical_default
```

**同样的「在 metrics[] 数组中匹配」诱导**。06-18 data-analyst spec 只删了参数审计（step 2.8）+ 收窄 outlier 搬运，**没动这段展示/判读元数据读取**。data-analyst 在 thread `3a41e483` 没炸只是因为那次 thinking 预算恰好够——隐患同构，下次数据量稍大就触发。

### 1.5 plan_metrics.json 为什么这么大：元数据按 subject 重复

`plan_metrics_to_dict`（[resolve.py:1391-1420](../../../packages/ethoinsight/ethoinsight/catalog/resolve.py#L1391-L1420)）的 `metrics[]` 每条都带全部字段：

```python
{
    "id", "script", "input", "output", "required", "reason", "subject_index",
    "display_name_zh", "unit_zh", "one_liner", "output_unit",
    "direction_for_anxiety", "statistical_default",
    "parameters_in_use", "args"
}
for m in pm.metrics   # 每 subject 一条 → 28 × 5 = 140 条
```

EPM 28 subject × 5 metric = 140 条，每条 ~950 字符 ≈ 133K。其中**展示/判读元数据按 metric id 去重后只有 5 条**（每 metric 一份，所有 subject 共享同一份元数据）。140 条里的元数据是 5 条的 28 倍冗余。

`plan_metrics.json` 是 **code-executor 的施工 SSOT**（它要 140 条的 `script/input/output/args/parameters_in_use/subject_index` 来逐 subject 跑计算）——**这份冗余对施工是必要的，不能去重**。但 report-writer / data-analyst 只读元数据，不该啃这份施工文件。

---

## 二、设计

### 2.1 生成去重元数据旁路文件 `_metric_metadata.json`

在 `plan_metrics.json` 落盘的同一处（resolve 完 plan 后），额外写一个去重的元数据投影文件。**元数据的 SSOT 仍是 catalog**（`MetricEntry.display_name_zh` 等），旁路文件是 catalog → plan_metrics.json → 去重投影的只读产物，不引入新 SSOT（守 `feedback_single_source_of_truth`：元数据只在 catalog 定义一份，旁路是投影）。

文件位置：`/mnt/user-data/workspace/_metric_metadata.json`（与 `plan_metrics.json` 同目录，`_` 前缀表明旁路文件，同 `_outliers.json` / `_outputs.json` 惯例）。

结构（按 metric id 去重，每 metric 一条）：

```json
{
  "paradigm": "epm",
  "metrics": {
    "open_arm_time_ratio": {
      "display_name_zh": "开放臂时间比例",
      "unit_zh": "%",
      "one_liner": "在开放臂停留时间占总时间比例，焦虑样行为核心指标（越低越焦虑）",
      "output_unit": "ratio",
      "direction_for_anxiety": "lower_is_anxious",
      "statistical_default": "groupwise_compare"
    },
    "open_arm_time": { ... },
    ...
  }
}
```

- key 是 metric id，value 是该 metric 的展示+判读元数据（report-writer 要的 3 字段 + data-analyst 要的 2 字段 + output_unit）。
- **5 条而非 140 条**，几 KB，单次 read 即达，无需估算行号/分段。
- 字段来源：`plan_metrics.json` 的 `metrics[]` 去重取首个（或直接从 `pm.metrics` 去重，实施时按方便选）。**不重新查 catalog**——plan_metrics.json 已是 catalog 的投影，旁路是投影的投影，避免绕过 plan 层的「实际 resolve 出来的 metric 集合」（plan 可能因列缺失 skip 掉某些 metric，旁路应反映 plan 实际包含的 metric，不是 catalog 全集）。

### 2.2 report-writer prompt 改读旁路文件

[report_writer.py:176-192](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py#L176-L192)「指标展示元数据查询」段改写：

```
## 指标展示元数据查询

每个指标的中文展示字段在去重元数据文件里（按 metric id 一条，已去重）:

read_file: /mnt/user-data/workspace/_metric_metadata.json

按 metric id 直接查 metrics[id]，读取 display_name_zh / unit_zh / one_liner。
one_liner 仅首次提及该指标时引用，不要在每段重复。

禁止 read plan_metrics.json 取展示字段 —— 那是 code-executor 的施工文件（按 subject 重复 140 条，133K），
展示元数据已在 _metric_metadata.json 去重（5 条）。read plan_metrics.json 取展示字段会撑爆 turn。
禁止 read catalog YAML 文件 —— 它在 Python 包内，sandbox 不暴露。

禁止在本 prompt 内硬编码任何指标的中文名或单位 —— 全部走 _metric_metadata.json。
```

> ⚠️ 措辞守 `feedback_skill_describing_tool_output_enables_hallucination`：用**正面指令**（「从 _metric_metadata.json 取」），「禁止 read plan_metrics.json」这句是**必要的边界**（否则模型会习惯性回退到旧路径），保留但紧跟正面替代。不要用「不要 X」「禁止 X」做主要指令。

### 2.3 data-analyst prompt 同改

[data_analyst.py:249-262](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py#L249-L262)「指标元数据查询」段同样改读 `_metric_metadata.json`，取 `direction_for_anxiety / statistical_default / output_unit`。删「按 metric id 在 metrics[] 数组中匹配」。

### 2.4 prompt 加 thinking 瘦身铁律（复用 06-18 spec §2.2）

两处 prompt 各加一句（与 06-18 data-analyst spec 同款）：

```
<thinking_discipline>
thinking 只用来做判断（写报告/判读），不用来在文件里扫匹配、估算行号、分段读。
元数据查询是一次性 read + 直接查 key，不是遍历。若发现自己在估算「这 metric 在文件第几行」，
立即停——你读错文件了，展示/判读元数据在 _metric_metadata.json（去重，按 id 直查）。
</thinking_discipline>
```

### 2.5 fallback：旁路文件缺失

`_metric_metadata.json` 缺失（老 plan / 降级路径未生成）时：report-writer / data-analyst **不硬崩**，在该 metric 的展示字段用 metric id 本身（如 `open_arm_time_ratio`），并在报告/finding 里注明「展示元数据未就绪」。**不回退去啃 plan_metrics.json**——那会重新触发 thinking 过载。

---

## 三、改动清单

### 3.1 `resolve.py` / plan 落盘处 —— 写 `_metric_metadata.json`

定位 `plan_metrics.json` 的落盘处（`prep_metric_plan` / `run_metric_plan` / resolve 完 plan 写盘的工具），在写 `plan_metrics.json` 之后，从 `pm.metrics` 按 id 去重生成 `_metric_metadata.json` 写同目录。

```python
def metric_metadata_to_dict(pm: PlanMetrics) -> dict:
    """去重的展示/判读元数据投影（按 metric id 一条）。"""
    metrics_meta: dict[str, dict] = {}
    for m in pm.metrics:
        if m.id in metrics_meta:
            continue  # 去重，取首个（所有同 id 行元数据一致）
        metrics_meta[m.id] = {
            "display_name_zh": m.display_name_zh,
            "unit_zh": m.unit_zh,
            "one_liner": m.one_liner,
            "output_unit": m.output_unit,
            "direction_for_anxiety": m.direction_for_anxiety,
            "statistical_default": m.statistical_default,
        }
    return {"paradigm": pm.paradigm, "metrics": metrics_meta}
```

> 实施位置：grep `plan_metrics.json` 的写出处（很可能在 `prep_metric_plan_tool.py` 或 resolve 的序列化层），在写主 plan 后追加写旁路。**旁路与主 plan 同源同次写**，避免漂移。

### 3.2 `report_writer.py` —— prompt 改读旁路 + thinking 铁律

- §2.2 改写「指标展示元数据查询」段。
- §2.4 加 `<thinking_discipline>` 段。
- `<workflow>` step 2 里 `read_file plan_metrics.json` 这行**删掉**（report-writer 不再读施工文件；它要的数值在 handoff_code_executor.json，要的元数据在 _metric_metadata.json，没有理由读 plan_metrics.json）。

### 3.3 `data_analyst.py` —— prompt 改读旁路 + thinking 铁律

- §2.3 改写「指标元数据查询」段。
- §2.4 加 `<thinking_discipline>` 段。

### 3.4 sandbox 白名单 —— 放行 `_metric_metadata.json` 读取

核实 subagent 能 read `/mnt/user-data/workspace/_metric_metadata.json`（同目录的 `_outliers.json` 已能读，应已放行）。若 sandbox 白名单按文件名前缀，确认 `_` 前缀旁路文件都可达（memory `feedback_subagent_consumption_via_first_party_tool` / `feedback_chart_maker_bash_guardrail_must_allow_resolve_dumpheaders` 同源：白名单别漏放包内/旁路路径）。

### 3.5 不改 handoff schema、不改 plan_metrics.json 结构

`plan_metrics.json` 的 `metrics[]` 仍按 subject 重复（code-executor 施工需要）。`_metric_metadata.json` 是新增的只读投影，不影响任何下游契约。

---

## 四、测试（红→绿坐实，TDD 强制）

新建 `packages/ethoinsight/tests/test_metric_metadata_sidecar.py`：

1. **`test_metric_metadata_sidecar_is_deduplicated`**（红线）：
   构造 EPM PlanMetrics（28 subject × 5 metric = 140 条），调 `metric_metadata_to_dict`，断言 `metrics` 字段**只有 5 个 key**（按 id 去重），每个 key 含 6 字段。

2. **`test_metric_metadata_sidecar_written_alongside_plan`**：
   调 plan 落盘工具，断言 workspace 里同时存在 `plan_metrics.json` 和 `_metric_metadata.json`，且后者是前者的去重投影（元数据一致）。

3. **`test_metric_metadata_reflects_resolved_metrics_not_catalog_full`**：
   构造一个 plan 因列缺失 skip 了某 metric，断言 `_metric_metadata.json` **不含**被 skip 的 metric id（旁路反映 plan 实际集合，非 catalog 全集）。

4. **report-writer / data-analyst prompt 静态契约（防回归）**：
   - 断言 `report_writer.py` system_prompt **含** `_metric_metadata.json` 措辞 + `<thinking_discipline>` 措辞。
   - 断言 system_prompt **不含** `按 metric id 在 metrics[] 数组中匹配`（旧诱导串）+ **不含** `read_file\n    /mnt/user-data/workspace/plan_metrics.json`（report-writer 不再读施工文件）。
   - 同样断言 `data_analyst.py`。

5. **裸导入两生产入口（改 subagents 核心，铁律）**：
   `PYTHONPATH=. python -c "import app.gateway"` + `python -c "from deerflow.agents import make_lead_agent"` 均 0 退出（memory `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`）。

6. **全量回归**：backend 全量 + ethoinsight 全量；已知 pre-existing 污染排除。

---

## 五、验收标准

1. **dogfood EPM 复跑**（thread `3a41e483` 同款 28 subject 数据）：report-writer **单次** model turn 内完成报告并调 `seal_report_writer_handoff`，**不超时、不重派**。
2. report-writer / data-analyst 的 thinking 不再出现「估算 metric 在文件第几行」「分段读 plan_metrics.json」——元数据查询是一次性 read `_metric_metadata.json` + 按 id 直查。
3. `_metric_metadata.json` 存在、按 metric id 去重（EPM 5 条而非 140 条）、字段与 catalog 一致。
4. data-analyst 同款数据复跑不触发同构超时（隐患一并根治）。
5. 全量回归绿（除已知污染）。

---

## 六、风险与注意事项

1. **SealGate 救不了 turn 内超时**（memory 第 4 类根因边界）：本 spec 是治本（让 turn 能正常结束），不要试图靠调 SealGate / 加 auto-seal 兜底解决——根因未隔离前别叠加兜底（memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`）。
2. **不调 thinking_enabled / timeout 旋钮**：调旋钮会把响亮的超时变哑故障（思考变浅、报告变水却不报错）。本 spec 落地后若仍偶发超时再单独评估，不在本 spec 内（同 06-18 spec §6.2）。
3. **守 SSOT**：`_metric_metadata.json` 是 catalog → plan_metrics.json 的去重投影，**不是新 SSOT**。元数据只在 catalog 定义（`MetricEntry` 字段），旁路只读复制。改元数据改 catalog，旁路自动跟（每次 plan 落盘重新投影）。不要让 subagent 或旁路成为元数据的第二处定义源（memory `feedback_single_source_of_truth`）。
4. **旁路反映 plan 实际集合**：去重源是 `pm.metrics`（resolve 后实际包含的 metric），不是 catalog 全集——plan 会因列缺失 skip metric，旁路应同步 skip，否则 report-writer 会引用 plan 里没有的 metric 元数据。
5. **覆盖两处 prompt**：report-writer + data-analyst 都要改，别只改 report-writer（data-analyst 同构隐患仍在，下次数据量稍大就触发）。
6. **fallback 不回退啃 plan_metrics.json**（§2.5）：旁路缺失时用 metric id 兜底 + 注明，绝不回退到 133K 施工文件。
7. **措辞守正面指令**（memory `feedback_skill_describing_tool_output_enables_hallucination`）：主指令用「从 _metric_metadata.json 取」，「禁止 read plan_metrics.json」仅作边界紧跟正面替代，不用否定句做主指令。

---

## milestone 建议

本 spec 让「harness 鲁棒性 / dogfood 根因治理」track 再进一步：第三轮 EPM dogfood（thread `3a41e483`）暴露 memory 第 4 类 seal 根因**复发**——report-writer 的 thinking 把「在 133K 文件里扫匹配找元数据」当产出，撞 turn 内超时，SealGate 结构性救不了。与 06-18 data-analyst 同构（对象换 report-writer），且** 06-18 spec 漏修了展示元数据读取这步**（只删参数审计），本 spec 一并补上 report-writer + data-analyst 两处。建议在该 milestone 记录：① 此根因 + 本 spec；② **可复用教训**：catalog 的展示/判读元数据按 metric id 去重后体积极小，但 `plan_metrics.json` 为施工把它按 subject 重复 N 倍——subagent 读元数据应走去重投影旁路，啃施工文件必触发 thinking 过载；③ 06-18 spec（删参数审计）+ 本 spec（元数据旁路）合起来才是「判读/报告 subagent thinking 瘦身」的完整治本；④ 判 seal 漏调先分「收尾漏 call / turn 内超时」两类（memory 第 4 类根因铁律），本例属后者。
