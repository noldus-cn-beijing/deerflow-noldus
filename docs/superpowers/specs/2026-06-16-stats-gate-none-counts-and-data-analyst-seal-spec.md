# Spec: 2026-06-16 EPM dogfood — 统计被误跳过 + data-analyst seal 死锁修复（#6）

> 日期：2026-06-16（#2/#3/#5 修复 merge 后 dogfood 复跑暴露）
> 起点：EPM dogfood（`Raw data-EPM-Xuhui-28`，thread `9b1bc49b`）。#5 验证生效（run_metric_plan `critical_count=0`、code-executor 干净 seal），但 **data-analyst seal 失败**（"terminated without emitting handoff"）。
> 实施者：新 worktree（`git worktree add -b feat/stats-gate-and-analyst-seal <path> origin/dev` 独立分支，别共享 dev）。注意 origin/dev 当前 `05090c55`（含 #5），用 origin/dev 作基线。

---

## 0. 因果链（两条链式 bug，先读）

```
#6a prep_metric_plan 调 resolve_metrics 不传 n_per_group/n_groups（传了 None）
    └─→ stats gate `n_per_group>=2 and n_groups>=2` 用 None 评估 → False
          └─→ plan.statistics.skip_reason 被写死 "...not met (n_per_group=None, n_groups=None)"
                └─→ run_metric_plan Step7 只认 plan 的 skip_reason、runtime 不重评 → 跳过统计
                      └─→ handoff statistics={}（即便 groups.json 有 control=7/treatment=21）
                            └─→ #6b data-analyst 无推断统计可解读 → 掉进手算补偿螺旋
                                  → 耗尽推理预算、最后一轮纯文本、忘调 seal → "terminated without emitting"
```

**#6a 是根（工程化根治）**：把"漏传组计数"做成**结构上不可能**——resolve 自读 groups_file 派生计数（计数从"调用方必传入参"降级为"权威点自动派生的内部量"），覆盖所有现存+未来调用方。修好后统计正常跑 → data-analyst 有真统计结果 → 不手算 → 正常 seal，#6b 链根本不触发。
**#6b 是 seal 健壮性纵深防御（用户裁决：本批做但非主修复）**：data-analyst 在 statistics 缺失时的手算螺旋，prompt 加固只能降低概率、不能根治（守 `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`）；它的价值是兜住"#6a 之外的合法 statistics 空场景"（单组、脚本失败），不是替代 #6a。**单改 prompt 会复发——根治必须在代码（#6a）。**

---

## 1. #6a（根，工程化根除）：让"漏传组计数"结构上不可能再发生

### 1.1 根因——不是"忘了一行"，是 API 允许自相矛盾的调用且沉默退化（已逐层坐实）

dogfood 实测（thread 9b1bc49b）：plan.statistics.skip_reason=`"...not met (n_per_group=None, n_groups=None)"`，但 groups.json=`{control:7, treatment:21}`（n 充分），handoff.statistics=`{}`。

链路（纯 Python，无 LLM 参与——所以 prompt 修不了）：
- `prep_metric_plan_tool.py:265` 调 `resolve_metrics(...)` 传了 `groups_file` 但**漏传 `n_per_group`/`n_groups`**（默认 None）。
- `resolve.py:309-320` stats gate `_evaluate_when("n_per_group>=2 and n_groups>=2", n_per_group=None, n_groups=None)` → False（`_evaluate_atomic_when` 对 None 返 False）→ 写死 skip_reason。
- `run_metric_plan_tool.py:214` Step 7 只认 plan 的 `skip_reason is None` 才跑统计、runtime 不重评 → 跳过 → `statistics={}`。

**三个结构性病灶（这才是"为什么会漏传"的真答案）**：
1. **语义耦合参数被拆成结构独立入参**：`groups_file`（路径）/ `n_per_group`（最小组 size）/ `n_groups`（组数）本质是"同一份分组事实的三个投影"，后两者完全可从第一者派生，但 API 让你能传前者而不传后者 → `resolve_metrics(groups_file="...", n_per_group=None, n_groups=None)` 是一个**合法但自相矛盾**的调用（"有分组"+"没有分组计数"）。这正是 `feedback_fable_pr115_stage_decisions_parambinding_optional` 的教训：**让非法状态不可表示**，别指望每个调用方手动同步两字段。
2. **不一致沉默退化、不响亮失败**：矛盾状态没人拦 → 门用 None 算 False → 统计静默跳过 → plan 照常 `status=completed`。哑故障。
3. **同一个洞在整个 API 面复制**：不止 prep——`catalog/cli.py:86/89/90` 也把 `--groups-file`/`--n-per-group`/`--n-groups` 做成三个独立 arg，且它**已经读了 `--groups-json` 内容**（cli.py:165）却仍要单独传计数（派生数据摆着没用）。任何新调用方都要重新记得手动同步 → 漏传会复发。

> **`n_per_group`/`n_groups` 语义（实施前对齐）**：gate `n_per_group>=2 and n_groups>=2`。`n_groups`=不同 group 数=`len(set(group_names))`；`n_per_group`=**最小组 size**（gate 语义"每组都≥2"，取 min）。`_evaluate_atomic_when` 当标量比较，`cli.py:240`/`cli.py` 也是标量。

### 1.2 修复方案（用户裁决：① 治本自派生 + ② 入口断言兜底 + ③ 回归探针，三者都做）

核心思路：**计数从"调用方必须记得传的入参"降级为"在拿得到 groups 内容的唯一权威点自动派生的内部量"**，groups 成为唯一真相源，调用方根本没有"漏传计数"这个选项。

#### ① 治本：`resolve` 自读 groups_file 派生计数（删除调用方传计数的必要）

`resolve_metrics` 既收 `groups_file` 路径、又有文件系统访问（已在 physical_workspace 写文件），让它**自己读 groups_file 派生计数**。新增内部 helper（放 `resolve.py`，纯函数）：

```python
def _derive_group_counts(groups_file: str | None) -> tuple[int | None, int | None]:
    """从 groups_file 派生 (n_per_group=最小组 size, n_groups=组数)。

    groups.json 两种形状都支持：
      - {subject_path: group_name}（prep_metric_plan 写的正向映射）→ Counter(values)
      - {group_name: [subject_path, ...]}（read_groups_json/charts 的反向）→ {g: len(lst)}
    None/读不到/空 → (None, None)（单组或无分组场景，gate 正确 skip，行为不变）。
    """
    if not groups_file:
        return (None, None)
    try:
        from ethoinsight.scripts._cli import resolve_sandbox_path
        import json
        data = json.loads(Path(resolve_sandbox_path(groups_file)).read_text(encoding="utf-8"))
    except Exception:
        return (None, None)  # fail-safe：读不到不阻断（与现状等价）
    if not isinstance(data, dict) or not data:
        return (None, None)
    vals = list(data.values())
    if all(isinstance(v, str) for v in vals):          # {subject: group}
        from collections import Counter
        counts = Counter(vals)
    elif all(isinstance(v, list) for v in vals):       # {group: [subjects]}
        counts = {k: len(v) for k, v in data.items()}
    else:
        return (None, None)
    if not counts:
        return (None, None)
    return (min(counts.values()), len(counts))
```

`resolve_metrics` 入口：**当调用方没显式传计数、但传了 groups_file 时，自派生**（显式传入仍优先，保留向后兼容 + CLI 兜底）：

```python
    # 自派生：groups_file 是组计数的唯一真相源。调用方漏传计数（prep 的 #6a bug）时
    # 从 groups_file 派生，使"有分组却 statistics={}"结构上不可能。显式入参优先（兼容/测试）。
    if groups_file and (n_per_group is None or n_groups is None):
        derived_npg, derived_ng = _derive_group_counts(groups_file)
        n_per_group = n_per_group if n_per_group is not None else derived_npg
        n_groups = n_groups if n_groups is not None else derived_ng
```

效果：
- prep_metric_plan **只需传 `groups_file`（它已经传了）** → 漏传计数 bug 结构上消失。**prep 侧可以不改一行**（自派生兜住），但建议 prep 也显式派生传入（双保险 + 可读），见下。
- CLI 的 `--n-per-group`/`--n-groups` 变为可选覆盖；不传则从 `--groups-file`/`--groups-json` 派生。
- 单组/无分组（groups_file=None）→ 派生 None → 正确 skip，**行为不变**。

> **为什么在 resolve 自派生、不在 prep 单点修**：①groups 是分组事实的源，resolve 是 gate 的唯一评估点，在评估点就近读源最内聚；②修 prep 一处只堵了一个调用方，CLI 那个洞还在、未来新调用方还会漏——自派生是 API 级根治，覆盖所有现存+未来调用方；③不引入 runtime 第二评估路径（守 #5 同族教训：plan/runtime 不一致），派生发生在 plan 期 resolve 内、仍是单一评估点。

#### ② 兜底：入口断言，残留不一致响亮失败

自派生后理论上不会再有"有 groups 无计数"，但为防 groups_file 读取失败（派生返回 None）后又静默 skip，在 stats gate 处加一条**可观测日志 + plan note**：当 `groups_file` 非空但派生/传入的计数仍为 None 时，`skip_reason` 里写明这是"**计数不可得**"而非"**n 真的不足**"——两者语义不同，前者是 bug 信号。

```python
# resolve.py stats gate（_resolve_statistics 附近）：区分两类 skip
if groups_file and (n_per_group is None or n_groups is None):
    skip_reason = (
        f"groups_file 提供但组计数不可得（n_per_group={n_per_group}, n_groups={n_groups}）"
        f"——这通常是 groups.json 读取/格式问题，非样本量不足；统计被跳过需排查"
    )
    logger.warning("[resolve] %s groups_file=%s", skip_reason, groups_file)
```

> 不直接 `raise ResolveError`（避免一个 groups.json 读取抖动就整盘失败）——但把"计数不可得的 skip"做成**响亮、可 grep 的信号**，区别于正常的"单组 skip"。这是哑故障→响亮的最小代价版（守 `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`）。

#### ③ prep 侧显式派生（双保险 + 可读性）+ 回归探针

prep_metric_plan 在写完 groups.json、调 resolve 前，**也**从 groups dict 显式派生并传入（与 resolve 自派生互为冗余校验，且让 prep 的 plan 意图自解释）：

```python
# prep_metric_plan_tool.py，写完 groups.json 后：
n_groups_val = n_per_group_val = None
if groups:
    from collections import Counter
    counts = Counter(groups.values())
    if counts:
        n_groups_val, n_per_group_val = len(counts), min(counts.values())
# 传入 resolve_metrics(..., n_per_group=n_per_group_val, n_groups=n_groups_val, ...)
```

### 1.3 #6a red 测试锚点（三层，对应三道防线）

**层 A（ethoinsight，自派生纯函数 + gate — 最干净的根因锚点）**：
```python
def test_derive_group_counts_from_subject_map(tmp_path):
    f = tmp_path/"g.json"; f.write_text(json.dumps({"a.xlsx":"control","b.xlsx":"control","c.xlsx":"treatment"}))
    assert _derive_group_counts(str(f)) == (1, 2)  # min组=1(treatment), 组数=2

def test_derive_group_counts_from_grouplist_shape(tmp_path):
    f = tmp_path/"g.json"; f.write_text(json.dumps({"control":["a","b"],"treatment":["c"]}))
    assert _derive_group_counts(str(f)) == (1, 2)

def test_resolve_self_derives_counts_so_stats_not_skipped(tmp_path):
    """red 锚点：只传 groups_file（不传 n）→ resolve 自派生 → statistics 不被 skip。
    修复前 resolve 不自派生 → None → skip。"""
    gf = _write_groups(tmp_path, control=7, treatment=21)
    pm = resolve_metrics(paradigm="epm", columns=<epm cols+aliases>, raw_files=[...28...],
                         workspace_dir=str(tmp_path), groups_file=gf)  # 注意:不传 n_per_group/n_groups
    assert pm.statistics.skip_reason is None, f"统计被误 skip: {pm.statistics.skip_reason}"

def test_resolve_single_group_still_skips(tmp_path):
    """守护：单组 → 正确 skip（不误伤）。"""
    gf = _write_groups(tmp_path, control=7)  # 1 组
    pm = resolve_metrics(..., groups_file=gf)
    assert pm.statistics.skip_reason is not None
```

**层 B（backend，prep_metric_plan 端到端 — 锚定整链不漏）**：
```python
def test_prep_metric_plan_stats_not_skipped_with_valid_groups(tmp_path, ...):
    """red 锚点：prep(groups=7control+21treatment) → plan.statistics.skip_reason is None。
    修复前 skip_reason 含 'n_per_group=None'。用 importlib 加载 worktree 源。"""
    groups = {f"/mnt/user-data/uploads/s{i}.xlsx": ("control" if i<7 else "treatment") for i in range(28)}
    prep_metric_plan_tool.func(runtime, paradigm="epm", uploaded_files=[...], groups=groups, ...)
    plan = json.loads((real_ws/"plan_metrics.json").read_text())
    assert plan["statistics"]["skip_reason"] is None
```

**层 C（兜底信号守护）**：groups_file 指向坏 JSON → skip_reason 含"组计数不可得"字样（响亮信号存在），而非沉默等同正常 skip。

> 回退验证：删 resolve 自派生 → 层 A 的 `test_resolve_self_derives...` + 层 B 应红（skip_reason 含 None）；删 prep 显式派生但留 resolve 自派生 → 层 B 仍绿（证自派生是真兜底，不是靠 prep）。

### 1.4 #6a 验证 + 红线
- `cd packages/ethoinsight && pytest tests/`（层 A）；`cd packages/agent/backend && PYTHONPATH=. pytest tests/`（层 B/C）。
- 改 `prep_metric_plan_tool.py` 后裸导入 `import app.gateway` + `make_lead_agent`；改 ethoinsight 后 `python -c "import ethoinsight"`。
- **不动** run_metric_plan Step7 `skip_reason is None` 判据 / catalog when 条件 / gate 比较逻辑——都对，#6a 只让 skip_reason 在 plan 期被正确计算。
- **不破 CLI**：自派生是"调用方没传才派生"，CLI 显式传 `--n-per-group` 时仍优先用显式值（向后兼容）。
- 单组/无分组行为不变。

### 1.5 #6a 端到端验收
dogfood 复跑 EPM：handoff `statistics` **非空**（groupwise_compare 结果），statistical_validity=ok 名副其实，data-analyst 有真统计可解读。

---

## 2. #6b（附带，seal 健壮性）：data-analyst 在 statistics 缺失时手算螺旋 → 忘 seal

### 2.1 现象与定位
- gateway.log thread 9b1bc49b trace 25f465cd：data-analyst 4 条 AI message（msg#4 单条 59s 巨型推理），**未撞 max_turns=12**，模型自行 "completed"——最后一轮纯文本/思考、**没调 `seal_data_analyst_handoff`**。seal-resume（补1轮）也没救回。
- frontend thinking 实录根因：`statistics={}` → data-analyst 判"没有推断统计可解读"→ 掉进**手算补偿螺旋**（手算各组 mean/SD、映射 28 subject、debug mean≠手算、逐 subject 离群检测），到 "Parameter Audit" 消息结束、忘 seal。
- 这是 `feedback_subagent_seal_deadlock_is_prompt_not_budget` 同款（把"写完分析"当"完成"而非调 seal），**由 statistics 缺失触发**。

### 2.2 修复方案（最小 prompt 加固，#6a 修好后这是纵深防御）

data-analyst prompt（`subagents/builtins/data_analyst.py` 的 fast-fail / 工作流段）加一条：当 `statistics` 为空 `{}`（无推断检验结果）时——**不要手算补偿组间统计**；走描述性 `partial` 路径，在 key_findings 写明"仅描述性、缺推断统计"，**立即调 `seal_data_analyst_handoff`**。正面措辞（deepseek）：

```
- **statistics 为空时**：若 handoff_code_executor.json 的 statistics 字段为空 {}，说明本次无推断统计结果。
  此时给出**描述性**判读（各组均值/中位数/离群观察），在 key_findings 标注"仅描述性分析、未做推断检验"，
  status="partial"，并**立即调用 seal_data_analyst_handoff 落库**。不要尝试手工重算组间 t 检验 / 效应量 /
  SD——手算既不可靠又会耗尽推理预算导致漏调 seal。封存是终止动作，分析写完必须调 seal 工具。
```

> **为什么 #6a 修好后还要 #6b（用户已裁决：本批做，但定位为纵深防御）**：#6a 让正常路径有统计（代码根治，是主防线）；但 statistics 仍可能在其他**合法**场景为空（单组分析、统计脚本运行失败）。#6b 确保 data-analyst 在那些场景也**有界且必 seal**，不再手算螺旋。这是 seal 健壮性的通用加固，**不是本 bug 的主修复**——主修复是 #6a 的代码根治。#6b 单独不足以治本（只在 prompt 层、有概率复发），但作为兜底层有价值：#6a 堵住"统计为什么缺失"，#6b 堵住"万一缺失了 data-analyst 也别崩"。

### 2.3 #6b 测试
prompt 文案守护（断言关键句存在，防 sync paraphrase 削弱，守 `feedback_sync_protected_file_paraphrase_merge_weakens_constitution`）：
```python
def test_data_analyst_prompt_has_empty_statistics_seal_rule():
    text = <render data_analyst prompt>
    assert "statistics 为空" in text or "statistics 字段为空" in text
    assert "立即调用 seal_data_analyst_handoff" in text or "立即" in text and "seal" in text
    assert "不要" in text and ("手算" in text or "手工重算" in text)
```

---

## 3. 实施顺序、验证、红线

### 3.1 PR 拆分
- **PR-A（#6a，根，治本）**：prep_metric_plan 派生并传组计数 + ethoinsight/backend 双层 red 锚点。这是让 dogfood 通过的关键。
- **PR-B（#6b，可选同批 / 可独立）**：data-analyst prompt 加固 + 文案守护测试。

### 3.2 每个 PR 必跑
- `packages/ethoinsight`：`pytest tests/`（#6a 层 A）。
- `packages/agent/backend`：`PYTHONPATH=. pytest tests/`（#6a 层 B / #6b）。**已知基线债 ~39 failed**（origin/dev `05090c55` 同 venv 复现，守 `feedback_known_full_suite_test_pollution`），别归因本次——用 origin/dev detached worktree 同 venv 对照坐实。
- **改 `prep_metric_plan_tool.py` 后裸导入两入口**：
  ```bash
  cd packages/agent/backend
  PYTHONPATH=. python -c "import app.gateway"
  PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
  ```
- **改 `resolve` 调用 / ethoinsight 后**：`cd packages/ethoinsight && python -c "import ethoinsight"`。

### 3.3 红→绿基线实证
- #6a：在 origin/dev `05090c55` 跑层 A/B 新锚点 → 确认**真红**（skip_reason 含 `n_per_group=None`），修复后转绿；单组守护测试改前改后**都绿**（不误伤单组 skip）。

### 3.4 红线（守 memory 纪律）
- **不动** resolve gate 逻辑 / catalog when 条件 / run_metric_plan Step7 `skip_reason is None` 判据——它们都对，#6a 只补 prep_metric_plan 漏传的入参。
- **SSOT**：组计数派生在 prep_metric_plan 一处（它是 groups 的源），不在 run_metric_plan 再造第二评估路径。
- **单组/无分组**：`groups=None` → 计数 None → 统计正确 skip，不受影响。
- **受保护文件**（data_analyst prompt）surgical + 正面措辞。

## 4. dogfood 验收（修复后复跑）
EPM `Raw data-EPM-Xuhui-28`：① handoff `statistics` 非空（groupwise_compare 结果）② statistical_validity=ok 名副其实 ③ data-analyst 拿真统计、不手算、**正常 seal**（无 "terminated without emitting"）④ 全链路走到 report-writer。

## 5. 关键文件/行号
- #6a 修复点：`packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py:265`（resolve_metrics 调用）+ `:245-261`（groups.json 写入处，派生计数插在其后）
- #6a 根因链：`packages/ethoinsight/ethoinsight/catalog/resolve.py:72-73`（resolve 入参）/ `:309-320`（stats gate 评估 + skip_reason）/ `:1052`（_evaluate_when）；`run_metric_plan_tool.py:214`（Step7 skip_reason 判据，不改）
- #6b：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`（fast-fail / 工作流段）
- 证据：`packages/agent/logs/gateway.log` thread 9b1bc49b trace 25f465cd（seal 失败）；handoff `statistics={}` + groups.json control7/treatment21

## 6. 关联 memory
- `feedback_prep_metric_plan_stats_skip_none_counts_poisons_data_analyst_seal`（本 bug 完整诊断）
- `feedback_subagent_seal_deadlock_is_prompt_not_budget`（#6b 同款 seal 死锁）
- `feedback_run_metric_plan_step8_validation_not_scoped_path_env`（#5，同属 plan/runtime 状态不一致族）
- `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`（为何先修 #6a）
- `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib` / `feedback_known_full_suite_test_pollution_4_tests`（测试加载 + 基线债对照）
