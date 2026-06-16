# Spec: 2026-06-15 EPM dogfood 四 bug 修复（路径 resolve / failed handoff 误判 / 模板判断反 / lead 熔断）

> 日期：2026-06-15
> 起点：EPM dogfood（数据 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`，thread `be25ce13`）
> 诊断会话已逐层坐实四个独立 bug（含 `gateway.log` 行号证据），本 spec 给出可直接实施的修复。
> 实施者：照本 spec 在新 worktree（`git worktree add -b feat/dogfood-0615-fixes <path> dev` 一步建独立分支，**别共享 dev**）逐块执行。

---

## 0. 背景与因果链（先读，理解四 bug 的关系）

这次 dogfood 的真实因果链（`gateway.log` L266/L288-361 坐实）：

```
#2 run_metric_plan 进程内跑脚本不 resolve /mnt 路径
   └─→ 140/140 FileNotFoundError（硬阻塞，是 #3/#4 的源头）
         └─→ code-executor 正确诊断"环境层错误"、诚实 seal status=failed   ✅ subagent 没做错
               └─→ #3 校验器 _check_code_executor_content 无视 status，把 failed 判 "incomplete/unusable"
                     └─→ seal-resume 救不回（合法失败，没东西可救）→ "terminated without emitting handoff"
                           └─→ #4 lead 按规则 #7 自动重派（最多 2 次）→ 每轮 run_metric_plan 再失败 → 烧会话

#1 是独立的、更早的 bug：identify/对齐阶段 lead 把 open/closed 列名误判成 NoZones，建议切模板。
```

**修复优先级与独立性**：
- **#2（治本，最高优先）**：修好后 140 指标能真算出来，#3/#4 链根本不触发。
- **#3（治本，独立必修）**：即使 #2 修好，任何真实的环境/数据失败仍会再次触发"failed handoff 被误判 → 死循环"。这是潜伏雷，独立于 #2 必须修。
- **#1（prompt 缺口）**：模板判断反，与 #2/#3 正交。
- **#4（lead 熔断，附带）**：#2/#3 修好后基本不触发；本 spec 只做最小加固，不过度设计。

四块改动**互相独立**，可分别实施、分别测试、分别 PR（建议 #2+#3 一个 PR，#1 一个 PR，#4 并入 #3 PR 或独立）。

---

## 1. Bug #2（最高优先）：run_metric_plan 进程内跑脚本不 resolve `/mnt` 虚拟路径

### 1.1 根因（已坐实）

- `run_metric_plan`（S4 新工具，`tools/builtins/run_metric_plan_tool.py`）改成**进程内** ProcessPoolExecutor 跑 compute 脚本，**无 bash sandbox 的 mount**。
- S4 的 env 隔离设计本身正确：worker 经 `_worker_init` 设 `DEERFLOW_PATH_*` env，意图让**脚本内** `resolve_sandbox_path` 把 `/mnt/...` 翻成真实路径。
- **但 compute 脚本根本不调 `resolve_sandbox_path`**：`compute_*.main()` 直接 `parse_trajectory(args.input)` + `save_output_json(args.output)`，两个 helper 都不 resolve `/mnt`。原样喂进去 → `FileNotFoundError: '/mnt/user-data/uploads/...'`。
- 为何旧 bash code-executor 没炸：bash sandbox **mount** 了 `/mnt/user-data/uploads`→真实目录，沙箱内 virtual==real。S4 搬进程内（无 mount），没让单文件 `--input`/`--output` 路径 resolve。
- 唯一已接 resolve 的是 stats 路径的 `read_inputs_json`(`--inputs`)/`read_groups_json`(`--groups`)（`_cli.py:147/165`）；**单文件 `--input`/`--output` 从未 resolve**（旧流程不需要）。

**规模**：41 个 compute 脚本 `parse_trajectory(args.input)` 裸路径 + 44 个 `save_output_json(args.output)` 裸路径——**全范式通病，非 EPM 独有**。

**实测证据**（诊断会话，主仓 venv 直接跑）：
- 不 resolve 时 `resolve('epm', cols, raw, ev19_template='PlusMaze-FewZones', column_aliases={'open':'open_arms','closed':'closed_arms'})` 能生成 plan，但脚本读 `/mnt/...` 路径 → FileNotFoundError。
- `resolve_sandbox_path` 设 `DEERFLOW_PATH_*` 后能把 `/mnt/user-data/uploads/x.xlsx` 翻成真实路径。

### 1.2 修复方案：在 I/O sink 收口 resolve（单点，覆盖 41+44 脚本）

**决策：在 I/O 边界（`parse_trajectory` 入口 + `save_output_json` 入口）resolve，不在每个脚本手加，也不在 argparse 层加。**

理由：
- I/O sink 是真正的 SSOT——所有读轨迹/写 JSON 的脚本都过这两个函数，无论路径由哪个 arg 携带（`--input` / `--inputs[0]` / plot 的 `resolve_per_subject_input`）。
- 与现有 `read_inputs_json`/`read_groups_json`（已在 I/O 边界 resolve）**对称**，是同一处方的补全。
- `resolve_sandbox_path` 是 **fail-safe 幂等**的（非 `/mnt` 原样返回、`/mnt` 匹配不到 env 也原样返回），所以在 sink 处 resolve 对所有现有调用方（喂真实路径的测试、bash 流程喂已 mount 路径）**零行为变化**，只对"进程内喂 `/mnt` + 设了 env"这一新场景生效。

#### 改动 A：`save_output_json` 入口 resolve（`packages/ethoinsight/ethoinsight/scripts/_cli.py`）

`save_output_json` 已在 `_cli.py`，`resolve_sandbox_path` 同文件，直接改：

```python
def save_output_json(path: str | Path, data: dict[str, Any]) -> None:
    """..."""
    p = Path(resolve_sandbox_path(path))   # <-- 新增：resolve /mnt 虚拟路径（fail-safe 幂等）
    p.parent.mkdir(parents=True, exist_ok=True)
    # ...（其余不变）
```

#### 改动 B：`parse_trajectory` 入口 resolve（`packages/ethoinsight/ethoinsight/parse/_core.py`）

`parse_trajectory` 有两条路径（xlsx 走 `_parse_path_and_sheet`；txt 又 `Path(file_path)` 重开 + `parse_header(file_path)`），且 `_parse_path_and_sheet` 是 `parse_header`/`parse_trajectory` 共用的单一入口（`_core.py:122/258`）。

**在 `_parse_path_and_sheet` 顶部 resolve**——一处覆盖 xlsx + txt + header 三条读路径：

```python
def _parse_path_and_sheet(file_path: str) -> tuple[Path, str | int]:
    """..."""
    # resolve /mnt 虚拟沙箱路径（进程内执行无 mount 时必需；fail-safe 幂等，
    # 对真实路径/bash-mounted 路径零影响）。注意先 resolve 再切 ::sheet。
    from ethoinsight.scripts._cli import resolve_sandbox_path  # 惰性 import 避免 parse→scripts 层级倒置环
    if "::" in file_path:
        path_str, sheet = file_path.rsplit("::", 1)
        return Path(resolve_sandbox_path(path_str)), sheet
    return Path(resolve_sandbox_path(file_path)), 0
```

> ⚠️ **层级注意**：`parse/` 是比 `scripts/` 更底层的模块，正常 `scripts` 依赖 `parse` 不反向。这里用**函数体内惰性 import** `resolve_sandbox_path`，避免顶层 `parse → scripts._cli` 倒置 import 在某些导入顺序下成环（守 CLAUDE.md「所有可能成环的 import 放函数体惰性」纪律）。实施后必须验证 `python -c "import ethoinsight.parse"` 与 `python -c "import ethoinsight.scripts._cli"` 两入口都 0 退出。

> **替代方案（若惰性 import 仍嫌别扭）**：把 `resolve_sandbox_path` + `_KNOWN_SANDBOX_PREFIXES` + `_sandbox_env_key_for_prefix` 下沉到一个更底层的中性模块（如 `ethoinsight/_sandbox_paths.py`），`parse/_core.py` 与 `scripts/_cli.py` 都从它 import。**本 spec 默认走惰性 import（改动小、风险低）；下沉方案留作 reviewer 判断**——若选下沉，`scripts/_cli.py` 的 `resolve_sandbox_path` 改为 re-export 保持向后兼容（已有 `test_resolve_sandbox_path.py` 不破）。

### 1.3 #2 的 red 测试锚点

现有 `test_run_metric_plan.py` 与 `test_resolve_sandbox_path.py` 都没覆盖"进程内喂 `/mnt` 路径真读文件"。新增 red 锚点（放 `packages/ethoinsight/tests/test_cli_resolves_sandbox_paths.py`）：

```python
"""2026-06-15 spec #2: compute 脚本经 I/O sink resolve /mnt 虚拟路径。

red 锚点：修复前 parse_trajectory('/mnt/user-data/uploads/x.xlsx') + 设 DEERFLOW_PATH_USER_DATA_UPLOADS
直接 FileNotFoundError（脚本拿原样 /mnt 路径）；修复后经 _parse_path_and_sheet resolve 能读到真实文件。
"""
import os
from pathlib import Path
import pytest
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import save_output_json


def test_parse_trajectory_resolves_mnt_path(tmp_path, monkeypatch, <一个真实 EV19 xlsx fixture>):
    # 把真实文件放到 tmp 下，模拟 sandbox 的真实 uploads 目录
    real_uploads = tmp_path / "real_uploads"
    real_uploads.mkdir()
    real_file = real_uploads / "x.xlsx"
    real_file.write_bytes(Path(<fixture>).read_bytes())
    # 设 DEERFLOW_PATH_*：/mnt/user-data/uploads -> real_uploads
    monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_UPLOADS", str(real_uploads))
    df = parse_trajectory("/mnt/user-data/uploads/x.xlsx")   # red: 修复前 FileNotFoundError
    assert len(df) > 0


def test_save_output_json_resolves_mnt_path(tmp_path, monkeypatch):
    real_ws = tmp_path / "real_workspace"
    real_ws.mkdir()
    monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", str(real_ws))
    save_output_json("/mnt/user-data/workspace/m_test.json", {"value": 1.0})   # red: 修复前写到不存在的 /mnt
    assert (real_ws / "m_test.json").exists()


def test_resolve_is_idempotent_and_failsafe(tmp_path):
    # 真实路径原样、/mnt 无 env 原样（不引入新失败）——保证对现有调用方零影响
    out = tmp_path / "m.json"
    save_output_json(out, {"value": 2.0})   # 真实路径，无 env：照常写
    assert out.exists()
```

> `DEERFLOW_PATH_*` 的 env key 由 `_sandbox_env_key_for_prefix` 生成（含 `MNT_` 段，已实算确认）：`/mnt/user-data/uploads` → `DEERFLOW_PATH_MNT_USER_DATA_UPLOADS`；`/mnt/user-data/workspace` → `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE`。
> fixture 用真实 EV19 xlsx：可复用 `packages/ethoinsight/tests/` 下已有的 EPM/OFT 测试数据；若无，从 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28` 拷一个小的进 tests fixtures（注意文件名多空格，测试里别手敲，用 glob）。

### 1.4 #2 端到端验收（dogfood 复跑）

实施后用 dogfood 数据复跑 EPM：`run_metric_plan` 应 `status=completed n_completed=140 n_failed=0`，5 指标算出真值（诊断会话实测单文件：开臂时间比 9.9%/进入 3 次/总进 14 次/进入比 21.4%）。

---

## 2. Bug #3（独立必修）：handoff 完整性校验器无视 `status`，把诚实 failed 误判 incomplete → 无限重派

### 2.1 根因（已坐实）

`subagents/executor.py:122` 的 `_check_code_executor_content`（及同类 data-analyst/chart-maker/report-writer 的 content check）**只看核心字段是否非空，完全无视 `status` 字段**：

```python
def _check_code_executor_content(d: dict) -> str | None:
    present = [f for f in _CODE_EXECUTOR_METRICS_FIELDS if d.get(f)]
    if not present:
        return "metrics data is empty (none of metrics_summary/metrics/metrics_results populated)"
    ...
```

当 code-executor 因环境错误**正确地** seal `status=failed`（failed handoff 本就没有指标数据可填——那是失败的定义），校验器却判 "incomplete/unusable/re-dispatch"。`gateway.log` L296/350 坐实：错误信息含 "terminated without emitting handoff"，触发 lead 规则 #7 自动重派 → 每轮 run_metric_plan 再失败 → 死循环。

**四个 handoff schema 都有 `status: Literal["completed", "partial", "failed"]`**（`handoff_schemas.py:411/523/611/684`），所以 status 三态感知有可靠字段依托。

这是与 memory `feedback_dataanalyst_reportwriter_handoff_status_missing_partial`、`feedback_handoff_metrics_field_divergence_mislabels_failed` **同一类病的第 N 次复发**——handoff 校验不分 status 三态，在不同校验点反复中招。

### 2.2 修复方案：content check 加 status 短路

**在每个 content check 函数开头加 status 短路**：`status=failed` 时不要求核心字段非空（failed 本就允许空核心字段）。

最干净的做法：在 `_validate_handoff_emitted` 调 content check **之前**统一短路（一处治四个 check），而非每个 check 函数各加：

定位 `_validate_handoff_emitted`（`executor.py`）解析出 `parsed` dict 后、调 `content_check(parsed)`（**当前 `executor.py:239`**）之前，插入：

```python
            # status=failed 的 handoff 合法地没有核心数据（失败的定义就是没产出）。
            # 不能用"核心字段非空"判据卡它——否则诚实的 failed handoff 被误判 incomplete，
            # seal-resume 救不回 → "terminated without emitting" → lead 无限重派（2026-06-15 EPM
            # dogfood 死循环根因；gateway.log L288-361）。failed 只要文件存在 + 是合法 JSON 即放行。
            if isinstance(parsed, dict) and parsed.get("status") == "failed":
                return None
            missing = content_check(parsed) if isinstance(parsed, dict) else "handoff is not a JSON object"
            ...
```

> **为什么放在 `_validate_handoff_emitted` 里、不放各 content check 函数**：四个 subagent 的 failed 语义一致（failed=没产出），一处短路覆盖全部，避免四个函数各写一遍 status 判断（SSOT）。content check 函数保持「只管核心字段非空」单一职责。
> **partial 不在本次放宽范围**：partial 仍要求核心字段非空（partial=部分成功，应有部分数据）。只短路 failed。若后续发现 partial 也有合法空核心场景，另开。

### 2.3 #3 的 red 测试锚点

放 `packages/agent/backend/tests/test_handoff_content_validation.py`（若已存在则追加）：

```python
def test_failed_status_handoff_passes_even_with_empty_metrics(tmp_path):
    """red 锚点：status=failed 的 code-executor handoff（核心字段空）必须放行，
    不能判 incomplete 触发重派（2026-06-15 EPM dogfood 死循环根因）。"""
    from deerflow.subagents.executor import _validate_handoff_emitted
    import json
    ws = tmp_path
    (ws / "handoff_code_executor.json").write_text(json.dumps({
        "status": "failed",
        "summary": "环境层错误，140 指标全失败",
        "metrics_summary": {},   # 失败本就该空
        "errors": ["FileNotFoundError ..."],
    }), encoding="utf-8")
    # 修复前：返回 "core content is incomplete..." → 触发重派死循环
    # 修复后：返回 None（放行）
    assert _validate_handoff_emitted("code-executor", str(ws)) is None


def test_completed_status_handoff_still_requires_metrics(tmp_path):
    """守护：status=completed 但核心字段空，仍须判 incomplete（不能因放宽 failed 误伤 completed）。"""
    from deerflow.subagents.executor import _validate_handoff_emitted
    import json
    ws = tmp_path
    (ws / "handoff_code_executor.json").write_text(json.dumps({
        "status": "completed",
        "metrics_summary": {},   # completed 却空 = 真残缺
    }), encoding="utf-8")
    result = _validate_handoff_emitted("code-executor", str(ws))
    assert result is not None and "incomplete" in result


def test_failed_data_analyst_handoff_passes_empty_key_findings(tmp_path):
    """四个 subagent 一致：failed data-analyst handoff（key_findings 空）也放行。"""
    # 同上结构，subagent="data-analyst"，status=failed，key_findings=[]
    ...
```

### 2.4 #3 与 auto-seal 的关系（核实，勿误改）

`code-executor` 在 `_AUTO_SEALABLE`（`executor.py:285`）里，但本次只改 content 校验的 status 短路，**不动 auto-seal 逻辑**。auto-seal 是「漏调 seal 时从磁盘产物机械重建」，与「校验已 seal 的 failed handoff」正交。run_metric_plan 自己 seal（`sealed_by="run_plan"`），不走 auto-seal 路径。

---

## 3. Bug #1（prompt 缺口）：lead 把列名非标准（open/closed）误判 NoZones、建议切模板

### 3.1 根因（已坐实）

数据列 `open`(在开臂=1,占9.9%)/`closed`(在闭臂=1,占70.4%)/`result_1`(恒1=全场默认区,无用)。lead 把"列名非标准"误读成"NoZones 数据结构"，反复建议切模板 FewZones→NoZones。

**这是错的且更糟**——EV19 模板变体由"录制时划没划分析区"定义，不看列名：
| 模板 | `zone_template` | 含义 | 你的数据 |
|---|---|---|---|
| FewZones | `Closed and open arms` | 划了开/闭臂 → 数据**有**归属列 | ✅ 正是这个 |
| AllZones | `Closed-, open arms, head dip zone` | 多 head dip 区 | ❌ |
| NoZones | `No zone template` | **完全没划区** → 数据**只有 x/y 轨迹、无任何归属列** | ❌ |

`open`/`closed` 是真·归属列 → 定义上就是 FewZones，列名非默认 `in_zone_open_arms_*` 而已。正解=列语义对齐（`open`→`open_arms`、`closed`→`closed_arms`、`result_1` 忽略），**不切模板**。切 NoZones 反而丢掉全部 5 个 zone 指标。

**链路全对，缺的是 prompt/skill 的一条正面铁律**（叙事黑洞，同 memory `feedback_code_executor_skill_writefile_contradicts_seal_tool`）：
- ✅ catalog(`epm.yaml`)/resolve 引擎/inspect(`assess_column_confidence`)/`prep_metric_plan` 的 columns_missing hint/column-confirmation skill **内容全对**（实测对齐后 5 指标全算真值）。
- ❌ **缺口**：lead prompt（`prompt.py:466` 只管"没选模板要重问"）+ column-confirmation skill **都没有铁律**「模板变体看划区与否、不看列名；数据里有归属列就不是 NoZones；列名非标准走列对齐，绝不因列名改判模板」。

### 3.2 修复方案：lead prompt + column-confirmation skill 各加正面铁律

> ⚠️ `prompt.py` 是**受保护文件**（Noldus 定制），surgical 编辑，**用正面指令**（deepseek 正面提示原则，CLAUDE.md 第 6 条）。

#### 改动 A：lead prompt（`agents/lead_agent/prompt.py`，在 466 行「自定义分析区列未对齐」条目附近追加）

新增一条铁律（正面措辞）：

```
- **模板变体由"录制时是否划分析区"决定，不由列名决定**：EV19 模板变体（AllZones / FewZones / NoZones）的区别是录制时划了哪些分析区——FewZones=划了开臂+闭臂（数据含开/闭臂归属列）、AllZones=另含 head dip 区、NoZones=完全没划任何分析区（数据只有 x/y 轨迹、无任何区归属列）。**只要数据里存在区归属列（哪怕列名非标准，如 open/closed/中心区/zone_A），该实验就属于划了区的变体（Few/AllZones），应走列语义对齐把这些列对齐到 catalog 概念（参考 ethoinsight-column-confirmation skill），保持已选模板不变。** 列名非标准是"对齐"问题，不是"换模板"问题。仅当数据确实只有轨迹列、没有任何区归属列时，才考虑 NoZones。
```

#### 改动 B：column-confirmation skill（`skills/custom/ethoinsight-column-confirmation/SKILL.md`，在「触发条件」或「方法论」段加一条防御）

```
## 重要：列名非标准 ≠ 换模板

当 inspect 报告未识别的自定义分析区列（如 open/closed/中心区）时，这是**列对齐**任务，
不是重新判断模板的信号。模板变体由"录制时是否划分析区"决定：数据里有区归属列就属于划了区的
变体（Few/AllZones）。**绝不能因为列名非标准（不是 in_zone_open_arms_*）就反过来建议把模板
改成 NoZones**——NoZones 指数据完全没有任何区归属列（只有 x/y 轨迹）。把已识别的归属列对齐到
catalog 概念即可，保持模板不变。
```

### 3.3 #1 的 red 测试锚点

prompt 铁律难直接单测；用**断言文案存在**的轻量守护（防回归被 sync 洗掉，守 memory `feedback_sync_protected_file_paraphrase_merge_weakens_constitution`）：

```python
def test_lead_prompt_has_template_by_zone_recording_not_column_name_rule():
    """守护 #1 铁律不被 sync paraphrase 削弱：模板变体看划区与否、不看列名。"""
    from deerflow.agents.lead_agent.prompt import <prompt 常量或 builder>
    text = <render prompt>
    assert "由\"录制时是否划分析区\"决定" in text or "不由列名决定" in text
    # 关键语义：有归属列 → Few/AllZones、走对齐不换模板
```

> 若 prompt 是 `@lru_cache` builder（`prompt.py:861`），渲染一次取文本断言即可。

### 3.4 #1 待坐实的加重因素（实施者顺手查，查不到不阻塞）

用户反映**最开始 identify 阶段**（还没看列、只看文件名/路径）就"一直强调 no zones"，疑似 `identify_ev19_template` 的 candidates 排序/措辞预先植入"NoZones"印象。实施 #1 时 **grep 一下 `identify_ev19_template_tool.py` 返回的 candidates 是否对 NoZones 有排序/措辞偏向**；若有，在 spec 外记一个 follow-up（本次不强行改，避免根因未隔离前叠加）。

---

## 4. Bug #4（附带加固）：lead 缺"同一环境错误连续失败 → 停手如实上报"熔断

### 4.1 现状与定位

lead 规则 #7（`prompt.py:257-272`）：error 含 "terminated without emitting" → 自动重派，**最多 2 次**，2 次后向用户报告。所以 #4 **已有 2 次上限兜底**——不是无限循环到天荒地老，是"2 次昂贵重派后才停"。`gateway.log` 看到的多轮是这 2 次重派 + seal-resume。

**#4 的真问题**：重派的根因是 #3 校验器误判喂给 lead 的错误指令（"re-dispatch"）。**#2+#3 修好后，failed handoff 正常放行，根本不会进 "terminated without emitting" 路径 → #4 不触发**。

### 4.2 修复方案（最小，不过度设计）

按 memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`：根因（#2/#3）未隔离前别叠兜底。**本次只做一处最小加固**——在规则 #7 补一句区分"漏 seal"与"诚实失败"：

在 `prompt.py` 规则 #7 末尾追加：

```
   注意区分两种失败：(a) "terminated without emitting" 且 subagent 从未产出任何结果 →
   按上述重派；(b) 若 subagent 的 handoff/上报明确是 status=failed 且给出了具体失败原因
   （如环境层文件访问错误、范式脚本缺失），这是**诚实的失败上报**，不是漏调 seal——
   此时不要机械重派，应把失败原因如实转达用户并停止（同规则 #6）。
```

> 这是**纯 prompt 防御**，是 #3 治本之外的纵深防御附带。不新增中间件、不加熔断计数器（#3 修好后无触发场景，加了也是死代码）。**若 reviewer 认为 #2+#3 已足够、#4 prompt 改动多余，可省略 #4**——记录在 PR 说明即可。

### 4.3 #4 测试

无独立单测（纯 prompt 文案）。并入 #1 的 prompt 文案守护测试，断言区分句存在即可（可选）。

---

## 5. 实施顺序、验证、红线

### 5.1 建议实施 + PR 拆分
1. **PR-A（#2 + #3，最高优先，治本）**：路径 resolve + failed handoff 短路。两者一起是因为它们共同决定 dogfood 能否跑通端到端。
2. **PR-B（#1 + 可选 #4）**：模板铁律 + lead 熔断 prompt。受保护文件 surgical。

### 5.2 每个 PR 必跑
- `packages/ethoinsight`：`pytest tests/`（#2 在此）。
- `packages/agent/backend`：`make test`（#3/#1/#4 在此）。**注意已知基线债 ~24 failed**（memory `feedback_known_full_suite_test_pollution_4_tests` + `test_subagent_executor` namespace 债），用 parent checkout 同 venv 对照坐实，别归因本次。
- **改 `subagents/executor.py`（#3）后必裸导入两生产入口**（CLAUDE.md 铁律）：
  ```bash
  cd packages/agent/backend
  PYTHONPATH=. python -c "import app.gateway"
  PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
  ```
- **改 `parse/_core.py`（#2）后验证无 import 环**：
  ```bash
  cd packages/ethoinsight
  python -c "import ethoinsight.parse"
  python -c "import ethoinsight.scripts._cli"
  python -c "import ethoinsight"   # 顶层 __init__ 串联 parse/metrics/...
  ```

### 5.3 红→绿基线证明
- #2：在 parent commit 上跑新增 `test_cli_resolves_sandbox_paths.py`，确认**真红**（FileNotFoundError），再在修复后转绿。
- #3：在 parent 上跑 `test_failed_status_handoff_passes_even_with_empty_metrics`，确认**真红**（返回 incomplete 串），修复后转绿；同时 `test_completed_status_handoff_still_requires_metrics` 在 parent 与修复后**都绿**（守护没误伤 completed）。

### 5.4 端到端验收（修复后 dogfood 复跑）
用 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28` 复跑 EPM：
- identify → 模板确认（#1：lead 不再建议切 NoZones，正常走 FewZones + 列对齐）
- 列对齐 `open`→`open_arms` / `closed`→`closed_arms` / `result_1` 忽略 → `prep_metric_plan` 成功
- `run_metric_plan`：**`status=completed n_completed=140 n_failed=0`**（#2 修好）
- 不再出现 "terminated without emitting → 重派" 死循环（#3 修好）
- 全链路走到 data-analyst / report-writer

### 5.5 红线（守 memory 既有纪律）
- **不动 catalog（`epm.yaml`）/resolve 引擎/column-confirmation skill 的对齐逻辑**——它们全对，#1 只补 prompt/skill 的防御铁律。
- **SSOT**：#2 在 I/O sink 单点 resolve，别每个脚本手加（41+44 处）；#3 在 `_validate_handoff_emitted` 单点短路，别四个 content check 各写。
- **受保护文件**（`prompt.py`）surgical + 正面措辞。
- **`resolve_sandbox_path` fail-safe 幂等**是 #2 安全的前提——确认没有任何调用方依赖"不 resolve"的旧行为（已核实：read_inputs_json/read_groups_json 早已 resolve，sink resolve 与之一致）。

---

## 6. 关键文件/行号速查
- #2：`packages/ethoinsight/ethoinsight/scripts/_cli.py`（`save_output_json:111` / `resolve_sandbox_path:77`）、`packages/ethoinsight/ethoinsight/parse/_core.py`（`_parse_path_and_sheet:76` / `parse_trajectory:249`）
- #3：`packages/agent/backend/packages/harness/deerflow/subagents/executor.py`（`_check_code_executor_content:122` / `_validate_handoff_emitted` content check 调用点 `content_check`取得 `:223`、调用 `:239`）、`handoff_schemas.py`（status Literal :411/523/611/684）
- #1：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:466`、`skills/custom/ethoinsight-column-confirmation/SKILL.md`、`references/by-template/PlusMaze.md`（模板语义权威来源，勿改，仅参照）
- #4：`prompt.py:257-272`（规则 #7）
- 证据：`packages/agent/logs/gateway.log` L266（#2 全败）/ L288-361（#3 死循环）

## 7. 关联 memory（实施者先读）
- `feedback_run_metric_plan_inprocess_scripts_dont_resolve_sandbox_path`（#2 完整诊断）
- `feedback_handoff_content_validator_rejects_failed_status_infinite_redispatch`（#3 完整诊断）
- `feedback_lead_inverts_fewzones_vs_nozones_by_column_name`（#1 完整诊断）
- `feedback_dataanalyst_reportwriter_handoff_status_missing_partial` + `feedback_handoff_metrics_field_divergence_mislabels_failed`（#3 同类病史）
- `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`（#4 为何只做最小加固）
- `feedback_dev_prod_behavior_alignment`（#2 同源警示：执行模型从 bash-mount 换 in-process-env 是行为变更）
- `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`（#2/#3 裸导入验证）
