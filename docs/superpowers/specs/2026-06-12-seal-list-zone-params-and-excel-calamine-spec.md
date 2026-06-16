# Spec: 修 seal schema bug（list 型 zone 参数）+ Excel 解析换 calamine 引擎

> 日期：2026-06-12
> 执行方式：新开 worktree 基于 `dev` 实施，A+B 同一 PR（commit 分两笔）
> 来源：dogfood thread `38be2753` seal 失败实证 + 用户性能反馈 + Fable-5 代码评审

---

## 0. 背景与根因（实证，非推测）

Dogfood thread `38be2753`（EPM，28 文件 4 组）跑到 code-executor 末尾**140 个指标全算完、`validate_catalog` 通过**，却被判 `子任务失败`。

**真根因**（gateway.log `trace=087f87a7` line 956-1225）：auto-seal 兜底**正常触发**，但构造 `CodeExecutorHandoff` 时 Pydantic 抛 **84 个 ValidationError**：

```
metrics_summary.XX.open_arm_entry_count.parameters_used.open_arm_zones.str
  Input should be a valid string [input_value=['open'], input_type=list]
```

`MetricStat.parameters_used`（[handoff_schemas.py:109](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py)）类型是标量 `dict[str, float|int|str|None]`，但 EPM zone-aware 指标如实产出 **list 值** `open_arm_zones=['open']`、`closed_arm_zones=['closed']`（28 个 list 参数 × 3 类型分支 ≈ 84 错误）→ 完整成功的 run 被误标 FAILED。日志那句 WARNING「LLM 忘了调 seal 工具」是**误导**，真因是其上方的 ValidationError。

这是「**过窄的 handoff schema 拒绝如实、完整数据**」bug 类（同 memory `feedback_handoff_metrics_field_divergence_mislabels_failed` / `feedback_dataanalyst_reportwriter_handoff_status_missing_partial`）。schema 为旧标量参数设计，列语义对齐工作引入 list 型 zone 参数后从未更新。

**非 EPM 独有（逐范式核实）**：

| 范式 | zone 参数类型 | 触发？ |
|------|-------------|-------|
| **EPM** | `open_arm_zones`/`closed_arm_zones` → `list[str]` | ✅ 本次实证 |
| **Zero Maze** | `open_zones`/`closed_zones` → `list[str]`（[metrics/zero_maze.py:73,149-150](packages/ethoinsight/ethoinsight/metrics/zero_maze.py)，同款 `emit_result` 透传）| ✅ 会同样炸 |
| OFT | `center_zone` → `str` | ❌ |
| LDB | `light_zone`/`dark_zone` → `str` | ❌ |
| FST/TST | pendulum/velocity → float | ❌ |

→ 采**防御性范围**（治本而非 EPM 局部补丁）。

同次 dogfood 用户反馈两个性能痛点（实测见 Part B）：inspect 每文件 ≈2s（28 文件 ≈57s）；code-executor 第二次 run **≈13 分钟**。

---

## Part A — 修 seal schema bug（红线，必须）

### A1. 抽共享类型别名 `ParamValue`，两个字段共用（治本，非改两处）

> **Fable-5 核心指正**：bug 类的本质是「同一 value-domain 的两份独立声明各自漂移」。不是改两个字段，是消除重复。

在 [handoff_schemas.py](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py) 顶部（import 后、第一个 model 前）定义：

```python
# 单一 value-domain 声明：metric 参数的合法值形状。
# 标量(旧 velocity_threshold=30.0)+ list[str](列语义对齐引入的多列 zone 聚合，如 open_arm_zones=['open'])。
# 两个字段(MetricStat.parameters_used / ParameterAuditFinding.used_value)共用此别名，
# 避免「同一 domain 两份声明独立漂移」——这正是本次 84-error seal 失败的 bug 类。
ParamValue = float | int | str | list[str] | None
```

**A1a** — [:109](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py)：
```python
# 改前
parameters_used: dict[str, float | int | str | None] = Field(
# 改后
parameters_used: dict[str, ParamValue] = Field(
```
docstring 增一句：list 值用于多列 zone 聚合参数（`open_arm_zones=['open']`）。

**A1b** — [:239](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py) `ParameterAuditFinding.used_value`：
```python
# 改前
used_value: float | int | str = Field(
# 改后
used_value: ParamValue = Field(   # 注意：这会同时引入 None 合法，见 A2 对 normalizer 的处理
```

> **为何 used_value 也要放宽（非投机）**：[data_analyst.py:210](packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py) 指示 LLM 把 `parameters_used[参数名]` 的**真实值**填进 `used_value`。`{"open_arm_zones":["open"]}` 在 A1a 落地后即合法，而 producer 被指示逐字拷贝 → 这是**有确定触发条件的潜伏 bug，不是投机**。本次 trace 没炸到这里仅因 run 没走到 data-analyst。
> **Fable 补充的更深理由**：窄 schema 面对 LLM 不是拒绝坏数据而是**洗白**——约束成标量时模型会 stringify 成 `"['open']"`（Python repr）静默通过，比响亮 ValidationError 更糟。schema 必须接受 instruction 能产出的一切。

### A2. `_normalize_audit_finding` 的 `None→""` 处理

`ParamValue` 含 `None`，与 [:289](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py) 现有的 `used_value=None → ""` 映射产生哲学冲突（Fable 指出：一个字段里两种哲学——list 保真透传 vs None 有损归一）。

**决策（保守，遵「根因未隔离前别叠加」）**：**保留** None→"" 映射不动，但在该处加注释说明它现在是**有意保留的 documented 例外**（非 bug）：
```python
# 注：used_value 类型(ParamValue)现含 None，但此处仍主动 None→"" 归一。
# 理由：下游 data-analyst 审计逻辑(prompt data_analyst.py:236 的 used_value>p90×3 数值比对)
# 在概念上不接受 None；prompt 已硬性要求"绝不填 None"(:210)。此归一是该硬约束的兜底，
# 是 documented 例外而非疏漏。删除它会改变一条未坏路径的行为(违"根因未隔离前别叠加")，本次不做。
```
**不**删除映射（Fable 建议可删，但删=改未坏路径行为，本项目纪律优先保守）。executor 实施时若发现删除更干净且有测试覆盖，可在 PR 说明里提请 review，但默认保留。

### A3. 不需改动（已核实下游容忍 list）
- `executor.py:493` `_attempt_auto_seal_from_artifacts` 逐字透传 `params`，无标量假设。
- `validate_catalog` 从不读 `parameters_used`（故 trace 里它通过）。
- ChartMaker/DataAnalyst/ReportWriter handoff 不内嵌 `MetricStat`。
- `observed_distribution: dict[str, float|int]`（:242）是计算统计量非参数值，**不动**。

### A4. Pydantic v2 union 注意事项（Fable 实测确认 pydantic 2.13.4，写进实施注意）
- **smart-union 无 footgun**：`str` 与 `list[str]` 互不坍缩——`"open"` 精确匹配 str 立即返回，`["open"]` 精确匹配 list[str]，`[]` 保持空 list；v2 绝不 list↔str 互转，**成员顺序无关**。**保持默认 smart 模式**。
- **禁止** 给该 union 设 `union_mode="left_to_right"`（Fable 实测：float 在前 + lax 会把 `1→1.0`、`"1"→1.0`、`"3.5"→3.5`，数字字符串和 int 身份全毁）。
- **bool 值域已现场核实无**（不加 `bool` 进 union）。证据：metric 函数签名零 bool 型参数（`_common.py:358` 的 `return_signal` 是内部信号开关不经 `--parameters-json`；`_pendulum.py:141` 的 `is_pendulum` 是输出非输入）；catalog 里的 `true/false` 全是 schema 元数据（`tunable_by_user` 等）不进 `parameters_used`。
  - ⚠️ **但通道不防 bool**：`scripts/_cli.py:212` `parse_parameters` 直接 `json.loads` 不做类型过滤，若 lead 手写 `{"flag": true}` 仍会漏进一个 Python bool。**Fable 实测**：bool 不在 union 时 smart 模式精确匹配全失败 → 走宽松路径被 **float 分支静默接走变 `1.0`**（哑故障）。**处置**：值域既无 bool 就不引入 bool 进 union（否则要管 bool 排序徒增复杂度）；在 spec/代码注释记此边界。若未来真出现 bool 型参数，需把 `bool` 加进 `ParamValue` **且放在 int/float 之前或用 `Strict` 标注**（否则永远轮不到）。
- **`list[int]` 仍非法**（`[1,2]` 全分支 fail）——zone 是列名 list[str]，这是想要的响亮失败，**不放宽到 list[int]**。

### A4b. 放宽前必查消费侧（Fable 唯一可能炸点）
放宽 union **只接受更多、不拒绝任何既有数据**，对存量零破坏。真正风险在**消费侧**：实施时 `grep -rn "used_value" + "parameters_used"` 所有消费者，确认无人假设它是标量（直接做算术 / 拼字符串）。已核实（见 A3）：`executor.py:493` 逐字透传、data-analyst 的数值比对在 prompt 层且有离散参数捷径分流、`validate_catalog` 不读——均不假设标量。实施时再 grep 一遍坐实。

### A5. 测试（TDD，先红后绿，普通断言不用 xfail）

**单元** — [tests/test_metric_stat_and_executor_schema.py](packages/agent/backend/tests/test_metric_stat_and_executor_schema.py) `TestMetricStatParametersUsed` 加：
```python
def test_list_valued_zone_params(self):
    """EPM/Zero Maze 多列 zone 聚合参数是 list[str]，必须合法（thread 38be2753 实证）。"""
    m = MetricStat(mean=None, parameters_used={"open_arm_zones": ["open"], "closed_arm_zones": ["closed"]})
    assert m.parameters_used["open_arm_zones"] == ["open"]
    assert m.parameters_used["closed_arm_zones"] == ["closed"]

def test_scalar_params_still_work(self):
    """放宽 union 不得破坏旧标量参数（smart-union 不坍缩 str↔list 回归）。"""
    m = MetricStat(parameters_used={"velocity_threshold": 30.0, "unit": "mm/s", "n": 25})
    assert m.parameters_used == {"velocity_threshold": 30.0, "unit": "mm/s", "n": 25}
```
新增 `ParameterAuditFinding` list used_value 一例（构造合法 finding，`used_value=["open"]`，断言不抛）。

**端到端**（**最贴 trace 的真实失败路径**）— [tests/test_auto_seal_from_artifacts.py](packages/agent/backend/tests/test_auto_seal_from_artifacts.py) 仿 line 479-516 `test_missing_does_not_block_other_reconstruction`（用真实 executor 经 `importlib.spec_from_file_location` 绕 conftest mock）加一例：
- 写 `plan_metrics.json`（paradigm=epm）+ `groups.json` + `experiment-context.json`
- 写含 `"parameters_used":{"open_arm_zones":["open"]}` 的 `m_*.json`
- 调 `_auto_seal("code-executor", str(ws_path))`，断言 `ok is True` 且 `handoff_code_executor.json` 写成功，`metrics_summary[组][指标]["parameters_used"]["open_arm_zones"]==["open"]`
- **此测试在 A1 改动前必须红（复现 84-error），改动后绿。**

### A6. `extra="allow"` + `default_factory=dict` 的 fail-open（Fable 实测，**记 TODO 本次不改**）
Fable 实测演示（pydantic 2.13.4）：`MetricStat` 是 `extra="allow"`（[:96](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py)）+ `parameters_used` 有 `default_factory=dict`。若 producer 把字段名打错（`parms` 代替 `parameters_used`），模型**零报错通过**，`parameters_used` 拿到空 dict，真实数据被静默隔离进 `model_extra` → 下游读到「合法的空参数」而非异常（哑故障）。同 memory `feedback_handoff_metrics_field_divergence_mislabels_failed` 同源。

**本次不改**（理由：`extra="allow"` 是**有意设计**——docstring [:92-93](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py) 明说「容纳 ethoinsight 新增 metric-level 统计字段如 median/IQR 不破旧 handoff」；改 `forbid` 会破此意图，且横切 10 处 `extra="allow"`，属独立大改，顺手改违「根因未隔离前别叠加」）。**记 TODO 独立评估**：候选处方（Fable）= 承重 schema 用 `extra="forbid"`，或校验后检查 `model_extra` 非空就告警，或加 `status=completed → parameters_used 非空` 的 model_validator 把哑故障换响亮故障。注意：值校验在 `allow` 下仍严格（往 `parameters_used` 塞嵌套 dict 照样报错），fail-open **只在字段名这一层**。

---

## Part B — Excel 引擎换 calamine（性能，同 PR）

### 实测证据（real file `Raw data-EPM-Xuhui-Trial 1.xlsx`, 9301×12, 742KB；best of 3）
| 操作 | openpyxl | calamine | 加速 |
|------|---------|----------|------|
| `pd.read_excel` 全量 | 0.651s | **0.081s** | **8.1×** |
| `parse_header` | 0.73s（且读全文件，见 B3）| — | — |
| `parse_trajectory` | 1.35s | ~0.1s | ~13× |
| 单次 inspect（header+trajectory）| 2.03s → 28 文件 ≈57s | 预估 ≈8s | — |
| 单 compute 冷启 import | 0.83s/process（不变）| — | — |

calamine 与 openpyxl 输出**逐格 `astype(str).equals` 完全一致**（已验证）。EV19 trajectory 数据区**全 float64/int64、零 datetime 列**（已验证：trial_time/recording_time 是秒不是 Excel 日期）→ Fable 警告的「日期轴」对真实数据**不咬**，但仍按下方做语料级差分测试防未来范式。

### B1. 加硬依赖 `python-calamine`
- 进 [packages/ethoinsight/pyproject.toml](packages/ethoinsight/pyproject.toml) 的 `dependencies`（**硬依赖非 optional**——装不上=build 时响亮失败，Fable 指正）。
- pin 版本（Rust dep，minor bump 可能移动 edge-case typing）：写 `python-calamine>=0.x,<0.y` 区间或精确 pin，实施时取当前装上的版本号。`uv sync` 固化进 lockfile。
- **注意环境状态**：上一会话调研时已 `uv pip install python-calamine` 进 `packages/agent/backend/.venv`（用于基准）。新 worktree 是干净环境，按 pyproject 正式装；旧 venv 那个 ad-hoc 安装无 pyproject 声明，可忽略（新环境不继承）。

### B2. 收口引擎选择到单一常量 + 显式传参（**无运行时 fallback**）

> **Fable-5 核心指正，推翻原 fallback 方案**：值得怕的是「calamine 成功但值不同」，而 `try calamine except openpyxl` 对这个**永不触发**（它只在异常时触发，而异常正是想暴露的响亮失败）。fallback 一无所获，还把 engine bug 静默转成无监控的性能回归 + 环境相关行为 + 逃逸的是 openpyxl 异常（误导 debug）+ 坏文件解析两遍。

在 [ethoinsight/parse/_core.py](packages/ethoinsight/ethoinsight/parse/_core.py) 顶部加模块常量：
```python
# Excel 读引擎：calamine(Rust)比默认 openpyxl 快 ~8×，值逐格等价(见 spec 2026-06-12)。
# 显式 pin 引擎名——不依赖 pandas 默认(engine=None 的自动选择上游可能变)。
# 不做运行时 fallback：calamine 失败应响亮 raise(带 path+engine)，而非静默退 openpyxl
# 掩盖 engine bug。openpyxl 仅留作 test 侧 fixture 写入(calamine 只读)。
EXCEL_ENGINE = "calamine"
```
三处 `pd.read_excel(...)`（[:91](packages/ethoinsight/ethoinsight/parse/_core.py) / [:195](packages/ethoinsight/ethoinsight/parse/_core.py) / [:320](packages/ethoinsight/ethoinsight/parse/_core.py)）显式加 `engine=EXCEL_ENGINE`。**不要** try/except 退 openpyxl。

> 若未来需运营逃逸阀，唯一可接受形态是**单一全局 override（env var，单点读、启动 log）由人设**——绝不 per-call 自动切换。本次**不建**，等事故再说。

### B3.（P2 可选，本 PR 可不做）`parse_header` 别读全文件
[:195](packages/ethoinsight/ethoinsight/parse/_core.py) `pd.read_excel(..., nrows=None)` 读整文件，但下一行 `df_header.iloc[0,1]` 就是 header 行数。换 calamine 后全读才 0.08s，收益变小。**列 P2，不阻塞 B1/B2。** 若做：先 `nrows=1` 拿 header_lines 再 `nrows=header_lines` 读。

### B4. 测试（差分/等价，Fable 实测强化）

> **Fable 实测结论（pandas 3.0.3 / python-calamine 0.6.2 / openpyxl 3.1.5）**：把经典分歧轴全测了一遍——int / 整型浮点(42.0) / float / bool / date / datetime / time-of-day / timedelta(duration) / 前导零字符串("0042") / 空 cell / error cell(#DIV/0!) / 超 2^53 大整数 / 浮点精度(0.1+0.2、1e-9)——**双引擎逐格值相等、列 dtype 全一致、浮点 bit 级一致、`assert_frame_equal(check_dtype=True)` 通过**。历史分歧轴（老 calamine 把整型浮点还原成 int 而 openpyxl 保留 float）在 pandas 3.0 **已收口**。大整数精度丢失发生在 xlsx 写入时（Excel 数字本质 f64），与引擎无关。
>
> **`astype(str)` 的原理性盲区**：`datetime64` 列与装 datetime 对象的 object 列，str 形式一模一样（`"2026-01-01 00:00:00"`），`astype(str)` 看不见这种 dtype 分歧，而下游按 dtype 分派的逻辑（`.dt` accessor / 日期算术）会炸。→ **必须用 `assert_frame_equal(check_dtype=True)`**（更严、不更贵，直接换）。我们 trajectory 数据区已核实全 float64/int64 无 datetime，但测试仍按严格 dtype 对拍防未来范式。

在 [packages/ethoinsight/tests/](packages/ethoinsight/tests/) 加 `test_excel_engine_equivalence.py`：
1. **语料级一次性迁移验收**：对 demo-data 下所有 EV19 xlsx + 现有测试 fixture 整批用 calamine 与 openpyxl 各读一遍，`pd.testing.assert_frame_equal(df_cal, df_open, check_dtype=True)`。calamine 不可用时 `pytest.skip`。
2. **合成边界 fixture 进 CI 常驻（版本升级金丝雀）**：用 openpyxl 写 fixture（仿 test_parse.py:474）覆盖 Fable 实测的分歧轴，重点盯**最易漂的三轴**：① 时刻 vs 时长（`[h]:mm` 等非常规 number format 两引擎启发式可能不同——Noldus 导出器格式不在 Fable 控制内，**这轴最该在真实语料上盯**）② 含空洞 int 列 → float64 提升 ③ error cell。calamine 或 pandas 升级引入分歧时此测试变红，而非悄悄漂进生产数据。
3. **parse 套回归**：`pytest tests/ -k parse` 换引擎后全绿、值不变。

> openpyxl 在 ethoinsight 仍是 **test/dev 依赖**（test_parse.py 用它写 fixture + 作对拍基准；calamine 只读不能写）——保留它在 dev-dependencies，不要从环境移除。另：`packages/agent/skills/public/data-analysis/scripts/analyze.py` 直接 import openpyxl（上游公共 skill，与 ethoinsight 无关），backend 环境仍需 openpyxl，**勿动**。

### B5. 引擎可观测性（Fable 建议，轻量做）
- **启动期断言**（非首次读文件时）：在 ethoinsight 解析模块 import 期或 backend 启动期断言 `python-calamine` 可导入——缺依赖在部署当场响亮失败，而非运行到第一次读才炸。配合 B1 硬依赖。
- **provenance 标记**（可选，低成本）：把引擎名+版本写进已有的运行产物元数据（handoff/manifest 任一已有位置即可），事后可查「这批数是哪个引擎读的」。**列轻量可选**，不阻塞。

---

## Part C — code-executor「好几分钟」（仅记录，本 PR 不做）

13 分钟里**真正计算只 ~2 分钟**（5 个 metric 批次 #7→#13 ≈18:16:40→18:18:43）。其余=LLM 逐 token 吐 28 行 bash×5 批+thinking（大头）+ stats 重试（run_groupwise_stats 18:25/18:27 跑两次）+ 140 次独立 process 冷启（每次 +0.83s import）。

B2 换 calamine 把每 compute 读盘 1.35s→0.1s，140 次省 ~170s。进一步治本（独立 sprint）：让 code-executor 一条命令单 process 循环 28 文件算一个指标（省 27 次 import+进程启动），而非 28 个 `python -m ... &`。**记 TODO，不进本 spec**（避免「根因未隔离前叠加机制」）。

---

## 实施顺序
1. 新 worktree 基于 `dev`，建分支 `fix/seal-list-zone-params-and-excel-perf`。
2. **Part A**：A5 先写红测试确认红 → A1（ParamValue 别名 + 两字段）+ A2（normalizer 加注释保留）→ 转绿。commit 1。
3. **Part B**：B1（加依赖）+ B2（EXCEL_ENGINE 常量 + 三处显式 engine，无 fallback）+ B4（差分测试）→ parse 套全绿 → `uv sync`。commit 2。
4. 全量回归 + 裸导入 → 提 PR（A+B 一个 PR）。

## 验证（端到端）
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
# A: 红测试先红后绿
PYTHONPATH=. .venv/bin/python -m pytest tests/test_metric_stat_and_executor_schema.py tests/test_auto_seal_from_artifacts.py -q
# B: ethoinsight parse + 引擎等价
cd ../../ethoinsight && pytest tests/ -k "parse or engine or equivalence" -q
# 全量回归（基线 6 failed 为已知债，见 2026-06-12 sync handoff §6；勿归因自己）
cd ../agent/backend && PYTHONPATH=. .venv/bin/python -m pytest -q
# 改了 subagents/手 schema → 裸导入两生产入口（conftest mock 藏循环导入）
PYTHONPATH=. .venv/bin/python -c "import app.gateway"
PYTHONPATH=. .venv/bin/python -c "from deerflow.agents import make_lead_agent"
```
最终：dogfood 复跑同一 EPM 数据，code-executor seal 成功、handoff 链走通、inspect 明显变快。

## 关键文件
- `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`（ParamValue 别名 + :109 + :239 + :289 注释）
- `packages/agent/backend/tests/test_metric_stat_and_executor_schema.py` + `tests/test_auto_seal_from_artifacts.py`（红测试）
- `packages/ethoinsight/ethoinsight/parse/_core.py`（EXCEL_ENGINE 常量 + :91/:195/:320 显式 engine，无 fallback）
- `packages/ethoinsight/pyproject.toml`（加 `python-calamine` 硬依赖 + pin）
- `packages/ethoinsight/tests/test_excel_engine_equivalence.py`（语料差分 + 边界探针，新建）

## 红线（勿违）
- A 部分：用 `ParamValue` 别名收口，**不**改两处裸 union；**不**设 left_to_right；**不**放宽到 list[int]/bool（值域已核实无 bool，见 A4）。
- A 部分：放宽前 grep 消费侧确认无人假设标量（A4b）；`extra="allow"` fail-open 记 TODO 本次不动（A6）。
- B 部分：**无运行时 fallback**（Fable 论证：fallback 对"成功但值不同"永不触发，纯负收益）；引擎显式传参不靠 pandas 默认；测试用 `assert_frame_equal(check_dtype=True)` 非 `astype(str)`（datetime dtype 盲区）；openpyxl 留 test/dev 依赖勿删。
- 改 schema 后必跑裸导入两生产入口（conftest mock 藏循环导入，pytest 假绿）。
- 全量 6 failed 是已知基线债，勿归因本次改动。

## Fable 核实结论（2026-06-12，实测 pydantic 2.13.4 / pandas 3.0.3 / calamine 0.6.2 / openpyxl 3.1.5）
两个修复方向**都确认正确**；Fable 实测补三点（已并入上文）：
1. **bool→1.0 静默转换**：bool 不在 union 时 smart 模式被 float 分支接走变 1.0。已现场核实 `parameters_used` 值域无 bool（metric 签名零 bool 参数），不加 bool 进 union；但 `parse_parameters` 的 `json.loads` 通道不防 bool，记边界（A4）。
2. **`extra="allow"` 是真 fail-open**：字段名打错静默吞进 model_extra。是有意设计（容纳新统计字段），本次不动，记 TODO（A6）。
3. **`astype(str)` 有 datetime dtype 盲区**：升级为 `assert_frame_equal(check_dtype=True)` 语料级对拍（B4）。pandas 3.0 下双引擎全轴一致（含历史分歧的整型浮点已收口），最该盯「时刻 vs 时长」非常规 number format。
