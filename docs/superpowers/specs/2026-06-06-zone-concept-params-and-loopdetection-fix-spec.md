# Spec：column_aliases → compute 参数注入通用化 + LoopDetection subagent 修复

> 状态：实施 spec（v2，经 Opus review 修订）
> 日期：2026-06-06
> 来源：EPM dogfood 28 文件全 None 失败分析
> 前置：PR #103（column_aliases 集成修复）已合 dev
> Review：Opus agent `aab44024b452bd290`（2026-06-06），P0-1/P0-2/P0-3 已在本版修正
> 铁律：`feedback_single_source_of_truth`、`feedback_pr_merge_must_run_full_suite_on_shared_logic`、`feedback_known_full_suite_test_pollution_4_tests`

---

## 0. 给实施 agent 的必读

### 0.1 一句话

EPM dogfood 失败由两个独立 bug 造成：① `_build_zone_aliases_overrides` 只在范式有 `anonymous_zone_override` 时工作，EPM 没有 → column_aliases 确认的概念永远到不了 compute 参数；② subagent LoopDetection 裸构造拿到 3/5 阈值，140 次合法 bash 到第 5 次就被掐死，到不了 seal。

### 0.2 Bug A 真根因

`_build_zone_aliases_overrides`（resolve.py:582-587）：

```python
azo = cat.anonymous_zone_override
if azo is None:
    return {}  # ← EPM 无此字段，直接返回
```

`anonymous_zone_override` 语义是"用户数据中有一个裸 `in_zone` 列，应该当作哪个 zone"。OFT/LDB/Zero Maze 有，EPM 没有（EPM zone 列是显式命名的 open/closed）。**双重失效**：参数注入不通 + compute 脚本 autodiscovery 正则 `in_zone.*open.?arm` 也匹配不到 `open` → 140 条全 None。

### 0.3 Bug B 真根因

executor.py:655 用 `LoopDetectionMiddleware()` 裸构造 → 模块常量 `_DEFAULT_TOOL_FREQ_HARD_LIMIT=5`。code-executor prompt "每个 metric 一次 bash" → 140 metrics → bash 被调 140 次 → 第 5 次 hard-stop，到不了 seal。

> **注意**：模块常量 3/5 是 Noldus fork 的**有意选择**（见 middleware L68 注释 `# P0 fix: lead 微调 bash command 让 hash 不同绕过 Layer 1`），目的是给 **lead agent** 更严的频次防护。subagent 需要的是自己的宽松阈值，**不能动全局常量**。

### 0.4 判断：100% 工程 bug，不需等行为学专家

EPM 的概念清单、参数名、requires_columns 全部已在 catalog/metrics 里写死自洽，HITL 也成功了。专家要补的是 Layer 3（多物理子区聚合），与本次 1:1 场景正交。

### 0.5 关键限制：metrics 层现状

EPM compute 函数签名（`epm.py`，已 `inspect.signature` 实测）：

| 函数 | 接受的 zone 参数 |
|------|-----------------|
| `compute_open_arm_time_ratio` | `open_arm_zones: list[str] \| None` |
| `compute_open_arm_time` | `open_arm_zones: list[str] \| None` |
| `compute_open_arm_entry_count` | `open_arm_zones: list[str] \| None` |
| `compute_open_arm_entry_ratio` | `open_arm_zones: list[str] \| None` |
| `compute_total_entry_count` | **无 zone 参数** |

辅助函数 `_get_closed_zone_cols(df)` **不接受 zone 参数**，只用正则。因此 `compute_open_arm_entry_ratio`（调 `compute_total_entry_count` → `_get_closed_zone_cols`）和 `compute_total_entry_count` 在用户列名不是 `in_zone_closed_arm_*` 时仍会失败。

**本次修复范围**：resolve 层注入 `open_arm_zones` + metrics 层给 `_get_closed_zone_cols`/`compute_total_entry_count`/`compute_open_arm_entry_ratio` 补 `closed_arm_zones` 参数。

---

## 1. Bug A：column_aliases 概念→参数注入通用化

### 1.1 设计

新增范式级 `zone_concept_params` 字段：概念→参数名 + 类型的显式映射。`_build_zone_aliases_overrides` 改为从 `zone_concept_params` 读取映射（新增），`anonymous_zone_override.target_param` 继续作为 fallback（保留）。

**为什么不靠 convention 推导？** `open_arms`→`open_arm_zones` 的字符串变换（去 s + 加 _zones）脆弱且隐式。显式声明把 catalog 作者已知的对应关系写成数据，可审计、不脆弱。

**为什么是范式级而非 metric 级？** zone 概念到参数的对应是范式属性（EPM 所有 5 个指标共享 open_arms→open_arm_zones），不是 per-metric 配置。对流到 `_compute_parameters_in_use` 的替换逻辑，范式级映射已足够。

**与 `anonymous_zone_override` 的关系**：两者职责正交——`anonymous_zone_override` 管"裸 in_zone → 哪个概念"，`zone_concept_params` 管"概念 → 哪个参数"。同时命中同一 param 时 `anonymous_zone_is` 优先。

### 1.2 schema.py 变更

新增 `ZoneConceptParam`：

```python
@dataclass(frozen=True)
class ZoneConceptParam:
    param: str              # compute 参数名，如 "open_arm_zones"
    wrap_list: bool = False # 参数是否为 list（多列合并 vs 单列）
```

`Catalog` 新增：
```python
zone_concept_params: dict[str, ZoneConceptParam] = field(default_factory=dict)
# key = concept keyword, e.g. "open_arms", "center"
```

### 1.3 loader.py 变更

`_parse_catalog`（约 L106-162）新增 `zone_concept_params` 解析块，仿 `anonymous_zone_override` 写法：

```python
zone_concept_params: dict[str, ZoneConceptParam] = {}
for concept_key, mapping in raw.get("zone_concept_params", {}).items():
    if not isinstance(mapping, dict):
        raise CatalogError(f"zone_concept_params.{concept_key}: must be a dict")
    param = mapping.get("param", "")
    if not isinstance(param, str) or not param:
        raise CatalogError(f"zone_concept_params.{concept_key}.param: must be a non-empty string")
    wrap_list = mapping.get("wrap_list", False)
    if not isinstance(wrap_list, bool):
        raise CatalogError(f"zone_concept_params.{concept_key}.wrap_list: must be a bool, got {type(wrap_list).__name__}")
    zone_concept_params[concept_key] = ZoneConceptParam(
        param=param,
        wrap_list=wrap_list,
    )
```

并在 `Catalog(...)` 构造中传入 `zone_concept_params=zone_concept_params`。

### 1.4 各范式 YAML 变更

**epm.yaml**（新增 `zone_concept_params`，范式级别）：
```yaml
zone_concept_params:
  open_arms:
    param: open_arm_zones
    wrap_list: true
  closed_arms:
    param: closed_arm_zones
    wrap_list: true
```

**epm.yaml 各 metric**（补参数声明。注意 default 不能是 null——loader 的 ParamSpec 校验只接受 int/float/str/list，用空字符串 `""`，falsy 语义等价于 None。**关键：声明必须与函数签名严格对齐**，否则 `**parameters` 展开时多余参数 → TypeError）：

```yaml
# open_arm_time_ratio / open_arm_time / open_arm_entry_count：
#   只接受 open_arm_zones，不声明 closed_arm_zones
parameters:
  open_arm_zones:
    default: ""
    unit: ""
    description: "开臂区域列名列表。由 HITL column_aliases 自动注入。"
    tunable_by_user: false
    valid_range: null

# open_arm_entry_ratio + total_entry_count：
#   接受 open_arm_zones + closed_arm_zones 两者
parameters:
  open_arm_zones:
    default: ""
    unit: ""
    description: "开臂区域列名列表。由 HITL column_aliases 自动注入。"
    tunable_by_user: false
    valid_range: null
  closed_arm_zones:
    default: ""
    unit: ""
    description: "闭臂区域列名列表。由 HITL column_aliases 自动注入。"
    tunable_by_user: false
    valid_range: null

# total_entry_count：同 open_arm_entry_ratio（两个参数都声明）
# 注：total_entry_count 是 epm.yaml 的 default_metrics[4]，需确认在清单内
```

**oft.yaml / ldb.yaml / zero_maze.yaml**：不动。`anonymous_zone_override` 继续覆盖它们。

**fst.yaml / tst.yaml**：不涉及，不动。

### 1.5 resolve.py 变更

**重写 `_build_zone_aliases_overrides`**：

```
新逻辑（支持多 concept 路由）：
1. 从 zone_concept_params + azo 收集所有 (concept → param, wrap_list) 映射
   - zone_concept_params 中的每条直接加入
   - 若 azo 存在且其 target_param 未被 zone_concept_params 覆盖 → 
     从 requires_columns 模式中推导 concept 关键词，加入映射
2. 收集 zone_patterns（从 catalog entries 的 requires_columns 提取 in_zone*）
3. 对 column_aliases 中每个概念匹配 zone_patterns 的物理列 → 按 concept 分组
4. 对每个 concept，查映射得 (param, wrap_list) → 物理列按 wrap_list 决定类型 → 注入 overrides
5. existing_overrides 含 anonymous_zone_is 或同名 target_param → 该 param 不覆盖
```

**删除**：以下在当前 dev 中但本次修复后不再需要的代码（如果有的话）：无。`anonymous_zone_override` 分支保留作为 fallback。

**不新增** `_derive_zone_concept_params_from_azo` 函数——两段式 fallback（zone_concept_params → azo.target_param）在 resolve 内联即可，无需预先转换。

### 1.6 metrics/epm.py 变更

给 `_get_closed_zone_cols` 和依赖它的函数补 `closed_arm_zones` 参数：

```python
def _get_closed_zone_cols(
    df: pd.DataFrame,
    closed_arm_zones: list[str] | None = None,
) -> list[str]:
    """Return closed-arm zone columns, preferring center-point suffix."""
    if closed_arm_zones:
        return [c for c in closed_arm_zones if c in df.columns]
    all_cols = [c for c in df.columns if re.search(r"in_zone.*closed.?arm", c, re.I)]
    return _prefer_center_suffix(all_cols)


def compute_total_entry_count(
    df: pd.DataFrame,
    open_arm_zones: list[str] | None = None,
    closed_arm_zones: list[str] | None = None,
    min_duration_frames: int = 0,
) -> int | None:
    if open_arm_zones:
        open_cols = [c for c in open_arm_zones if c in df.columns]
    else:
        open_cols = _get_open_zone_cols(df)
    if closed_arm_zones:
        closed_cols = [c for c in closed_arm_zones if c in df.columns]
    else:
        closed_cols = _get_closed_zone_cols(df)
    # ... 其余不变


def compute_open_arm_entry_ratio(
    df: pd.DataFrame,
    open_arm_zones: list[str] | None = None,
    closed_arm_zones: list[str] | None = None,
) -> float | None:
    open_count = compute_open_arm_entry_count(df, open_arm_zones)
    total_count = compute_total_entry_count(
        df,
        open_arm_zones=open_arm_zones,
        closed_arm_zones=closed_arm_zones,
    )
    # ... 其余不变
```

### 1.7 测试

在 `test_column_semantics.py` 新增（red anchor，现在跑必红）：

```python
class TestEPMZoneConceptParams:
    """EPM column_aliases → open_arm_zones / closed_arm_zones 注入。"""

    def test_open_arm_zones_injected(self, tmp_path):
        """open→open_arms alias → open_arm_zones=["open"]。"""
        plan = resolve_metrics(
            paradigm="epm",
            columns=["trial_time", "x_center", "y_center", "distance_moved",
                     "velocity", "open", "closed", "result_1"],
            raw_files=["/mnt/user-data/uploads/test.xlsx"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases={"open": "open_arms", "closed": "closed_arms"},
        )
        m = next(m for m in plan.metrics if m.id == "open_arm_time_ratio")
        assert m.parameters_in_use["open_arm_zones"] == ["open"]

    def test_closed_arm_zones_injected(self, tmp_path):
        """closed→closed_arms alias → closed_arm_zones=["closed"]。"""
        plan = resolve_metrics(..., column_aliases={"open": "open_arms", "closed": "closed_arms"})
        m = next(m for m in plan.metrics if m.id == "open_arm_entry_ratio")
        assert m.parameters_in_use["closed_arm_zones"] == ["closed"]

    def test_multi_concept_routing(self, tmp_path):
        """两个概念分别落到两个不同参数。"""
        plan = resolve_metrics(..., column_aliases={"open": "open_arms", "closed": "closed_arms"})
        m = next(m for m in plan.metrics if m.id == "open_arm_entry_ratio")
        assert m.parameters_in_use["open_arm_zones"] == ["open"]
        assert m.parameters_in_use["closed_arm_zones"] == ["closed"]

    def test_no_alias_standard_columns_still_work(self, tmp_path):
        """无 column_aliases 时（标准 in_zone_open_arms_center 列）不回归。"""
        plan = resolve_metrics(
            paradigm="epm",
            columns=["trial_time", "x_center", "y_center", "distance_moved",
                     "velocity", "in_zone_open_arms_center", "in_zone_closed_arms_center", "result_1"],
            raw_files=["/mnt/user-data/uploads/test.xlsx"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
        )
        m = next(m for m in plan.metrics if m.id == "open_arm_time_ratio")
        # 无 alias 时参数应为空字符串（default），由 autodiscovery 处理
        assert m.parameters_in_use.get("open_arm_zones", "") == ""
        # 关键：open_arm_time_ratio 函数不收 closed_arm_zones → 不应出现在参数中
        assert "closed_arm_zones" not in m.parameters_in_use, (
            "open_arm_time_ratio 不接受 closed_arm_zones 参数"
        )

    def test_parameter_subset_matches_function_signature(self, tmp_path):
        """每个 metric 的 parameters_in_use 仅含其函数支持的参数子集。"""
        plan = resolve_metrics(
            paradigm="epm",
            columns=["trial_time", "x_center", "y_center", "distance_moved",
                     "velocity", "open", "closed", "result_1"],
            raw_files=["/mnt/user-data/uploads/test.xlsx"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases={"open": "open_arms", "closed": "closed_arms"},
        )
        # 只接受 open_arm_zones 的 metric
        for mid in ("open_arm_time_ratio", "open_arm_time", "open_arm_entry_count"):
            m = next(m for m in plan.metrics if m.id == mid)
            assert "open_arm_zones" in m.parameters_in_use
            assert "closed_arm_zones" not in m.parameters_in_use, \
                f"{mid} 不接受 closed_arm_zones，但出现在了 parameters_in_use"
        # 接受两者的 metric
        for mid in ("open_arm_entry_ratio", "total_entry_count"):
            m = next(m for m in plan.metrics if m.id == mid)
            assert "open_arm_zones" in m.parameters_in_use
            assert "closed_arm_zones" in m.parameters_in_use

    def test_explicit_override_wins(self, tmp_path):
        """显式 overrides 优先于 zone_concept_params。"""
        plan = resolve_metrics(
            ...,
            column_aliases={"open": "open_arms"},
            overrides={"open_arm_zones": ["in_zone"]},
        )
        m = next(m for m in plan.metrics if m.id == "open_arm_time_ratio")
        assert m.parameters_in_use["open_arm_zones"] == ["in_zone"]

    def test_oft_anonymous_zone_override_not_regressed(self, tmp_path):
        """OFT anonymous_zone_is 路径不回归。"""
        # 复用现有 fixture，断言 center_zone 仍正确注入
        ...


class TestLoaderZoneConceptParams:
    """loader 正确解析 zone_concept_params。"""

    def test_epm_zone_concept_params_loaded(self):
        cat = load_catalog("epm")
        assert "open_arms" in cat.zone_concept_params
        assert cat.zone_concept_params["open_arms"].param == "open_arm_zones"
        assert cat.zone_concept_params["open_arms"].wrap_list is True

    def test_invalid_zone_concept_params_raises(self, tmp_path):
        """非法 shape 报 CatalogError。"""
        ...
```

**不改动**：现有 OFT/LDB/Zero Maze 的 alias 测试保留作为回归锚点。

---

## 2. Bug B：LoopDetection subagent 配置错配

### 2.1 修复（局部传参，不动全局常量）

**executor.py:655** — 当前 `LoopDetectionMiddleware()`，改为显式传值：

```python
LoopDetectionMiddleware(
    tool_freq_warn=30,
    tool_freq_hard_limit=50,
)
```

**为什么不动全局常量？** 常量 3/5 是 Noldus fork 对 lead agent 的有意防护（middleware L68 注释 + test L371 注释），改它会同时放宽 lead agent。subagent 只需自己宽松。

**loop_detection_middleware.py** — docstring L208-210 修正为与常量一致的 3/5（不改常量值，只修 docstring 的误标）。

### 2.2 Batch bash（P0 配套，与 2.1 同批上线）

30/50 对 140 次 bash 仍不够（140 > 50，第 50 次仍 hard-stop）。必须同时改 prompt 让 code-executor 批量执行。

**code_executor.py prompt** — 从"每个 metric 一次 bash"改为允许批量：

```
对同指标类型的不同 subject，用一条 bash 并行执行：
python -m <script> --input s1.xlsx --output m1_s0.json --parameters-json '...' &
python -m <script> --input s2.xlsx --output m1_s1.json --parameters-json '...' &
...
wait
echo 'ALL_DONE_<metric_id>'
```

效果：140 次 bash → ~5-8 次（按指标类型数）。

**实施前必做**：`git show -S` 实证 `ScriptInvocationOnlyProvider` 的白名单正则放行 `&`/`wait` 复合命令——参考 memory `feedback_chart_maker_bash_guardrail_must_allow_resolve_dumpheaders` 的同类教训。

### 2.3 SKILL.md 收拢

- code_executor prompt：删除"开工前必读 output-constitution"步骤（如已通过 skill render inline 进 system prompt）
- SKILL.md workflow Step 1 → "读一次 plan_metrics.json，然后立即开始跑第一个脚本"
- 加"每个文件最多读一次"正面指令

### 2.4 测试

- 后端 guardrail 测试确认 batch bash 不被拦
- 新增 subagent 跑 60+ 次不同 bash 不被硬停的回归测试
- 新增 Layer 1 identical-set 3/5 仍生效的测试（真循环防护未被削弱）
- `make test` 全量通过

---

## 3. 不改的部分

- loop_detection 全局常量（3/5 保持，subagent 局部传参 30/50）
- `anonymous_zone_override` 全路径逻辑
- OFT/LDB/Zero Maze YAML
- FST/TST YAML
- `experiment_context.py`
- 前端

---

## 4. 验收标准

1. EPM resolve_metrics + column_aliases → `open_arm_zones=["open"]` / `closed_arm_zones=["closed"]` 出现在 parameters_in_use
2. 每个 metric 的 `parameters_in_use` 仅含其函数签名支持的参数子集（`open_arm_time_ratio`/`open_arm_time`/`open_arm_entry_count` 无 `closed_arm_zones`；`open_arm_entry_ratio`/`total_entry_count` 有两者）
3. EPM compute 脚本用物理列名 `open`/`closed` 产出非 null 值
4. OFT/LDB/Zero Maze anonymous_zone_is 路径不回归
5. 无 alias 标准 EV19 EPM 列（in_zone_open_arms_center）不回归
6. subagent 跑 60+ 次合法 bash 不被硬停
7. subagent 真死循环（连续 5 次相同 tool_call）仍在 5 次被 Layer 1 硬停
8. ethoinsight 全量绿（除已知 4 污染）
9. 后端 guardrail + executor 测试绿
10. EPM dogfood（Xuhui 28 文件）端到端指标非 null

---

## 5. 合并前检查

- `cd packages/ethoinsight && pytest` 全量（改 resolve.py + loader.py + epm.yaml + metrics/epm.py = 6 范式承重墙）
- `cd packages/agent/backend && make test`（executor + guardrail）
- 手动验证：`resolve_metrics(paradigm="epm", ..., column_aliases={"open":"open_arms","closed":"closed_arms"})` 断言两参数均注入
- `grep -rn "compute_total_entry_count(" packages/ethoinsight/` 确认无位置参数调用（新增 `open_arm_zones`/`closed_arm_zones` 参数后 `min_duration_frames` 从第 2 位移到第 4 位）

---

## 6. 文件清单

| 文件 | 操作 | 预计规模 |
|------|------|---------|
| `packages/ethoinsight/ethoinsight/catalog/schema.py` | 新增 `ZoneConceptParam` + `Catalog` 加 `zone_concept_params` | ~15 行 |
| `packages/ethoinsight/ethoinsight/catalog/loader.py` | `_parse_catalog` 解析 `zone_concept_params` + 传入 `Catalog(...)` | ~15 行 |
| `packages/ethoinsight/ethoinsight/catalog/epm.yaml` | 新增 `zone_concept_params`（范式级）+ 4 个 metric 补 `parameters` | ~30 行 |
| `packages/ethoinsight/ethoinsight/catalog/resolve.py` | 重写 `_build_zone_aliases_overrides`（多 concept 路由） | ~60 行 |
| `packages/ethoinsight/ethoinsight/metrics/epm.py` | `_get_closed_zone_cols`/`compute_total_entry_count`/`compute_open_arm_entry_ratio` 补 `closed_arm_zones` 参数 | ~25 行 |
| `packages/ethoinsight/tests/test_column_semantics.py` | 新增 `TestEPMZoneConceptParams` + `TestLoaderZoneConceptParams` | ~80 行 |
| `packages/ethoinsight/tests/test_epm_metrics.py` | 新增带 `closed_arm_zones` 参数的 metric 测试 | ~20 行 |
| `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` | L655 局部传参 `tool_freq_warn=30, hard_limit=50` | 1 行改动 |
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py` | docstring 修正（3/5 非 30/50） | 2 行 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` | prompt batch bash + 删"先读宪法" | ~15 行 |
| `packages/agent/skills/custom/ethoinsight-code/SKILL.md` | workflow 收拢 | ~5 行 |
| `packages/agent/backend/tests/test_loop_detection_middleware.py` | 更新 L371 stale 注释 + 新增 subagent 60+ bash 不硬停测试 | ~30 行 |
