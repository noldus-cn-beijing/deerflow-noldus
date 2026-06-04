# 2026-06-04 匿名 `in_zone` 区检测三范式泛化 + 统一 override key 修复设计（实施 spec）

> **本 spec 用途**：把 2026-06-03 O迷宫（zero_maze）dogfood 全链路失败的修复，连同对 OFT/LDB 同一地基缺陷的根治，落成一份**自包含、可整段交给独立 agent 执行**的实施 spec。经一手证据 + 几何实测 + 多轮核实定稿，**纠正了 dogfood 现场的危险误判**（见 §0）。
>
> **配套问题来源**：2026-06-03 O迷宫 dogfood 失败 thread `packages/agent/backend/.deer-flow/users/cd95effa-.../threads/6dea6ae9-.../`
> **前序相关 spec**：[2026-06-03-oft-pipeline-failure-fix-design.md](2026-06-03-oft-pipeline-failure-fix-design.md)（PR-1 修了 OFT 的 `center_zone`，本 spec 把机制泛化到三范式）
> **当前基线**：ethoinsight 全量 **570 passed, 65 skipped**（实测，非旧 spec 的 553）

---

## 0. 现场核实结论（执行 agent 必读 —— 推翻了 dogfood 现场的假设）

1. **🔴 几何实测：`in_zone=1` = 开放臂（不是 dogfood 里用户答的"封闭区"）**。对真实 O迷宫数据实测：in_zone=1 三组占时 **10.7% / 14.2% / 6.4%**（少数占时），且角度上占据**两段相对弧**（如 Ser4-saline 在 30-60° 和 270-300°）——这正是零迷宫开放臂几何 + 动物回避开放臂的行为学铁律。dogfood 中 deepseek 在死循环里诱导用户二选一，用户答了"in_zone=1=封闭区"，**答反了**。**执行 agent 不要采信那个答案。**
2. **因此本次不需要"取反"语义**。三范式值语义统一：EthoVision `in_zone` 恒 **1 = 在该分析区内**。zero_maze 与 OFT 完全同构——in_zone=1 就是"在目标分析区（开放臂）内"，开放区时间 = in_zone==1 占比。dogfood 里出现的 `open_zone: "~in_zone"` 取反语法是 deepseek 在错误前提下脑补的伪需求，全链路无任何代码认识它。
3. **真根因 = OFT 的修复没泛化 + zero_maze/LDB catalog 没声明 zone 参数**。`resolve.py:495` 放行逻辑**硬编码 `if "center_zone" in overrides`**，只认 OFT；zero_maze/LDB catalog `paradigm_parameters` 没有任何 zone 参数，`_compute_parameters_in_use`（resolve.py:856-858）replace-only，未声明的 override key 被静默丢弃。LLM 无可用 key → 脑补 → 死循环 → 撞 read_file 5 次安全限 FORCED STOP。
4. **同一地基缺陷波及 OFT / zero_maze / LDB 三范式**（真实 EV19 导出都把区域列退化成单匿名 `in_zone`，源列头无括号 `分析区中`，`utils.py:55` 归一化）。EPM 命名良好（`in_zone_open_arms_center`）不受影响。**本次一次性根治三范式。**
5. **用户决策（已拍板）**：
   - **一次覆盖三范式**（OFT 已修 center_zone，本次补 zero_maze + LDB，并把放行逻辑泛化成范式无关）。
   - **遇匿名区：问 + 附几何占时证据**。抛 zone_unnamed 让 lead 反问，但要附 in_zone 各值占时比例 + 行为学常识帮用户判断、防止再被诱导答错；最终用户拍板（守 SSOT 不猜铁律）。
   - **统一 override key**：agent 永远只写 `{"anonymous_zone_is": "in_zone"}`，不记 center/light/open、不分 str/list；后端在 catalog 翻译层吸收差异。这是为消除 agent 接口困扰的明确选择。

---

## 1. 设计总览

**核心机制（范式无关 + agent 零困扰）**：

```
lead 反问用户 → 用户确认匿名区代表目标区 → lead 写统一 key:
        parameter_overrides = {"anonymous_zone_is": "in_zone"}    ← 三范式同一形态
                ↓ 写入 experiment-context.json
prep_metric_plan → resolve_metrics(overrides={"anonymous_zone_is":"in_zone"})
                ↓
  ① _detect_anonymous_zone：本范式声明了 anonymous_zone_override 且 overrides 含统一 key → 放行
  ② _compute_parameters_in_use：按本范式 catalog 元字段把统一 key 翻译成真实参数：
        OFT       → {"center_zone": "in_zone"}        (wrap_list=false)
        zero_maze → {"open_zones": ["in_zone"]}       (wrap_list=true ← list 包装在此发生)
        LDB       → {"light_zone": "in_zone"}          (wrap_list=false)
                ↓ 真实参数 key 已在 result（metric.parameters default 注入），replace-only 正确替换
  生成 plan（args 带 --parameters-json {真实参数}）
                ↓ code-executor 照 plan 跑 → compute 收到真实参数 → 算出真值
                  （zero_maze open_zone_time_ratio ≈ 0.10）
```

**关键不变量**：
- compute 库函数签名**完全不动**（仍 `center_zone: str` / `open_zones: list[str]|None` / `light_zone: str`）。str/list 差异被 catalog 的 `wrap_list` 吸收，agent 与 LLM 永不感知 list。
- catalog `requires_columns` **保持命名 zone pattern 不动**（`in_zone_center_*`/`in_zone_open*`/`in_zone_light*`）。契约正确地拒绝匿名数据，触发反问的逻辑在 resolve 层，不靠放宽契约。
- 翻译规则与范式定义同源（catalog yaml 顶层元字段），不散落——SSOT。未来加范式只填两行（target_param + wrap_list）。

---

## 2. 已核实事实（执行 agent 必读，纠正了探索阶段的多处错报）

| 事实 | 证据 |
|---|---|
| `_detect_anonymous_zone` **唯一调用点 = resolve.py:199**（resolve_charts 不调） | grep 核实，探索 agent 误报"两处" |
| `schema.py` 的 `Catalog` 是 `@dataclass(frozen=True)`（102-103 行） | 加字段用 `field(default=...)`；`ParamSpec`(36-44)/`MetricEntry`(61-73) 均 frozen dataclass |
| **zero_maze compute 参数是 list 形态** `open_zones: list[str]\|None`（metrics/zero_maze.py:73、`_get_open_zone_cols`:23） | 误传 str 会逐字符迭代 → 必须包 list |
| OFT `center_zone: str="in_zone_center"`（oft.py:60，`_find_center_zone_column` hint）；LDB `light_zone: str="in_zone_light"`/`dark_zone`（ldb.py:32-46） | str 形态 |
| resolve / prep **都只有列名没有 DataFrame** | resolve.py:59-75 签名只收 `columns: list[str]`；prep 只 `parse_header` 拿 columns（prep_metric_plan_tool.py:195），不读 trajectory → 几何证据不能在这两层算 |
| `inspect_uploaded_file_tool` **已读文件、有全量 df**（xlsx/csv 路径 inspect:375/429 调 `parse_trajectory`）；**txt 路径只读前 5 行**未全量 parse | 几何证据注入点选 inspect |
| `compute_analysis_config_id` 用 `json.dumps(sort_keys=True)`（experiment_context.py:78-79）→ **list override 安全**（不哈希、不 set 化） | 已核实，排除 list 炸裂风险 |
| loader `_parse_param_spec` 的 `default` 校验只收 `(int,float,str)`；`valid_range` 数值分支已有 `isinstance(default,(int,float))` 守卫 | list default 需放宽类型，但不会误触数值校验 |
| `validate_catalog_consistency` 把 `(pname, default)` 作 dict key（list 不可哈希），但**只遍历 paradigm_parameters** | zone 参数必须放 **per-metric `parameters:`**，否则 list default `TypeError: unhashable` |
| `compute_*.py` 走 `value = compute_xxx(df, **parse_parameters(args))`，`parse_parameters` 用 `json.loads` | list 类型完整保留透传给 compute kwarg |

---

## 3. 改动点（ethoinsight 库 + backend，两域源文件零重叠可并行，但合并需统一回归）

### 3.1 ethoinsight 库（跨范式共享逻辑 + catalog 契约）

#### (a) `ethoinsight/catalog/schema.py` —— 新增翻译规则 dataclass + Catalog 字段
- 新增 `@dataclass(frozen=True) AnonymousZoneOverride`：字段 `target_param: str`、`wrap_list: bool = False`。
- `Catalog`（102-112）加字段 `anonymous_zone_override: AnonymousZoneOverride | None = None`（紧邻 `paradigm_parameters`）。

#### (b) `ethoinsight/catalog/loader.py` —— 解析元字段 + 放宽 list default
- `_parse_catalog`（~105-134）解析顶层 `anonymous_zone_override`（dict → `AnonymousZoneOverride(target_param=..., wrap_list=...)`，缺省 None），传入 `Catalog(...)`。校验 `target_param` 是 str；`wrap_list` 是 bool（缺省 False）。
- `_parse_param_spec` 的 `default` 类型校验放宽 `(int,float,str)` → `(int,float,str,list)`（zero_maze `open_zones` 的 list default 需要）。确认 `valid_range` 数值校验分支保持 `isinstance(default,(int,float))` 守卫（list 自动跳过，安全）。

#### (c) `ethoinsight/catalog/resolve.py` —— 放行泛化 + 统一 key 翻译
两处，都范式无关：
- **放行**：`_detect_anonymous_zone`（469-510）签名加形参 `anonymous_zone_override: AnonymousZoneOverride | None`；把 495 行 `if "center_zone" in overrides: return True` 改为：
  ```python
  if anonymous_zone_override is not None and "anonymous_zone_is" in overrides:
      return True
  ```
  唯一调用点 resolve.py:199 传 `cat.anonymous_zone_override`。`zone_unnamed` 的 `message` 保持技术事实（不塞用户话术——职责分层）。
- **翻译**：`_compute_parameters_in_use`（805-860）在第 4 步 override 循环（856-858）**之前**，把统一 key 翻译成真实参数。建议改函数签名加 `anonymous_zone_override` 形参（调用方 resolve.py 内传 `cat.anonymous_zone_override`），逻辑：
  ```python
  # translate unified zone key → paradigm's real param, before replace-only loop
  effective_overrides = dict(overrides)
  azo = anonymous_zone_override
  if azo is not None and "anonymous_zone_is" in effective_overrides:
      val = effective_overrides.pop("anonymous_zone_is")
      effective_overrides[azo.target_param] = [val] if azo.wrap_list else val
  # 然后第 4 步用 effective_overrides 替换 result 中已存在的 key
  ```
  注意：真实参数 key（center_zone/open_zones/light_zone）必须已由 852-853 行 `metric.parameters` default 注入 `result`，replace-only 才命中——这是 (d) 必须声明 per-metric parameters 的原因。

#### (d) 三个 catalog yaml —— 声明真实 zone 参数（per-metric）+ 顶层翻译元字段
- **真实 zone 参数放 per-metric `parameters:`**（非 paradigm_parameters，否则 list default 触发 unhashable）：
  - `oft.yaml`：各 center default 指标（5 个）声明 `center_zone`（default `"in_zone_center"`，str）。**若 PR-1 已声明则保持不动。**
  - `zero_maze.yaml`：4 个 open 指标（open_zone_time_ratio / open_zone_time / open_zone_distance / hesitation_count）各声明 `open_zones`（default `["in_zone_open"]`，**list**，`valid_range: null`）。
  - `ldb.yaml`：3 个 light 指标（light_time_ratio / transition_count / light_latency）各声明 `light_zone`（default `"in_zone_light"`，str）。
- **顶层翻译元字段**（每范式一行结构）：
  ```yaml
  # oft.yaml
  anonymous_zone_override: {target_param: center_zone, wrap_list: false}
  # zero_maze.yaml
  anonymous_zone_override: {target_param: open_zones, wrap_list: true}
  # ldb.yaml
  anonymous_zone_override: {target_param: light_zone, wrap_list: false}
  ```
- **LDB 只覆盖 light_zone**：匿名区只有一个 in_zone，暗区=非 in_zone；`compute_transition_count` 在 dark_zone 列缺失时容错只数 light 列穿越（ldb.py:63-80），不为 dark 无中生有第二区。
- **EPM/FST/TST 不声明 `anonymous_zone_override`**（默认 None）→ 匿名区检测对它们不触发（命名列齐全本就不会走到该分支），零误伤。

### 3.2 backend（受保护 Noldus 工具，surgical 编辑）

#### (e) `tools/builtins/inspect_uploaded_file_tool.py` —— 附几何占时证据
- 新增 `_compute_anonymous_zone_evidence(df) -> dict | None`：
  ```python
  def _compute_anonymous_zone_evidence(df):
      if "in_zone" not in df.columns: return None
      s = df["in_zone"].dropna()
      if s.empty: return None
      vc = s.value_counts(normalize=True)
      return {
          "column": "in_zone",
          "occupancy_ratio": {str(int(k)): round(float(v), 4) for k, v in vc.items()},
          "n_frames": int(len(s)),
          "note": "in_zone=1 表示在该匿名区内；动物在焦虑回避区（中心/开放臂/亮室）通常占时较低。",
      }
  ```
- 在三个 `_inspect_*` 的 EV19 分支拿到 df 后，当 `columns` 含裸 `in_zone` 时，把结果塞进返回 dict 的 `anonymous_zone_evidence` 字段。
- **txt 路径**当前只读 5 行，需在"columns 含裸 in_zone"条件下补一次 `parse_trajectory(real_path)` 算全程占时（仅此条件触发，不拖慢正常命名文件）。

#### (f) `tools/builtins/prep_metric_plan_tool.py` —— zone_unnamed hint 改统一 key + 正面措辞
改写 `_ERROR_HINTS["zone_unnamed"]`（约 43-50），正面措辞（禁用"不要/禁止"），含：
1. 指引 lead 先调 `inspect_uploaded_file` 查看 `anonymous_zone_evidence` 的占时比例。
2. 用 `ask_clarification` 把占时证据呈现给用户、请其确认该匿名区代表哪个目标区。
3. 行为学常识（只此一处，SSOT）：动物在焦虑回避区（旷场中心区 / 零迷宫开放臂 / 明暗箱亮室）通常停留时间较短，占时低的一侧更可能是目标区；以用户确认为准。
4. **统一 key 操作指南**：用户确认后写 `parameter_overrides={"anonymous_zone_is": "in_zone"}` 重调 prep_metric_plan（三范式同一形态）。
5. 若用户判断该区不是目标区 → 说明数据需在 EthoVision 重新命名区域后导出再分析。

参考文案（执行 agent 可微调，保持正面 + 含统一 key + 含 ask_clarification + 含占时指引）：
> "数据里有一个未命名分析区(in_zone)，需要确认它代表哪个目标区域后再分析。第一步：调 inspect_uploaded_file 查看该文件的 anonymous_zone_evidence，它给出 in_zone=1 与 in_zone=0 的占时比例。第二步：用 ask_clarification 把占时证据呈现给用户并请其确认。行为学常识可辅助判断：动物在焦虑回避区（旷场中心区 / 零迷宫开放臂 / 明暗箱亮室）通常停留时间较短，占时低的一侧更可能是目标区，最终以用户确认为准。第三步：用户确认后写 parameter_overrides={\"anonymous_zone_is\": \"in_zone\"} 再重调 prep_metric_plan。若用户判断该区不是目标区，请说明数据需在 EthoVision 重新命名区域后导出，以保证分析建立在明确区域定义之上。"

#### (g) `agents/middlewares/experiment_context.py` —— docstring 举统一 key 例
- `set_experiment_paradigm_tool` 的 `parameter_overrides` docstring（~229）加示例 `{"anonymous_zone_is": "in_zone"}`。
- **类型注解 `dict[str, float|int|str]` 维持不变**（统一 key 的 value 是 str，list 包装在后端 resolve 层发生，LLM 永不写 list）。

---

## 4. 测试清单（TDD 强制，参照 `tests/test_oft_zone_unnamed_detection.py` 形态）

### ethoinsight 侧（新增 `tests/test_zone_unnamed_detection_all_paradigms.py`）
- **裸 in_zone 无 override → zone_unnamed**：OFT / zero_maze / LDB 三范式各一例（带 details 断言）。
- **统一 key 翻译（核心）**：三范式各传 `overrides={"anonymous_zone_is":"in_zone"}` → plan 成功，`parameters_in_use` 落到真实参数：
  - OFT：`center_zone == "in_zone"`
  - LDB：`light_zone == "in_zone"`
  - **zero_maze：`open_zones == ["in_zone"]` 且 `isinstance(..., list)`**（验证 wrap_list 生效）
- **list 被 compute 正确消费**：`compute_open_zone_time_ratio(df, open_zones=["in_zone"])` 对只含 `in_zone` 列的 df 出真值（非 None、非逐字符迭代 bug）。
- **`_detect_anonymous_zone` 直测**：范式声明 `anonymous_zone_override` + overrides 含统一 key → True；声明了但 overrides 不含统一 key → ResolveError；范式未声明（None）→ None（不触发，落入 columns_missing 分支）。
- **真缺列不误判**：无任何 in_zone 列 → `columns_missing`；有命名 `in_zone_open` 但缺 `distance_moved` → `columns_missing`。
- **override 不泄漏**：override 不进无 zone 参数的 optional metric（参照 OFT leak 测试）。
- **loader**：收 list default 不抛 CatalogError；正确解析 `anonymous_zone_override` 成 dataclass（缺省 None）；`load_all_catalogs()` 不触发一致性重复检测（守护 zone 参数在 per-metric 而非 paradigm_parameters）。
- **不误伤**：EPM/FST/TST 命名列齐全仍正常出 plan（无 `anonymous_zone_override` 声明）。

### backend 侧（扩展现有测试）
- `_ERROR_HINTS["zone_unnamed"]` 含统一 key `anonymous_zone_is` + `ask_clarification` + 占时证据指引，且**不含"不要/禁止"**（正面措辞断言）。
- inspect 对裸 in_zone fixture（txt 或 csv）返回含 `anonymous_zone_evidence.occupancy_ratio`；命名 zone 列文件**不带**该字段（不污染正常路径）。

---

## 5. 验收 + 回归

### 5.1 单元/契约回归（合并前必跑 —— 跨范式共享逻辑铁律 `feedback_pr_merge_must_run_full_suite_on_shared_logic`）
改了 `resolve.py`/`schema.py`/`loader.py` 是跨范式共享逻辑，**必须跑全量**（PR #66 血泪：只跑新测试致共享 fixture 静默 fail）：
```bash
cd packages/ethoinsight && .venv/bin/python -m pytest tests/ -q -p no:cacheprovider          # 基线 ≥570 passed + 新增全绿
cd packages/agent/backend && DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/ -q -p no:cacheprovider -o addopts=""
make lint   # ruff line-length 240
```

### 5.2 5 范式 demo data prep 冒烟（无误降级）
对 `/home/wangqiuyang/DemoData/newdemodata/{旷场_小鼠_三点,高架十字迷宫_小鼠_三点,明暗箱,O迷宫,强迫游泳_大鼠}` 各跑一次 prep，确认命名列齐全的范式仍出 plan、不被新 zone 逻辑误伤；FST/TST 无 zone 不受影响。

### 5.3 O迷宫完整 dogfood（本次根因数据）
1. 上传裸 in_zone 的 O迷宫文件 → lead 调 inspect → 返回含 `anonymous_zone_evidence`，in_zone=1 占时落在 **6-14%**（几何实测开放臂）。
2. prep 首调 → `error_code=="zone_unnamed"`，hint 含占时指引 + 统一 key 形态。
3. lead 用 ask_clarification 把"in_zone=1 占时约 N%，占时低侧通常是开放臂"呈现给用户 → 用户确认开放臂。
4. lead 调 `set_experiment_paradigm(parameter_overrides={"anonymous_zone_is":"in_zone"})`（统一 key）→ prep 重调，后端翻译成 `open_zones=["in_zone"]` → 出 plan。
5. code-executor 跑 `compute_open_zone_time_ratio` → 值 ≈ in_zone=1 占时（非 None、非逐字符 bug）→ 走完 data/chart/report。
6. 验收点：**用户不再被诱导答"封闭区"**（证据前置）；agent 全程只写统一 key（不分 str/list）；list 参数全链路类型正确。

### 5.4 改后重启 dev
改了 backend tools + ethoinsight 库（lead 进程内 import ethoinsight）→ `cd packages/agent && make stop && make dev`（system_prompt / tool 定义在 agent 创建时构建）。

---

## 6. 风险 / 边界 / 受保护文件

- **受保护文件**：`prep_metric_plan_tool.py`/`inspect_uploaded_file_tool.py`/`experiment_context.py` 是 Noldus 独有工具（非 deerflow 上游原生），**surgical 编辑**保留其余。三者不在 `sync-deerflow.sh` 的 22 个上游受保护清单里，但仍只改目标行。
- **zone 参数必须放 per-metric `parameters:`**（非 paradigm_parameters），否则 list default 进 `validate_catalog_consistency` 的 dict key 触发 `TypeError: unhashable`。测试守护。
- **catalog 契约不放宽**：`requires_columns` 保持命名 zone pattern 不动，触发反问靠 resolve 层 `_detect_anonymous_zone`（spec grill 定论）。
- **不引入"取反"**：几何实测已证 in_zone=1=目标区，三范式值语义统一 1=在区内；"取反"是 dogfood 错误前提下的伪需求。
- **deepseek 正面措辞铁律**（CLAUDE.md §6）：所有 hint/prompt 改动正面描述想要的行为，禁令交给 harness gate，不写"不要/禁止"。
- **统一 key value 始终是 str**：LLM 只写 `{"anonymous_zone_is": "in_zone"}`，list 包装只在 resolve 翻译层发生；故 `set_experiment_paradigm` 类型注解无需放宽到 list。

---

## 7. 关键路径速查
- catalog 翻译/放行：`packages/ethoinsight/ethoinsight/catalog/resolve.py`（`_detect_anonymous_zone` 469-510、唯一调用点 199、`_compute_parameters_in_use` 805-860 replace-only）
- catalog schema：`packages/ethoinsight/ethoinsight/catalog/schema.py`（`Catalog` 102-112、`ParamSpec` 36-44、`MetricEntry` 61-73）
- catalog loader：`packages/ethoinsight/ethoinsight/catalog/loader.py`（`_parse_catalog` ~105-134、`_parse_param_spec` default 校验、`validate_catalog_consistency` 仅遍历 paradigm_parameters）
- 三 catalog yaml：`packages/ethoinsight/ethoinsight/catalog/{oft.yaml,zero_maze.yaml,ldb.yaml}`
- compute（签名不动）：`metrics/oft.py:_find_center_zone_column`+`center_zone`；`metrics/zero_maze.py:_get_open_zone_cols`+`open_zones: list`（23/73）；`metrics/ldb.py:light_zone/dark_zone`（32-46/63-80）
- 列归一化（zone 语义源头）：`packages/ethoinsight/ethoinsight/utils.py:49-82`（无括号 `分析区中`→裸 `in_zone`）
- prep override rail：`tools/builtins/prep_metric_plan_tool.py`（`_ERROR_HINTS` 24-58、`parse_header` 拿 columns 195、读 ctx overrides 203-207、调 resolve_metrics 255-262）
- inspect：`tools/builtins/inspect_uploaded_file_tool.py`（EV19 parse_trajectory 375/429、txt 只读 5 行 `_build_data_preview_txt`）
- override 写入：`agents/middlewares/experiment_context.py`（`set_experiment_paradigm_tool` 204、`compute_analysis_config_id` 62-81 json.dumps 安全）
- 真实 demo data：`/home/wangqiuyang/DemoData/newdemodata/`
- 失败 thread（回放素材）：`packages/agent/backend/.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/6dea6ae9-121e-43e1-b71f-4ddc19e8bcd6/`（experiment-context.json 存有 LLM 脑补的 `open_zone:"~in_zone"`）
