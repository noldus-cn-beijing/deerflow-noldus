# Catalog 判读字段下沉到 plan_metrics.json — Design

- **日期**: 2026-05-27
- **作者**: claude(opus-4-7-1m)
- **目标分支**: dev
- **状态**: Draft,等用户 review
- **前置 handoff**: [docs/handoffs/2026-05/2026-05-27-channel-todos-bug-resolved-handoff.md](../../handoffs/2026-05/2026-05-27-channel-todos-bug-resolved-handoff.md)("遗留问题"段)
- **相关 spec**: [2026-05-13-metric-catalog-architecture-design.md](2026-05-13-metric-catalog-architecture-design.md)(catalog 架构原始设计,本 spec 是它的运行时契约修正)

## 1. 背景与问题

### 1.1 用户可见症状

ECS 生产环境 FST 强迫游泳端到端请求中,`data-analyst` subagent thinking 出现约 10 步反复猜路径:

```
The catalog YAML file wasn't found at that path.
Let me try to find it via the python import approach mentioned in the skill...
The catalog YAML files are not physically present in the skill directory.
The skill says python -c ... would give the path. But I can't run bash.
...
```

LLM 最终走 fallback — 用 `handoff_code_executor.json` 中已有的 `display_name_zh` + `ethovision-paradigm-knowledge/references/by-experiment/<paradigm>.md` 凑出报告。**业务结果出来了,但**:

1. 推理 token 浪费在猜路径
2. **绕过 single source of truth**:catalog YAML 里的 `direction_for_anxiety` / `statistical_default` 等判读权威字段没被读到,LLM 用范式 markdown 的散文描述替代
3. 这次是单样本场景统计 skipped,judgement 字段恰好不需要;**多样本组间比较是 v0.1 真正目标场景,fallback 不再可行**,LLM 会要么编造判读、要么报告漏判读段

### 1.2 真根因(契约错位)

机制级复现已坐实(在本地 dev 跑过):

```python
# 真实 validate_local_tool_path + 真实 thread_data
REJECTED:  /app/backend/.venv/.../ethoinsight/catalog/fst.yaml  (PermissionError)
REJECTED:  /home/wangqiuyang/.../ethoinsight/catalog/fst.yaml   (PermissionError)
ACCEPTED:  /mnt/skills/custom/ethoinsight-metric-catalog/SKILL.md
ACCEPTED:  /mnt/user-data/workspace/handoff.json
```

精确根因 — `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py:700` 的 `validate_local_tool_path` 是一道白名单闸门,只放行 5 类虚拟路径前缀:`/mnt/user-data/*` / `/mnt/skills/*` / `/mnt/acp-workspace/*` / `/mnt/shared/*` / `config.yaml.sandbox.mounts` 配的 custom mounts。Catalog YAML 物理位置 `/app/backend/packages/ethoinsight/ethoinsight/catalog/*.yaml`(或本地 editable install 后的源码路径)**不在任何一类前缀下**,read_file 第一道闸门就被拒。

设计层面看,这是两个独立子系统的契约在 catalog 这条消费路径上交叉,谁都没意识到对方的约束:

| 子系统 | 它对 catalog 的设定 | 它没意识到 |
|---|---|---|
| ethoinsight 库 | catalog 是 Python 包内 YAML,通过 `import` + `__file__` 拿路径 — 在普通 Python 进程里正常 | subagent 不是普通 Python 进程,它走 sandbox |
| deerflow sandbox | 安全闸门白名单 — subagent 只能访问明确暴露的虚拟路径前缀 | catalog 物理路径不在白名单 |
| `ethoinsight-metric-catalog` SKILL.md | 桥接两边的文档,但**写错了** — 教 subagent 用 `__file__` 的包内路径,把 subagent 推向 sandbox 会拒的方向 | — |

dev / prod 行为一致(都走同一个 `LocalSandboxProvider` + 同一份 `tools.py` 白名单 + 同一份 `config.yaml`);本地 make dev 也会撞同样的墙。

### 1.3 已经成立的派遣前路径(本 spec 复用)

`packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py` 已经是 deerflow 内的 first-party tool,lead agent 调它时,工具在**沙箱外**(deerflow tool 进程内)直接 `import ethoinsight.catalog.resolve` 跑解析,把结果写到 `/mnt/user-data/workspace/plan_metrics.json`(白名单允许)。这条路径目前已下沉了一个字段(`display_name_zh`)给 subagent,**只是没下沉完整**。

本 spec 的核心思路是:**复用并完善这条已经成立的派遣前路径**,把 catalog YAML 从 subagent 消费路径上彻底移除。

## 2. 设计目标与非目标

### 2.1 目标

1. data-analyst 和 report-writer 不再尝试 read catalog YAML 文件
2. 这两个 subagent 需要的所有 catalog 判读 / 展示字段,都从 `plan_metrics.json` 一次读出
3. catalog YAML 仍是 single source of truth — 字段下沉只是派生(plan 由 resolve 阶段产生)
4. 未来 catalog YAML 加新字段时,**plan_metrics.json 自动带新字段**(零 prompt 改动),只在新字段语义不直观时才考虑更新 SKILL.md 字段字典

### 2.2 非目标

- **不**改 sandbox 白名单 / 不加 catalog 挂载(违反 dev/prod 行为对齐原则,见 `feedback_dev_prod_behavior_alignment`)
- **不**新增 dump CLI(每次 subagent 多一步 bash,且 data-analyst 没有 bash 工具)
- **不**改 catalog YAML 内容或 schema(`MetricEntry` schema 已经齐全,本 spec 只在 plan 序列化阶段透传)
- **不**触碰 `prep_metric_plan_tool` 的入参 / 出参 schema(只通过 `plan_metrics_to_dict()` 间接受益)
- **不**改 charts / statistics 相关字段下沉(本次只解决 metric 判读字段;chart 路径已经通过 `display_name_zh` 透传足够;statistics 走另一条 handoff 通道,不依赖 catalog 读取)

## 3. 设计方案

### 3.1 一句话总结

`packages/ethoinsight/ethoinsight/catalog/resolve.py` 的 `_metric_to_plan()` 和 `plan_metrics_to_dict()` 在生成 PlanMetric / dict 时,把 `MetricEntry` 已有的 5 个判读 / 展示字段透传下去;同步改 SKILL.md / data-analyst / report-writer 的 prompt,引导走 plan_metrics.json,删除原 catalog YAML 读取指引。

### 3.2 字段下沉清单

**MetricEntry**(`schema.py:32-41`,catalog YAML 源,**不改**)已有字段:

| 字段 | 当前在 plan? | 谁需要 | 备注 |
|---|---|---|---|
| `id` | ✅ | 所有 | — |
| `script` | ✅ | code-executor | — |
| `requires_columns` | (在 resolve 期消费,不下沉) | resolve.py | 用于列匹配,不需要 plan 携带 |
| `output_unit` | ❌ | report-writer | 物理单位标识(seconds / count),与 unit_zh 配对 |
| `display_name_zh` | ✅ | report-writer / data-analyst | 已下沉 |
| `unit_zh` | ❌ | report-writer | 中文单位 |
| `one_liner` | ❌ | report-writer | 首次提及指标时的解释 |
| `direction_for_anxiety` | ❌ | data-analyst | **核心判读字段** |
| `statistical_default` | ❌ | data-analyst | **核心判读字段** |

需要新增下沉的 5 个字段:`output_unit` / `unit_zh` / `one_liner` / `direction_for_anxiety` / `statistical_default`。

### 3.3 改动清单

#### 改动 1:`PlanMetric` dataclass 扩字段

文件:`packages/ethoinsight/ethoinsight/catalog/schema.py`

在 `PlanMetric`(line 97-105)新增 5 个字段,默认值用空字符串 / `None` 保持向前兼容(老代码序列化的 plan 反序列化时不必带新字段)。

```python
@dataclass
class PlanMetric:
    id: str
    script: str
    input: str
    output: str
    required: bool
    reason: str
    subject_index: int = 0
    display_name_zh: str = ""
    # 新增 5 个判读 / 展示字段(catalog MetricEntry 透传)
    unit_zh: str = ""
    one_liner: str = ""
    output_unit: str = ""
    direction_for_anxiety: str | None = None
    statistical_default: str = ""
```

#### 改动 2:`_metric_to_plan()` 写入新字段

文件:`packages/ethoinsight/ethoinsight/catalog/resolve.py` line 427-463

每个 PlanMetric 构造处加上 5 个字段透传(从入参 `m: MetricEntry` 取):

```python
plans.append(
    PlanMetric(
        id=m.id,
        script=m.script,
        input=raw_file,
        output=output_path,
        required=required,
        reason=reason,
        subject_index=idx,
        display_name_zh=m.display_name_zh,
        unit_zh=m.unit_zh,
        one_liner=m.one_liner,
        output_unit=m.output_unit,
        direction_for_anxiety=m.direction_for_anxiety,
        statistical_default=m.statistical_default,
    )
)
```

#### 改动 3:`plan_metrics_to_dict()` 输出新字段

文件:`packages/ethoinsight/ethoinsight/catalog/resolve.py` line 745-785

`metrics[]` 序列化项加 5 个 key:

```python
"metrics": [
    {
        "id": m.id,
        "script": m.script,
        "input": m.input,
        "output": m.output,
        "required": m.required,
        "reason": m.reason,
        "subject_index": m.subject_index,
        "display_name_zh": m.display_name_zh,
        "unit_zh": m.unit_zh,
        "one_liner": m.one_liner,
        "output_unit": m.output_unit,
        "direction_for_anxiety": m.direction_for_anxiety,
        "statistical_default": m.statistical_default,
    }
    for m in pm.metrics
],
```

同步改老 `plan_to_dict()`(line 698-742,backward-compat wrapper)的 metrics 段保持一致。

#### 改动 4:`prep_metric_plan_tool` 不动

`prep_metric_plan_tool.py` 通过 `plan_metrics_to_dict()` 写 JSON,改动 3 落地后**自动**包含新字段,无需改 tool 本身。

#### 改动 5:`ethoinsight-metric-catalog/SKILL.md` 重写

文件:`packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`

**删除**:
- Line 16-22 "catalog 物理位置" 段(教 LLM 用 `import + __file__` 的歪路)
- Line 80-96 "### data-analyst" 段中 "Step 1: 定位 catalog YAML 路径" / "Step 2: read_file 对应的 YAML 文件"(走 catalog YAML 路径)
- Line 97-103 "### report-writer" 段中 "按 metric id read catalog YAML" 引导

**新增**:plan_metrics.json `metrics[]` 字段字典,作为 subagent 消费契约的唯一文档。结构示例:

```markdown
## metric 字段字典(plan_metrics.json `metrics[]` 元素)

每个 metric 元素含以下字段(由 lead 在派遣前通过 prep_metric_plan 生成,白名单允许直接 read):

| 字段 | 类型 | 用途 | 谁消费 |
|---|---|---|---|
| id | str | metric 唯一标识 | 所有 |
| script | str | Python 调用路径 | code-executor |
| input / output / required / reason / subject_index | … | 派遣元数据 | code-executor |
| display_name_zh | str | 中文展示名 | report-writer / data-analyst |
| unit_zh | str | 中文单位(如 "秒" / "次") | report-writer |
| one_liner | str | 一句话指标解释 | report-writer(仅首次提及) |
| output_unit | str | 物理单位标识(seconds / count) | report-writer |
| direction_for_anxiety | "lower_is_anxious" / "higher_is_anxious" / null | 焦虑判读方向 | data-analyst |
| statistical_default | "groupwise_compare" / "paired_compare" | 默认统计入口标识 | data-analyst(校验 code-executor 是否走了正确入口) |
```

#### 改动 6:`data_analyst.py` prompt 改写

文件:`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` line 137-148

**删除**:
- "判读某个指标时,read catalog YAML"
- "read_file: /path/to/ethoinsight/catalog/<paradigm>.yaml"
- "(catalog 物理路径由 lead 提供给你,或从 ethoinsight-metric-catalog skill 的 SKILL.md 顶部读取定位方法)"

**改为**:

```
## 指标元数据查询

每个指标的判读字段已由 lead 在派遣前 resolve 到 plan_metrics.json,从那里取:

read_file:
    /mnt/user-data/workspace/plan_metrics.json

按 metric id 在 `metrics[]` 数组中匹配,读取以下字段:
- direction_for_anxiety: "lower_is_anxious" / "higher_is_anxious" / null
- statistical_default: "groupwise_compare" / "paired_compare"

注:多 subject 场景下同一 metric id 会出现多次(subject_index 区分),
判读字段在所有同 id 行上一致,取首个即可。

不要尝试 read catalog YAML 文件 — 它在 Python 包内,不暴露给 sandbox。
plan_metrics.json 已经包含 subagent 需要的全部字段,详见 ethoinsight-metric-catalog
skill 的字段字典。
```

#### 改动 7:`report_writer.py` prompt 改写

文件:`packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` line 163-175

**删除**:
- "写'Results / Discussion'段时,按 metric id read catalog YAML"
- "read_file: /path/to/ethoinsight/catalog/<paradigm>.yaml"

**改为**:

```
## 指标展示元数据查询

每个指标的中文展示字段已下沉到 plan_metrics.json,从那里取:

read_file:
    /mnt/user-data/workspace/plan_metrics.json

按 metric id 在 `metrics[]` 数组中匹配,读取以下字段:
- display_name_zh: 中文展示名
- unit_zh: 中文单位
- one_liner: 一句话解释(仅首次提及该指标时引用,不要在每段重复)

注:多 subject 场景下同一 metric id 会出现多次(subject_index 区分),
展示字段在所有同 id 行上一致,取首个即可。

禁止在本 prompt 内硬编码任何指标的中文名或单位 —— 全部走 plan_metrics.json。
不要尝试 read catalog YAML 文件 — 它在 Python 包内,不暴露给 sandbox。
```

### 3.4 回归测试

#### 测试 1:`tests/test_plan_metrics_has_interpretation_fields.py`(新增)

位置:`packages/ethoinsight/tests/`(因为字段下沉发生在 ethoinsight 库内)

```python
import json
from pathlib import Path
from ethoinsight.catalog.resolve import resolve_metrics, plan_metrics_to_dict

PARADIGMS = ["epm", "oft", "fst", "ldb", "zero_maze"]

EXPECTED_INTERP_FIELDS = {
    "unit_zh", "one_liner", "output_unit",
    "direction_for_anxiety", "statistical_default",
}

def test_plan_metrics_carries_interpretation_fields(tmp_path: Path) -> None:
    """每个范式 resolve 后,plan_metrics.json 的 metrics[] 项必须含 5 个判读/展示字段。"""
    for paradigm in PARADIGMS:
        # 用足以触发该范式 default_metrics 的最小 columns 集合
        # 测试时只关心 dict 输出,不实际执行 metric script
        columns = _minimal_columns_for_paradigm(paradigm)
        pm = resolve_metrics(
            paradigm=paradigm,
            columns=columns,
            raw_files=["/mnt/user-data/uploads/dummy.txt"],
            workspace_dir=str(tmp_path),
        )
        d = plan_metrics_to_dict(pm)
        assert d["metrics"], f"{paradigm}: metrics 空"
        for m in d["metrics"]:
            missing = EXPECTED_INTERP_FIELDS - set(m.keys())
            assert not missing, f"{paradigm} / {m['id']}: 缺字段 {missing}"
            # 字段类型契约
            assert isinstance(m["unit_zh"], str)
            assert isinstance(m["one_liner"], str)
            assert isinstance(m["output_unit"], str)
            assert m["direction_for_anxiety"] in (None, "lower_is_anxious", "higher_is_anxious")
            assert m["statistical_default"] in ("groupwise_compare", "paired_compare")
```

`_minimal_columns_for_paradigm` helper 视各范式 default_metrics 的 `requires_columns` 取并集生成最小列集。如果实现负担大,可改成只测 5 范式各一个有 metric 输出的 fixture。

#### 测试 2:扩展 `tests/test_prep_metric_plan_tool.py`(如存在则扩,不存在则新建)

位置:`packages/agent/backend/tests/`

跑 `prep_metric_plan_tool` 走完整链路,验证写入 workspace 的 JSON 文件确实包含 5 个新字段。这是 e2e 防线 — 改动 3 后 tool 自动受益,但有测试保底。

#### 测试 3:prompt 字符串扫描(可选,低优先级)

新增一个 lint 用 test,扫描 data_analyst.py / report_writer.py 的 prompt 字符串:**不应再含** "catalog/<paradigm>.yaml" 字面量。防止后续编辑回滚走错方向。

### 3.5 文档对齐

- `packages/ethoinsight/ethoinsight/catalog/__init__.py` docstring 现写"被 lead / data-analyst / report-writer 多方共读" — 改成"被 lead 通过 prep_metric_plan 工具消费,在 sandbox 外 resolve 后下沉到 plan_metrics.json 供 subagent 使用"
- `docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md` 在文末加 "## Addendum 2026-05-27" 段,指向本 spec,标注"subagent 不直接读 catalog YAML"为最终运行时契约

## 4. 替代方案与未选原因

### A. 加 docker-compose volume mount + config.yaml sandbox.mounts

把 catalog 目录挂到 `/mnt/skills/_ethoinsight_catalog`,改 SKILL.md 引导走这条挂载路径。

未选原因:违反 dev/prod 行为对齐(`feedback_dev_prod_behavior_alignment`)。本地 LocalSandboxProvider 读 config.yaml.sandbox.mounts,Docker 走 docker-compose.yaml 的 volumes,**两份配置同时维护**正是 2026-05-27 channel-todos bug 刚踩过的坑。即便配齐,catalog 也仍是"sandbox 外代码包内的文件",架构上把代码包暴露给 LLM 直接读是缩短 abstraction 距离,长期会牵出别的源码也撞墙(`ethoinsight.scripts.*` / `ev19_facts.py`)。

### B. 新增 `python -m ethoinsight.catalog.dump --paradigm <X> --output <JSON>` CLI

subagent 用 bash 跑 dump,把 catalog 翻成 JSON 后再 read_file。

未选原因(两个独立硬伤):
1. `data_analyst.py:158-160` `disallowed_tools=[... "bash" ...]` — data-analyst **本来就没有 bash 工具**。新 CLI 走不通。
2. 即便给 data-analyst 加回 bash,**两份消费契约同时存在**(派遣前 resolve + 派遣后 dump),哪条改了哪条会被 LLM 优先用是无法预测的;只是把契约错位从"catalog YAML 路径"转移到了"dump 命令 vs plan 字段"。

### C. plan_metrics.json schema 扩展(本 spec 选择,即方案 D 在 brainstorm 中的命名)

见上文 §3。这是 brainstorm 阶段标的"方案 D",由原 A/B/C 三选项中演化而来;原 C(仅 schema 扩展)缺"删 SKILL.md 引导"和"改 prompt"两步,会让两份契约同时存在;最终方案把这三步绑成一组改动,才能真正"把 catalog YAML 从 subagent 消费路径上移除"。

## 5. 风险与缓解

### 风险 1:LLM 看到 plan_metrics.json 字段变多产生噪声

每个 metric 项从 8 字段(现状)变成 13 字段(本 spec 后)。

缓解:5 个新字段语义直观(中文名 / 单位 / 一句话解释 / 焦虑方向 / 统计入口),不引入嵌套结构,LLM 消费成本不高。subagent prompt 也改成"按 metric id 在 metrics[] 中匹配,读取以下字段"显式列名,LLM 不需要扫全 JSON。

### 风险 2:回归 — 老 plan_metrics.json 文件(已写盘的)不带新字段

PlanMetric dataclass 给新字段配默认值(`""` / `None`),老 plan_metrics.json 在反序列化时不会炸。但 data-analyst / report-writer 改 prompt 后会主动 read plan_metrics.json — 老文件被读会发现缺新字段。

缓解:plan_metrics.json 不跨 thread 持久化(每 thread workspace 独立),不存在"老线程持久 plan + 新 subagent prompt"的混合场景。即便 dev 调试期间留下老 plan,subagent 收到的 task prompt 是 lead 当次派遣时拼的,会跟着新一次 prep_metric_plan 生成新 plan。

### 风险 3:字段从 catalog YAML 下沉到 plan,plan_metrics.json 序列化后字段值可能不一致

如果 resolve.py 透传出 bug(比如下沉 `direction_for_anxiety` 时写错字段名),subagent 拿到错值。

缓解:回归测试 §3.4 测试 1 在 5 个范式各跑一次 resolve_metrics + plan_metrics_to_dict,断言字段值与 catalog YAML 同源(直接 load_catalog 取对照值)。

### 风险 4:未来 catalog 加新判读字段时,plan_metrics.json 自动带,但 subagent prompt 没说该消费它

例:加 `effect_size_threshold` 字段。subagent prompt 不更新,LLM 看到 JSON 里的新 key 是否会自然消费?

缓解:语义直观的字段(如 `effect_size_threshold`),LLM 看到 JSON key 会自然用;语义不直观的字段需要更新 SKILL.md 字段字典 + 触及该字段的 subagent prompt。这是单文件维护成本,可接受。本 spec 推荐:每次给 MetricEntry 加新字段时,在 `plan_metrics_to_dict()` 的 dict 字面量也加 key 是强约束(防漏),SKILL.md 是软约束(可选)。

### 风险 5:scope creep — 顺手改 charts / statistics 字段下沉

charts 已经透传 `display_name_zh`,statistics 走另一条 handoff 通道,不在本 spec 范围内。如果实施时发现 chart-maker 也有类似猜路径问题(目前 chart-maker 走 `ethoinsight-chart-maker` skill,但本会话未取证它是否也有 catalog YAML 读取需求),开独立 spec,不在本次合并。

## 6. 实施顺序与验证

1. 改动 1-3(ethoinsight catalog 库代码改动):一个 commit
2. 测试 1(ethoinsight 库回归):同 commit 内
3. 改动 5-7(skill + 两个 subagent prompt):一个 commit
4. 测试 2(prep_metric_plan_tool e2e):同 commit 内
5. 文档对齐(§3.5):一个 commit
6. **手工验证步骤**:本地 make dev 跑一次完整 FST 单样本(复现原症状的最小场景),观察 data-analyst thinking — 不再出现"catalog YAML file wasn't found"猜路径,直接走 plan_metrics.json

最终 commit 前跑:
- `cd packages/ethoinsight && pytest tests/`
- `cd packages/agent/backend && make test`
- `cd packages/agent/backend && make lint`

## 7. milestone 影响

本 spec 完结后,deploy 链路 + catalog 消费契约两条线都达到稳态。建议在 `docs/milestone/` 下相关 track 加 checkpoint 摘要,记一句"2026-05-27: catalog 消费契约固化 — subagent 不再直接读 catalog YAML,所有判读字段统一走 plan_metrics.json"。
