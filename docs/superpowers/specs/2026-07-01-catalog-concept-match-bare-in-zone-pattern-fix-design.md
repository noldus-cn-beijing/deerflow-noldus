# 设计 spec：catalog `_concept_matches_pattern` 裸 `in_zone*` 越界 bug 修复（2026-07-01）

> 治 EZM/LDB/OFT 的 box/bar 图因裸 `in_zone*` pattern 被 catalog 误判「缺列」而 skip 的纯函数 bug。**一处纯函数下标 bug，跨三范式共 7 处 chart 命中。**
>
> 来源：prod thread `28fc60ee`（zero_maze「箱线图无法生成」）triage → `/home/wangqiuyang/triage-28fc60ee/ROOT_CAUSE.md`。根因经**本会话独立复现坐实**（不信 handoff 二手叙述，跑最小复现 + 对当前源码逐条核）。走 ROOT_CAUSE §4 路 B（改纯函数）。

---

## 根因（独立复现坐实）

`packages/ethoinsight/ethoinsight/catalog/resolve.py:643`：

```python
if pat.startswith("in_zone") and "*" in pat:
    keyword = pat[len("in_zone_"):].rstrip("*").rstrip("_")   # len("in_zone_") == 8
    if keyword and concept == keyword:
        return True
```

`len("in_zone_") == 8`。对裸 `in_zone*`（长 7）→ `pat[8:] == ""` → `keyword == ""` → `if keyword and ...` 短路 → False。对齐后的 `open`/`closed`/`center`/`light` concept **匹配不上** `in_zone*` → box/bar 图被 `_missing_columns` 判缺列 → `resolve_charts` 塞进 `skipped[]`、`charts` 保持空。

**最小复现（本会话实跑，坐实）**：
```
open vs in_zone*    : False   ← bug（应 True）
open vs in_zone_open*: True   ← 对照（带 concept 不踩）
pat[8:] for in_zone*: ''      ← 越界成空串
```

**不对称根源**：metric 的 requires_columns 用 `in_zone_open*`（带 concept，长度 ≥ 8，keyword 提取正确）→ metric 100/100 算成功；chart 用裸 `in_zone*`（无 concept）→ 踩 bug → "指标能算、图不能画"。

**跨范式命中（本会话 grep 坐实，非 EZM 孤例）**：裸 `in_zone*` 用于
- `zero_maze.yaml:109/119`（box_open_zone / bar_open_zone，开放区）
- `ldb.yaml:80/90`（box_light / bar_light，亮室）
- `oft.yaml:183/190/197`（center 箱线/柱状/汇总，中心区）

共 **7 处 chart**。→ **排除路 A（只改 zero_maze.yaml）**：改 EZM yaml 漏 LDB/OFT。

## 修法（路 B + 宽松语义 + 裸形态直接放行）

改 `_concept_matches_pattern` 一处纯函数。

**核心洞察**：该函数收到的 `concept` **必然是 `column_aliases` 已对齐产物**——上游 `_any_concept_matches_pattern`（resolve.py:599-620）只遍历 `column_aliases` 的对齐 concept，已排除 `None`/`__ignore__`。所以 concept 能走到这里，就说明有真实列对齐到了它。

**语义决策（宽松）**：裸 `in_zone*` 是「存在分析 zone 列即可」的**弱声明**（作者若要精确会写 `in_zone_open*` 像 metric 那样）。门（`_concept_matches_pattern`）只管「有没有 zone 列」，**挑对/算对是 chart 脚本层的责任**（脚本已用 aliases 对齐，如 `plot_box_open_zone` 走 `compute_paradigm_metrics`）。

**判定机制（裸形态直接放行）**：利用上述洞察——
- **裸 `in_zone*`**（`rstrip("*")` 后 == `"in_zone"`，无 concept keyword）→ **直接返回 True**（能走到这里就说明有对齐 zone 列）。无关键词白名单、无 zone 集判定、无签名改动、零漂移。
- **带 concept 的 `in_zone_open*`** → 精确 keyword 匹配，**行为完全不变**。

**改动骨架（示意，实施细化）**：
```python
if pat.startswith("in_zone") and "*" in pat:
    base = pat.rstrip("*")                          # "in_zone*"→"in_zone"; "in_zone_open*"→"in_zone_open"
    if base == "in_zone":                           # 裸形态：弱声明「有分析 zone 列即可」
        return True                                 # concept 必是已对齐 zone 产物（上游保证）
    keyword = base[len("in_zone_"):].rstrip("_")    # 带 concept：原精确匹配，行为不变
    if keyword and concept == keyword:
        return True
return False
```

关键：修法**只对裸 `in_zone*` 放行**（判 `base == "in_zone"`），与带 concept 形态区分，**不误伤** metric。

## 排除的替代方案

- **路 A（改 zero_maze.yaml 的 `in_zone*` → `in_zone_open*`）**：只修 EZM，漏 LDB/OFT 同 bug（7 处跨三范式）；且给别范式留坑、易漂移。❌
- **方案 B（收紧：`in_zone*` 只匹配范式「主分析 zone」）**：引入「范式→主 zone」新映射层 = 新 SSOT = 新漂移源；为边缘情形（数据缺主 zone 列，实践几乎不发生——真缺 metric 早崩在前面）引入常驻复杂度，YAGNI 违例。❌
- **判定机制 A（查是否在已对齐 zone 集）/ B（zone 关键词白名单）**：前者冗余（concept 必已是对齐产物），后者硬编码名单必漏。裸形态直接 True（本 spec 采用）最干净。❌

## 测试策略（守 TDD + 防 seesaw + 防 vacuous + reward-hacking 自检）

放 `packages/ethoinsight/tests/`：
1. **EZM**：`_concept_matches_pattern("open", "in_zone*") == True`、`("closed", "in_zone*") == True`（原 bug 处，现绿）。
2. **LDB**：light concept + `in_zone*` → True。
3. **OFT**：center concept + `in_zone*` → True。
4. **对照防回归（关键，防 seesaw）**：`in_zone_open*` + concept `open` → 仍 True；`in_zone_open*` + concept `closed` → 仍 False（带 concept 的精确匹配**行为不变**，证没把精确匹配也放松）。
5. **端到端验收（守 reward-hacking 自检——看真产物非自述）**：`resolve_charts("zero_maze", ezm_columns, ..., column_aliases={"in_open":"open","in_closed":"closed"})` 的返回 `charts` **非空且含 `box_open_zone`**（看 resolve 真实输出，不看 handoff 自述）。
6. **防 vacuous 探针（必须实跑观察红）**：临时删/注释修复行（`if base == "in_zone": return True`）→ 跑测试 1/2/3 + 端到端，**观察它们变红并贴出红证据**，再恢复跑绿（仅写测试不算——要证明「修复不存在时测试会红」）。

## 影响面 / sync / 边界

- **改动面**：`packages/ethoinsight/ethoinsight/catalog/resolve.py` 一个纯函数 `_concept_matches_pattern` + 测试。`ethoinsight` 库，**deerflow 子树外**，sync 无关。
- **catastrophic forgetting 自检**：改前 grep 全范式 catalog 的 `in_zone*` 用法（已做：zero_maze/LDB/OFT 共 7 处），改后全量跑 `cd packages/ethoinsight && pytest tests/`。
- **under-exploration 自检**：这是结构 bug（纯函数下标越界），上门改纯函数是对的，不是「加 prompt 提醒」。

## 不做什么（YAGNI）

- ❌ 不改任何 catalog yaml（路 A 排除）。
- ❌ 不引「主 zone」映射层（方案 B 排除）。
- ❌ 不碰 metric 的 `in_zone_open*` 精确匹配（行为不变，仅对裸 `in_zone*` 放行）。
- ❌ 不碰 chart-maker/lead 体验层——ROOT_CAUSE §5 的「chart-maker charts=[] 应 fail-loud 标注 catalog 0 命中」+「lead 反复 read 同一 handoff 触发 FORCED STOP」是**诱因层**，另立 issue，优先级低于本根因修复。

## 验收标准

1. EZM/LDB/OFT 的裸 `in_zone*` box/bar 图不再被误 skip（三范式各有断言）。
2. metric `in_zone_open*` 精确匹配行为不变（防回归对照绿）。
3. `resolve_charts` 真实 `charts` 输出含 box 图（端到端，不看自述）。
4. 防 vacuous 探针实跑观察红 + 全量 ethoinsight 测试绿。

## 关联

- triage：`/home/wangqiuyang/triage-28fc60ee/ROOT_CAUSE.md`（含 thread.json/plan_charts.json/handoff/experiment-context 证据）；prod thread `28fc60ee-74ae-4977-9705-6a45419ab708`。
- 现有代码：`packages/ethoinsight/ethoinsight/catalog/resolve.py`（`_concept_matches_pattern:623` / `_any_concept_matches_pattern:599` / `_missing_columns` / `resolve_charts`）、`zero_maze.yaml:109/119`、`ldb.yaml:80/90`、`oft.yaml:183/190/197`。
- 守 memory：`feedback_chart_requires_columns_gate_distinct_from_zone_alias_overrides`（chart requires_columns gate 与 zone alias 对齐正交，本 bug 正是此门崩了）、`feedback_grill_handoff_must_be_verified`（接 triage 先现场核实——已独立复现）、`feedback_no_cross_paradigm_reuse_accept_duplication`（不引跨范式「主 zone」抽象）、`feedback_pr_merge_must_run_full_suite_on_shared_logic`（改共享 helper 全量跑 + grep 消费方）、`reference_harnessx_report_and_etho_spec_application`（reward hacking / catastrophic forgetting / under-exploration 三自检）。
- CLAUDE.md 判读哲学第 9 条无关（本 bug 纯 catalog 列匹配，不涉判读）。
