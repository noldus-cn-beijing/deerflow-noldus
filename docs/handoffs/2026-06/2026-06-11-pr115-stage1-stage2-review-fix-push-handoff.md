# 交接文档 — 2026-06-11 PR#115 地基 Stage 1/2 review+修复+push

> 写给下一位 AI Agent。本次会话把 PR#115 catalog 概念整合地基的 **Stage 1 + Stage 2** 实现 review 通过、修掉问题、push 到远程。**两分支未建 PR、未合 dev**。

---

## 一句话现状

PR#115（行为学同事 issue #98 列聚合方法论 + #63 调参指南落地）的**地基重构**分 4 个 stage。Stage 1（Q1 门控 CNF）+ Stage 2（Q3 前置 concept 统一内部模型）**已实现、已 review、已修复、已 push**，**等建 PR 合 dev**。Stage 3/4 spec 已就绪未实施。

dev HEAD：`000927f2`（两分支都干净基于它）。

---

## 本次会话做了什么

### 1. Review 两分支（对照各自 stage spec）
- **Stage 1** `feature/pr115-stage1-requires-columns-cnf`：核心实现正确（CNF 嵌套列表、flatten 全消费者覆盖、`_missing_columns` str-path 字节保留、loader CNF 校验、schema 类型、catalog yaml 零改动、import 惰性）。
- **Stage 2** `worktree-pr115-stage2-zone-concept-normalization`：核心实现正确（`ParamBinding`+`ResolvedZoneConcept(binding: ParamBinding|None)` 是 Fable 决策门 1 形态、loader 规范化与旧 resolve 1a/1b 逻辑等价、resolve 改读统一模型 + `if rc.binding is None: continue`、关注点2 零改动、import 惰性+函数名直接 import 避开 `__init__` 遮蔽与闭环）。

### 2. 修掉 review 发现的问题

**Stage 1 🔴 C1（阻塞，已修，commit `6949ba4b`）**：
- `test_pure_list_of_str_resolve_unchanged` 调错 `resolve()` 签名（传 Catalog 对象非 paradigm str + 缺 `raw_files`/`workspace_dir`）→ `TypeError` 被宽 `except` 吞 → **6 范式 `tests/fixtures/cnf_baseline/*.json` 全是同一句 TypeError 错误串当基线** → 比的是"改前改后都抛同样 TypeError"，对真实 resolve 零覆盖（763 全绿里的真·假绿）。
- 修法：改用 `resolve_metrics(paradigm, columns, raw_files, workspace_dir)` + `plan_metrics_to_dict`、**移除吞错的宽 except**（调用错必须响亮失败）、抽 `_resolve_snapshot` helper（strip 易变 `generated_at`）、每范式给覆盖其 requires globs 的真实列集、**baseline 在 dev 干净态（改动前）重采**、加防假绿断言（baseline 必须含 `metrics` 或 `resolve_error`，拒绝裸 `error` 串）。
- 验证：含 CNF 的 worktree 跑该测试 20 passed（证明 CNF 未改纯 str 路径=字节等价）；故意篡改 baseline → 测试转 red（负控通过）。

**Stage 2 🟡 两项（非阻塞，已修，commit `c7751619`）**：
1. `ParamBinding`/`ResolvedZoneConcept` 加进 `catalog/__init__.py` 的 import 块 + `__all__`（否则 Stage 3/4 `from ethoinsight.catalog import ResolvedZoneConcept` 会 ImportError）。
2. **跨 stage 软缝加固**：loader 规范化 `anonymous_zone_override` 时迭代 `requires_columns` 找 in_zone glob、对 pat 直接调 `.startswith`。Stage 1（CNF）合入后 `requires_columns` 可能含 list[str] OR-组 → AttributeError。加**内联 flatten**（item is list 则展开子 pattern），不依赖 Stage 1 的 `_flatten_requires_columns`（本分支无）、**合并顺序无关**。配 `test_loader_zone_collection_tolerates_cnf_nested_requires_columns` 回归守护。

### 3. push（已完成）
```
feature/pr115-stage1-requires-columns-cnf            e41b2d82..6949ba4b  ✅ pushed
worktree-pr115-stage2-zone-concept-normalization     321e6d23..c7751619  ✅ pushed
```
远程 tip 已确认匹配本地。**两分支都未建 PR。**

### 全量测试（修复后）
- Stage 1 worktree：**763 passed, 70 skipped**（exit 0）
- Stage 2 worktree：**759 passed, 70 skipped**（exit 0，+1 软缝回归测试）
- 跑法：`cd <worktree>/packages/ethoinsight && uv run pytest tests/ -q`（**注意：无裸 `python`，必须 `uv run`**）

---

## 下一位 Agent 的待办（按优先级）

### P0 — 建两个 PR 合 dev
```
两分支 base = dev (000927f2)，均已 push，review 通过。
- feature/pr115-stage1-requires-columns-cnf → PR to dev（PR-地基-1）
- worktree-pr115-stage2-zone-concept-normalization → PR to dev（PR-地基-2）
```
**合并顺序：谁先合都安全**——Stage 2 的 loader zone 收集已对 CNF 嵌套表主动加固（本次 commit `c7751619`），不再依赖合并顺序。`schema.py`/`loader.py`/`resolve.py` 三文件两分支行区不重叠，git 自动合并应无冲突；若有冲突，`schema.py` 的 `requires_columns: list[str | list[str]]`（Stage 1）与 `resolved_zone_concepts` 字段 + `ParamBinding`/`ResolvedZoneConcept`（Stage 2）都要保留。

**合并后必做**：在 dev 上跑一次全量 `cd packages/ethoinsight && uv run pytest tests/ -q`，确认两 stage 合并后无交互回归（尤其那处 loader flatten 软缝）。

### P1 — 派 Stage 3（B 轨，依赖 Stage 2 已合）
- spec：`docs/superpowers/specs/2026-06-11-pr115-stage3-complement-zone-concepts-spec.md`
- 内容：补 OFT `border`(binding=None) / LDB `dark`(ParamBinding("dark_zone")) / ZM `closed`(ParamBinding("closed_zones")) 三概念进 `resolved_zone_concepts`。**两决策门已闭合，无待拍板。**
- **硬前置**：Stage 2 必须先合 dev（Stage 3 的 red 测试依赖 `resolved_zone_concepts` 字段存在，否则误红为 AttributeError）。

### P2 — 派 Stage 4（B 轨末尾，依赖 Stage 3）
- spec：`docs/superpowers/specs/2026-06-11-pr115-stage4-generate-concept-menu-spec.md`
- 构建期生成概念菜单（独立 `.generated.md` + CI staleness），删 SKILL.md/answer-mapping.md 手写表。

### P3 — 旧 worktree 清理（不急）
两 stage 合入后可删：
```bash
git worktree remove .claude/worktrees/pr115-stage1-requires-columns-cnf
git worktree remove .claude/worktrees/pr115-stage2-zone-concept-normalization
```

---

## 关键文档（全在 dev）

- **总纲/索引**：`docs/superpowers/specs/2026-06-11-pr115-catalog-concept-consolidation-and-gate-cnf-spec.md`（Fable 两轮裁决沉淀 + 4 stage 索引表 + 依赖图 + 边界声明）
- **4 篇 stage spec**：`docs/superpowers/specs/2026-06-11-pr115-stage{1,2,3,4}-*.md`
- Fable 裁决 memory：`feedback_fable_pr115_arch_verdicts_gate_cnf_ssot_generate`（第一轮三问）+ `feedback_fable_pr115_stage_decisions_parambinding_optional_and_buildtime_gen`（第二轮两决策门）
- 本次 review 教训 memory：`feedback_pr115_stage1_equivalence_baseline_is_hollow_error_string`

---

## 风险与注意

1. **等价性/快照测试的 baseline 必须打开看内容**（本次 C1 教训）：全是同一串错误 = 调用签名错被 except 吞 = 假绿。Stage 3 spec 也有"resolve 输出字节等价"验收项，落地时务必确认 baseline 是真实 plan 而非错误串，且采于改动前。
2. **无裸 `python`**：ethoinsight 测试用 `uv run pytest`（venv 在 `packages/ethoinsight/.venv`）。
3. **Stage 3 OFT border 是 `binding=None`**（Fable 决策门 1）：Stage 2 已加 `if rc.binding is None: continue`（resolve 注入点），Stage 3 落 border 后该 continue 才真正触发——Stage 3 的等价性回归要确认 border 进 dict 后不污染现有 resolve 注入输出。
4. **本次未碰 Stage 3/4 代码**，只 review+修+push 了 Stage 1/2。

---

## milestone 建议

PR#115 地基 track 到达 checkpoint（Stage 1/2 review 通过 push、等合）。若有 `docs/milestone/` 下对应 track（如 column-semantics-alignment 或 catalog 概念整合），建议更新：Stage 1/2 已实现 review 通过待合，Stage 3/4 spec 就绪待实施，两决策门已由 Fable 闭合。
