# Spec：模板候选 A/B 顺序不确定 —— 候选列表加确定性排序（推荐项恒为 A）

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-23
> 性质：正确性 + 体验（确定性）。同一份数据两轮 E2E，模板反问的 A/B 选项对调（Run1 A=AllZones/B=FewZones，Run2 反过来）。根因：`identify_ev19_template` 构造候选列表时**没有确定性排序**，顺序跟随 `EV19_TEMPLATE_PARADIGM_MAP` 的 dict 值列表迭代序；A/B 标签按候选索引分配 → 候选序一变，A/B 就对调，且「推荐项」绑在 `target_ids[0]` 也随之漂移。**修法一刀：候选列表 sort（推荐项恒为 A，其余按 template_id 字典序），让同输入永远同 A/B 序。**
> 关联：
> - 来源：`~/ETHOINSIGHT_BUGS.md` ETHO-8；根因对照 `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md`
> - 共享病根：ETHO-7（决策点数量不一致）、ETHO-9（options 缺失）同属「交互非确定性」，但**本条是纯确定性数据修复**，与 7/9 的 guardrail 改造正交、可独立先合（故单独成 spec，见对照文档粒度决策）
> - SSOT：候选→范式映射在 `packages/ethoinsight/ethoinsight/ev19_facts.py` 的 `EV19_TEMPLATE_PARADIGM_MAP`（不在本 spec 改它，只在消费端排序）

---

## 〇、给实施 agent 的一句话

`identify_ev19_template_tool.py` 的 Step 7（[L596-638](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L596)）遍历 `target_ids` 构造 `candidates` 列表，**遍历前没排序**；A/B 字母标签在 `_build_clarification_question` 里按 `candidates` 的索引顺序分配。`target_ids` 来自 `EV19_TEMPLATE_PARADIGM_MAP.get(paradigm_key, [])`（dict 值列表），其顺序取决于该 map 初始化时的填充序，不保证跨进程/跨调用稳定 → 两轮 A/B 对调。

**修法**：在 Step 7 遍历前对 `target_ids` 做**确定性排序**——推荐项（`experiment_recs` 命中的）排最前（恒为 A），其余按 `template_id` 字典序。`is_recommended` 的判定（L630）和 A/B 标签都跟着这个稳定序走。**不碰** `EV19_TEMPLATE_PARADIGM_MAP`、不碰 `_filter_candidates_by_zone`、不碰工具返回契约。

---

## 一、根因（逐字节实证）

### 1.1 现象（ETHO-8）

同一份 EPM 数据两轮 E2E：
- Run1 模板反问：A=AllZones / B=FewZones
- Run2 模板反问：A=FewZones / B=AllZones

用户按「选 A」的肌肉记忆会选错，必须每次读语义判断。破坏可复现性 + 自动化脚本无法假设固定选项。

### 1.2 真因：候选列表无确定性排序

[`identify_ev19_template_tool.py`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py) Step 7：

```python
# L597-600
candidates: list[dict] = []
target_ids = filtered_ids if filtered_ids else candidate_ids   # ← 来源 dict 值列表，顺序不稳定

for tid in target_ids:                                         # ← 遍历前无 sort
    ...
    # L630
    is_recommended = tid in experiment_recs or (tid == target_ids[0] if target_ids else False)
    #                                                  ^^^^^^^^^^^^^^ 推荐项绑首位，顺序变则推荐项也变
    candidates.append({"template_id": tid, ..., "recommended": is_recommended, ...})
```

`target_ids` 的两个来源都不保证确定序：
- `candidate_ids` ← `EV19_TEMPLATE_PARADIGM_MAP.get(paradigm_key, [])`（`ethoinsight.ev19_facts` 的 dict 值列表）
- `filtered_ids` ← `_filter_candidates_by_zone(candidate_ids, zone_info)`（保留 `candidate_ids` 的相对序，故同样不稳定）

### 1.3 A/B 标签按候选索引分配 → 候选序决定字母

`_build_clarification_question`（[L329-391 一带](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L329)）用 `labels = "ABCDEFGHIJ"` 按 `candidates` 列表的索引位给每个候选分配字母。`candidates` 顺序 = `target_ids` 遍历序。**候选序变 → A/B 对调**，这是现象的直接成因。

### 1.4 为什么是「确定性数据修复」而非「交互流程改造」

ETHO-7/9 的根因是「LLM 自由裁量反问形态」，需要 guardrail/path_registry 才能约束。**ETHO-8 不同**：A/B 顺序由**纯代码的列表顺序**决定，不经 LLM——只要列表排序确定，输出就确定。这是个独立的、几行的确定性修复，不需要等 7/9 的大改造。

---

## 二、设计

### 2.1 修法：Step 7 遍历前对 target_ids 确定性排序

在 [L598](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L598) 之后、`for` 循环之前插入排序。排序键：**(不是推荐项, template_id 字典序)** —— 推荐项排最前（恒得字母 A），同优先级内按 `template_id` 字典序：

```python
target_ids = filtered_ids if filtered_ids else candidate_ids

# 确定性排序：推荐项恒在最前（→ 字母 A），其余按 template_id 字典序。
# 修同输入两轮 A/B 对调（ETHO-8）。experiment_recs 来自 by-experiment 推荐，
# 是稳定集合；据它把推荐项提到首位，保证「推荐项 = A」跨调用一致。
target_ids = sorted(
    target_ids,
    key=lambda t: (t not in experiment_recs, t),
)
```

- `t not in experiment_recs` 为 `False`(=0) 的（即推荐项）排在 `True`(=1) 之前 → 推荐项在前。
- 同组内 `t`（template_id 字符串）字典序 → 稳定且人类可读。
- 若有多个推荐项，它们之间也按 template_id 字典序，确定。

### 2.2 is_recommended 判定自然跟随（无需额外改）

排序后 `target_ids[0]` 必是推荐项（若存在推荐项）。L630 `is_recommended = tid in experiment_recs or (tid == target_ids[0] ...)` 的语义不变，但现在 `target_ids[0]` 是确定的那一个，不再漂移。**保留 L630 原样**（排序已让它确定）。

> 注：若 `experiment_recs` 为空（无 by-experiment 推荐），则排序退化为「全部按 template_id 字典序」，`target_ids[0]` 是字典序最小的那个、稳定。`is_recommended` 对该首位仍标 True（保持现有「至少推荐一个」行为），确定。

### 2.3 不改的东西（守约束）

- **不改** `EV19_TEMPLATE_PARADIGM_MAP`（SSOT 在 ev19_facts，本 spec 只在消费端排序，不动数据源）。
- **不改** `_filter_candidates_by_zone`（它的过滤逻辑正确，只是输出序不稳定——在它之后排序即可）。
- **不改** 工具返回的 `status` / `evidence` / `candidates` 的字段结构，只改 `candidates` 的**顺序**。
- **不改** `_build_clarification_question` 的字母分配逻辑（它按索引分配本身没错，错的是上游顺序）。

---

## 三、改动清单

### 3.1 `identify_ev19_template_tool.py` —— Step 7 加排序
- 在 [L598](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L598)（`target_ids = ...`）后加一行 `target_ids = sorted(target_ids, key=lambda t: (t not in experiment_recs, t))`。
- L600 起的 `for` 循环、L630 的 `is_recommended`、L632 的 `candidates.append` 全部不动。

### 3.2 不改其他文件
- `_build_clarification_question`、`ev19_facts.py`、lead prompt 均不动。

---

## 四、测试（红→绿坐实，TDD 强制）

测试文件：`packages/agent/backend/tests/test_identify_ev19_template_ab_order.py`（新增）。

> ⚠️ worktree 借主仓 venv 时，按 memory `feedback_worktree_shares_main_venv...` 用 `importlib.spec_from_file_location` 加载 worktree 源，或 `PYTHONPATH=packages/harness` 覆盖，**别假绿读到主仓旧代码**；先 `cat .pth` 确认指向。

1. **test_candidate_order_is_deterministic_across_calls**（核心，红→绿）
   - 构造一个会返回多候选（≥2）的 paradigm（如 EPM 的 AllZones + FewZones）。
   - 用 monkeypatch 把 `EV19_TEMPLATE_PARADIGM_MAP` 的值列表**故意打乱两种顺序**（模拟 dict 迭代序漂移），各调一次工具。
   - 断言：两次返回的 `candidates` 的 `template_id` 顺序**完全一致**。（改前红：顺序跟随打乱的输入；改后绿。）

2. **test_recommended_candidate_is_first**
   - 构造 `experiment_recs` 含某个 template_id。
   - 断言：返回 `candidates[0].template_id` == 那个推荐项，且 `candidates[0].recommended is True`。

3. **test_non_recommended_sorted_lexicographically**
   - 无 experiment_recs（或推荐项之外的候选）。
   - 断言：非推荐候选按 template_id 字典序排列。

4. **test_ab_label_stable_in_clarification_question**
   - 调 `_build_clarification_question` 或检查工具 `clarification_question` 文本。
   - 断言：字母 A 对应的 template_id 跨两次（打乱输入）调用一致。

5. **test_single_candidate_unaffected**
   - 单候选场景，断言行为不变（排序对单元素无副作用）。

---

## 五、验收标准

1. 上述 5 个测试全绿；改前测试 1 在乱序输入下红。
2. 同一份数据多次调 `identify_ev19_template`，模板反问的 A/B 字母↔模板映射**完全稳定**。
3. 推荐项恒为 A。
4. ethoinsight + backend identify 相关测试邻域全绿（`test_identify_*`），裸导入 `app.gateway` / `make_lead_agent` 0 退出（守 import 环铁律）。
5. 不改变 `status` 判定、不改变候选集合内容（只改顺序）。

---

## 六、风险与注意事项

1. **排序键依赖 experiment_recs 的稳定性**：`experiment_recs` 来自 by-experiment markdown 解析，是确定输入 → 稳定。若将来它变成非确定来源，需复查；当前是稳定集合。
2. **不要改成「按 zone_config 复杂度排序」之类语义排序**——那会引入新的判读逻辑、违反「只做确定性顺序修复」的边界。字典序 + 推荐优先即可，简单稳定。
3. **A/B 稳定 ≠ 解决 ETHO-7/9**：本 spec 只让「已经发出的模板反问」选项序稳定；「该不该发这个反问、发几个、带不带 options」是 ETHO-7/9 的范畴，另 spec。别在本 spec 里扩张到决策点编排。
4. **守 import 铁律**：identify tool 不在顶层新增跨模块 import（本 spec 只加一行 sorted，无新 import，天然安全）。

---

## milestone 建议
- 「EPM dogfood 图表/交互流水线打磨」track：ETHO-8（A/B 确定性）作为「交互非确定性」系列的第一个确定性子修复 checkpoint；ETHO-7/9（决策点编排）待后续 guardrail spec。
