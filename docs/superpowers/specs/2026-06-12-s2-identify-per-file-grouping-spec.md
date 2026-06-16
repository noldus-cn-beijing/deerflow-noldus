# Spec S2: identify_ev19_template 一次性返回全部文件的分组字段（消除逐个 inspect 试探）

> 日期：2026-06-12
> 顺序：第 2 份（共 4 份，按序实施）。承接 S1：S1 修了「逐个 inspect 撞 loop limit 卡死」，S2 治本——让 lead 一次拿到全部分组，**根本不必逐个 inspect**。
> 来源：dogfood thread `a5b97c00` 卡死的第二层根因 + 代码坐实
> 实施方式：新开 worktree 基于**已合 S1 的最新 dev**，单 PR
> 前置：S1 已合 dev（loop_detection 接 config）；calamine 引擎已合 dev（PR #125，`parse_header` 已加速到 ~0.09s/文件）

---

## 0. 背景与根因（实证，非推测）

dogfood thread `a5b97c00`：用户说「你帮我看下数据的 group 信息决定」，lead 逐个 `inspect_uploaded_file` 探查 28 个文件找分组边界（XX/YY/YZ 在哪个 Trial 切换），调到第 5 次撞 loop limit 卡死（S1 已修 limit，但**逐个试探本身是错的路径**）。

**第二层根因（代码坐实）**：

1. **identify 只读第 1 个文件**：[identify_ev19_template_tool.py:424-425](packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py) `first_file = uploaded_files[0]` —— identify 为「模板识别」只解析第一个文件的列结构，**根本没读其他 27 个文件**，返回结构（`ev19_template`/`candidates`/`evidence`/`domain_summary`，三个 return 点 :531/:552/:561）**完全不含任何文件的分组字段**（grep `grouping_fields`/`Treatment`/`Group` 在 identify 工具里零命中）。

2. **prompt 与现实矛盾**：[prompt.py:470](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) 说「identify 已返回每个文件的列结构/元数据，不需要再 inspect」——**不准确**，identify 没返回分组。

3. **分组字段只有 inspect 提取**：单文件工具 `inspect_uploaded_file` 有 `_extract_grouping_fields`（[inspect_uploaded_file_tool.py:278](packages/agent/backend/packages/harness/deerflow/tools/builtins/inspect_uploaded_file_tool.py)），从 `parse_header` 的 `raw_metadata` 提取 `Treatment/Group/Dose/...`。

→ lead 要拿 28 个文件的分组，**只能逐个调单文件 inspect 试探边界**，本该一次批量拿到。

**关键证据（实测）**：EV19 头的 `raw_metadata` 含分组字段（如 `{'Group': 'XY'}`）；calamine 合入后 `parse_header` 仅 **0.09s/文件**，遍历 28 文件提分组 **≈2.6s**，完全可接受（只读 header 不读 trajectory）。

> **教训已落 memory**：`feedback_loop_detection_config_not_wired_runs_hardcoded_default.md`（第二层根因段）+ `feedback_ev19_header_has_treatment_field`（EV19 头自带分组字段该一次批量提取）。

---

## 1. 修复：identify 遍历全部文件，返回 `per_file_grouping`

### 1.1 复用纯函数（不重写、不双存）
`_extract_grouping_fields(raw_metadata) -> dict[str, str]`（inspect_uploaded_file_tool.py:278）是**纯函数**（只吃 `raw_metadata` dict），`_GROUPING_METADATA_KEYS`（:36，含 `Treatment/Group/Drug/Dose/Condition/Compound` 中英 + Animal ID）。

**复用方式**：把 `_extract_grouping_fields` + `_GROUPING_METADATA_KEYS` **抽到共享位置**，identify 和 inspect 都 import（避免两份副本漂移，守 SSOT 铁律 `feedback_single_source_of_truth`）。建议抽到一个轻量模块（如 `tools/builtins/_ev19_grouping.py` 或 inspect 工具暴露为可 import）。**不在 identify 里复制一份**。

### 1.2 identify 新增「Step 3.5：遍历全部文件提取分组」
在 [identify 工具 Step 3](packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py)（解析 first_file 之后，:443 zone 检测附近）加一步：
```python
# Step 3.5: 遍历全部 uploaded_files 提取每个文件的分组字段（EV19 头自带 Treatment/Group/...）。
# 让 lead 一次拿到全部分组，无需逐个 inspect_uploaded_file 试探边界。
# 只读 header（parse_header ~0.09s/文件，calamine 已生效），不读 trajectory。
per_file_grouping: dict[str, dict[str, str]] = {}
for f in uploaded_files:
    real_f = replace_virtual_path(f, thread_data)
    if not Path(real_f).exists():
        continue  # 缺文件不阻断模板识别；记空即可
    try:
        h = parse_header(real_f)
        gf = _extract_grouping_fields(h.get("raw_metadata", {}) or {})
        if gf:
            per_file_grouping[Path(f).name] = gf  # 用文件名做 key（lead 按文件名对照分组）
    except Exception:
        continue  # 单文件解析失败不阻断（防御性，同 inspect 风格）
```
- **key 用文件名**（`Path(f).name`），不是虚拟路径——lead 按文件名（如 `Raw data-...-Trial 1.xlsx`）对照分组更直观。
- **失败容忍**：单文件缺失/解析失败 `continue`，不阻断模板识别主流程（分组是 best-effort 增强，不是 identify 的核心职责）。

### 1.3 三个 return 点都挂 `per_file_grouping`
identify 有三个 return 点（:531 unknown / :552 ok / :561→ambiguous 的 `_write_template_candidates` 后），**ok 和 ambiguous 两个 return dict 加 `"per_file_grouping": per_file_grouping`**。unknown/unsupported/error 路径可不带（那些路径 lead 不会走到派遣分析）。
- ok return（:552）：加 `"per_file_grouping": per_file_grouping`
- ambiguous return（:570 附近的 return）：加 `"per_file_grouping": per_file_grouping`

### 1.4 性能边界（写进代码注释）
- 遍历只调 `parse_header`（读元数据），**绝不调 `parse_trajectory`**（读全表）——后者每文件 ~0.1s（calamine）但仍是 trajectory 全读，分组不需要。
- 28 文件 ≈2.6s，是 identify 一次调用内的同步成本，可接受。若未来文件数 >100，再考虑并行（本次不做，记 TODO）。

---

## 2. 更新 prompt：让 lead 优先用 identify 的 per_file_grouping

[prompt.py:470-471](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)（⚠️ 实施前确认 prompt.py 是干净 dev 版——S1 review 时发现它曾被过时 stash 污染过，已还原；S2 worktree 基于合 S1 的 dev 应已干净，但实施时 grep 确认无 `<<<<<<<` 冲突标记）：

改「identify 已返回元数据不用 inspect」段，明确：
- **identify_ev19_template 现在返回 `per_file_grouping`**（每个文件名 → 分组字段 dict，如 `{"Raw data-...-Trial 1.xlsx": {"Group": "XX"}, ...}`）。
- lead 构造分组时**优先用 identify 的 `per_file_grouping`**——它一次给出全部文件的分组，**不要逐个调 `inspect_uploaded_file` 试探边界**。
- 仅当 `per_file_grouping` 为空（EV19 头无分组字段）或不足以推断分组时，才 fallback 到 inspect_uploaded_file 看数据预览行 / ask_clarification 问用户。
- 用正面指令（deepseek 正面提示铁律，CLAUDE.md 第 6 条）：描述「优先用 per_file_grouping 一次性提取」，而非「禁止逐个 inspect」。

> **不删 inspect_uploaded_file 的分组能力**——它仍是 per_file_grouping 为空时的 fallback（如自定义分组标签需看预览行）。S2 只是让 identify 成为分组的**首选批量来源**，inspect 退为兜底。

---

## 3. 测试（TDD）

### 单元测试（harness）— 新建 `tests/test_identify_per_file_grouping.py`
1. **per_file_grouping 提取（红→绿）**：mock 多个 EV19 文件（或用真实 fixture，可复用 #125 入库的 `packages/ethoinsight/tests/fixtures/原始数据-Elevated Plus Maze XT190-Trial 1.xlsx`），调 identify，断言返回的 `per_file_grouping` 含每个文件的分组字段。改前红（identify 不返回该字段），改后绿。
2. **多文件遍历**：构造 3 个文件，identify 返回的 `per_file_grouping` 有 3 个 key（文件名），每个含正确分组。
3. **失败容忍**：一个文件缺失/损坏，identify 不抛、不阻断，`per_file_grouping` 含其余文件。
4. **只读 header 不读 trajectory**：mock/spy `parse_trajectory` 断言**未被调用**（性能契约），只 `parse_header` 被调。
5. **共享纯函数无双存**：断言 identify 和 inspect 用的是同一个 `_extract_grouping_fields`（import 自同一模块）。

### prompt 契约测试 — `tests/test_lead_prompt_grouping.py`（或并入现有 prompt 测试）
6. prompt 含「优先用 identify per_file_grouping」指引，含正面措辞（非「禁止 inspect」）。

### 回归
7. 现有 identify 测试全绿（`pytest tests/ -k identify`）——确认加 Step 3.5 不破坏模板识别主流程（ok/ambiguous/unknown/unsupported 四态）。
8. inspect 测试全绿（抽共享函数不破坏 inspect）。

---

## 4. 验证（端到端）
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. .venv/bin/python -m pytest tests/test_identify_per_file_grouping.py -q
PYTHONPATH=. .venv/bin/python -m pytest tests/ -k "identify or inspect" -q
# 改了 tools/builtins 核心 → 裸导入两生产入口
PYTHONPATH=. .venv/bin/python -c "import app.gateway"
PYTHONPATH=. .venv/bin/python -c "from deerflow.agents import make_lead_agent"
# 全量回归（基线含 test_subagent_executor 的 ModuleNotFoundError 环境债 + 其它已知，勿归因本次）
PYTHONPATH=. .venv/bin/python -m pytest -q
```
最终：复跑 dogfood，lead 调一次 identify 即拿到全部 28 文件分组（per_file_grouping），构造 groups dict 无需逐个 inspect。

---

## 5. 关键文件
- `packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py`（Step 3.5 遍历 + 三 return 挂 per_file_grouping）
- `packages/agent/backend/packages/harness/deerflow/tools/builtins/_ev19_grouping.py`（**新建**，抽 `_extract_grouping_fields` + `_GROUPING_METADATA_KEYS` 共享）或等价共享位置
- `packages/agent/backend/packages/harness/deerflow/tools/builtins/inspect_uploaded_file_tool.py`（改为 import 共享函数，删本地副本）
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（:470 段更新指引）
- `packages/agent/backend/tests/test_identify_per_file_grouping.py`（新建，红→绿）

## 6. 红线（勿违）
- `_extract_grouping_fields` **抽共享、不复制副本**（守 SSOT，inspect 和 identify 同一份）。
- identify 遍历**只调 `parse_header` 不调 `parse_trajectory`**（性能契约，测试断言后者未被调用）。
- prompt 用**正面指令**（"优先用 per_file_grouping"），不用"禁止逐个 inspect"（deepseek 反向激活）。
- **不删 inspect 的分组 fallback**——per_file_grouping 为空时仍需它。
- 实施前 grep prompt.py / identify 工具确认无 `<<<<<<<` 冲突标记（S1 review 教训：worktree 曾被过时 stash 污染）。
- 改 tools/builtins 后必跑裸导入两生产入口（conftest mock 藏循环导入）。
- `test_subagent_executor` 的 ModuleNotFoundError 失败是纯 dev 环境债，勿归因本次。
