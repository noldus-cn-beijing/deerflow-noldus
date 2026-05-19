# EV19 模板识别地基 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Lead Agent 通过对话识别 EthoVision XT 19 真实模板（20 大类 / 62 变体），把 `ev19_template` 字段写入 `experiment-context.json`，作为 PRD v2 MVP 6 范式的工程地基。

**Architecture:** 新建 `ethovision-paradigm-knowledge` skill（渐进披露 markdown 知识）+ 升级 `set_experiment_paradigm` 工具签名（加 `ev19_template` 必填 + 白名单）+ 新增 `Ev19TemplateGuardrailProvider`（拒绝 `ev19_template=null` 时的 `task('code-executor')` 派遣）+ 删除 lead agent prompt 旧的「7 大类 18 范式」段。复用 deerflow 现成的 `LoopDetectionMiddleware` 和 `ClarificationMiddleware`，不重复造轮子。

**Tech Stack:** Python 3.12（agent backend）/ Python 3.10+（ethoinsight）/ LangGraph + langchain agent middleware / pytest / ruff.

**Spec:** [docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md](../specs/2026-05-08-ev19-template-skill-foundation-design.md)

---

## 文件结构（决策预先锁定）

### 创建

| 路径 | 责任 |
|---|---|
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md` | skill 入口：决策树 + 大类索引 + 何时反问指引 |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/_facts.md` | 62 变体事实表（人类可读） |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/identification-decision-tree.md` | agent 决策流程 + 反问质量准则 |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md` | 范式 → 默认变体降级表 |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-template/` | 20 大类知识（从 review 包搬入） |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/` | 20 实验范式知识（从 review 包搬入） |
| `packages/ethoinsight/ethoinsight/ev19_facts.py` | Python 模块：62 变体白名单 + paradigm 兼容性映射（事实表的代码版） |
| `packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py` | `Ev19TemplateGuardrailProvider`，实现 `GuardrailProvider` 协议 |
| `packages/agent/backend/tests/test_ev19_template_guardrail_provider.py` | provider 单元测试 |
| `packages/ethoinsight/tests/test_ev19_facts.py` | 事实表单元测试 |
| `packages/agent/backend/tests/test_set_experiment_paradigm_ev19.py` | 工具签名升级单元测试 |

### 修改

| 路径 | 修改点 |
|---|---|
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` | `set_experiment_paradigm_tool` 加 `ev19_template` 必填参数 + 白名单校验 + paradigm 兼容性 warning |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 删除「7 大类 18 范式」段（约 100 行），引导 agent 用新 skill |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` | 注册 `GuardrailMiddleware` + `Ev19TemplateGuardrailProvider`（仅当 `guardrails.enabled=true`） |
| `packages/agent/config.yaml` | 启用 `guardrails.enabled=true`；指定 provider 类路径 |
| `packages/agent/extensions_config.json` | 启用 `ethovision-paradigm-knowledge` skill |
| `packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md` | 引用 `ev19_template` 字段（取代单独的 paradigm 字段引用） |

### 不修改（保留）

- `GateEnforcementMiddleware` — 继续负责 `paradigm` 字段拦截，与新 GuardrailMiddleware 职责正交
- 5 个 ethoinsight subagent — 流程不变
- handoff JSON 契约 — subagent 之间通信方式不变

---

## Task 1: 行为学知识包搬迁到 skill 目录

**Files:**
- Create: `packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md`
- Create: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-template/` (20 .md files copied)
- Create: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/` (20 .md files copied)

- [ ] **Step 1: 创建 skill 目录结构**

```bash
mkdir -p /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-template
mkdir -p /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment
```

- [ ] **Step 2: 复制 review 包所有 markdown 文件到 skill references**

```bash
cp /home/wangqiuyang/noldus-insight/docs/review-packages/2026-04-29-ev19-templates/by-template/*.md \
   /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-template/

cp /home/wangqiuyang/noldus-insight/docs/review-packages/2026-04-29-ev19-templates/by-experiment/*.md \
   /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/
```

- [ ] **Step 3: 验证文件搬迁完成**

```bash
ls /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-template/ | wc -l
ls /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/ | wc -l
```

Expected: 两个数字都 ≥ 20。

- [ ] **Step 4: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethovision-paradigm-knowledge/
git commit -m "新增 ethovision-paradigm-knowledge skill 骨架并搬入 review 包 markdown"
```

---

## Task 2: 编写 SKILL.md 入口文件

**Files:**
- Create: `packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md`

- [ ] **Step 1: 写入 SKILL.md**

写入以下完整内容到 `packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md`：

```markdown
---
name: ethovision-paradigm-knowledge
description: >
  EthoVision XT 19 模板知识库（20 大类 / 62 变体）+ 学术实验范式映射。
  用于在用户上传数据并请求分析时识别其使用的 EV19 模板变体（如 PlusMaze-AllZones），
  并把识别结果作为 ev19_template 字段写入 experiment-context.json。
  使用对话识别（read 用户消息 + 文件名 + 必要时 read raw txt meta），
  必要时通过 ask_clarification 反问，反问失败时按 default-template-fallback.md 兜底。
version: 0.1.0
author: noldus-insight
---

# EthoVision Paradigm Knowledge — EV19 模板识别 + 学术范式映射

## 何时使用此 skill

**必须使用**：用户提到任何 EthoVision 实验数据分析需求时（含上传 raw txt 文件 + 请求分析/统计/可视化/报告）。

**可跳过**：纯知识问答（无数据上传 + 概念性问题）；追问已有分析结果；闲聊。

## 核心原则

1. **EV19 模板 = 用户语言**（agent 与用户对话时使用），**学术范式 = 内部分析路径**（agent 调 set_experiment_paradigm 时填这个）。
2. **不要硬猜**——如果信息不足，**用 ask_clarification 给结构化选项**让用户选；不要瞎填导致下游分析跑错路径。
3. **反问最多 1 次**——LoopDetectionMiddleware 会在重复反问时强制中断；如果第一次反问后用户答 "不知道"，按 references/default-template-fallback.md 选默认值进入分析。
4. **反问前必读 raw 文件 meta**——用 read_file 读用户上传的第一个 raw txt 前 50 行，看单位（毫米=鱼 / 厘米=啮齿）、追踪点（单点/三点）、zone 列结构，把候选缩到 ≤3 个再问。

## Workflow

### Step 1: 收集证据

读以下信息（由 agent 综合判断）：
- 用户消息文本（"高架十字迷宫"、"EPM"、"焦虑测试" 等关键词）
- 上传文件名（"轨迹-EPM-Trial 1...txt" 等）
- 文件数量 + Subject 数（5 Subject = shoaling / 三箱社交，2 Arena = 三箱社交）
- 必要时 read_file 第一个 raw txt 前 50 行查 meta + 列结构

### Step 2: 决策

按 `references/identification-decision-tree.md` 决策：
- 候选 = 1 高置信度 → 直接 set_experiment_paradigm（不反问）
- 候选 2-3 → ask_clarification 给结构化选项 + 推荐项放第一位 + 默认值兜底说明
- 候选 0 或 ≥4 → ask_clarification 先问大实验类型

### Step 3: 调 set_experiment_paradigm

```
set_experiment_paradigm(
    paradigm="epm",                    # 学术范式 key（snake_case 英文）
    paradigm_cn="高架十字迷宫",         # 中文显示名
    category="anxiety",                # 大类
    subject="rodent",                  # rodent | fish | insect | other
    ev19_template="PlusMaze-AllZones", # EV19 变体 ID（白名单内）
)
```

工具会校验 `ev19_template` 在 62 变体白名单内；如不在，会返回错误 + 候选模板。

## 知识资源（按需 read_file 加载）

- `references/_facts.md` — 62 变体事实表（机器抽取的 arena/zone/subject 字段，最权威）
- `references/identification-decision-tree.md` — 决策流程详解 + 反问质量准则
- `references/default-template-fallback.md` — 范式 → 默认变体降级表（反问失败时用）
- `references/by-template/<大类>.md` — 单个 EV19 大类的变体差异 + 推荐场景（同事 PR 持续补充）
- `references/by-experiment/<范式>.md` — 单个学术范式的指标 / 模板候选 / 解读语言（同事 PR 持续补充）

**Token 节省提示**：不要一次性加载所有 references，按对话需要 read_file 单个文件。
```

- [ ] **Step 2: 启用 skill**

修改 `/home/wangqiuyang/noldus-insight/packages/agent/extensions_config.json`，在 `skills` 字段加入：

```json
{
  "skills": {
    "ethovision-paradigm-knowledge": {"enabled": true}
  }
}
```

如果文件不存在，先 `cat` 看 example 文件并参照创建。如果其他 skill 已经在 `skills` 下，把这一行加到现有 map 里，**不要覆盖其他 skill 的 enabled 状态**。

- [ ] **Step 3: 验证 skill 系统能识别新 skill**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.skills.loader import load_skills
skills = list(load_skills(enabled_only=True))
names = {s.name for s in skills}
assert 'ethovision-paradigm-knowledge' in names, f'skill not loaded, found: {names}'
print('OK: ethovision-paradigm-knowledge loaded')
"
```

Expected: `OK: ethovision-paradigm-knowledge loaded`

- [ ] **Step 4: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md packages/agent/extensions_config.json
git commit -m "编写 ethovision-paradigm-knowledge SKILL.md 并启用 skill"
```

---

## Task 3: 把 _facts.json 转为 _facts.md（人类可读版）

**Files:**
- Create: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/_facts.md`

- [ ] **Step 1: 写一个一次性脚本生成 _facts.md**

创建临时脚本 `/tmp/gen_facts_md.py`：

```python
import json
from pathlib import Path

facts_json = Path("/home/wangqiuyang/noldus-insight/docs/review-packages/2026-04-29-ev19-templates/_facts.json")
out = Path("/home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/references/_facts.md")

data = json.loads(facts_json.read_text(encoding="utf-8"))
variants = data["variants"]

lines = [
    "# EV19 模板事实表（62 变体）",
    "",
    f"**来源**: `{data['source']}`",
    f"**变体总数**: {data['total_variants']}（{data['total_categories']} 大类）",
    "**生成方式**: 从 `templateMetaData.xml` 自动抽取，请勿手工修改",
    "",
    "## 字段说明",
    "",
    "- `template_id`: 完整变体 ID（用作 `ev19_template` 字段值）",
    "- `category`: 大类",
    "- `arena_template`: EthoVision 软件中的 arena 类型",
    "- `zone_template`: zone 配置",
    "- `inferred_subject_hint`: 推测的动物种类",
    "- `inferred_zone_config`: 推测的 zone 配置缩写",
    "- `inferred_array_size`: 阵列规模（Single / 96w / 16x / Quad / 1cubicle / 4cubicles）",
    "",
    "## 变体清单",
    "",
    "| template_id | category | arena_template | zone_template | subject | zone_config | array_size |",
    "|---|---|---|---|---|---|---|",
]

for v in variants:
    lines.append(
        f"| `{v['template_id']}` | {v['category']} | {v['arena_template']} | "
        f"{v['zone_template']} | {v['inferred_subject_hint']} | "
        f"{v['inferred_zone_config']} | {v['inferred_array_size']} |"
    )

out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {out} ({len(variants)} variants)")
```

运行：

```bash
cd /home/wangqiuyang/noldus-insight
python3 /tmp/gen_facts_md.py
```

Expected: `Wrote /home/.../references/_facts.md (62 variants)`

- [ ] **Step 2: 验证 markdown 渲染**

```bash
head -30 /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/references/_facts.md
```

Expected: 看到表格头 + 前 N 行变体。

- [ ] **Step 3: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethovision-paradigm-knowledge/references/_facts.md
git commit -m "生成 _facts.md（62 变体事实表，人类可读版）"
```

---

## Task 4: 创建 ev19_facts.py Python 模块（白名单 + 兼容性映射）

**Files:**
- Create: `packages/ethoinsight/ethoinsight/ev19_facts.py`
- Create: `packages/ethoinsight/tests/test_ev19_facts.py`

- [ ] **Step 1: 写测试 test_ev19_facts.py**

创建 `/home/wangqiuyang/noldus-insight/packages/ethoinsight/tests/test_ev19_facts.py`：

```python
"""Unit tests for EV19 facts table (62 variants)."""

from ethoinsight.ev19_facts import (
    EV19_VARIANTS,
    EV19_CATEGORIES,
    EV19_TEMPLATE_PARADIGM_MAP,
    is_valid_ev19_template,
    get_template_facts,
    suggest_nearby_templates,
    get_default_template_for_paradigm,
)


def test_ev19_variants_count():
    """62 variants are loaded."""
    assert len(EV19_VARIANTS) == 62


def test_ev19_categories_count():
    """20 unique categories."""
    assert len(EV19_CATEGORIES) == 20


def test_known_variant_is_valid():
    """A known variant ID passes validation."""
    assert is_valid_ev19_template("PlusMaze-AllZones") is True


def test_unknown_variant_is_invalid():
    """An unknown variant ID fails validation."""
    assert is_valid_ev19_template("Bogus-Template") is False


def test_get_template_facts_returns_full_record():
    """Facts include arena_template, zone_template, etc."""
    facts = get_template_facts("PlusMaze-AllZones")
    assert facts is not None
    assert facts["arena_template"] == "Elevated plus maze"
    assert facts["category"] == "PlusMaze"


def test_get_template_facts_returns_none_for_unknown():
    assert get_template_facts("Bogus") is None


def test_suggest_nearby_templates_for_typo():
    """Typo in PlusMaze suggests close matches."""
    suggestions = suggest_nearby_templates("PlusMze-AllZones")
    assert any("PlusMaze" in s for s in suggestions)


def test_paradigm_map_contains_known_paradigms():
    """epm and forced_swim are in paradigm map."""
    assert "epm" in EV19_TEMPLATE_PARADIGM_MAP
    assert "forced_swim" in EV19_TEMPLATE_PARADIGM_MAP


def test_paradigm_map_epm_includes_plusmaze_variants():
    """EPM should map to PlusMaze variants."""
    epm_templates = EV19_TEMPLATE_PARADIGM_MAP["epm"]
    assert "PlusMaze-AllZones" in epm_templates
    assert "PlusMaze-FewZones" in epm_templates
    assert "PlusMaze-NoZones" in epm_templates


def test_get_default_template_for_paradigm():
    """epm default = PlusMaze-AllZones."""
    assert get_default_template_for_paradigm("epm") == "PlusMaze-AllZones"


def test_get_default_template_for_unknown_paradigm_returns_none():
    assert get_default_template_for_paradigm("nonexistent") is None
```

- [ ] **Step 2: 运行测试，确认 fail**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_ev19_facts.py -v
```

Expected: ImportError 或 ModuleNotFoundError，test failed because `ethoinsight.ev19_facts` 不存在。

- [ ] **Step 3: 实现 ev19_facts.py**

创建 `/home/wangqiuyang/noldus-insight/packages/ethoinsight/ethoinsight/ev19_facts.py`：

```python
"""EV19 模板事实表 + 范式兼容性映射。

数据来源：docs/review-packages/2026-04-29-ev19-templates/_facts.json（自动抽取自 EV19 demodata）。
模块在 import 时把 JSON 加载到内存，作为只读字典提供给：
- set_experiment_paradigm 工具的白名单校验
- Ev19TemplateGuardrailProvider 的检查逻辑
- agent skill 中范式 → 默认变体降级
"""

from __future__ import annotations

import difflib
import json
from functools import lru_cache
from pathlib import Path

# 仓库内相对路径（不依赖 deerflow runtime / app context）
_FACTS_JSON_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "docs"
    / "review-packages"
    / "2026-04-29-ev19-templates"
    / "_facts.json"
)


@lru_cache(maxsize=1)
def _load_facts() -> dict:
    """Load _facts.json from canonical location. Cached after first call."""
    return json.loads(_FACTS_JSON_PATH.read_text(encoding="utf-8"))


def _build_variant_index() -> dict[str, dict]:
    data = _load_facts()
    return {v["template_id"]: v for v in data["variants"]}


# 公开常量 ---------------------------------------------------------------------

EV19_VARIANTS: dict[str, dict] = _build_variant_index()
"""62 变体的字典（template_id → 完整 facts 记录）。只读。"""

EV19_CATEGORIES: set[str] = {v["category"] for v in EV19_VARIANTS.values()}
"""20 大类的集合。"""

# paradigm_key → 推荐变体列表（手工维护；行为学同事 PR 后会扩展）
# 顺序很重要：第一个是该范式的默认变体（用于 agent 反问失败时降级）
EV19_TEMPLATE_PARADIGM_MAP: dict[str, list[str]] = {
    # 焦虑样行为（PRD MVP 4 个）
    "epm": ["PlusMaze-AllZones", "PlusMaze-FewZones", "PlusMaze-NoZones"],
    "open_field": [
        "OpenFieldRectangle-AllZones",
        "OpenFieldRectangle-NoZones",
        "OpenFieldCircle-AllZones",
        "OpenFieldCircle-NoZones-Rodents-Other",
    ],
    "zero_maze": ["ZeroMaze-AllZones", "ZeroMaze-NoZones"],
    "light_dark_box": [
        # LDB 在 EV19 表里没有独立大类，先用矩形 OFT 子集兜底
        # 等行为学同事 PR 后修正
        "OpenFieldRectangle-Subdivided2x2",
        "OpenFieldRectangle-AllZones",
    ],
    # 抑郁样行为（PRD MVP 2 个）
    "tail_suspension": ["NoTemplate"],  # TST 不需要 zone，仅活动度
    "forced_swim": ["PorsoltCylinder-AllZones", "PorsoltCylinder-NoZones"],
    # 其他（保留，shoaling 已建成）
    "shoaling": ["OpenFieldCircle-NoZones-Fish", "AquariumTrack3D"],
    "novel_object": [
        "OpenFieldCircle-NovObjZones",
        "OpenFieldRectangle-NovObjZones",
    ],
    "y_maze": ["Y-Maze-AllZones", "Y-Maze-NoZones"],
    "barnes_maze": ["BarnesMaze-20Holes", "BarnesMaze-NoZones"],
    "morris_water_maze": ["MWM-AllZones", "MWM-AFewZones", "MWM-NoZones"],
    "sociability": ["Sociability-AllZones", "Sociability-NoZones"],
    "radial_arm_maze": ["Radial-8-arm-AllZones", "Radial-8-arm-NoZones"],
}


# 公开函数 ---------------------------------------------------------------------

def is_valid_ev19_template(template_id: str) -> bool:
    """Check if a template_id is in the 62-variant whitelist."""
    return template_id in EV19_VARIANTS


def get_template_facts(template_id: str) -> dict | None:
    """Return full facts record for a template_id, or None if unknown."""
    return EV19_VARIANTS.get(template_id)


def suggest_nearby_templates(template_id: str, max_results: int = 3) -> list[str]:
    """Return up to max_results template IDs that are close matches (for typo correction)."""
    return difflib.get_close_matches(
        template_id, list(EV19_VARIANTS.keys()), n=max_results, cutoff=0.6
    )


def get_default_template_for_paradigm(paradigm_key: str) -> str | None:
    """Return the recommended default variant for a paradigm, or None if unknown."""
    candidates = EV19_TEMPLATE_PARADIGM_MAP.get(paradigm_key)
    if not candidates:
        return None
    return candidates[0]


def is_paradigm_template_compatible(paradigm_key: str, template_id: str) -> bool:
    """Check if template_id is in the recommended list for paradigm_key."""
    candidates = EV19_TEMPLATE_PARADIGM_MAP.get(paradigm_key, [])
    return template_id in candidates
```

- [ ] **Step 4: 运行测试，确认 pass**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_ev19_facts.py -v
```

Expected: 所有测试 PASS。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/ev19_facts.py packages/ethoinsight/tests/test_ev19_facts.py
git commit -m "新增 ev19_facts.py: 62 变体白名单 + 范式兼容性映射 + 单元测试"
```

---

## Task 5: 升级 set_experiment_paradigm 工具加 ev19_template 必填参数

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py`
- Create: `packages/agent/backend/tests/test_set_experiment_paradigm_ev19.py`

- [ ] **Step 1: 写测试 test_set_experiment_paradigm_ev19.py**

创建 `/home/wangqiuyang/noldus-insight/packages/agent/backend/tests/test_set_experiment_paradigm_ev19.py`：

```python
"""Tests for set_experiment_paradigm tool with ev19_template parameter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deerflow.agents.middlewares.experiment_context import set_experiment_paradigm_tool


def _runtime_with_workspace(workspace: Path):
    runtime = MagicMock()
    runtime.state = {"thread_data": {"workspace_path": str(workspace)}}
    return runtime


def test_valid_ev19_template_writes_context_with_field(tmp_path):
    """Valid ev19_template is written to experiment-context.json."""
    runtime = _runtime_with_workspace(tmp_path)

    result = set_experiment_paradigm_tool.invoke({
        "paradigm": "epm",
        "paradigm_cn": "高架十字迷宫",
        "category": "anxiety",
        "subject": "rodent",
        "ev19_template": "PlusMaze-AllZones",
        "runtime": runtime,
    })

    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert parsed["ev19_template"] == "PlusMaze-AllZones"

    written = json.loads((tmp_path / "experiment-context.json").read_text(encoding="utf-8"))
    assert written["paradigm"] == "epm"
    assert written["ev19_template"] == "PlusMaze-AllZones"


def test_invalid_ev19_template_returns_error_with_candidates(tmp_path):
    """Unknown ev19_template returns error + suggested close matches."""
    runtime = _runtime_with_workspace(tmp_path)

    result = set_experiment_paradigm_tool.invoke({
        "paradigm": "epm",
        "paradigm_cn": "高架十字迷宫",
        "category": "anxiety",
        "subject": "rodent",
        "ev19_template": "PlusMze-AllZones",  # typo
        "runtime": runtime,
    })

    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert "candidates" in parsed
    assert any("PlusMaze" in c for c in parsed["candidates"])

    # Context file MUST NOT be written on error
    assert not (tmp_path / "experiment-context.json").exists()


def test_paradigm_template_mismatch_writes_warning_but_proceeds(tmp_path):
    """If ev19_template is valid but not in the recommended list for paradigm, a warning is included but the write succeeds."""
    runtime = _runtime_with_workspace(tmp_path)

    result = set_experiment_paradigm_tool.invoke({
        "paradigm": "epm",
        "paradigm_cn": "高架十字迷宫",
        "category": "anxiety",
        "subject": "rodent",
        "ev19_template": "PorsoltCylinder-AllZones",  # 抑郁范式模板用在 EPM 上
        "runtime": runtime,
    })

    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert "warning" in parsed

    # Context file IS written
    written = json.loads((tmp_path / "experiment-context.json").read_text(encoding="utf-8"))
    assert written["ev19_template"] == "PorsoltCylinder-AllZones"
```

- [ ] **Step 2: 运行测试确认 fail**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_set_experiment_paradigm_ev19.py -v
```

Expected: 所有测试 FAIL（`ev19_template` 参数尚不存在）。

- [ ] **Step 3: 修改 experiment_context.py 加 ev19_template 参数**

修改 `/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` 中 `set_experiment_paradigm_tool` 函数，从第 95 行开始替换：

```python
@tool("set_experiment_paradigm", parse_docstring=True)
def set_experiment_paradigm_tool(
    paradigm: str,
    paradigm_cn: str,
    category: str,
    subject: str,
    ev19_template: str,
    workspace_dir: str = "/mnt/user-data/workspace/",
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Record the user's experiment paradigm choice for the analysis pipeline.

    Call this after the user has confirmed their experiment type via ask_clarification.
    Writes experiment-context.json to the workspace so downstream agents know the paradigm.

    Args:
        paradigm: English paradigm name key (e.g. "shoaling", "epm", "open_field")
        paradigm_cn: Chinese display name (e.g. "斑马鱼鱼群行为")
        category: Category name (e.g. "zebrafish", "anxiety", "spatial_memory")
        subject: Subject type — "rodent" | "fish" | "insect" | "other"
        ev19_template: EV19 template variant ID (e.g. "PlusMaze-AllZones"). Must be one of
            the 62 known variants. See ethovision-paradigm-knowledge skill references/_facts.md
            for the full list.
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/"

    Returns:
        JSON: on success {status:"ok", ev19_template, paradigm, path} (with optional warning);
        on error {status:"error", reason, candidates}.
    """
    # 白名单校验 — 在写入文件前 fail-fast
    try:
        from ethoinsight.ev19_facts import (
            is_valid_ev19_template,
            suggest_nearby_templates,
            is_paradigm_template_compatible,
        )
    except ImportError as e:
        logger.error("Failed to import ev19_facts: %s", e)
        return json.dumps({
            "status": "error",
            "reason": f"ev19_facts import failed: {e}",
            "candidates": [],
        }, ensure_ascii=False)

    if not is_valid_ev19_template(ev19_template):
        return json.dumps({
            "status": "error",
            "reason": f"Unknown EV19 template '{ev19_template}'. Must be one of 62 known variants.",
            "candidates": suggest_nearby_templates(ev19_template, max_results=3),
        }, ensure_ascii=False)

    # paradigm × ev19_template 兼容性检查（不阻塞，仅警告）
    warning: str | None = None
    if not is_paradigm_template_compatible(paradigm, ev19_template):
        warning = (
            f"模板 '{ev19_template}' 通常不用于范式 '{paradigm}'。"
            f"如果是有意为之请忽略此警告；否则请重新调用 set_experiment_paradigm 或反问用户。"
        )
        logger.warning("paradigm-template mismatch: %s × %s", paradigm, ev19_template)

    # Resolve the actual host workspace path from thread state.
    actual_workspace = workspace_dir
    if runtime is not None and runtime.state is not None:
        thread_data: ThreadDataState | None = runtime.state.get("thread_data")
        if thread_data is not None:
            host_workspace = thread_data.get("workspace_path")
            if host_workspace is not None:
                actual_workspace = host_workspace

    data = {
        "paradigm": paradigm,
        "paradigm_cn": paradigm_cn,
        "category": category,
        "subject": subject,
        "ev19_template": ev19_template,
        "paradigm_confirmed_at": datetime.now(UTC).isoformat(),
        "gate_completed": ["gate1_paradigm"],
    }
    path = Path(actual_workspace) / "experiment-context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    response = {
        "status": "ok",
        "path": str(path),
        "paradigm": paradigm,
        "ev19_template": ev19_template,
    }
    if warning:
        response["warning"] = warning
    return json.dumps(response, ensure_ascii=False)
```

- [ ] **Step 4: 运行测试确认 pass**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_set_experiment_paradigm_ev19.py -v
```

Expected: 3/3 PASS。

- [ ] **Step 5: 跑老测试确认无回归**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_paradigm_degradation.py tests/test_gate_enforcement_middleware.py -v
```

Expected: 全 PASS。如有 fail，可能是老测试调 `set_experiment_paradigm_tool` 时没传 `ev19_template`，需要更新这些老测试加入 `ev19_template="<合理默认>"`（如 shoaling 测试用 `OpenFieldCircle-NoZones-Fish`）。

- [ ] **Step 6: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py packages/agent/backend/tests/test_set_experiment_paradigm_ev19.py
# 如果 step 5 修了老测试也加上：
git add packages/agent/backend/tests/test_paradigm_degradation.py packages/agent/backend/tests/test_gate_enforcement_middleware.py 2>/dev/null
git commit -m "set_experiment_paradigm 工具加 ev19_template 必填参数 + 白名单校验"
```

---

## Task 6: 编写 default-template-fallback.md 降级表

**Files:**
- Create: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md`

- [ ] **Step 1: 写入 default-template-fallback.md**

创建文件，内容如下：

```markdown
# 范式 → 默认 EV19 模板降级表

## 何时使用

当 agent 在 ask_clarification 反问后用户答 "不知道" / "随便" / "你决定"，或者 LoopDetectionMiddleware 阻断了第二次反问时，按此表选默认变体进入分析。

**重要**：填到 `set_experiment_paradigm(ev19_template=...)` 之前，先在用户面前确认一次：
> "您的实验我会按 EPM 标准模板（PlusMaze-AllZones）分析。这是 90%+ EPM 实验的默认配置。如果您的实验有特殊设置，分析后告诉我我会重做。"

不要默不作声地默认。

## 范式 → 默认变体

| 学术范式 (paradigm key) | 默认 ev19_template | 选择理由 |
|---|---|---|
| `epm` | `PlusMaze-AllZones` | 90%+ EPM 实验用 AllZones（含开闭臂 + 头探出区） |
| `open_field` | `OpenFieldRectangle-AllZones` | 大多数 OFT 实验。圆形 arena 看 demodata 几何形状判断是否切到 `OpenFieldCircle-AllZones` |
| `zero_maze` | `ZeroMaze-AllZones` | 同上 |
| `light_dark_box` | `OpenFieldRectangle-Subdivided2x2` | **未来由行为学同事确认**。LDB 在 EV19 表里无独立大类，2x2 子分区可手工指明明暗箱区 |
| `tail_suspension` | `NoTemplate` | TST 不用 zone，仅活动度（不动时间） |
| `forced_swim` | `PorsoltCylinder-AllZones` | FST 标准（圆柱形容器） |
| `shoaling` | `OpenFieldCircle-NoZones-Fish` | 斑马鱼鱼群（多动物 2D） |
| `novel_object` | `OpenFieldCircle-NovObjZones` | NOR 实验，圆形 arena + 物体区 |
| `y_maze` | `Y-Maze-AllZones` | 三臂 + 中央交汇区 |
| `barnes_maze` | `BarnesMaze-20Holes` | 标准 20 孔配置 |
| `morris_water_maze` | `MWM-AllZones` | 平台 + 象限 + 走廊 + 边缘 |
| `sociability` | `Sociability-AllZones` | 三箱社交（社交区 + 对照区） |
| `radial_arm_maze` | `Radial-8-arm-AllZones` | 标准 8 臂 |

## 决策流程

```
用户答 "不知道" / 反问被 LoopDetection 阻断
    ↓
1. 看用户文字 + 文件名能否推断 paradigm_key（epm / open_field / ...）
    ↓
2. 查上表 → 默认 ev19_template
    ↓
3. 在与用户的下一条消息里告知："我会按 <默认模板> 分析。如有特殊设置，分析后告诉我我会重做。"
    ↓
4. 调 set_experiment_paradigm(paradigm=<推断>, ev19_template=<查表>, ...)
```

## 此表的更新

行为学同事 PR 中的 `by-experiment/<范式>.md` 一旦填写"适用模板（按推荐顺序）"，本表的相应行应同步更新。**保持单一事实源**：未来若数据飞轮启动 + agent 自学偏好，可考虑由 agent 自己改这个文件（启用 update_agent / Skill Evolution 后）。
```

- [ ] **Step 2: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md
git commit -m "新增 default-template-fallback.md: 范式→默认 EV19 模板降级表"
```

---

## Task 7: 编写 identification-decision-tree.md 决策流程

**Files:**
- Create: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/identification-decision-tree.md`

- [ ] **Step 1: 写入 identification-decision-tree.md**

```markdown
# EV19 模板识别决策流程

## 决策树（agent 严格按此流程执行）

```
用户上传文件 + 提问
    ↓
Step 1: 收集证据（不调任何工具，只看上下文）
    - 用户消息中是否提到学术范式名（"高架十字迷宫" / "EPM" / "焦虑测试" / "斑马鱼" 等）？
    - 上传文件名中是否带范式简写（"轨迹-EPM-Trial 1...txt" 等）？
    - 上传文件数量？Subject 数？
    ↓
Step 2: 推测候选 paradigm_key（snake_case 英文）
    - 命中明确范式 → paradigm_key 确定（如 "epm"）
    - 模糊（"焦虑测试"）→ paradigm_key 多候选（["epm", "open_field", "zero_maze", "light_dark_box"]）
    - 完全没线索 → 进入 Step 6（必反问）
    ↓
Step 3: 读 by-experiment/<paradigm_key>.md（已确定单一 paradigm 时）
    - 看 "适用模板" 字段，得到 ev19_template 候选列表
    - 候选 ≥ 2 → 进入 Step 4
    - 候选 = 1 → 进入 Step 7（直接 set）
    ↓
Step 4: 读用户上传的第一个 raw txt（前 50 行）
    - 用 read_file 看 meta + 列结构
    - 单位（"毫米"/"厘米"）→ 区分鱼/啮齿
    - 追踪点（仅 X 中心 = 单点 / X 鼻点 + X 中心 + X 尾 = 三点）→ 区分鱼/啮齿
    - 列名是否含 "In zone(...)" → 区分 AllZones / NoZones
    - 列名是否含 "Nose within object zone(...)" → NovObj 模板
    - 据此把候选缩到 ≤ 3 个
    ↓
Step 5: 候选数判断
    - 候选 = 1 → 进入 Step 7（直接 set）
    - 候选 2-3 → 进入 Step 6（结构化反问）
    ↓
Step 6: ask_clarification（最多 1 次）
    - 必须给 ≤3 个结构化选项
    - 推荐项放第一位，标 "(推荐)"
    - 每个选项标差异（"含开闭臂 + 头探出区，最常见"）
    - 兜底说明（"如不确定，选 A，绝大多数 EPM 用这个"）
    - 用户回复后 → 进入 Step 7
    - 用户答 "不知道" / "随便" → 查 default-template-fallback.md → 进入 Step 7
    ↓
Step 7: 调 set_experiment_paradigm(paradigm=<key>, ev19_template=<选定>, ...)
    - 工具校验白名单
    - 通过 → 写 experiment-context.json，进入分析
    - 失败（白名单不通过）→ 看错误的 candidates 字段重选 → 重调（这一步算入 LoopDetection 计数，避免反复）
```

## 反问质量准则

### 反例（不精准）

> "请问您用的是 EthoVision 哪个模板？"

用户懵：他根本不熟 EthoVision 模板表。

### 正例（精准）

> 我从您的数据看到：
> - 实验 = 高架十字迷宫 (EPM) — 文件名含 "EPM"
> - 数据有 zone 列（开臂/闭臂标记）
>
> 您用的是哪个 EV19 模板？
> A. **PlusMaze-AllZones**（推荐，含开闭臂 + 头探出区，90%+ EPM 实验用这个）
> B. **PlusMaze-FewZones**（只有开闭臂，无头探出区）
> C. **PlusMaze-NoZones**（仅坐标无 zone — 但您的数据有 zone 列，可能不是这个）
>
> 如果不确定，选 A。

要点：
1. **先告知 agent 已经收集到的证据**（让用户知道你不是瞎问）
2. **结构化选项 ≤3 个**
3. **推荐项放第一位 + 解释为什么推荐**
4. **每个选项标差异**
5. **兜底说明**（让用户能在 5 秒内回答）

## 反问后的处理

| 用户回复 | agent 行为 |
|---|---|
| 选了某个选项（A/B/C）| 直接 set_experiment_paradigm |
| "我不知道" / "随便" / "你定" | 查 default-template-fallback.md，告知用户默认值后 set |
| 提供了表里没有的模板名 | 用 suggest_nearby_templates 函数（在工具返回的 candidates 字段）反问澄清 |
| 完全跑题（用户开始说别的事） | 把模板识别挂起，回应用户当前问题；下次需要 set 时回到此流程 |

## 不要做的事

- ❌ 一上来不读 raw txt 直接反问（候选不缩小，烂问）
- ❌ 反问 ≥2 次模板相关问题（LoopDetectionMiddleware 会拦截）
- ❌ 给开放性问题（"您的实验是什么类型？"）
- ❌ 自己拼 ev19_template 字符串（必须从白名单选）
- ❌ 在用户没明确同意时使用默认值（默认值要在用户面前说出来，不要默不作声）
```

- [ ] **Step 2: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethovision-paradigm-knowledge/references/identification-decision-tree.md
git commit -m "新增 identification-decision-tree.md: agent 决策流程 + 反问质量准则"
```

---

## Task 8: 实现 Ev19TemplateGuardrailProvider

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py`
- Create: `packages/agent/backend/tests/test_ev19_template_guardrail_provider.py`

- [ ] **Step 1: 写测试**

创建 `/home/wangqiuyang/noldus-insight/packages/agent/backend/tests/test_ev19_template_guardrail_provider.py`：

```python
"""Tests for Ev19TemplateGuardrailProvider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from deerflow.guardrails.ev19_template_provider import Ev19TemplateGuardrailProvider
from deerflow.guardrails.provider import GuardrailRequest


@pytest.fixture
def workspace_with_ev19(tmp_path):
    """Workspace with experiment-context.json containing ev19_template."""
    ctx = {
        "paradigm": "epm",
        "ev19_template": "PlusMaze-AllZones",
        "gate_completed": ["gate1_paradigm"],
    }
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")
    return tmp_path


@pytest.fixture
def workspace_without_ev19(tmp_path):
    """Workspace with experiment-context.json missing ev19_template."""
    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}  # ev19_template missing
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")
    return tmp_path


@pytest.fixture
def empty_workspace(tmp_path):
    """Workspace without experiment-context.json."""
    return tmp_path


def _make_request(tool_name: str, args: dict) -> GuardrailRequest:
    return GuardrailRequest(tool_name=tool_name, tool_input=args, agent_id=None, timestamp="2026-05-08T00:00:00Z")


def _provider_with_workspace(ws: Path) -> Ev19TemplateGuardrailProvider:
    return Ev19TemplateGuardrailProvider(workspace_resolver=lambda: str(ws))


def test_allows_non_task_tools(workspace_without_ev19):
    """Provider only inspects task() calls, others pass through."""
    p = _provider_with_workspace(workspace_without_ev19)
    decision = p.evaluate(_make_request("read_file", {"path": "x"}))
    assert decision.allow is True


def test_allows_task_to_non_code_executor_subagents(workspace_without_ev19):
    """Provider only blocks task(code-executor); other subagents pass."""
    p = _provider_with_workspace(workspace_without_ev19)
    decision = p.evaluate(_make_request("task", {"subagent_type": "knowledge-assistant", "prompt": "..."}))
    assert decision.allow is True


def test_blocks_task_code_executor_when_ev19_template_missing(workspace_without_ev19):
    """Block task(code-executor) when ev19_template field is missing."""
    p = _provider_with_workspace(workspace_without_ev19)
    decision = p.evaluate(_make_request("task", {"subagent_type": "code-executor", "prompt": "..."}))
    assert decision.allow is False
    assert decision.reasons[0].code == "ethoinsight.no_ev19_template"


def test_blocks_task_code_executor_when_workspace_has_no_context(empty_workspace):
    """Block task(code-executor) when experiment-context.json doesn't exist at all."""
    p = _provider_with_workspace(empty_workspace)
    decision = p.evaluate(_make_request("task", {"subagent_type": "code-executor", "prompt": "..."}))
    assert decision.allow is False


def test_allows_task_code_executor_when_ev19_template_set(workspace_with_ev19):
    """Allow task(code-executor) when ev19_template is set."""
    p = _provider_with_workspace(workspace_with_ev19)
    decision = p.evaluate(_make_request("task", {"subagent_type": "code-executor", "prompt": "..."}))
    assert decision.allow is True


@pytest.mark.asyncio
async def test_async_evaluate_matches_sync(workspace_with_ev19):
    """aevaluate returns the same decision as evaluate."""
    p = _provider_with_workspace(workspace_with_ev19)
    sync_dec = p.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    async_dec = await p.aevaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert sync_dec.allow == async_dec.allow
```

- [ ] **Step 2: 运行测试确认 fail**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_ev19_template_guardrail_provider.py -v
```

Expected: ImportError（provider 未实现）。

- [ ] **Step 3: 实现 provider**

创建 `/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py`：

```python
"""Guardrail provider that blocks task(code-executor) when ev19_template is unset.

Works alongside the existing GateEnforcementMiddleware (which checks the `paradigm`
field). The two have orthogonal responsibilities:
  - GateEnforcementMiddleware: paradigm field present and valid
  - Ev19TemplateGuardrailProvider: ev19_template field present in 62-variant whitelist

The provider only blocks task(code-executor). Other tool calls pass through.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from deerflow.guardrails.provider import GuardrailDecision, GuardrailReason, GuardrailRequest

logger = logging.getLogger(__name__)


def _default_workspace_resolver() -> str | None:
    """Default workspace resolver — caller should pass a callable that returns the host workspace path."""
    return None


class Ev19TemplateGuardrailProvider:
    """Block task(code-executor) when experiment-context.json lacks ev19_template.

    Agent sees the error reason and is expected to call set_experiment_paradigm
    (with ev19_template) or ask_clarification before retrying.
    """

    name = "ev19-template-guardrail"

    def __init__(self, workspace_resolver: Callable[[], str | None] | None = None):
        self._resolve_workspace = workspace_resolver or _default_workspace_resolver

    # --- core check (sync) ---

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Only inspect task() calls
        if request.tool_name != "task":
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # Only inspect task(code-executor) — other subagents are unaffected
        subagent = request.tool_input.get("subagent_type", "") if request.tool_input else ""
        if "code-executor" not in subagent:
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        workspace = self._resolve_workspace()
        if workspace is None:
            # No workspace context available — fail-open (don't block)
            logger.debug("Ev19TemplateGuardrailProvider: workspace unresolvable, allowing task call")
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        ctx = self._read_context(workspace)
        if ctx is None or not ctx.get("ev19_template"):
            return GuardrailDecision(
                allow=False,
                reasons=[
                    GuardrailReason(
                        code="ethoinsight.no_ev19_template",
                        message=(
                            "EV19 模板尚未设置。请先调用 set_experiment_paradigm(..., ev19_template=...) "
                            "确定模板变体（参考 ethovision-paradigm-knowledge skill 中 references/_facts.md "
                            "的 62 变体白名单）。如果信息不足，先 ask_clarification 反问用户。"
                        ),
                    )
                ],
                policy_id="ev19-template-guardrail",
            )

        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

    # --- async wrapper ---

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)

    # --- helpers ---

    def _read_context(self, workspace: str) -> dict | None:
        path = Path(workspace) / "experiment-context.json"
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read experiment-context.json: %s", e)
            return None
```

- [ ] **Step 4: 安装 pytest-asyncio（如未装）+ 运行测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_ev19_template_guardrail_provider.py -v
```

如果出现 `pytest-asyncio` 缺失错误，先安装：
```bash
uv pip install pytest-asyncio
```
然后在 `tests/test_ev19_template_guardrail_provider.py` 顶部加 `pytestmark = pytest.mark.asyncio` 或在 `pyproject.toml` 配置自动 asyncio 模式（参考已有的 `test_guardrail_middleware.py` 写法，照抄）。

Expected: 所有测试 PASS。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py packages/agent/backend/tests/test_ev19_template_guardrail_provider.py
git commit -m "新增 Ev19TemplateGuardrailProvider: 拒绝 ev19_template=null 时的 code-executor 派遣"
```

---

## Task 9: 在 lead agent 中间件链注册新 GuardrailMiddleware

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`
- Modify: `packages/agent/config.yaml`

- [ ] **Step 1: 在 config.yaml 启用 guardrails**

打开 `/home/wangqiuyang/noldus-insight/packages/agent/config.yaml`，找到（或在 `subagents`、`memory` 等同级位置新增）`guardrails:` 段，加：

```yaml
guardrails:
  enabled: true
  provider:
    use: deerflow.guardrails.ev19_template_provider:Ev19TemplateGuardrailProvider
  fail_closed: false  # 不让 guardrail provider 内部错误阻塞流程
```

如果 config 里已有 `guardrails:` 段（之前可能用过 AllowlistProvider），把 provider 替换为新的。

- [ ] **Step 2: 修改 agent.py 注册 GuardrailMiddleware**

打开 `/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`，搜索 `GateEnforcementMiddleware`（应该在第 305-310 行附近）。在 GateEnforcementMiddleware 之前（或之后，但要在 ToolErrorHandling 之后）加：

```python
# Ev19TemplateGuardrail — 复用 deerflow GuardrailMiddleware 框架，专门管 ev19_template 字段
guardrails_cfg = config.get("guardrails", None)
if guardrails_cfg and getattr(guardrails_cfg, "enabled", False):
    from deerflow.guardrails.middleware import GuardrailMiddleware
    from deerflow.guardrails.ev19_template_provider import Ev19TemplateGuardrailProvider
    from deerflow.agents.middlewares.experiment_context import resolve_workspace_from_state

    # workspace_resolver 在 middleware tool_call 时通过 state 解析
    # 但 GuardrailProvider.evaluate 不接收 state — 需要在 middleware 层桥接
    # 看现有 GuardrailMiddleware 怎么处理；可能需要 stateful 包装
    # 简化处理：用 contextvar 存当前 thread 的 workspace
    # （详细实现取决于 GuardrailMiddleware 现有代码，agent 看一下 deerflow_noldus 上游或本地的 middleware.py）
    provider = Ev19TemplateGuardrailProvider(workspace_resolver=...)
    middlewares.append(GuardrailMiddleware(provider=provider, fail_closed=False))
```

**注意**：上面的代码是骨架。实际实现时**先看 `deerflow.guardrails.middleware.GuardrailMiddleware` 现有签名**——可能它会在 `wrap_tool_call` 时把 state 传给 provider，也可能不传。如果不传，需要：
- 方案 A: 在 ThreadDataMiddleware 里把 workspace 写到 contextvar（如已存在则用）
- 方案 B: 自己写一个继承自 `GuardrailMiddleware` 的子类，在 `wrap_tool_call` 里把 state 注入到 provider

读 `/home/wangqiuyang/deerflow-noldus/backend/packages/harness/deerflow/guardrails/middleware.py` 第 30-100 行参考；本地路径 `/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/guardrails/middleware.py` 也可能存在（如果同步过）。

如果发现需要写 state 桥接代码（方案 B），把它放进 `ev19_template_provider.py` 文件末尾，作为一个 `Ev19GuardrailMiddleware` 子类。

- [ ] **Step 3: 跑全部 backend 测试确认无回归**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest -x --tb=short
```

Expected: 全 PASS（除非引入了循环 import 或依赖缺失，按错误信息逐一修）。

- [ ] **Step 4: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/config.yaml packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
# 如果在 ev19_template_provider.py 加了 middleware 桥接子类，也加上：
git add packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py 2>/dev/null
git commit -m "在 lead agent 中间件链注册 Ev19TemplateGuardrailProvider"
```

---

## Task 10: 删除 lead_agent/prompt.py 中旧的 7 大类 18 范式段

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

- [ ] **Step 1: 找到要删除的段**

```bash
grep -n "7 大类\|18 范式\|斑马鱼鱼群行为\|焦虑迷宫\|抑郁绝望" /home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py | head -30
```

找到的行号是删除范围的边界。具体行号每个 fork 不一样，需 agent 实际看了再操作。

- [ ] **Step 2: 用 read_file + Edit 删除整段**

读取该段（一般约 100 行），确认这是「识别实验范式与实验设计类型」章节中的旧分类表。**保留章节标题**，但**把整张分类表替换为引导段**：

替换前（旧）：
```
## 识别实验范式与实验设计类型

### 7 大类 18 范式分类表
[此处约 100 行表格 + 选择规则]
```

替换后（新）：
```
## 识别实验范式与实验设计类型（EV19 模板地基）

EV19 真实模板体系：20 大类 / 62 变体（详见 `ethovision-paradigm-knowledge` skill）。

**当用户上传分析数据时**：
1. 读 `ethovision-paradigm-knowledge` skill 的 SKILL.md 决策树
2. 综合用户文字 + 文件名 + 必要时 read raw txt 推测 EV19 模板变体
3. 候选 ≤3 时用 ask_clarification 给结构化选项（详见 skill references/identification-decision-tree.md）
4. 调 `set_experiment_paradigm(paradigm, paradigm_cn, category, subject, ev19_template)` 写入 experiment-context.json

**反问最多 1 次**——如用户答 "不知道"，按 skill references/default-template-fallback.md 选默认变体。

**ev19_template 字段未设置时，task("code-executor") 会被 GuardrailMiddleware 拦截**——必须先调 set_experiment_paradigm。
```

- [ ] **Step 3: 验证 prompt 仍能正常加载**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.agents.lead_agent.prompt import apply_prompt_template
# 简单 smoke check：函数能调用
print('prompt module imports OK')
"
```

Expected: `prompt module imports OK`。

- [ ] **Step 4: 跑 prompt 相关测试（若存在）+ 全测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest -k prompt -v
PYTHONPATH=. uv run pytest -x --tb=short
```

Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "删除 lead agent prompt 旧的 7 大类 18 范式段，引导走新 EV19 skill"
```

---

## Task 11: 给 5 个 ethoinsight 分析步骤入口加"软门"

**Files:**
- Modify: 5 个文件 in `packages/ethoinsight/ethoinsight/templates/` — 这些是 code-executor 通过 sandbox bash 调用的分析步骤入口
- Create: `packages/ethoinsight/tests/test_template_soft_gate.py`

> **注意**：截至本 plan 写作时，`packages/ethoinsight/ethoinsight/templates/` 下只有 `shoaling.py`。本任务给 shoaling.py 加软门作为模板示范；其他 5 个范式（epm/open_field/zero_maze/light_dark_box/tail_suspension/forced_swim）的 templates/<x>.py 是明天起 E2 task 由后续 agent 完成时同时加软门。**软门代码模式必须在本 task 中写完整给 shoaling，作为后续 6 个范式实施时的复制基准**。

- [ ] **Step 1: 写测试 test_template_soft_gate.py**

创建 `/home/wangqiuyang/noldus-insight/packages/ethoinsight/tests/test_template_soft_gate.py`：

```python
"""Soft gate tests — analysis entrypoints fail-fast when ev19_template is missing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_context(workspace: Path, *, with_ev19: bool):
    ctx = {"paradigm": "shoaling", "category": "zebrafish", "subject": "fish"}
    if with_ev19:
        ctx["ev19_template"] = "OpenFieldCircle-NoZones-Fish"
    (workspace / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")


def test_shoaling_template_fails_fast_without_ev19_template(tmp_path):
    """shoaling.run_analysis returns structured error when ev19_template missing."""
    from ethoinsight.templates.shoaling import run_analysis

    _write_context(tmp_path, with_ev19=False)

    result = run_analysis(workspace_dir=str(tmp_path))
    assert result["status"] == "error"
    assert "ev19_template" in result["reason"]
    assert "remediation" in result


def test_shoaling_template_proceeds_with_ev19_template(tmp_path):
    """shoaling.run_analysis runs (or fails for other reasons) when ev19_template is set."""
    from ethoinsight.templates.shoaling import run_analysis

    _write_context(tmp_path, with_ev19=True)

    # 此处可能因为没有真实数据而失败（OK），但失败原因必须不是 "ev19_template missing"
    result = run_analysis(workspace_dir=str(tmp_path))
    if result.get("status") == "error":
        assert "ev19_template" not in result.get("reason", "")
```

- [ ] **Step 2: 运行测试确认 fail**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_template_soft_gate.py -v
```

Expected: FAIL（软门未实现）。

- [ ] **Step 3: 在 shoaling.py 加软门 + 抽出公用 helper**

先看 `shoaling.py` 现有入口签名：

```bash
head -30 /home/wangqiuyang/noldus-insight/packages/ethoinsight/ethoinsight/templates/shoaling.py
```

如果有 `run_analysis(workspace_dir, ...)` 之类入口，在函数体最开头加：

```python
def run_analysis(workspace_dir: str, ...) -> dict:
    # 软门 — 早退检查 ev19_template 字段
    from ethoinsight.templates._gate import require_ev19_template
    gate_error = require_ev19_template(workspace_dir)
    if gate_error is not None:
        return gate_error

    # ...原有分析逻辑
```

如果 shoaling.py 没有统一的 `run_analysis` 入口（按 5 个细粒度 step 调用），把软门加在每个细粒度函数（`parse_trajectories`、`compute_metrics`、`run_statistics`、`generate_charts`、`assess_and_handoff`）的开头。

抽公用 helper 到 `packages/ethoinsight/ethoinsight/templates/_gate.py`：

```python
"""Shared soft gate for paradigm template entrypoints.

Each template's analysis steps must check ev19_template is set before doing work,
to avoid silently writing wrong-template results when the lead agent skipped
set_experiment_paradigm.
"""

from __future__ import annotations

import json
from pathlib import Path


def require_ev19_template(workspace_dir: str) -> dict | None:
    """Return None if ev19_template is set; return structured error dict if missing.

    Caller (template entrypoint) returns the dict directly to its caller,
    short-circuiting the analysis. The error dict contains a `remediation` field
    so the lead agent (reading it via handoff) knows what to do next.
    """
    ctx_path = Path(workspace_dir) / "experiment-context.json"
    if not ctx_path.exists():
        return {
            "status": "error",
            "reason": "experiment-context.json 不存在 — ev19_template 字段未设置",
            "remediation": (
                "lead agent 应先调用 set_experiment_paradigm(paradigm, ..., ev19_template) "
                "确定模板。如不能确定，先 ask_clarification 反问用户。"
            ),
        }
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {
            "status": "error",
            "reason": f"无法解析 experiment-context.json: {e}",
            "remediation": "lead agent 应重新调用 set_experiment_paradigm 写入正确的 context。",
        }
    if not ctx.get("ev19_template"):
        return {
            "status": "error",
            "reason": "experiment-context.json 缺少 ev19_template 字段",
            "remediation": (
                "lead agent 应调用 set_experiment_paradigm(..., ev19_template=...) 补齐字段。"
                "参考 ethovision-paradigm-knowledge skill 的 _facts.md 选择白名单内变体。"
            ),
        }
    return None
```

- [ ] **Step 4: 运行测试确认 pass**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_template_soft_gate.py -v
```

Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/templates/_gate.py packages/ethoinsight/ethoinsight/templates/shoaling.py packages/ethoinsight/tests/test_template_soft_gate.py
git commit -m "ethoinsight templates 加 ev19_template 软门 + shoaling.py 集成示范"
```

---

## Task 12: 在 Guardrail Provider 中加锁定逻辑（防止 ev19_template 中途修改）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py`
- Modify: `packages/agent/backend/tests/test_ev19_template_guardrail_provider.py`

> **目标**：当 `experiment-context.json` 已经写入 `ev19_template` 后，禁止 agent 通过再次调用 `set_experiment_paradigm` 改变模板（避免下游分析中途切换基线）。除非工具调用包含明确的 `confirm_template_change=True` 参数。

- [ ] **Step 1: 写测试 — 锁定行为**

在 `tests/test_ev19_template_guardrail_provider.py` 末尾追加：

```python
def test_blocks_set_experiment_paradigm_when_already_set(workspace_with_ev19):
    """Block set_experiment_paradigm if ev19_template is already in context."""
    p = _provider_with_workspace(workspace_with_ev19)
    decision = p.evaluate(_make_request("set_experiment_paradigm", {
        "paradigm": "open_field",
        "ev19_template": "OpenFieldRectangle-AllZones",
        # confirm_template_change NOT provided
    }))
    assert decision.allow is False
    assert decision.reasons[0].code == "ethoinsight.template_already_set"


def test_allows_set_experiment_paradigm_with_confirm_flag(workspace_with_ev19):
    """Allow re-set if confirm_template_change=True is provided."""
    p = _provider_with_workspace(workspace_with_ev19)
    decision = p.evaluate(_make_request("set_experiment_paradigm", {
        "paradigm": "open_field",
        "ev19_template": "OpenFieldRectangle-AllZones",
        "confirm_template_change": True,
    }))
    assert decision.allow is True


def test_allows_first_set_experiment_paradigm(workspace_without_ev19):
    """First-time set is always allowed."""
    p = _provider_with_workspace(workspace_without_ev19)
    decision = p.evaluate(_make_request("set_experiment_paradigm", {
        "paradigm": "epm",
        "ev19_template": "PlusMaze-AllZones",
    }))
    assert decision.allow is True
```

- [ ] **Step 2: 跑测试确认 fail**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_ev19_template_guardrail_provider.py -v
```

Expected: 3 个新测试 FAIL（锁定逻辑未实现）。

- [ ] **Step 3: 在 provider 加锁定逻辑**

修改 `ev19_template_provider.py` 的 `evaluate` 方法，在原有 task() 检查之外加入 set_experiment_paradigm 拦截：

```python
def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
    # ── 锁定检查：set_experiment_paradigm 不能改已设置的 ev19_template ──
    if request.tool_name == "set_experiment_paradigm":
        workspace = self._resolve_workspace()
        if workspace is not None:
            ctx = self._read_context(workspace)
            if ctx and ctx.get("ev19_template"):
                # 已经设置过；除非 confirm_template_change=True 否则拒绝
                args = request.tool_input or {}
                if not args.get("confirm_template_change"):
                    return GuardrailDecision(
                        allow=False,
                        reasons=[
                            GuardrailReason(
                                code="ethoinsight.template_already_set",
                                message=(
                                    f"ev19_template 已设置为 '{ctx['ev19_template']}'，"
                                    f"不允许中途修改以保持分析一致。"
                                    f"如确实需要修改，请向 set_experiment_paradigm 传 confirm_template_change=True；"
                                    f"或建议用户开新 thread 重新分析。"
                                ),
                            )
                        ],
                        policy_id="ev19-template-guardrail",
                    )
        # 首次设置 — 放行
        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

    # ── task(code-executor) 检查（原有逻辑） ──
    if request.tool_name != "task":
        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

    subagent = request.tool_input.get("subagent_type", "") if request.tool_input else ""
    if "code-executor" not in subagent:
        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

    workspace = self._resolve_workspace()
    if workspace is None:
        logger.debug("Ev19TemplateGuardrailProvider: workspace unresolvable, allowing task call")
        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

    ctx = self._read_context(workspace)
    if ctx is None or not ctx.get("ev19_template"):
        return GuardrailDecision(
            allow=False,
            reasons=[
                GuardrailReason(
                    code="ethoinsight.no_ev19_template",
                    message=(
                        "EV19 模板尚未设置。请先调用 set_experiment_paradigm(..., ev19_template=...) "
                        "确定模板变体（参考 ethovision-paradigm-knowledge skill 中 references/_facts.md "
                        "的 62 变体白名单）。如果信息不足，先 ask_clarification 反问用户。"
                    ),
                )
            ],
            policy_id="ev19-template-guardrail",
        )

    return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])
```

- [ ] **Step 4: 跑测试确认 pass**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_ev19_template_guardrail_provider.py -v
```

Expected: 全部 PASS（包括新加的 3 个测试）。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py packages/agent/backend/tests/test_ev19_template_guardrail_provider.py
git commit -m "Ev19TemplateGuardrailProvider 加 ev19_template 锁定 — 防止中途切换模板"
```

---

## Task 13: 更新 ethoinsight-planning skill 的 quality-gates.md 引用 ev19_template

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md`

- [ ] **Step 1: 看现有内容**

```bash
cat /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md | head -100
```

- [ ] **Step 2: 更新 Gate 1 段引用 ev19_template 字段**

在 quality-gates.md 中找到描述 Gate 1（paradigm 确认）的段落，把现有的 "调用 set_experiment_paradigm" 描述更新为带 `ev19_template` 必填参数的新签名。具体编辑要求：

- 把 `set_experiment_paradigm(paradigm, paradigm_cn, category, subject)` 改为 `set_experiment_paradigm(paradigm, paradigm_cn, category, subject, ev19_template)`
- 加一行说明："`ev19_template` 必须在 62 变体白名单内（见 `ethovision-paradigm-knowledge` skill）"
- 如果文件中提到 GateEnforcementMiddleware 的拦截，加一段说明 GuardrailMiddleware 也会拦 `task("code-executor")` 当 `ev19_template` 字段缺失，且会拦 `set_experiment_paradigm` 的二次修改（除非 `confirm_template_change=True`）

- [ ] **Step 3: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md
git commit -m "ethoinsight-planning quality-gates.md 引用 ev19_template 字段"
```

---

## Task 14: 端到端验证 + 写交接文档

**Files:**
- Create: `docs/handoffs/2026-05-08-ev19-template-skill-foundation-handoff.md`

- [ ] **Step 1: 跑 lint**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make lint
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
ruff check ethoinsight/ tests/
```

修任何 lint 错误，commit 后再继续。

- [ ] **Step 2: 跑全部测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test

cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/
```

Expected: 全 PASS（agent backend + ethoinsight）。

- [ ] **Step 3: 启服务做手工 e2e**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make dev
```

打开 `http://localhost:2026`，手工验证：

1. **场景 A — 信息明确**：上传 `golden-cases/case-001-shoaling-baseline/raw-data/` 中任意 5 个 txt，输入 "帮我分析鱼群行为数据"
   - 预期：agent 直接调 `set_experiment_paradigm(paradigm="shoaling", ev19_template="OpenFieldCircle-NoZones-Fish", ...)` 不反问

2. **场景 B — 候选多，需要反问**：上传 EPM 数据（如有 demo data 中 EPM raw txt），输入 "帮我分析这个数据"
   - 预期：agent 反问 1 次，给 PlusMaze 三个变体的结构化选项 + 推荐 PlusMaze-AllZones

3. **场景 C — Guardrail 拦截**：在某 thread 中通过工具直接调 `task("code-executor", ...)` 而不先调 `set_experiment_paradigm`
   - 预期：GuardrailMiddleware 返回 `ethoinsight.no_ev19_template` 错误 ToolMessage，agent 看到后转去调 set_experiment_paradigm 或 ask_clarification

4. **场景 D — 工具白名单拒绝**：让 agent 调 `set_experiment_paradigm(ev19_template="Bogus-Template")`
   - 预期：工具返回 status="error" + candidates 字段，agent 看到候选后重选

记录每个场景的实际行为到交接文档。

- [ ] **Step 4: 写交接文档**

创建 `/home/wangqiuyang/noldus-insight/docs/handoffs/2026-05-08-ev19-template-skill-foundation-handoff.md`：

```markdown
# 2026-05-08 EV19 模板识别地基 — 实施完成

## TL;DR

按 spec [docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md] 完成 D1-D10：
- 新建 `ethovision-paradigm-knowledge` skill（4 篇 references + 20+20 篇按大类/实验组织的 markdown）
- 新增 `ethoinsight.ev19_facts` 模块（62 变体白名单 + 范式兼容性映射）
- `set_experiment_paradigm` 工具升级 — 加 `ev19_template` 必填参数 + 白名单校验 + 兼容性 warning
- 新增 `Ev19TemplateGuardrailProvider` — 通过 deerflow GuardrailMiddleware 框架拦截 `task("code-executor")` 当 ev19_template 缺失
- 删除 lead_agent/prompt.py 旧的「7 大类 18 范式」段，引导 agent 走新 skill

## 改动清单

[列出本次所有 commit 列表 + 关键文件]

## 验证

[填入 Task 12 step 3 的 e2e 实际结果]

## 后续 / 不在本次范围（明天起）

- E1: 同事 review PR 进来后，更新 skill `references/by-template/*.md` 和 `references/by-experiment/*.md`
- E2: 6 个 PRD 范式分析模板补全（templates/epm.py 等）— 依赖同事 PR
- E3: shoaling golden-case 校验
- E4: 抽象 templates/_base.py 基类

## 已知遗留

- `Ev19TemplateGuardrailProvider` 的 workspace_resolver 实现方式 — 看 Task 9 step 2 的方案 A/B 实际选了哪个
- LDB 默认变体 (`OpenFieldRectangle-Subdivided2x2`) 是临时兜底，等行为学同事 PR 修正
- LoopDetectionMiddleware 对 ask_clarification 的 hash 区分度未在 e2e 中验证（如反复反问被检测到则免做；否则需扩展 _stable_tool_key）
```

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05-08-ev19-template-skill-foundation-handoff.md
git commit -m "EV19 模板识别地基实施完成交接文档"
```

---

## 验收清单（实施 agent 自查）

完成 Task 1-12 后，逐项打勾确认：

- [ ] D1 完成：`ls packages/agent/skills/custom/ethovision-paradigm-knowledge/` 显示 SKILL.md + references/
- [ ] D2 完成：`ls packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-{template,experiment}/` 各 ≥20 篇 .md
- [ ] D3 完成：`packages/ethoinsight/ethoinsight/ev19_facts.py` 存在，`pytest tests/test_ev19_facts.py` 全绿
- [ ] D4 完成：SKILL.md 含决策树 + 大类索引 + 何时反问指引
- [ ] D5 完成：`grep ethovision-paradigm-knowledge packages/agent/extensions_config.json` 返回 enabled:true
- [ ] D6 完成：`grep -c "7 大类\|18 范式" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` = 0
- [ ] D7 完成：`set_experiment_paradigm_tool` 签名含 `ev19_template`
- [ ] D8a 完成：白名单校验在工具内（test_set_experiment_paradigm_ev19 全绿）
- [ ] D8b 完成：5 个 ethoinsight 分析步骤入口含软门（test_template_soft_gate 全绿；shoaling.py 集成示范完成）
- [ ] D8c 完成：`Ev19TemplateGuardrailProvider` 已注册到 GuardrailMiddleware（agent.py 中）
- [ ] D8d：LoopDetectionMiddleware 已启用（手工 e2e 中观察到反复反问会被警告/中断）
- [ ] D8e 完成：`default-template-fallback.md` 含 13 个范式 → 默认变体的映射
- [ ] D8f 完成：`Ev19TemplateGuardrailProvider` 拦截 set_experiment_paradigm 二次修改（除非 confirm_template_change=True）；test_blocks_set_experiment_paradigm_when_already_set 全绿
- [ ] D9 完成：`identification-decision-tree.md` 含决策流程 + 反问质量准则
- [ ] D10 完成：手工 e2e 4 个场景全部按预期行为
- [ ] `make test` 全绿（backend + ethoinsight）
- [ ] `make lint` 0 错误
- [ ] 交接文档写完

---

## 已知风险与回退

如果实施过程中发现：

1. **`Ev19TemplateGuardrailProvider` 的 workspace 解析复杂**：先把 GuardrailMiddleware 注册临时改为 `enabled=false`，单独把 set_experiment_paradigm 工具白名单校验先 ship。GuardrailMiddleware 单独作为 v0.1+1 task 处理。

2. **删除旧 18 范式段后 e2e 显示 agent 不会调 set_experiment_paradigm**：可能是 prompt 引导不够明确。回到 Task 10 step 2 在 prompt 末尾加更强的指令（"必须先 read ethovision-paradigm-knowledge SKILL.md"）。

3. **行为学同事 review PR 进来时和当前 markdown 冲突**：他们改的是同一份 markdown，git merge。冲突点应该集中在 🟡 字段（同事填的内容），不会和我们这次改的 SKILL.md 决策树冲突。

回退方案（任何 step 后想撤销）：

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline -20  # 找到本次第一个 commit 的 hash
git reset --hard <prev_commit_hash>
```

或单独 revert 某次 commit：

```bash
git revert <commit_hash>
```
