# Spec B — 输出宪法术语对齐：松绑定性行为术语，保留绝对阈值禁令

> 日期：2026-06-08 ｜ 目标分支：从 `dev` 新建 worktree（独立于 Spec A / Spec C）
> 来源：EPM n=1 dogfood 第三轮（thread `7d4d9b8e`）根因 B / C 的最深 SSOT 矛盾
> 性质：**SSOT 修订**（改输出宪法）。⚠️ 这是判读语言的权威定义，**实施后须与行为学同事确认**（见 §6）。
> 这是给执行 agent 的施工单，不是给用户的总结。

---

## 0. 一句话目标

输出宪法当前禁止"焦虑样行为"等**定性行为术语**，但行为学同事维护的 paradigm 知识（epm.md）正用这些词做标准判读 → SSOT 自相矛盾，逼得消费方（report-writer / data-analyst）陷入"改写违禁词"地狱（seal 黑洞根因之一）。

**修复**：松绑**定性行为术语**（描述行为模式的词），**保留绝对阈值禁令**（拿数值比固定标准的词）。这两类是不同的东西，宪法当前把它们混在一起禁了。

---

## 1. 背景与证据（已现场核实）

### SSOT 矛盾实锤

- **输出宪法**（`packages/agent/skills/custom/ethoinsight/references/output-constitution.md` line 18）禁止：
  ```
  | 绝对焦虑判读 | "高焦虑"、"低焦虑"、"焦虑样行为"、"提示焦虑"、"焦虑水平" |
  ```
- **行为学同事的 paradigm 知识**（`skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md`）**用了 2 次"焦虑样行为"**做标准判读（grep 实测）。epm.md 是同事维护的领域 SSOT。
- **两个 subagent 都读 epm.md**：data-analyst（`data_analyst.py:147` read by-experiment/<paradigm>.md）+ report-writer（step 2.6 读 paradigm 文档）。**它们读到"焦虑样行为"是判读标准，却被宪法禁止输出它** → 认知僵局。
- 本次 dogfood report-writer 第 1 次 seal 失败，前端 thinking 实证它反复纠结：*"焦虑样行为 is banned... but knowledge_response.md uses these terms extensively... I need to rephrase"* → 烧光 turn → 叙述黑洞（详见 Spec C）。

### 决策（已与用户对齐）

**知识为准，松绑宪法 —— 但只松定性术语，保留阈值禁令。** 宪法 line 17-19 的三类禁止词里：

| 行 | 类别 | 当前禁词 | 决策 |
|----|------|---------|------|
| 17 | 绝对阈值判读 | "典型值"、"常模"、"参考范围"、"金标准"、"文献典型"、"基线水平"、"正常小鼠"、"典型低焦虑小鼠"、"正常范围 X-Y%" | **保留全禁**（这些是拿数值比固定标准，违反"组间比较不用绝对阈值"铁律 = CLAUDE.md §9） |
| 18 | 绝对焦虑判读 | "高焦虑"、"低焦虑"、"焦虑样行为"、"提示焦虑"、"焦虑水平" | **拆分**：松绑定性行为术语，保留绝对程度判断（见 §2） |
| 19 | 绝对正常判读 | "正常活动"、"正常行为"、"典型正常" | **保留全禁**（"正常"隐含一个绝对标准） |

**判别原则（写进宪法，供未来判断）**：
- ✅ **可用 — 定性行为术语**：描述**行为模式/机制**，不绑定数值标准。如"焦虑样行为"（anxiety-like behavior，行为学公认术语）、"趋近-回避冲突"、"回避倾向"、"探索行为减少"。
- ❌ **仍禁 — 绝对阈值/绝对程度**：把某个数值跟一个**固定的、脱离对照组的标准**比。如"正常范围 15-25%"、"典型值"、"基线水平"、"高焦虑/低焦虑"（这是给焦虑定**绝对程度等级**，仍须靠组间比较）。

> 微妙边界（写进宪法）：
> - "焦虑样行为" ✅（描述行为类型）vs "高焦虑" ❌（给绝对程度定级）。
> - "实验组开臂时间显著低于对照组，提示焦虑样行为增加（组间）" ✅（组间比较 + 定性术语）。
> - "开臂 8% 属于高焦虑水平" ❌（绝对阈值 + 绝对程度）。

---

## 2. 改动清单

### 改动 B1：拆分宪法 line 18，松绑定性术语

**文件**：`packages/agent/skills/custom/ethoinsight/references/output-constitution.md`

把 line 17-19 的禁止词表 + 周围说明改为（保留 17、19 不动，**只改 18 这一行 + 补判别原则**）：

当前 line 18：
```
| 绝对焦虑判读 | "高焦虑"、"低焦虑"、"焦虑样行为"、"提示焦虑"、"焦虑水平" |
```

改为：
```
| 绝对焦虑程度判读 | "高焦虑"、"低焦虑"、"焦虑水平"（给焦虑定脱离对照组的绝对程度等级） |
```
（即从禁词里**移除 "焦虑样行为"、"提示焦虑"**——这两个是定性行为术语/组间提示语，不是绝对程度判断。"提示焦虑样行为增加"这类组间提示语应允许。）

并在禁止词表**下方**补一段判别原则（新增）：
```
**定性行为术语 vs 绝对阈值/程度的区分**（判断某个词能否用）：

- ✅ 允许 — 定性行为术语：描述行为模式或机制，不绑定固定数值标准。
  例："焦虑样行为(anxiety-like behavior)"、"趋近-回避冲突"、"回避倾向"、"探索行为减少/增加"。
  这些是行为学公认的描述性术语，且 paradigm 知识文档(by-experiment/<paradigm>.md)在用。
- ❌ 仍禁 — 绝对阈值：把数值跟脱离对照组的固定标准比。
  例："正常范围 15-25%"、"典型值"、"基线水平"、"文献典型"。
- ❌ 仍禁 — 绝对程度判读：给行为状态定脱离对照组的绝对等级。
  例："高焦虑"、"低焦虑"、"焦虑水平偏高"。

口诀：可以说"这是什么行为"(定性)，不可以说"这个数值高/低/正常"(绝对阈值)或"焦虑程度是高/低"(绝对程度)。
判读结论的力度仍来自组间比较(control vs treatment) + 效应量，而非术语本身。
```

并在"正确的写法"示例（line 23-25）补一条，明确组间 + 定性术语的正确组合：
```
- "实验组开臂停留时间显著短于对照组，提示焦虑样行为增加（p=0.03, d=0.8）" ← 组间比较 + 定性术语，✅
```

> ⚠️ **不要动 line 17（绝对阈值）和 line 19（绝对正常）**——那些保留全禁。只改 line 18 + 补判别原则。

### 改动 B2：epm.md（及其他 paradigm md）用词复核 —— 确认只含定性术语，不含阈值

**文件**：`packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/*.md`

epm.md 已用"焦虑样行为"（松绑后合法 ✅）。但要**核实它没有偷偷用绝对阈值**（如"正常范围 10-30%"），因为那些仍禁、且会再次污染下游：
```bash
for f in skills/custom/ethovision-paradigm-knowledge/references/by-experiment/*.md; do
  echo "=== $f ==="
  grep -nE '正常范围|典型值|基线水平|常模|参考范围|[0-9]+\s*[-~]\s*[0-9]+\s*%' "$f" 2>/dev/null
done
```
- 若发现 paradigm md 含绝对阈值表述 → **不在本 spec 直接改**（那是同事的 SSOT），而是**列入 §6 的"与同事确认清单"**，让同事决定如何改写成组间语言。本 spec 只改宪法 + 标记问题。

### 改动 B3（关联 Spec A，但此处兜底）：knowledge-assistant 若仍可能判读，须读宪法

Spec A 已收窄 knowledge-assistant 不从零判读。但作为**纵深防御**（万一 Spec A 未完全堵住，或 knowledge-assistant 场景 A 解释时仍可能用词）：在 `knowledge_assistant.py` 的 prompt 补一句让它遵守输出宪法的判读语言规范（读 output-constitution.md 或内联关键原则）。

> 实现提示：knowledge-assistant 当前完全不提宪法。补一句：
> ```
> ## 判读语言规范
> 涉及对用户数据的任何解读时，遵守输出宪法判读语言规范：可用定性行为术语（如焦虑样行为、趋近-回避冲突），但禁用绝对阈值（正常范围 X-Y%、典型值、基线水平）和绝对程度（高/低焦虑）。结论力度来自组间比较，不是绝对标准。
> ```
> 这样即使 knowledge-assistant 解释已有结论时，也不会产"正常范围 10-30%"这种违禁阈值词。

---

## 3. 测试（TDD）

放 `packages/agent/backend/tests/` 或 `packages/ethoinsight/tests/`（宪法是 skill 文件，测试可放 backend 侧验证内容）。

新建 `test_output_constitution_terms.py`：
```python
from pathlib import Path

OC = Path("<repo>/packages/agent/skills/custom/ethoinsight/references/output-constitution.md")

def test_qualitative_terms_no_longer_banned():
    """焦虑样行为 等定性术语应从禁词表移除（松绑）。"""
    text = OC.read_text(encoding="utf-8")
    # 找到禁词表区域（line 13-19 的 table）
    # 断言：禁词行不再把'焦虑样行为'列为违规
    ban_table = text.split("正确的写法")[0]  # 禁词表在'正确的写法'之前
    # "焦虑样行为" 不应出现在禁词表里（但可出现在'允许'判别原则里）
    # 更稳妥：检查 line 18 那一行不含'焦虑样行为'
    assert "焦虑样行为" not in _extract_ban_row(text, "绝对焦虑")

def test_absolute_threshold_still_banned():
    """绝对阈值禁令保留（铁律不动）。"""
    text = OC.read_text(encoding="utf-8")
    assert "正常范围" in text  # 仍在禁词表
    assert "典型值" in text
    # 判别原则段也明确'仍禁'
    assert "绝对阈值" in text

def test_constitution_has_qualitative_vs_threshold_distinction():
    """新增的判别原则段存在。"""
    text = OC.read_text(encoding="utf-8")
    assert "定性行为术语" in text
    assert "趋近-回避" in text or "anxiety-like" in text.lower() or "焦虑样行为" in text
```
> `_extract_ban_row` 等 helper 按宪法实际 markdown 结构写。执行 agent 先读宪法确认 table 行格式。

---

## 4. 影响面与风险

- **下游自动受益**：data-analyst / report-writer 读 epm.md 的"焦虑样行为"后，不再因宪法冲突陷入改写 → 直接缓解 Spec C 的触发路径（但 C 仍独立修）。
- **不放松铁律**：CLAUDE.md §9 "组间比较不用绝对阈值"完全保留——line 17/19 禁令不动，line 18 只移除定性术语、保留绝对程度判断。
- **SSOT 单点**：宪法是判读语言的唯一权威（[[feedback_single_source_of_truth.md]]）。本 spec 只改宪法这一份，不在各 subagent prompt 内联重复禁词表（避免双存）。B3 让 knowledge-assistant **引用**宪法，不是复制。
- **风险**：松绑边界靠 LLM 理解"定性 vs 绝对程度"。"高焦虑"禁、"焦虑样行为"准——这个区分较细，deepseek 可能偶尔越界。缓解：判别原则给口诀 + 正反例；dogfood 验证 report 措辞。

---

## 5. 验收标准

1. 宪法内容测试绿（§3）。
2. paradigm md 复核完成（B2）：要么确认无绝对阈值，要么列入 §6 清单。
3. 全量回归：`cd packages/agent/backend && make test`（已知污染 + config symlink，同 Spec A）。skill 文件改动一般不影响 Python 测试，但跑一遍确认。
4. **dogfood 验证**：重跑 EPM，data-analyst/report-writer 产出的判读**可以**含"焦虑样行为/趋近-回避"（不再纠结改写），但**不含**"正常范围 X-Y%/高焦虑/典型值"。report-writer 不再因术语冲突 seal 失败（与 Spec C 协同验证）。

---

## 6. ⚠️ 与行为学同事确认清单（实施后必做）

宪法是判读语言 SSOT，松绑前/后须与同事对齐（避免我们单方面改了同事认可的判读规范）：

1. **确认松绑范围**：同事是否同意"焦虑样行为/趋近-回避冲突/回避倾向"等定性术语可用于对用户的报告输出？（同事的 epm.md 在用，应同意，但要确认。）
2. **确认保留范围**：同事是否同意"高焦虑/低焦虑/正常范围 X-Y%/典型值"仍禁（坚持组间比较哲学）？
3. **paradigm md 阈值复核**（B2 产出）：若 epm.md 等含绝对阈值表述（如"正常范围 10-30%"），请同事决定改写成组间语言，还是确认这些只作"批次质检参考"不进判读（参照 CLAUDE.md §9 `_DEFAULT_THRESHOLDS` 的定位）。
4. 落实方式：在 PR 描述里 @ 同事 review 宪法 diff，或单独发 review-package。

---

## 7. 提交

- worktree 名建议：`worktree-output-constitution-term-alignment`
- commit message（中文）：`fix(constitution): 松绑定性行为术语(焦虑样行为等)，保留绝对阈值/程度禁令`
- 全量绿 + 同事确认后建 PR 合入 dev。

---

## 8. 关联

- dogfood findings：`docs/handoffs/2026-06/2026-06-08-epm-dogfood-findings.md`
- memory：`project_2026-06-08_epm_dogfood_routing_and_constitution_leak.md`（根因 B）、`feedback_single_source_of_truth.md`（宪法单点）
- 判读哲学铁律：CLAUDE.md 第 9 条（组间比较，不用绝对阈值）
- Spec A（路由收窄）/ Spec C（seal 硬保障）：同批
- paradigm 知识 SSOT 在 review-packages：`feedback_ssot_lives_in_review_packages.md`
