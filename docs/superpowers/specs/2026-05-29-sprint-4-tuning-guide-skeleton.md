# Sprint 4 设计骨架 — 调参指南进 by-experiment md（⚠️ SSOT 归属边界）

**类型**：设计骨架版
**对应**：[roadmap v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 4 + 2026-05-29 grill 复审 🔴
**估期**：工程部分 ~0.5 周（内容部分依赖行为学同事，不计入）
**前置**：Sprint 3（参数审计产 mismatch 警告）

---

## 0. 目标

agent 警告了"阈值不合适"（Sprint 3），但不知道"调到多少"。paradigm md 提供权威调参指南。knowledge-assistant 回答"为什么是 30mm/s"时**解释权衡**而非背诵权威。

---

## 1. 🔴 grill 复审核心发现：SSOT 归属边界（实施前必读，最重要）

roadmap 原文要往 `by-experiment/{forced_swim,...}.md` 末尾加 `## 参数调整指南`。**但这撞了项目最硬的 SSOT 原则**：

> [[feedback_ssot_lives_in_review_packages]]：范式/图表/指标 SSOT 在**行为学同事的 review-packages 下**，不是 catalog 也不是 skill。
> [[feedback_ssot_skill_deployment_distinction]]：`by-experiment/*.md` 是同事写的、走 review-packages → 搬入 skill 的流程。

**所以 Sprint 4 必须拆成两半，归属不同的人：**

| 部分 | 归属 | 内容 |
|---|---|---|
| **内容**（"velocity_threshold 调到多少、为什么、权衡是什么"） | **行为学同事**（走 review-packages） | 这是领域判断，工程团队没有资格/知识写。工程写了会：①被同事下次从 review-packages 同步覆盖 ②越权编造领域知识 |
| **通路**（data-analyst/knowledge-assistant 怎么 read 到这段内容） | **工程团队**（本 sprint） | grep 该 md 段、注入 prompt 工作流 |

**实施 agent 铁律**：
- ❌ **不要直接往 `skills/.../by-experiment/*.md` 写"参数调整指南"的实质内容**（调多少、为什么）
- ✅ **只做工程通路**：在 data_analyst.py / knowledge_assistant prompt 加"read paradigm md 时 grep `## 参数调整指南` 段"
- ✅ 内容部分：在 review-packages 下开占位/issue，**等同事 PR**；或与同事确认后由同事填
- ✅ 若工程要先放占位结构（空的 `## 参数调整指南` 标题 + "待行为学同事补充"），必须确认 review-packages 同步流程不会冲突，且标注"内容待 review"

---

## 2. 设计（骨架）

**关键原则（roadmap）**：参数默认值在 catalog YAML（SSOT），调参权衡解释在 paradigm md（SSOT 归同事），**两者不重复**。`ev19-dependent-variables.md` 不动（它是公式 SSOT，不混应用指南）。

**工程改动（只做通路）**：
- `subagents/builtins/data_analyst.py`：workflow 里"按范式 read 判读文档"那步（已存在，5.7 spec 见过 step 2.6），加"grep `## 参数调整指南` 段，若有则纳入 recommendations"
  - **注意热点**：又是改 data_analyst.py workflow → grep 编号唯一（同 Sprint 3）
- `knowledge_assistant` prompt：回答参数类问题时 read 对应 paradigm md 的参数调整指南段
- **不碰** ev19-dependent-variables.md

**内容侧（同事，非工程）**：
- `by-experiment/{forced_swim,tail_suspension,epm,open_field,zero_maze,light_dark_box}.md` 末尾的 `## 参数调整指南` —— **内容由行为学同事经 review-packages 提供**

---

## 3. 实施前核验清单
1. **先与用户/行为学同事确认**：参数调整指南的内容谁写、走不走 review-packages（这是 go/no-go——工程不该擅自写）
2. data_analyst.py 当前"read 判读文档"步骤的真实结构（在哪加 grep）
3. by-experiment md 当前是否已有该段（同事可能已经在 review-packages 写了）
4. review-packages → skill 的同步机制（确认工程加的通路不被覆盖）
5. **🐛 参数继承卡点（2026-05-29 核实补入，原骨架漏写）**：一个范式的可调参数 = **该范式 YAML 自有 + `_common.yaml` 的 `shared_parameters` 共享继承**，不能只看范式本体文件。实测 `tunable_by_user` 分布：`_common.yaml` 13 处 / `epm/ldb/zero_maze` 各 1 处 / **`fst/oft/tst` 各 0 处**。
   - **fst/oft/tst 本体 0 个 tunable，不代表"无参数可调"**——它们的核心可调参数（如 FST 的 `velocity_threshold`，immobility 判定命脉）**全部继承自 `_common.yaml`**。这是 SSOT 设计（共享参数不在 6 个范式文件重复），不是 catalog bug。
   - **坑**：若实施 agent 只看 `fst.yaml`（0 处）→ 写出空的"可调参数表"→ 漏掉 velocity_threshold；且 §2.7 的 CI 哨兵 `test_tuning_section_lists_catalog_tunable_params` 会因"md 空表 vs 实际继承了 _common 参数"而红。
   - **正确**：列某范式可调参数时，**合并范式 YAML 自有 + `_common.shared_parameters`**；CI 哨兵也必须按"自有 + 继承"的合集校验，而非只查范式本体文件。

---

## 4. 验收（骨架）
1. **工程通路**：data-analyst 遇到 Sprint 3 的参数 mismatch 警告 + paradigm md 有调整指南段 → recommendations 引用该指南（而非工程编的数字）
2. knowledge-assistant 回答"为什么阈值是 X"时解释权衡（来自 md），不背诵
3. **内容准确性由同事背书**，非工程团队
4. ev19-dependent-variables.md 未被改动

## 5. 不在范围（SSOT 边界）
- ❌ **工程团队编写参数调整指南的实质内容**（领域知识，归同事/review-packages）
- ❌ 改 catalog YAML 的参数默认值（那是 2a SSOT，本 sprint 不碰）
- ❌ 改 ev19 公式 SSOT
