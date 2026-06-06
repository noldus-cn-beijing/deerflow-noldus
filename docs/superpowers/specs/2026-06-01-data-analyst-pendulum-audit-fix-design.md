# 2026-06-01 设计 spec — 修复 data-analyst 在 pendulum 范式 + 小样本时 step 2.8 参数审计卡死

**类型**：可实施版（已对 dev HEAD `065b4180` 核验；实施前 `git pull` 复核行号）
**对应**：2026-06-01 E2E dogfood 复现的 data-analyst 卡死故障 + [Sprint 3 参数审计 spec](2026-05-29-sprint-3-parameter-audit-skeleton.md)
**估期**：~0.5-1 天（纯 prompt 降级逻辑 + 一份 FST pendulum 判据搬迁）
**前置**：无硬前置；与 TST spec 协调 pendulum 文档（见 §5）

> **故障现象**（2026-06-01 用户两次 E2E dogfood 实证）：跑 FST（PorsoltCylinder-NoZones，Drug vs Saline，每组 n=1）时，data-analyst **稳定卡死**：烧光 `max_turns=12`、Sprint 5.8 的 seal-resume 补轮也救不回 → lead 重派 → **再次卡死**。「以前从没兜底也失败过」。

---

## 0. 目标与原则

**目标**：消除 data-analyst 在「pendulum 范式（FST/TST）+ 参数判据缺失/小样本」路径上的系统性卡死，让它在无法完成参数审计时**优雅降级**（诚实记 info、立即 seal），而不是在死胡同里烧光 turn。

**铁律**：
- **只警告不调参**（Sprint 3 铁律，不变）
- **判据内容归同事/review-packages，工程只做通路 + 降级**（[[feedback_ssot_lives_in_review_packages]] / [[feedback_single_source_of_truth]]）。**工程绝不编 pendulum/velocity 的领域阈值数字**（issue #63 归同事）
- **解读按范式独立**（用户 2026-06-01 锁定）

---

## 1. 根因链（已用证据闭环，实施前必读）

故障**不是 harness 兜底逻辑坏了**，是 data-analyst 的 **step 2.8 参数审计在 FST/TST + n=1 路径上是个无判据可依的死胡同**，LLM 反复纠结烧光 12 轮 → 连 seal 都没走到。三条死路叠加：

| # | 死路 | 证据 |
|---|---|---|
| R1 | step 2.8 让 read 的判据文档 `tst-pendulum-algorithm.md` **subagent read 不到** | 文档真实存在于 `docs/review-packages/2026-0521-feedbacks/tstYoyo/`（9683B），但**未搬进 skill references**；subagent sandbox 只能 read `/mnt/skills/`，read 不到 docs/。`find packages/agent/skills -iname "*pendulum*"` 返回空 |
| R2 | **n<2 时 p90/p10 数学判据算不出** | data-analyst thinking 反复出现 `I can't compute p10/p90 with n=2`。每组 n=1 → per_subject 每指标只 1 个值 → 百分位无意义 |
| R3 | step 2.8 让提示「参数调整指南段」**不存在** | `forced_swim.md`（44 行）无此段；Sprint 4 调参指南未产出（[Sprint 4 spec](2026-05-29-sprint-4-tuning-guide-skeleton.md)）|

**机制**（`executor.py` 已核实）：
- 主循环 `_aexecute` 在 AI message 数达 `max_turns=12`（`data_analyst.py:225`）时 **硬 break**（`executor.py:909`）
- LLM 在 step 2.8 烧光 12 轮（thinking 全是 `Actually wait... Let me reconsider... I'm overcomplicating this`），最后一条 message 是「准备 seal」纯文本，**不是 seal tool_call**
- Sprint 5.8 seal-resume 补轮（`executor.py:701` `_attempt_seal_resume`）续用已耗尽预算的 final_state，补轮也被同样的困惑吃掉 → FAILED
- **每次 FST/TST + 小样本必复现**（不是随机）

**为什么以前没失败**：第一次同时命中「判据死胡同 × n=1」双重放大器。n>1 的 FST 案例 p10/p90 能算、不会卡。

### 🔴 架构是健康的（横扫已确认 — 排除「解读被通用化」假设）

用户曾怀疑「指标解读被设计成跨范式通用」。**横扫 6 范式确认：没有**。
- 指标解读在 **catalog YAML（各范式独立 .yaml）+ by-experiment md（各范式独立）** 两层都按范式独立
- `_common.yaml` 只共享**参数数值**（velocity_threshold=30 等）和 fallback 图表，**不共享解读**
- step 2.6（`data_analyst.py:87`「按范式 read 判读文档 forced_swim→forced_swim.md」）是**对的**（按图索骥）
- **病灶只有 step 2.8 一处**：它没按图索骥读当前范式文档，而是写死跳到跨范式的 `tst-pendulum-algorithm.md`（且 read 不到）

---

## 2. 修法（三层，从治本到加固）

### 修法 A（治本）— step 2.8 改「按范式索骥 + 判据缺失/小样本优雅降级」

改 `data_analyst.py` 的 step 2.8（约 `:113-154`）：

1. **删掉对跨范式 `tst-pendulum-algorithm.md §3/§4` 的写死引用**（`:129` 附近）。改为：「pendulum 参数（FST/TST 共用**钟摆算法**，但**解读判据按范式独立**）的判据，从**当前范式自己的 by-experiment 文档**取（step 2.6 已 read）」
2. **判据缺失 → 优雅跳过**：「若当前范式文档无 pendulum 参数判据段 → 对每个 pendulum 参数记一条 `info` finding（`mismatch_kind` 留空或用 schema 允许值，suggestion=『该范式 pendulum 参数判据待补，issue #63』）并**立即继续**，不要自行判定数值」
3. **n<2 → velocity 类也跳过**：「per_subject 不足 2 个值无法算 p10/p90 → 该参数记 info 跳过」
4. **拆「逐条比对每个 metric」的铁律**：改为「**判据可用才比对**；判据不可用（文档缺/n<2）即记 info 跳过，不阻塞」

### 修法 B（加固）— step 2.7/2.8 加轮次硬约束

在 step 2.8 末尾 + step 2.7 加一句：
> **参数审计是辅助步骤，至多占 2-3 轮思考。无论审计是否产出 finding，都必须留出轮次走 step 3 调 `seal_data_analyst_handoff`。seal 是必达，审计是尽力。**

用 deepseek 正面提示（CLAUDE.md §6）：「先确保 seal 完成」而非「不要在审计上纠结」。

### 修法 C（建议同做）— FST pendulum 判据进 skill（与 TST spec 各管一份）

用户决策 **FST/TST 各自一份独立 pendulum 判据**。本 spec 管 **FST 那份**：
- 把 `tst-pendulum-algorithm.md` 的判据**内容**适配进 **FST 自己的**范式知识位（建议 `references/by-experiment/forced_swim-pendulum-params.md` 或 forced_swim.md 末尾 `## 🟡 pendulum 参数判据` 段）
- **🔴 SSOT 边界**：这是「把同事已写的判据**部署**到 FST 范式位」，**不改判据数值、不创作领域内容**（[[feedback_ssot_lives_in_review_packages]]）。若不确定 FST 是否该用和 TST 完全相同的 pendulum 判据数值（物理刺激不同可能不同），**标记同事确认，不工程拍脑袋**
- forced_swim.md 加指引「pendulum 参数判据见 [配套文档]」，让 step 2.6/2.8 能按图索骥找到

> **若 C 暂不做（判据仍缺）**：修法 A 的降级逻辑保证 data-analyst **不卡死**（记 info 跳过）。所以 **A+B 是必做且足以根治卡死**；C 是让参数审计「真能跑」的增强，可与 TST spec 协调后做或留待同事补判据。

---

## 3. 实施前核验清单

1. `git pull` 复核 `data_analyst.py` step 2.8 真实行号（5.7 血泪：改 workflow 必 `grep -nE "^\s*[0-9]+\.|^\s*[0-9]+\.[0-9]+" data_analyst.py` 确认编号唯一连续 2/2.5/2.6/2.7/2.8/3，不撞 L29 input 说明段的 `2.`）
2. `ParameterAuditFinding` schema 当前定义（`handoff_schemas.py` + 原始 spec `2026-05-28-sprint-3-data-analyst-parameter-audit-design.md` §2.1）— 确认 info severity + mismatch_kind 允许「跳过/缺判据」语义怎么表达（**schema 是 SSOT，不要新发明枚举值**；若现有 5 元枚举无「判据缺失」语义，用 info severity + suggestion 文字表达，不改枚举）
3. `data_analyst.py:225` `max_turns=12` 现值（修法 B 是 prompt 约束，**不建议单纯调大 max_turns** — 那是治标，不拆死胡同照样烧光）
4. 复现验证：用 `/home/wangqiuyang/DemoData/newdemodata/悬尾/` 或 FST n=1 数据先复现卡死（取基线），改后验证不卡

---

## 4. 验收

- [ ] **修法 A**：step 2.8 删跨范式 `tst-pendulum-algorithm.md` 引用；改为按当前范式文档索骥 + 判据缺失/n<2 记 info 跳过；`grep` 编号唯一（防 5.7 seal bug）
- [ ] **修法 B**：step 2.7/2.8 有「seal 必达、审计至多 2-3 轮」硬约束
- [ ] **🔴 复现测试（核心）**：FST 或 TST n=1 数据跑 data-analyst → **不卡死、不烧光 12 轮、正常调 seal 产出 handoff_data_analyst.json**（key_findings 非空）。这是本 spec 成败判据
- [ ] **降级正确性**：n=1 时 parameter_audit_findings 含 info 级「判据待补」条目，无 critical 误判；data-analyst 没自己编 pendulum 阈值数字
- [ ] **不退化**：n≥2 的正常 FST/EPM 案例参数审计仍正常产出 finding（修法 A 不能把能跑的也降级掉）
- [ ] （若做 C）FST pendulum 判据在 skill 可 read；forced_swim.md 有指引
- [ ] 全量 `make test` 不退化（基线先取真值；**改 data_analyst.py 是共享逻辑，必跑全量** [[feedback_pr_merge_must_run_full_suite_on_shared_logic]]）

---

## 5. 与其他工作的关系（实施 agent 必读）

- **🔴 与 TST spec（`2026-06-01-tst-paradigm-e2e-support-design.md`）协调 pendulum 文档**：
  - 用户决策 FST/TST **各自一份**。TST spec 管 TST 那份，**本 spec 管 FST 那份**（修法 C）。命名各自带范式名（`forced_swim-pendulum-params.md` / `tail_suspension-pendulum-params.md`），不共用一个文件 → 不冲突
  - **顺序**：本 spec（卡死修复）**应先于或同期 TST dogfood** — TST dogfood 会触发 step 2.8，若卡死未修则 TST 也卡。两个 agent 若并行，注意都不要去改对方那份范式文档
- **issue #63（@Qukoyk）**：velocity + 焦虑范式精确判据在此。本 spec 降级逻辑「判据缺失记 info 跳过」**不等 #63 也能根治卡死**；#63 回来后参数审计精度才提升
- **Sprint 4（调参指南）**：step 2.8 的 suggestion 指向「参数调整指南」，Sprint 4 未做时 suggestion 文字仍合理（用户找不到指南内容而已），不影响降级
- **data_analyst.py 是多 sprint 热点 + 受保护文件**：Sprint 3/5/5.8/6 都改过它。改前 grep 编号、改后全量测试

---

## 6. 不在范围

- ❌ 工程编 pendulum/velocity 领域阈值数值（归 issue #63 / 同事）
- ❌ 单纯调大 max_turns 当唯一修法（治标，不拆死胡同照样烧光）
- ❌ 改 ParameterAuditFinding 的 mismatch_kind 枚举（schema SSOT，用现有值 + info severity 表达「判据缺失」）
- ❌ 改 step 2.8 以外的 workflow 步骤（除 step 2.7 加一句轮次约束）
- ❌ 改 executor.py 的 seal-resume / max_turns 机制（那是兜底层，根因在 prompt 死胡同，不在兜底）
