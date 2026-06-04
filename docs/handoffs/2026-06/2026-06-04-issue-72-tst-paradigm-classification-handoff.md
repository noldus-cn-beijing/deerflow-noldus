# 2026-06-04 issue #72 TST 范式归类遗留项收口 交接文档

> **面向下一个 AI Agent**。本会话处理 GitHub issue #72（TST/悬尾实验范式上线遗留的 3 项需行为学同事确认项）。**工程侧已收口、无遗留**。仓库 HEAD = `08d36542`（已 push origin/dev）。

---

## 1. 当前任务目标（已达成）

用户起点：「解决之前遗留的问题，行为学同事已经 commit 了一些」+ 指向 [issue #72](https://github.com/noldus-cn-beijing/noldus-insight/issues/72)。

issue #72 = TST 范式上线时 3 项需同事领域裁决的事（Q1 模板归类 / Q2 NoTemplate.md 补字段 / Q3 golden-case）。本会话：同事已在 issue 回复裁决 → 工程把裁决落地为代码边界锁 + 把领域文档改法作为建议回给同事。**全部完成，无半成品。**

---

## 2. 当前进展（全部 ✅）

| 项 | 状态 |
|---|---|
| **Q1 TST 模板归类** | ✅ 工程锁定 + 同事「同意」。commit `08d36542` 加 4 条回归测试钉死「TST 不跨范式归 FST 的 PorsoltCylinder」 |
| **Q2 NoTemplate.md** | ✅ 留同事填（领域 SSOT），issue 里已确认语义口径 |
| **Q3 TST golden-case** | ✅ 同事自填（"我会填在对应位置"），`golden-cases/case-006-tst-baseline/` 目录已就位 |
| **指标/脚本层独立性核查** | ✅ 确认 TST 完全独立于 FST（见 §4） |
| **issue 回复** | ✅ 贴了 2 条（首条 4620232440 + 收口 4621137805） |
| **Memory 落档** | ✅ 新建 `feedback_no_cross_paradigm_reuse_accept_duplication` |

### 唯一代码改动（commit `08d36542`，已 push）
- `packages/ethoinsight/tests/test_ev19_facts.py` **+34 行，只加测试**：
  - `test_tail_suspension_not_classified_under_porsolt_nozones`（compat==False）
  - `test_tail_suspension_not_classified_under_porsolt_allzones`（compat==False）
  - `test_tail_suspension_map_has_single_unambiguous_candidate`（`== ["NoTemplate"]`）
  - `test_forced_swim_keeps_its_own_porsolt_templates`（对照：FST 独占 Porsolt）
- **未碰生产代码**：`ev19_facts.py:62` 映射本就是 `tail_suspension:["NoTemplate"]`，本会话确认它已正确，只补回归锁。
- 红锚点验证过：误加 Porsolt 入 TST 映射 → 测试立即 fail；恢复 → green。
- 全量 `ethoinsight pytest tests/` = **602 passed, 69 skipped, 0 failed**。

---

## 3. 关键上下文

### 仓库状态
- 分支 `dev`，与 origin/dev 同步，HEAD `08d36542`
- **工作区有一批不属于本会话的 modified/untracked 文件**（并行工作的脏改动）：`CLAUDE.md`、`docs/milestone/*`、`packages/agent/scripts/deploy-via-tar.sh`、`packages/agent/docker/nginx/nginx.conf`、`packages/agent/frontend/src/**`（utils.ts/message-list.tsx/subtask-card.tsx/markdown-content.tsx）、`golden-cases/case-00X/`（空 TODO 模板脚手架）、一堆 `docs/handoffs|specs|plans/` untracked。**全部不要提交、不要回退** —— 它们是别的 track 的在途改动。

### issue #72 同事裁决原文（关键）
- **Q1**：`PorsoltCylinder-NoZones` 和 `NoTemplate` 本质上一个东西（都是单观察区、无分析区）；TST 用的就是强迫游泳模板。→ 后续追加「**Q1：同意**」（同意工程定 TST=NoTemplate）
- **Q2**：`NoTemplate` 不**对应**任何试验，但任何试验都能用它；真不用模板时用户一般一开始选「自定义新建」而非 NoTemplate 模式
- **Q3**：「我会填在对应位置」

### 用户校正的两条硬性认知（本会话最值钱，已存 memory）
1. **产品原则「不跨范式复用/归类」**：不同范式绝不跨范式用指标/归类，**即使结构 overlap 也接受重复（各自一份）**，而非复用。→ 所以 TST 只归 `NoTemplate`，不挂 FST 专属的 `PorsoltCylinder`，哪怕同事说「本质一个东西」。
2. **`NoTemplate` 准确语义**：= **用户做的实验根本不出自 EV 现有内建模板**（off-template/自定义），不是「通用单 zone」。实测佐证：真实 TST demo `tstHelperDemoVideo` 文件头**无 arena/zone 模板字段**（仅 `试验控制设置 1`）→ 本就是不套内建模板的设置 → 归 NoTemplate 正确。

---

## 4. 关键发现（已验证）

1. **TST 库层完全独立于 FST**（用户原则覆盖指标层，核查通过）：
   - `catalog/tst.yaml` / `metrics/tst.py`（`compute_*_tst`）/ `scripts/tst/*` 全指向 `scripts.tst.*`，dispatcher（`metrics/dispatcher.py:148-151`）按 `paradigm=="tail_suspension"` 分发本范式函数。**零 FST 交叉引用**（grep 实证）。
   - TST 有自己的实验模板名 `Tail Suspension Test XT190`（catalog `ev19_templates` 字段，与 FST 的 `Forced Swim Test XT190` 平行，不复用）。
   - 唯一共享 = `metrics/_common.py` 范式无关原语（immobility RLE 等）+ `dispatcher.py:373` 的 `n<5 功效不足` 通用统计警告。**这是工具原语/通用统计，不是跨范式指标复用**，不违反原则。

2. **catalog `ev19_templates` 字段 ≠ ev19_facts 的 template_id**：前者是 EV19 实验/协议名（`*XT190`，`loader.py:108` 消费），后者是结构 template_id（`PorsoltCylinder-NoZones` 等，identify 用）。两套命名空间，别混。

3. **识别走的是 skill copy 不是 source**：`identify_ev19_template_tool.py` 读 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/`（`_resolve_skills_ref_dir`），**不是** `docs/review-packages/`。两份已分岔、无自动 sync（只有一次性手动 cp）。改 source 文档对运行时零效果，除非 re-sync。

4. **文档三处归类口径矛盾仍在**（领域 SSOT 归同事，工程未改，只回建议）：
   - `docs/review-packages/2026-04-29-ev19-templates/by-experiment/tail_suspension.md:13` 适用模板写 `PorsoltCylinder-NoZones`（建议改 `NoTemplate`）
   - `by-template/PorsoltCylinder.md` 的 NoZones 变体「对应学术范式」列了 TST（建议移除）
   - 解读层文档已正确：`forced_swim.md`/`tail_suspension.md` 都写「FST/TST 解读各自独立不可交叉引用」

5. **Explore subagent 曾幻觉一处 source 内容**（声称 tail_suspension.md 有「NoTemplate—默认推荐」行，实际没有）。**接 subagent 报告必现场核实**（已 grep 坐实）。

---

## 5. 未完成事项（按优先级）

### 🟡 中（可选，本会话用户未要）
- **TST 端到端 dogfood 复测**：chart-maker guardrail 修复（上一会话 commit `12cf2c79`）后**从未验过 TST 全链路**。可用 `/home/wangqiuyang/DemoData/newdemodata/悬尾/`（`tstHelperDemoVideo`，单观察区单对象）跑：Gate 1 识别 TST 为已支持 → 指标（不动时间/潜伏期/次数）→ chart-maker 出图 → 报告。注意：TST 共用 pendulum 路径，若 n 小可能命中 data-analyst step 2.8（已修 `e5a0d53b`/`4caa78b8`，应不卡）。dogfood 前必 `cd packages/agent && make stop && make dev` 重启。
- 参考 dogfood 指南：`docs/sop/2026-06-04-playwright-multi-paradigm-dogfood-test-guide.md`

### 🟢 低（等同事）
- 同事填 `by-template/NoTemplate.md` 🟡 字段（Q2）
- 同事 PR TST golden-case `golden-cases/case-006-tst-baseline/`（Q3）
- 同事按建议改 review-package 两处归类文档（Q1 文档侧）；若改了，记得 re-sync 到 skill copy（见 §4-3）才运行时生效——但 identify 不依赖该字段，影响小

---

## 6. 建议接手路径

**本 issue 工程侧已收口，无依赖。** 下一步取决于用户开什么方向：
- 若用户要验 TST 全链路 → 走 §5 dogfood（最高验证价值）
- 若用户开新方向 → 直接开，本线干净
- 若同事回了文档/golden-case → 核实后按需 re-sync skill copy

**第一步永远先做**：`cd packages/ethoinsight && .venv/bin/python -m pytest tests/test_ev19_facts.py -q`（应 22 passed），确认 TST 回归锁在位。

---

## 7. 风险与注意事项

1. **别因同事「TST 用强迫游泳模板」的措辞就把 Porsolt 加进 TST 映射**——用户硬性原则不跨范式归类，且加了会让 identify 在真实无 zone 数据 ambiguous 多弹反问。回归测试已钉死，误改会立刻红。
2. **工作区那批脏改动不属于任何本会话 track**（frontend/nginx/deploy/CLAUDE.md/milestone/golden-cases 脚手架），**不提交不回退**。
3. **领域文档（review-packages / NoTemplate.md / golden-case）是同事 SSOT**，工程只回建议不擅改（[[feedback_ssot_lives_in_review_packages]] / [[feedback_single_source_of_truth]]）——哪怕同事 GitHub 措辞与产品原则相反，给改法让其定。
4. **改 ev19_facts 共享逻辑要跑全量** + grep 所有调用方（`get_default_template_for_paradigm`/`is_paradigm_template_compatible`/`is_valid_ev19_template`/`get_template_facts` + `EV19_TEMPLATE_PARADIGM_MAP` 直接索引）。本会话已映射调用方：`experiment_context.py`（软校验，mismatch 只 warning 不 block）+ `identify_ev19_template_tool.py`（`list()` 候选，不假设单元素）。
5. **接同事/subagent 自述必现场核实**（本会话 Explore 幻觉一处 source 内容）。

---

## 8. 下一位 Agent 的第一步建议

1. 读本文档 + memory `feedback_no_cross_paradigm_reuse_accept_duplication`（理解为什么 TST 只归 NoTemplate）
2. `cd packages/ethoinsight && .venv/bin/python -m pytest tests/test_ev19_facts.py -q` 确认回归锁（22 passed）
3. 确认用户新意图——本 issue 工程侧已收口；若用户提「验 TST」走 §5 dogfood，否则按用户新方向开

---

## 关键路径速查

```
# 本会话唯一改动（已 push origin/dev, commit 08d36542）
packages/ethoinsight/tests/test_ev19_facts.py   # +4 回归测试（TST 不跨范式归类锁）

# 决策核心（未改，确认正确）
packages/ethoinsight/ethoinsight/ev19_facts.py:62   # tail_suspension: ["NoTemplate"]

# TST 库层（独立于 FST，核查通过）
packages/ethoinsight/ethoinsight/catalog/tst.yaml
packages/ethoinsight/ethoinsight/metrics/tst.py
packages/ethoinsight/ethoinsight/scripts/tst/
packages/ethoinsight/ethoinsight/metrics/dispatcher.py:148-151,373

# 领域文档（同事 SSOT，工程只回建议）
docs/review-packages/2026-04-29-ev19-templates/by-experiment/tail_suspension.md:13
docs/review-packages/2026-04-29-ev19-templates/by-template/{PorsoltCylinder,NoTemplate}.md

# identify 运行时读的 skill copy（非 source）
packages/agent/skills/custom/ethovision-paradigm-knowledge/references/

# TST demo 数据（off-template，文件头无模板字段）
/home/wangqiuyang/DemoData/newdemodata/悬尾/

# issue
https://github.com/noldus-cn-beijing/noldus-insight/issues/72
# 同事裁决 comment 4621099943；工程回复 4620232440 + 4621137805

# TST spec（背景）
docs/superpowers/specs/2026-06-01-tst-paradigm-e2e-support-design.md

# 全量回归
cd packages/ethoinsight && .venv/bin/python -m pytest tests/ -q
```

---

## milestone 建议

本会话让 TST 范式上线 track 到达一个 checkpoint（issue #72 三项需同事确认事的工程侧收口）。建议更新/创建 milestone：
- **标题**：TST（悬尾）范式 — issue #72 领域确认项工程侧收口
- **关键摘要**：同事裁决 3 项（Q1 模板归类同意 TST=NoTemplate / Q2 NoTemplate.md 同事填 / Q3 golden-case 同事 PR）；工程 commit `08d36542` 加 4 回归测试钉死「不跨范式归类」原则（TST 不挂 FST 的 PorsoltCylinder）；核查确认 TST 库层完全独立于 FST；文档三处归类矛盾作为建议回给同事（领域 SSOT）。沉淀新 memory `feedback_no_cross_paradigm_reuse_accept_duplication`。剩余：TST 端到端 dogfood 复测（可选）+ 同事补 NoTemplate.md/golden-case。
