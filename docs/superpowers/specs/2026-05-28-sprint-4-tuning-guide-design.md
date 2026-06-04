# Sprint 4 实施 spec — 调参指南进 by-experiment md

**关联**：[2026-05-28 SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 4
**估期**：1 周
**前置**：**Sprint 2a 必须先合 dev**（Sprint 2a 把参数下沉进 catalog YAML 并打 `tunable_by_user: true` 标记；Sprint 4 的 md 文档要照着 catalog 实际可调参数清单写，不能写 catalog 没有的参数）；Sprint 3 已合更佳但非强制
**执行者**：交给独立 agent 执行 + **行为学同事 review**

---

## 1. 背景与目标

### 当前状态（Sprint 2a 之后）

- `catalog/_common.yaml.shared_parameters` 含 `velocity_threshold`、`velocity_min_duration`（跨范式共享）
- 各 paradigm yaml（fst/tst/epm/oft/ldb/zero_maze）含 paradigm 独有参数（如 fst.yaml 的 9 个 pendulum 常量、zero_maze.yaml 的 `zm_low_distance_threshold`）
- 所有可调参数都标了 `tunable_by_user: true/false`
- Sprint 3（如果已合）：data-analyst 警告"参数不合适"时，suggestion 字段指向"请参考 paradigm md 的参数调整指南段"

### Sprint 4 要解决的问题

研究者听到 agent 说"velocity_threshold 30 mm/s 偏高"——但**调到多少合适**？为什么？谁有权威建议？

| 来源 | 当前 | Sprint 4 后 |
|---|---|---|
| catalog YAML | 只有 default + valid_range，没有"为什么是 30" | 不变（catalog 是 SSOT，只放数字） |
| paradigm md | 只有"该范式有哪些指标"，没有"如何调参" | 末尾新增 `## 参数调整指南` 段，列每个 tunable 参数的物理含义、调参权衡、典型修改场景 |
| knowledge-assistant | 回答"为什么 velocity 阈值是 30" → 只能背 catalog 数字 | 能解释权衡（"30 mm/s 适合啮齿动物自发活动，鱼类可能要降到 5 因为静水中游速本就慢"） |

### 关键原则（grill 锁定 + 教训）

1. **catalog 数字 vs md 权衡，不重复** — catalog 是 SSOT 放可执行数据（default、valid_range、unit、tunable_by_user）；md 放无法机器化的领域权衡（生理基础、跨物种差异、典型边界）。检查机制：grep catalog 数字字面量在 md 里**不应该出现**（出现 = 重复）
2. **paradigm md 由行为学同事维护** — Sprint 4 agent 只产**第一版骨架**和**示例段落**，最后必须经同事 review。md 标 `🟡` 表示待同事补充（沿用现有 by-experiment md 的惯例）
3. **物种特异性必须显式标注** — 同一参数在啮齿动物和鱼类的"合适值"差 10 倍是常态，调参指南必须写清"啮齿动物典型 30，鱼类典型 5"，而不是"调到合适为止"
4. **不动 ev19-dependent-variables.md** — 那是公式 SSOT，不混应用指南；调参指南放 by-experiment/ 各范式 md

---

## 2. 文件改动清单

### 2.1 在每个 paradigm md 末尾追加 `## 参数调整指南` 段

**位置**：`packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/`

**v0.1 范围** — 仅 6 个已支持范式：
- `forced_swim.md`
- `tail_suspension.md`
- `epm.md`
- `open_field.md`
- `zero_maze.md`
- `light_dark_box.md`

**其他 14 个 md（鱼类 / 学习记忆类等）** — Sprint 4 不动；v0.1 不支持的范式，按 CLAUDE.md 项目状态保留为存档

### 2.2 `## 参数调整指南` 段模板

每个 md 在最后一个 `## 🟡 与其他实验的区分` 章节之后插入：

```markdown
## 🟡 参数调整指南

> **本段由行为学同事审核维护**。catalog YAML 是参数 default/valid_range 的 SSOT；本段是"为什么是这个值"和"什么时候应该调"的权威解释。两者不重复。

### 可调参数清单（由 Sprint 2a catalog 自动列出）

下表来自 `packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml`，
所有 `tunable_by_user: true` 的参数都在此列出（按 catalog 里出现的顺序）。

| 参数 | 物理含义 | catalog default | 典型变化范围 | 适用物种 |
|---|---|---|---|---|
| `<param_name>` | <一句话物理含义> | <default> | <典型范围> | <物种标签> |
| ... | ... | ... | ... | ... |

### 调参权衡

对每个 tunable 参数，回答下列问题（不必每个都答，但调过的必须答）：

**`<param_name>`**

- **生理基础**：这个参数物理上对应什么？（如 "velocity_threshold 是判断动物'移动 vs 不动'的速度阈值"）
- **调高会怎样**：（如 "阈值调高 → 更多帧被判为不动 → immobility 时间膨胀 → 抑郁样行为表型被过度估计"）
- **调低会怎样**：（如 "阈值调低 → 噪声运动也算移动 → immobility 被低估 → 抑郁样表型被错过"）
- **物种典型值**：列出 3-5 个常见物种的推荐范围（啮齿动物、鱼类、昆虫 …）
- **何时应调**：列出 1-2 个明确触发场景（如 "本批数据 velocity 中位数 < default 的 1/3 时考虑下调"）
- **何时不应调**：1 个反例（如 "default 在 valid_range 内，且数据分布与 default 物种匹配，则不调"）

### 与 data-analyst 警告的对接（Sprint 3 衔接）

data-analyst 的 `ParameterAuditFinding.suggestion` 会指引研究者来看本段。
本段的目标是让研究者**自己**做出调参决定，**不是**给出"调到 X"的命令式答案。
催调参的最终一步在 Sprint 4.5：写到 `experiment-context.json.parameter_overrides`
重跑 plan。

### 参考文献（行为学同事补）

🟡 待补充
```

### 2.3 第一版骨架示例：FST 的"参数调整指南"

Sprint 4 agent 必须为每个 paradigm md 至少写**一个完整调参权衡示例段**作为种子（其他参数留 🟡 等同事补）。FST 的示例（agent 必须写，同事 review）：

```markdown
**`velocity_threshold`**

- **生理基础**：判断动物"移动"vs"不动"的速度门限。FST 中"不动"是关键 readout（抑郁样行为指标）；不动 = 漂浮+小幅平衡运动。
- **调高会怎样**：阈值升 → 更多帧判为"不动" → immobility_time 上升 → 可能误判为"高抑郁样"
- **调低会怎样**：阈值降 → 平衡运动被算成"运动" → immobility_time 下降 → 可能错过真实表型
- **物种典型值**：
  - 啮齿动物（小鼠、大鼠）: 30 mm/s（catalog default）
  - 鱼类（斑马鱼）: 5-10 mm/s（静水中游速本就慢）
  - 注：物种差异源于运动学差异，不能跨物种用同一阈值
- **何时应调**：本批数据 velocity 分布（中位数/p90）与 default 物种不匹配时——data-analyst Sprint 3 会自动给出 ParameterAuditFinding 提示
- **何时不应调**：default 在数据分布合理范围内时，保留 default（可复现性 > 调参收益）

**`velocity_min_duration`**

🟡 待行为学同事补
```

### 2.4 data-analyst workflow 加 "read paradigm md 时 grep 参数调整指南段"

**位置**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`

在步骤 2.6（按范式 read 判读文档）的指令末尾加：

```
   - 读完 paradigm md 后，grep "## 🟡 参数调整指南" 段；如果存在：
     - 把该段内容缓存到 working memory（不在 method_warnings 重复展示）
     - Sprint 3 的 ParameterAuditFinding.suggestion 已经指向本段；
       本段内容用于：a) 在 method_warnings 写更具体的解读（"参数 velocity_threshold
       的物种典型值 30，当前数据物种为鱼类，建议参考调参指南段"）；b) 透传给 report-writer
   - 如果该段不存在（说明该 paradigm md 还没补 Sprint 4 内容），按"⚠️ paradigm md 暂无调参指南，仅给出参数审计警告" 处理
```

### 2.5 knowledge-assistant workflow 加 "解释参数权衡"

**位置**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py`

knowledge-assistant 当前能读 paradigm md。Sprint 4 完成后它会自然能回答"为什么 velocity 阈值是 30"——只需 prompt 引导：

在 knowledge_assistant.py system_prompt 加一句：

```
当用户问"为什么参数是 X"时：
1. read paradigm md 对应章节 → 找 "## 🟡 参数调整指南" 段
2. 引用 md 里的"生理基础 / 物种典型值 / 调参权衡" 三个角度回答
3. 严禁背 catalog YAML 里的数字 — catalog 是机器读的，md 才是人读的解释
```

### 2.6 review 流程（关键，非代码改动）

Sprint 4 是 v0.1 唯一显式依赖行为学同事补内容的 sprint。流程：

1. **agent 写骨架** — 6 个 paradigm md 各加 `## 🟡 参数调整指南` 段、含可调参数表（自动从 catalog 同步）和**至少一个完整示例**（FST 的 velocity_threshold、ZM 的 zm_low_distance_threshold 等）
2. **agent 写 review-package** — `docs/review-packages/2026-05-28-sprint-4-tuning-guides/README.md` 列：每个 paradigm 需要同事补的内容点、问题模板（"你认为 FST velocity_threshold 的鱼类合理范围是？"）、参考文献占位
3. **同事 review** — 收同事 PR 后**直接**搬入 by-experiment md（不要二次解读，sync 上游教训：同事的领域知识不能被 agent 再加工）

### 2.7 单元测试（轻）

新增 `packages/agent/backend/tests/test_paradigm_md_tuning_section.py`：

| 测试 | 验证 |
|---|---|
| `test_all_v01_paradigms_have_tuning_section` | 6 个 v0.1 范式 md 都含 `## 🟡 参数调整指南` 段 |
| `test_tuning_section_lists_catalog_tunable_params` | grep md 中的可调参数清单表与 `catalog/<paradigm>.yaml` 里 `tunable_by_user: true` 的参数一致 |
| `test_tuning_section_does_not_duplicate_catalog_defaults` | 至少一个 default 数字字面量（如 "30"）不在 md 段落正文中作为推荐值出现（防 SSOT 漂移） |

**注**：测试 2 是关键的 CI 哨兵——catalog 加新 tunable 参数时如果忘了在 md 同步，CI 红；反之 md 写了 catalog 没有的"假参数"也会红。

---

## 3. 实施顺序（task 拆分）

| Task | 内容 | 估时 |
|---|---|---|
| T1 | 读 6 个 paradigm md + Sprint 2a 后的 catalog YAML，整理每个 paradigm 的 tunable 参数清单 | 0.5 天 |
| T2 | 写 review-package 模板（含每 paradigm 的问题清单 + 参数表）| 0.5 天 |
| T3 | 6 个 paradigm md 末尾加骨架（参数表 + 至少 1 个完整示例段，其他参数留 🟡）| 1.5 天 |
| T4 | data_analyst.py prompt 加步骤 2.6 末尾的"grep 调参指南段"逻辑 | 0.25 天 |
| T5 | knowledge_assistant.py prompt 加"解释参数权衡"指令 | 0.25 天 |
| T6 | 单元测试 3 个（CI 哨兵） | 0.5 天 |
| T7 | dogfood：FST data-analyst 警告 + knowledge-assistant 回答"为什么是 30" → 引用 md 段 | 0.5 天 |
| T8 | 全量回归 + 修退化 | 0.25 天 |
| **合计** | | **4.25 天 ≈ 1 周（不含同事 review 时间）** |

---

## 4. 风险与缓解

| 风险 | 缓解 |
|---|---|
| catalog 数字字面量在 md 重复（SSOT 漂移）| T6 测试 3 用 grep 哨兵；agent 写 md 时只写"catalog default"占位符，不写具体数字 |
| 同事 review 拖延 | T2 review-package 用问题模板 + 占位 🟡 让 md 上线时已有结构、只缺内容；agent 不阻塞，先让 v0.1 dogfood 用"⚠️ 调参指南待补"消息 |
| 6 个 md 同步修改导致 conflict | T3 顺序处理（FST → TST → EPM → OFT → ZM → LDB），每个 md 独立 commit |
| data-analyst 把同事还没补的 🟡 段当真"调参建议" | T4 prompt 明确"如该段含 🟡 占位，按'paradigm md 暂无调参指南'处理"——agent 不能拿占位当内容 |
| md 写得太啰嗦导致 prompt 注入超 token | 每个 paradigm 的调参指南段总字数控制在 800-1500 中文字符；超出的写到 review-package 子文件分批 read |
| 跨物种参数差异未标注（用户照啮齿动物默认值跑鱼数据）| 物种典型值表是 spec §2.3 强制项；T6 测试可加哨兵"调参指南段必须含'物种'关键字" |

---

## 5. 验收 checklist

实施完成时，确认全部通过：

- [ ] 6 个 v0.1 paradigm md 全部含 `## 🟡 参数调整指南` 段
- [ ] 每段含：可调参数表 + 至少 1 个完整调参权衡示例 + 与 data-analyst 警告对接说明 + 参考文献占位
- [ ] 每段表中的参数与 `catalog/<paradigm>.yaml` 里 `tunable_by_user: true` 的参数完全一致
- [ ] 没有 catalog default 数字字面量被复制到 md 正文（grep 哨兵全过）
- [ ] data_analyst.py 步骤 2.6 含"grep 调参指南段"逻辑
- [ ] knowledge_assistant.py system_prompt 含"解释参数权衡"指令
- [ ] 单元测试 3 个 CI 哨兵全绿
- [ ] dogfood：FST velocity 中位数低 → data-analyst 警告 → 用户问"为什么阈值 30" → knowledge-assistant 引用 md 段回答
- [ ] dogfood：knowledge-assistant 回答中**不直接**说"30"这个具体数字（应说"啮齿动物典型值，参考 catalog"）
- [ ] review-package 已生成、待同事补充内容
- [ ] 全量测试通过（agent backend ≥3111+Sprint 3+~3）

---

## 6. 不在 Sprint 4 范围

明确**不做**的事：

- ❌ 鱼类/学习记忆类等 14 个 v0.1 不支持范式的调参指南
- ❌ 自动调参（"data-analyst 警告 → 自动改 overrides"）— Sprint 4 仍坚持"只警告不调参"原则
- ❌ paradigm md 重组（不动现有章节结构，只追加 1 段）
- ❌ ev19-dependent-variables.md（公式 SSOT 不混应用指南）
- ❌ catalog YAML 加新字段（如 `human_explanation` 字符串）— 那样会让 catalog 变 md，违反"catalog 只放可执行数据"原则
- ❌ frontend 渲染调参指南（用户走 knowledge-assistant 路径访问 md，不直接看 md 文件）

---

## 7. 参考

- [SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) — Sprint 4 章节
- [Sprint 2a spec](2026-05-28-sprint-2a-catalog-parameters-design.md) — catalog YAML 的 `tunable_by_user` 标志
- [Sprint 3 spec](2026-05-28-sprint-3-data-analyst-parameter-audit-design.md) — `ParameterAuditFinding.suggestion` 字段指向本段
- by-experiment md 模板：参考已有 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md`
- Skills 渐进披露设计：`packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md`
- CLAUDE.md 注意事项第 10 条"范式体系正在重构"——by-experiment 是同事维护范畴
