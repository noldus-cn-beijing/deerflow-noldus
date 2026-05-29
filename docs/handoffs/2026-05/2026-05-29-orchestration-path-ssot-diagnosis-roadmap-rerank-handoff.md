# 2026-05-29 编排路径 SSOT 诊断 + roadmap 三档重排 + 剩余 sprint spec 补齐 + stash 事故清理 — handoff

**会话主体**：Dynamic Workflows 文章引发的架构深挖（编排路径 SSOT）→ 诊断 + 阶段 A 实施 spec → roadmap 三档重排 → 补齐 6/7 骨架 + Sprint 3 升可实施版 → Sprint 4 卡点修正 → 给同事提 issue #63 → 处理一起 stash pop 残留事故 + 清理 8→3 worktree → 写多 agent 纪律 SOP
**触发分支**：dev
**dev HEAD**：`625a47f5`（本地 = origin/dev，已同步 push）
**本会话 commit 链**（接手点 `e9f694a5` 之后）：
- `8a8f77b7` roadmap grill 复审修订 + 补齐 5.5/3/4/5 spec（前一 agent 故障接手后补完）
- `e7724e5b` 编排路径 SSOT 诊断 + roadmap 三档重排 + Sprint 5 迁移注记
- `599ec558` 编排 SSOT 阶段 A 实施 spec + Sprint 4 参数继承卡点修正
- `1501ca65` 解决 ev19_facts stash pop 残留冲突
- `6c86162a` Sprint 3 spec 升级为可实施版
- `625a47f5` 多 agent worktree 纪律 SOP

**测试基线**：未跑（本会话纯 docs + 一处 1 行冲突解决，无代码逻辑改动）。主仓权威基线沿用：backend 3145 / ethoinsight 504 / frontend tsc 0。

---

## 1. 本会话的核心产出：编排路径 SSOT（架构主线）

### 1.1 缘起与第一性洞察
用户给了一篇讲 **Claude Code Dynamic Workflows**（命令式 JS 编排脚本，主 agent 睡觉、agent() 点临时雇 LLM）的文章，问对我们框架有什么启发。深挖收敛出一个**用代码证据坐实的诊断**：

> 我们把"算什么"（范式→指标→图表）工程化成了 **catalog SSOT（正面范例，已工程化）**，但"按什么顺序算、何时停下问用户"（**编排路径**）还停留在 `lead_agent/prompt.py:286-294` 的**自然语言箭头图**里。这一个月所有鲁棒性 bug（seal 漏调、lead 不读 handoff、意图误判、ask 漏问）的统一根因 = **编排路径"三层各说各话"**。

**关键澄清（务必传给下一 agent，防修错对象）**：诊断针对**编排路径**，**不针对**范式/指标/图表知识（后者 catalog YAML + review-packages 已是 SSOT，是要学的范例，绝不重做）。

### 1.2 三层 head-to-head 证据（已核实）
| 层 | 内容 | 覆盖路径多少 |
|---|---|---|
| prompt 箭头图（`prompt.py:286-294`） | 8 条 INTENT 完整路径 | 唯一完整描述，但是**自然语言**，LLM 读 |
| provider 规则（`guardrails/`） | intent_classification 只管"声明了没"；ask_gate 只保护 8 个 ask 点里的 **1 个**（viz）；handoff_auth 管依赖不管顺序 | 零星打补丁 |
| `required_upstream_handoffs`（`subagents/builtins/*.py`） | DAG 的**边（依赖）** | 是依赖不是路径 |

**三个可指认的洞**：① 跳过 data-analyst 直接派 chart-maker **无人拦**（顺序无 provider）；② 8 个 ask 点只硬保护 viz 1 个；③ 三层命名/粒度不一致，无 CI 哨兵校验。

### 1.3 关键决策（已锁定，写进 spec）
- **SSOT 形态 = 声明式 Python 数据**（dataclass + dict），**不是命令式代码/JS**。理由：path SSOT 要被 3 个消费者读取（prompt 渲染 / 顺序 provider / ask provider），命令式代码只能"执行"不能"多视角读"。这是 **catalog 模式的同构物**，不是 Workflow 的 JS 模式。
- **SSOT 位置 = `subagents/path_registry.py`**（与 SubagentConfig 同处，依赖+路径 single source）
- **分两阶段**：A（prompt 从 SSOT 渲染纯重构 + 派遣/ask provider 从 SSOT 生成，P0）/ B（`interrupt()` 根治唤醒点状态丢失，P2）
- **interrupt 地基已核实**：deerflow 有 checkpointer（`runtime/checkpointer/` + `langgraph.json` 注入）但**全仓从未用过 `interrupt()`**，唤醒点现在是 `Command(goto=END)` + 重入。根治是中等改造非配置开关。

### 1.4 产出文档
- `docs/superpowers/specs/2026-05-29-orchestration-path-ssot-diagnosis-design.md`（诊断 + 设计骨架，含 §3.3 interrupt 地基核实）
- `docs/superpowers/specs/2026-05-29-orchestration-path-ssot-phase-a-impl.md`（**阶段 A 可实施 spec**，含 Step/PATHS schema、A1 渲染 + A2 两 provider、CI 哨兵、TDD 7 task ~3 天）

---

## 2. roadmap 三档重排（已落地 roadmap v2）

用户拍板"借机重排优先级"+ 关键纠正（**golden-cases 最不重要；端到端 dogfood 是每 sprint 收尾纪律不进 roadmap**）。落地为 roadmap v2 新增"2026-05-29 优先级重排"段：

| 档 | 内容 |
|---|---|
| **P0** | ① 编排 SSOT 阶段 A（**最前**，地基先行）→ ② Sprint 3 → ③ Sprint 4 → ④ Sprint 5 |
| **P1** | Sprint 5.5、Sprint 4.5 |
| **P2（v0.2）** | Sprint 6、7、8、编排 SSOT 阶段 B |

理由：SSOT-A 放最前 → Sprint 5 的 quality gate provider 可"从 SSOT 生成"而非又一块手写补丁（已给 Sprint 5 spec 加迁移注记）。

---

## 3. 各 sprint spec 文档状态（全景）

| Sprint | spec 状态 | 卡点 |
|---|---|---|
| 0/1/2a/2b/5.7/5.8 | ✅ 已实施 | — |
| **编排 SSOT 阶段 A** | ✅ 可实施版 | 无，**正在被 agent 实施**（worktree `orchestration-path-ssot-phase-a`，已建 path_registry.py） |
| **3 参数审计** | ✅ 可实施版（本会话升级） | 不卡同事可实施（pendulum 判据现成 + velocity 用保守默认兜底，精确判据待 #63） |
| **5.5 handoff 内容校验** | ✅ 核验版 | **未实施**（5.8 注释立的案，可直接 TDD，挂载点 executor.py:146；独立不卡任何人） |
| 4 调参指南 | ✅ 骨架版 | **内容卡同事**（issue #63）；工程只做通路 |
| 5 数据质量门 | ✅ 骨架版 + 迁移注记 | `workflow_mode` 已验通（per-run configurable，GateEnforcement 已示范 manual 挂载）；撞编排悖论建议等 SSOT-A |
| 4.5 / 6 / 7 | ✅ spec 在（4.5 是 5-28，6/7 骨架） | 6/7 是 P2 上层建筑 |
| **8 feedback 回流** | ✅ 骨架版（本会话补写） | 最低优先 P2（v0.2）；自标短期补丁，微调到位后收益减半；前置 feedback 表加 paradigm 字段（Alembic migration） |

---

## 4. Sprint 3 的行为学知识需求（本会话核实结论）

参数审计要"判据"（数据满足什么条件算参数不匹配），分三类：
- 🟢 **pendulum 8 参数判据 = 够了**：同事 `docs/review-packages/2026-0521-feedbacks/tstYoyo/tst-pendulum-algorithm.md` §3/§4/§5/§7 写全（periodicity 钟摆>0.5/挣扎<0.3 等），工程直接照用
- 🔴 **velocity_threshold 物种值 + mismatch 判据 = 缺，最急**：5 范式共用命脉参数，issue #63 最高优先
- 🟡 **焦虑范式质检阈值（motor_low_entries=8 / signal_low_transition=4 / zm_low_distance=10）= 缺，可暂缓**：用保守默认先跑

**结论：Sprint 3 不卡同事也能实施**（pendulum 现成 + velocity/焦虑用保守数学默认），精确版待 #63。

---

## 5. 给同事的 issue #63（@Qukoyk）

- **#63 OPEN，0 评论（同事还没回）**：https://github.com/noldus-cn-beijing/noldus-insight/issues/63
- **为什么提到主 repo `noldus-insight` 而非同事的 `noldus-insight-ruo`**：ruo 的 Issues 功能关闭（`hasIssuesEnabled: false`），主 repo 开着；正文里说明内容 PR 回 ruo 的 review-packages
- **内容**：6 范式调参指南（Sprint 4）+ 追加评论拆解 Sprint 3 优先级（velocity 最急🔴 / pendulum 够🟢 / 焦虑质检可缓🟡）
- **SSOT 边界**：同事只写"调多少/为什么/物种差异"，**不写默认值数字**（那在 catalog YAML）

---

## 6. 本会话的一起 git 事故 + 清理（重要，影响 git 操作纪律）

### 6.1 事故：主仓 dev 上 stash pop 残留卡死提交
- 提交 Sprint 3 时报"未合并文件 `ev19_facts.py`（UU）"——**不是本会话造成**
- 真因：`stash@{0}: WIP: ev19_facts path fallback (kept aside for shoaling-removal PR)` 在主仓 dev 被 `git stash pop` 冲突未解决（`<<<<<<< Updated upstream / Stashed changes` 是 stash 标记，非 merge）
- **不是编排 SSOT agent 干的**（已核实它在自己 worktree 正常建 path_registry.py）
- 处理：备份自己改动到 /tmp → 不擅自解冲突 → 报告用户 → 用户拍"保留上半"（包内自包含路径 `_FACTS_JSON_PATH`，与已合 fa568499 一致；删死变量 `_FACTS_JSON_ALT_PATH`）→ 解决 + drop 该 stash

### 6.2 清理结果
- worktree 8 → 3：删 6 个已合并+干净的（sprint57/paradigm-key/pr3/retire-shoaling/sync-cleanup/sprint-2b）+ 删对应孤儿分支
- **剩 3 个**：主仓 dev + `sprint-0-handoff-schema`（未合，勿删）+ `orchestration-path-ssot-phase-a`（活跃 agent）
- stash：drop 了 ev19 那条；**保留 3 条不是本会话的**（PR#29 / subagent-role-split / g4-frontend）—— 下个 agent 别动，不清楚来源

### 6.3 沉淀为 SOP
`docs/sop/multi-agent-worktree-discipline-sop.md`：5 条纪律——① 主仓 dev 绝不 stash/pop ② worktree 用完即删 ③ agent 隔离走 PR 不直推 dev ④ stash 当天清 ⑤ 提交前 status 检查，发现非己脏状态即停手报告。

---

## 7. 下一会话执行清单

### A. 可立即派的 agent（spec 就绪，文件交集已分析）
1. **Sprint 5.5**（核验版，最干净）：handoff 内容非空校验，挂 executor.py:146，独立不卡任何人，可直接 TDD
2. **Sprint 3**（可实施版）：参数审计，用保守判据 + pendulum 现成判据，不卡同事。**⚠️ 实施 agent 必读 2026-05-28 原始 spec**（`2026-05-28-sprint-3-data-analyst-parameter-audit-design.md`）作为 schema 权威——5-29 这份是增量升级版，`ParameterAuditFinding` 完整定义（mismatch_kind 5 值 = threshold_too_high/low + window_too_wide/narrow + category_mismatch；severity 由受影响 subject 比例定义非由类型定义；判据阈值 p90×3 等）在原始 spec，5-29 spec §3.5.4 已补指引。只读增量版会凭推断踩偏（本会话实施 agent 已踩过一次，已修）
3. **编排 SSOT 阶段 A**：已在跑（worktree `orchestration-path-ssot-phase-a`）
- **并发前务必 head-to-head 验文件交集**（SOP §3）：Sprint 3 改 data_analyst.py + handoff_schemas.py + lead prompt 播报段；编排 SSOT 改 lead prompt 意图状态机段 + path_registry + provider。**两者都碰 lead prompt 但不同片区**——约定谁先合、Sprint 3 在播报段留 `# TODO(orchestration-ssot)`

### B. 待同事答复 issue #63 后再修补（修补 = 换判据数字，不动结构）
issue #63（@Qukoyk，OPEN）：velocity 物种判据（Sprint 3 精度）+ 6 范式调参指南（Sprint 4 内容）。

**工作节奏（已与用户确认）**：现在全速做 Sprint 3/4 的**结构**（不等同事），#63 答复后只做**精度修补**，因为修补是"换数字不重做"：
- **Sprint 3**：step 2.8 里的 velocity/焦虑保守判据阈值（`p90×3` 等）→ 换成同事的精确物种值。schema/workflow/seal/gate_signals **全不动**
- **Sprint 4**：md 里占位的 `## 参数调整指南` 段 → 搬入同事写的内容。工程通路（grep 逻辑）本就已做好
- 关键：**现在做结构不浪费**——一次到位，#63 回来只提精度，不返工。等同事会阻塞结构开发，保守默认让结构先落地、精度异步补

### C. 可选补
- ~~Sprint 8 spec~~ ✅ 已补（骨架版，`2026-05-29-sprint-8-feedback-reflow-skeleton.md`）。**至此 0-8 + 4.5 + 编排 SSOT 全部有 spec**
- 清理：/tmp/probe_*.py（旧）、/tmp/sprint*-*.md + /tmp/issue63-*.md（本会话临时正文）

### D. demo 判断（用户关心）
- **缺 Sprint 4 不影响 demo 端到端**：主干（上传→算→解读→图→报告）不依赖 3/4（5-28 FST dogfood 已证）。唯一风险=demo 用"参数不匹配数据"触发 Sprint 3 警告、警告指向空的 Sprint 4 调参指南。**demo 用标准匹配数据（小鼠标准范式）即可全程绕开**。建议 3+4 当一对处理

---

## 8. 给下一 agent 的友情提示
1. **dev HEAD = `625a47f5`**，本地=origin/dev，全部 push GitHub（GitHub-only，无 GitLab）
2. **遵守新 SOP**（`docs/sop/multi-agent-worktree-discipline-sop.md`）：主仓不 stash、提交前 git status、发现非己脏状态停手报告
3. **3 条 stash 别动**（不是本会话的）；2 个 worktree 别删（sprint-0 未合 + 编排 SSOT 活跃）
4. **编排 SSOT 的根纪律**：path_registry 是**声明式数据**，若看到 `await`/`while`/`if` 控制流就是方向错了（那是 Workflow JS 模式，不是我们的）
5. **catalog 是 SSOT 正面范例，不要重做**；要工程化的是"编排路径"不是"指标/图表知识"
6. **方法论**：本会话全程"先核实再下结论"——3 次修正了自己基于印象的判断（"同事没写调参"→其实 pendulum 写了；"Sprint 3 全卡同事"→其实判据分两类；"都写完了"→其实 Sprint 8 没写）。延续这个纪律，别凭印象答
7. `.env.wecom` 永不 commit

---

## 9. milestone 建议
本会话让两个 track 到达 checkpoint：
- **sota-agent-v2**：roadmap 三档重排（P0/P1/P2）+ 剩余 sprint spec 补齐（除 Sprint 8）。状态加"2026-05-29 重排 + 编排 SSOT 立项"
- **新建 `orchestration-path-ssot` milestone**：记录这条新 track 的完整脉络——Dynamic Workflows 文章引发 → 三层 head-to-head 诊断 → 声明式 SSOT 决策 → 两阶段（A 渲染+provider / B interrupt 根治）→ interrupt 地基核实。这是 v0.2 鲁棒性根治的核心 track
