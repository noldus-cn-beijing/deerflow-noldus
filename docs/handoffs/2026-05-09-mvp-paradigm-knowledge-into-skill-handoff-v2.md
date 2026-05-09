# 2026-05-09 MVP 范式知识搬入 skill — v2（含 grill-me 修订全程）

> **前置文档**：
> - [v1 初版交接](./2026-05-09-mvp-paradigm-knowledge-into-skill-handoff.md) — 仅覆盖搬入 + 初版 SKILL.md（含反模式），后被 grill-me 修订
> - 本文档（v2）= 完整会话记录：搬入 → 初版 SKILL.md → grill-me 修订 → push

## 当前任务目标

让行为学同事 5月8日 PR `0ec87dc1 "MVP范围内模板信息更新"` 中填好的 MVP 6 范式（EPM / OFT / ZeroMaze / LDB / TST / FST）领域知识进入 `ethovision-paradigm-knowledge` skill，让 agent 在分析流水线里能用上这些 markdown。

## 当前进展（已完成）

✅ **commit `96d6e2c4`** — 搬入同事 PR `0ec87dc1` 的 11 个覆盖 + 1 个新增 markdown 到 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-{experiment,template}/`
- by-experiment 覆盖：epm / open_field / zero_maze / light_dark_box / forced_swim
- by-experiment 新增：tail_suspension（5月8日地基搬入时此文件不存在）
- by-template 覆盖：PlusMaze / OpenFieldRectangle / OpenFieldCircle / ZeroMaze / PorsoltCylinder

✅ **commit `60cccd02`** — 初版 SKILL.md 加"分析阶段使用本 skill"段（**含反模式，被后续 commit 修订**）
- 这版引导写的是 "lead 在派遣 code-executor / report-writer 前 read by-experiment 抽段拷进 task() prompt"
- 用户随后明确指出该设计违反**多 agent 框架的核心理由**（上下文隔离 + 并行能力），lead 替 subagent 干领域判断会让 lead context 膨胀

✅ **commit `474fcf04`** — v1 交接文档（仅记录搬入 + 初版 SKILL.md，不含 grill-me 修订）

✅ **commit `8e6a6c1c`** — 按 grill-me 7 题最终决定修订 SKILL.md（**最终形态**）
- 删 "## Workflow" 段（Step 1/2/3 是 identification-decision-tree.md 详细版的简化重复）
- 删 "## 知识资源（按需 read_file 加载）" 段，替换为 "## 工作场景" 对照表
- 删 "## 分析阶段使用本 skill" 段（剔除"lead 抽段"反模式）
- 删核心原则第 4 条（反问前必读 raw meta — 识别阶段技术细节，归 identification-decision-tree.md 详细版）
- description 覆盖识别 + 分析两阶段（影响 agent 是否 read SKILL.md body 的关键字段）
- 工作场景对照表 2 行（识别 / 分析 → references 路径，不绑定 agent 角色）+ 路径占位符规则（`<范式>` / `<大类>` 取法 + 文件不存在兜底）

✅ **4 个 commit 全部 push 到 origin/dev**（`75d460ce..8e6a6c1c`）

## 关键上下文

### 项目状态
- 当前分支：`dev`（已与 origin 同步）
- 5月8日 EV19 模板识别地基 commit 已落地（spec [docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md](../superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md)）：skill 骨架 + ev19_facts.py + GuardrailMiddleware + set_experiment_paradigm 升级

### Skill 系统机制（CLAUDE.md backend 第 252 行）
- **SKILL.md 全文** = 进 system prompt（自动激活），每个 enabled skill 都加载到 lead + subagent system prompt
- **references/** = 不进 system prompt，agent 主动 `read_file` 才加载（渐进披露）
- 各 agent 的 `available_skills` 白名单：
  - **lead** = `None` 全继承（看到全部 5 个 enabled skill，含 `ethovision-paradigm-knowledge`）
  - **code-executor** = `["ethoinsight-analysis"]` 显式白名单（**看不到** `ethovision-paradigm-knowledge`）
  - **data-analyst / report-writer / knowledge-assistant** = `None` 默认全继承

### Review 包的角色（澄清后）
- review 包 `docs/review-packages/2026-04-29-ev19-templates/` = 行为学同事填写领域知识的工作目录
- 同事按 MVP 节奏分批补充：本批是 MVP 6 范式（已合 `0ec87dc1`），其他 14+14 是 MVP 范围外故意未填
- 工程侧每次同事 PR 后**搬运一次**到 skill references；不写双向同步脚本（频次低）
- shoaling.md / AquariumTrack3D.md 等 MVP 范围外文件保留 5月8日地基搬入的占位形态（同事故意未填，不要去推同事补）

## 关键发现 / 决策记录

### 多 agent 框架的核心理由（贯穿 grill-me 全程）
1. **上下文隔离** — subagent 拿专属 system prompt + 任务 prompt，避免 lead 历史污染
2. **并行执行** — 多 subagent 同时干活而非串行

**反模式**：lead 抢着自己做（如 read by-experiment 抽段拷进 task() prompt），把多 agent 退化成单 agent。`60cccd02` 那版就是反模式，`8e6a6c1c` 把它纠正回来。

### Skill 设计原则（grill-me 沉淀）
- **不绑定 agent 角色**：SKILL.md 工作场景对照表写"做 X 工作时 read Y"，不写"如果你是 lead/code-executor 就..."。让 agent 自己根据当下任务上下文判断（保留 LLM 的鲁棒性 + 随机性，不是代码 1+1=2）
- **渐进披露**：SKILL.md 入口本身要短，不复述 references 内容；by-experiment 文件内部段落结构（必算指标 / 报告解读语言 / 与其他实验的区分等）由 agent read 后自己看到，不在 SKILL.md 预先抄出
- **场景颗粒度中粒度**（不是粗到"分析阶段"也不是细到"必算指标段"）：让 agent 看见自己当下在做的具体动作能识别

### 同事的 markdown 形态判断
- 同事填好的 by-experiment（如 epm.md / PlusMaze.md）含 🟡/🟢 emoji 锚点 + 少量 HTML 注释残留
- 这些 emoji 是同事**有意保留的视觉锚点**（区分机器抽取 vs 同事填写），不是 noise
- "（待补充）" 占位段只在**同事尚未填写**的文件里出现（shoaling/AquariumTrack3D 等 MVP 外）
- 直接 cp 是 OK 的，**不**需要写转换脚本去 emoji / 注释 / 占位符

### EV19 模板体系（精确表述）
- **20 大类 / 62 变体**（大类下细分小类合计 62）
- ❌ 不要说 "62 种 EV19 模板"
- ❌ 不要说 "62 变体" 单独使用，要说 "20 大类 / 62 变体"

## 未完成事项（按优先级）

### P0 — 立即可做（地基已就位）

1. **写 `templates/epm.py`**（最高优先级）
   - 同事 EPM ground truth 已就位（`by-experiment/epm.md` 写明：开臂时间百分比 / 开臂进入百分比 / 开臂进臂次数 / 开臂进臂时间 / 总进臂次数 + n<5 警告 / 总进臂<8 警告）
   - 模式：照 `packages/ethoinsight/ethoinsight/templates/shoaling.py` + `_gate.py` 软门
   - 路径：`packages/ethoinsight/ethoinsight/templates/epm.py`
   - TDD：先写 `tests/test_template_epm.py`

2. **e2e 验证 EPM 全链路**
   - 用 demodata（如有 EPM raw txt）跑 `make dev` 启服务
   - 上传 EPM 数据 → lead 识别为 PlusMaze 大类 → ask_clarification 选变体 → set_experiment_paradigm 写 ev19_template → task("code-executor") → templates/epm.py → handoff
   - 重点观察：data-analyst 解读时是否主动 read `references/by-experiment/epm.md` 拿"判读哲学"+"与其他实验的区分"段；report-writer 写报告时是否 read 拿"报告解读语言"段
   - 如果观察到不主动 read，说明 SKILL.md 工作场景对照表 + description 引导仍不够；回到 SKILL.md 调强度

### P1 — EPM 验证后批量做

3. **写其他 5 个 PRD MVP 范式**：`templates/open_field.py` / `templates/zero_maze.py` / `templates/light_dark_box.py` / `templates/tail_suspension.py` / `templates/forced_swim.py`
   - 全部跟 EPM 模式相同
   - LDB 在 EV19 没独立大类，默认走 `OpenFieldRectangle-Subdivided2x2` 兜底（待行为学同事确认）
   - TST 走 `NoTemplate`（不需要 zone）

4. **shoaling golden-case 校验**（v0.1 demo 防翻车）
   - case-001 已存在，跑一次端到端确认 agent 输出符合 expected-analysis.yaml

### P2 — 异步推进

5. **streaming PR 上游 review**（`bytedance/deer-flow` 提的 PR `fix-streaming-word-animation-remount`）
   - 上游一 merge，按 [docs/handoffs/2026-05-09-streaming-fade-in-fixed-handoff.md](2026-05-09-streaming-fade-in-fixed-handoff.md) "sync 注意事项"回退 noldus-insight 的 `d4171eed`，恢复 word fade-in
   - 不阻塞 v0.1，但要持续跟进

6. **抽象 `templates/_base.py` 基类**（地基 plan E4，所有范式 templates 写完后再做）

7. **`docs/sop/aliyun-deployment-guide.md`** working tree 里有这个 untracked 文件，不是本任务范围；下次会话先确认归属再决定 add 或 .gitignore

## 建议接手路径

### 新会话第一步

1. **读这份交接文档全文** — 别只看 TL;DR
2. **读 SKILL.md 最终形态**：`/home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md`（41 行）
3. **看 `8e6a6c1c` commit message**：完整记录了为什么修订 + grill-me 7 题决定
4. **抽查同事填的 EPM ground truth**：`packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md`
5. **再看一次 CLAUDE.md 第 10 条**：EV19 双层架构 + 现有 5 个 ethoinsight skill 列表

### 开干 EPM 模板

```bash
# 看 shoaling.py 模板示范
cat /home/wangqiuyang/noldus-insight/packages/ethoinsight/ethoinsight/templates/shoaling.py

# 看 _gate.py 软门
cat /home/wangqiuyang/noldus-insight/packages/ethoinsight/ethoinsight/templates/_gate.py

# 看同事填的 EPM 必算指标 + 数据质量风险
cat /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md

# TDD：先写测试
# packages/ethoinsight/tests/test_template_epm.py
```

## 风险与注意事项

### 容易混淆的点

1. **review 包不是长期同步源** — 是同事每批 PR 进来后工程一次性搬入 skill references。不写双向 sync 脚本。
2. **不要往 review 包里加东西** — 它是同事的工作目录，工程侧只读。同事下批 PR 时同样按搬运流程。
3. **不要为 by-experiment 写转换脚本去 🟡 emoji / 注释** — 同事填好的 markdown 直接搬就行。emoji 是有意的视觉锚点。
4. **MVP 范围外的 markdown 不要去填** — shoaling / AquariumTrack3D 等保持空模板形态。同事会异步补；agent read 到空模板时回 `default-template-fallback.md` 兜底。
5. **`code-executor` 看不到 `ethovision-paradigm-knowledge` skill**（`skills=["ethoinsight-analysis"]` 显式白名单）— 这是**有意设计**：code-executor 是工具调用者，领域指标应该长在 `templates/<范式>.py` 工具实现里（确定性、不依赖 LLM 现场判断），不是让 code-executor 临时 read markdown 去"知道要算什么"。**不要**给 code-executor 加这个 skill。

### 不建议的方向

1. ❌ **lead 抽段拷进 task() prompt 给 subagent** — 反模式（违反上下文隔离），已在 `8e6a6c1c` 删除
2. ❌ **SKILL.md 写"如果你是 lead 就 read X，如果你是 code-executor 就 read Y"** — 违反 LLM 鲁棒性 + 随机性原则，agent 自己根据任务上下文判断
3. ❌ **把 by-experiment 的段落结构（必算指标段名等）抄到 SKILL.md** — 违反渐进披露
4. ❌ **强迫同事补 shoaling / AquariumTrack3D markdown** — 同事 MVP 收敛是有意的，shoaling 领域知识在 `case-001-shoaling-baseline` golden-case
5. ❌ **接受 deerflow 上游同步的"安全文件"分类不审视** — CLAUDE.md 第 154 行血泪教训

## 下一位 Agent 的第一步建议

**最具体的起始动作**：开 `templates/epm.py` TDD 流程

```bash
cd /home/wangqiuyang/noldus-insight

# 1. 看 shoaling.py 当模板（177 行左右）
cat packages/ethoinsight/ethoinsight/templates/shoaling.py | head -80

# 2. 看 _gate.py 软门 helper（已封装）
cat packages/ethoinsight/ethoinsight/templates/_gate.py

# 3. 读同事 EPM 必算指标 ground truth
cat packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md

# 4. 看现有 ethoinsight 库结构
ls packages/ethoinsight/ethoinsight/

# 5. 创建 tests/test_template_epm.py（TDD 第一步）
```

EPM 是 PRD MVP 6 范式里**领域知识最完整**的一个（同事填得最仔细）。先把 EPM 跑通端到端，作为后续 5 范式的样板。

---

## 关联 commit

```
8e6a6c1c 按 grill-me 讨论修订 ethovision-paradigm-knowledge SKILL.md：场景对照表替代分阶段流程描述
474fcf04 MVP 范式知识搬入 skill 交接文档（v1，仅含搬入）
60cccd02 ethovision-paradigm-knowledge SKILL.md 加分析阶段渐进披露引导（含反模式，被 8e6a6c1c 修订）
96d6e2c4 搬入行为学同事 MVP 6 范式领域知识到 ethovision-paradigm-knowledge skill
```

## 关联文档

- [docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md](../superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md) — 5月8日地基设计
- [docs/handoffs/2026-05-08-ev19-template-skill-foundation-handoff.md](2026-05-08-ev19-template-skill-foundation-handoff.md) — 5月8日地基交接
- [docs/plans/2026-04-29-ev19-template-paradigm-design.md](../plans/2026-04-29-ev19-template-paradigm-design.md) — 产品级 EV19 双层设计
- [docs/review-packages/2026-04-29-ev19-templates/README.md](../review-packages/2026-04-29-ev19-templates/README.md) — 行为学同事 review 包入口
- [CLAUDE.md](../../CLAUDE.md) 第 10 条 — EV19 模板体系重构架构决策
- [packages/agent/backend/CLAUDE.md](../../packages/agent/backend/CLAUDE.md) 第 252 行起 — Skills System 机制
