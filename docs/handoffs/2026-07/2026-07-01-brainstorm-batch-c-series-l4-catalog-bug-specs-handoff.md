# Handoff：连续 brainstorm 一批 spec/plan（C2/C2b/C3 + L4-1 + catalog bug + L4-2/3/4）全 push dev（2026-07-01）

> 本会话是纯 brainstorm + 写 spec/plan 会话（承接前一 handoff `2026-06-30-d0-d2-d3-specs-and-two-bug-diagnosis-handoff.md`）。开局 read-handoff 校准 → 逐个 brainstorm 待办 → 写 spec/plan → push。**实施全交别的 agent（用户自派）。** 本会话没写任何生产代码，只写设计文档。

---

## 〇、一句话现状

- **git：本地 = origin/dev，无分叉。** dev HEAD = `d1e80c91`（本 handoff commit 后会前进）。今天 2026-07-01。
- 本会话产出全已 commit+push（见 §一）。工作区只剩历史常驻 `M`（CLAUDE.md / e2e scripts / 两 plan）+ 历史 untracked（`docs/reports/`、`reports/`、`scripts/repro/`、`triage-28fc60ee` 在 ~ 下非仓库内）——**勿碰**。
- **D0 findings 现已全部转成 spec**（L4-1 结构门 + L4-2/3/4 脱敏），findings 从「诊断」变「全部有 spec 可派」。
- 被打断在「写完 handoff、新 session 继续 brainstorm」处——用户要在**新 agent** 继续 brainstorm 别的方向。

---

## 一、本会话已完成（✅）—— 全是设计文档，全 push dev

| 子项 | 文件 | commit | 状态 |
|---|---|---|---|
| C3 依赖契约对 C1 复核更正 | `2026-06-30-c3-metrics-selection-followup-design.md` | `a217a829` | ✅ 更正（指标值嵌 `per_subject[].values` 非平铺） |
| C2 画廊布局重做 spec | `2026-07-01-c2-gallery-layout-redesign-design.md` | `2dd836dc` | ✅ spec（实施前置 D1✅+D2） |
| C2b 后端注册表 spec | `2026-07-01-c2b-artifact-registry-backend-design.md` | `2dd836dc` | ✅ **已被实施合入 #258** |
| C2b + D2 实施 plan | `2026-07-01-{c2b-artifact-registry-backend,d2-design-system-kit}-impl.md` | `d93e8bc8` | ✅ plan |
| L5-1 复诊更正 + e2e 护栏 | findings.md + `noldus-insight-e2e/SKILL.md` | (前会话 push) | ✅ L5-1 作废（非 bug，采图用错 prod 形态） |
| **L4-1 剂量幻觉 provenance 门 spec** | `2026-07-01-l4-1-group-semantics-provenance-gate-design.md` | `67e6c36a`+`34baf48f` | ✅ spec，可即派 |
| **catalog in_zone* bug 修复 spec** | `2026-07-01-catalog-concept-match-bare-in-zone-pattern-fix-design.md` | `f4948b77` | ✅ spec，可即派（P0 prod bug） |
| **L4-2/3/4 信息脱敏 spec** | `2026-07-01-l4-2-3-4-researcher-view-desensitization-design.md` | `d1e80c91` | ✅ spec，可即派 |

### 各 spec 一句话核心
- **C3 更正**：C1(#255) 实际产出 `metrics_table.json` 把指标值嵌 `per_subject[].values` 子对象（非平铺行键）；C3 选区层挂已存在的 `metrics-table-card.tsx`。
- **C2**：在 C1 卡片之上建概览优先布局（不推倒重写卡片功能），6 dogfood 行为不变式逐条回归；前置 D1✅+D2。
- **C2b**：artifacts.py 四端点散落 kind/ext 逻辑收敛成 `ArtifactRegistry` SSOT + 薄门面（**已实施合入 #258**）。
- **L4-1**（结构门主治+宪法辅）：`set_experiment_paradigm` 拆 `group_structure`（照跑）+ `group_semantics`（带 source），未确认语义确定性降级中性名「实验组N」，report 写不出剂量-反应。软门不阻断分析。**治越权非消歧**。复用 #254 provenance 模式。
- **catalog bug**（P0 prod）：`resolve.py:643` `pat[len("in_zone_"):]` 对裸 `in_zone*`(长7) 越界成空串→keyword 短路→open/closed 匹配不上→box/bar 图误 skip。修法=裸 `in_zone*` 直接 True。**治 zero_maze/LDB/OFT 三范式共 7 处**。
- **L4-2/3/4**：删 `prompt.py:800` config id 展示指令（零消费）+ quality banner code→人读标签（含未知 fallback）+ 移除对话区 token 指示器。

---

## 二、关键发现/决策（下个 agent 直接用）

### 1. dev 期间被别的 agent 推进的实施（本会话校准出）
- **C1（#255）、C2b（#258）、D1（#257）、HITL Bug②（#254）、code-executor 假失败 Bug①（#256）** 全已实施合入 dev。
- **Bug① 真根因坐实**（#256）：不是 TOCTOU、不是 guard 漏拦，是 **ls 工具三态歧义**（list_dir 对文件/不存在/空目录都返 `[]`，code-executor 误读「ls 已存在文件得空」为「文件缺失」）。前 handoff 的「待坐实」已闭环。

### 2. 完整依赖链 + 当前可派状态
```
C1✅ → C2b✅(#258) ┐
D1✅ → D2(plan就绪,可派) ┴→ C2(spec就绪,等D2) → C3(spec就绪,等C2)
```
- **现在无阻塞、可并行派的实施**：**D2**（前端 kit，plan 就绪）、**L4-1**（后端结构门）、**catalog bug**（ethoinsight 库，P0）、**L4-2/3/4**（1后端+2前端脱敏）。四个不同子系统、互不撞车。
- **C2** 等 D2 落地后可派（spec 就绪，届时写 plan）；**C3** 等 C2。

### 3. triage 核实方法论（本会话实践，留档）
- 用户给了 `/home/wangqiuyang/triage-28fc60ee/ROOT_CAUSE.md`（别的 agent 对 prod thread 28fc60ee zero_maze 出图 bug 的调研）。**本会话没直接信**——按 `feedback_grill_handoff_must_be_verified` 独立复现坐实根因（跑最小复现 + 对当前源码逐条核），**且补了关键证据**：bug 跨 zero_maze/LDB/OFT 三范式（ROOT_CAUSE 主要讲 EZM），这直接排除了它列的「路 A（改 yaml）」。→ catalog bug spec 走路 B。

### 4. spec「补硬」代替另写 plan 的方法论（用户确立）
- 用户问「spec 直接派没问题，还要 plan 吗」。结论：**看 spec 类型**——大多数（CRUD/前端/加字段）spec 本身够；只有高假绿风险的（字节对账重构如 C2b）plan 有真实价值。
- **做法**：派 spec 前看「最易糊弄的那步（通常是测试要非 vacuous）」有没有写到 agent 照做就假绿不了。不够硬就**就地在 spec 里补硬**（非另写 plan）。本会话对 L4-1 和 catalog bug 和 L4-2/3/4 都这么做了：防 vacuous 探针要求「实跑观察红再恢复」（非声称）、端到端断言看真产物/结构不变式（非 substring 巧合命中）。

---

## 三、未完成事项（按优先级）

| # | 事项 | 状态/依赖 |
|---|---|---|
| 1 | **继续 brainstorm 别的方向** | 用户要在**新 agent** 继续。候选见 §四 |
| 2 | 派实施：D2 / L4-1 / catalog bug / L4-2-3-4 | spec/plan 就绪，**用户自派**（不是本会话职责） |
| 3 | C2 写实施 plan | 等 D2 落地后（届时接口已实） |
| 4 | C3 实施 | 等 C2 |

---

## 四、下一位 Agent 的第一步（继续 brainstorm）

1. `git fetch origin dev` + 核分叉。读本 handoff。
2. **brainstorm 候选方向**（本会话末尾给用户的清单，尚未选）：
   - **Experiment 跨范式对比 Phase 1**（愿景方向）：thread archive 归一为实验 + 快照提取 + 并列展示（不含 synthesizer 强判读）。Phase 0（路径承重墙 #229）已落地、骨架不阻塞；synthesizer 判读挂同事方法论 #226。spec = `2026-06-26-experiment-cross-paradigm-comparison-init-design.md`（**init 立项，非实施 spec**，要拆 Phase 1 实施 spec）。
   - **noldus-kb MCP 搜索工具**：346 篇论文已 embedding，建 grep+SQL+vector 搜索工具。独立轨道、与范式正交。handoff = `2026-06-10-noldus-kb-redesign-handoff.md`。
   - 或用户的新想法。
3. brainstorm 走 `superpowers:brainstorming`（一次一问 + 定稿用户审 + 写 spec push）。写完 spec 做「补硬检查」（§二.4）而非默认另写 plan。

---

## 五、关键陷阱 / 注意事项

1. **`git add` 永远精确路径**：工作区常驻别人的 `M` + 历史 untracked，**绝不 -A/.**。本会话所有 commit 都单文件精确 add。
2. **每次 git 操作前 fetch 核分叉**：dev 多并发推进（本会话期间 C1/C2b/D1/#254/#256 被并发合入）。
3. **改 harness 核心（L4-1 改 experiment_context / prompt）后裸导入两入口**：`import app.gateway` + `make_lead_agent`（spec 已写进测试步骤）。
4. **triage / handoff 先核实**：别信二手叙述，独立复现 + 对当前源码核（本会话对 catalog bug 就是这么抓出「跨三范式」的）。
5. **sync 命脉**：前端改动只通过 token(globals.css) + workspace/ 结构；`prompt.py`/`experiment_context.py` 受保护但改它们（纯加/纯删指令、Noldus 定制文件）是前向 feature 非违 sync。
6. **triage 目录在 ~ 下非仓库内**（`/home/wangqiuyang/triage-28fc60ee/`），别误提进 git。

---

## 六、milestone 建议

本会话让 D0 findings 全部转成 spec（L4-1/L4-2/3/4）、C 系列呈现层设计链补齐（C2/C2b/C3 spec + C2b/D2 plan）。可回流：
- D0 audit track：findings 已全部有 spec 可派（L4-1 结构门 / L4-2-3-4 脱敏 / L5-1 作废）。
- catalog bug 修复进「chart/catalog 可靠性」相关 milestone 或 dogfood-fixes 批次（prod thread 28fc60ee 驱动）。
- 前端设计语言 track：D1✅实施 / D2 plan就绪 / C2 spec / C2b✅实施 / C3 spec。
