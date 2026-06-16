# Handoff — EPM dogfood 第三轮诊断 + 5 份修复 spec + Spec B 落地

> 日期：2026-06-08 ｜ 主仓库 dev HEAD：`0a1f8663`（含 PR #105 Spec2 + #106 Spec1）
> 这是给下一位 AI Agent 的交接，不是给用户的总结。

---

## 1. 当前任务目标

从「EPM 单文件 n=1 dogfood」第三轮发现的问题出发，诊断根因 → 写修复 spec → 逐个 review 实施分支并干净落地。本会话已：诊断出 3 个新问题 + 2 个遗留 → 写成 **5 份 spec** → 其中 **Spec B 已实施 + review 通过 + 落到干净分支待 PR**。

---

## 2. 已完成 ✅

1. ✅ **第三轮 EPM dogfood 诊断**（thread `7d4d9b8e`，gateway.log 实证）：
   - Spec1（schema partial）+ Spec2（guardrail n=1）**已合 dev 且生效**：n=1 不再因 schema 卡死、chart-maker 不再被拦、指标文件 644。
   - **新发现 3 问题 + 2 遗留**（见 §4）。
2. ✅ **5 份 spec 全部写好**（都基于现场核实，非臆测）：
   | Spec | 文件 | 状态 |
   |------|------|------|
   | A 路由 | `docs/superpowers/specs/2026-06-08-data-insight-routing-spec.md` | 待实施 |
   | B 术语 | `docs/superpowers/specs/2026-06-08-output-constitution-term-alignment-spec.md` | **✅ 已实施+review+落干净分支** |
   | C seal | `docs/superpowers/specs/2026-06-08-seal-harness-hard-guarantee-spec.md` | 待实施 |
   | D 路径 | `docs/superpowers/specs/2026-06-08-validate-catalog-virtual-path-fix-spec.md` | 待实施 |
   | E 图展示 | `docs/superpowers/specs/2026-06-08-chart-display-consistency-spec.md` | 待实施（两阶段：先调查） |
3. ✅ **Spec B review + 干净落地**（本会话重点，见 §3）。

---

## 3. Spec B 当前状态（关键，PR 待推送）

**改动质量已 review 通过、测试全绿、落到干净分支。等待用户手动推送 + 建 PR。**

- **干净分支**：`specB-constitution-on-dev`，commit **`990edf2f`**，worktree 在 `.claude/worktrees/specB-clean-on-dev/`
- **基于最新 dev `0a1f8663`**，`dev..HEAD` 恰好 **1 commit / 3 文件 / +122 -2**，**无误删除**。
- 改动内容（3 文件）：
  - `output-constitution.md`：line 18 "绝对焦虑判读"→"绝对焦虑程度判读"，移除"焦虑样行为/提示焦虑"（松绑定性术语），保留"高/低焦虑/焦虑水平"（绝对程度）；line 17 阈值禁令、line 19 正常禁令**一字未动**；补"定性术语 vs 绝对阈值/程度"判别原则+口诀+正反例。
  - `knowledge_assistant.py`：补"判读语言规范"段（纵深防御）。
  - `test_output_constitution_terms.py`：新增，5 测试全绿。
- **B2 复核已做**：paradigm md（by-experiment/*.md）无偷藏绝对阈值，同事无需改阈值。
- **全量回归**：`5 failed, 3802 passed`。5 个失败 = 前两轮 Spec1/Spec2 review 同款**预存在基线**（deferred_tool_registry_promotion×2 + inspect_gate/paradigm_identification 的 test_async + chart_maker_config），**我的 commit 没碰这些文件**，与本改动无关。

### ⚠️ 用户的下一步（本会话末用户说"我自己手动推送"）
1. `cd .claude/worktrees/specB-clean-on-dev && git push -u origin specB-constitution-on-dev`
2. 建 PR `specB-constitution-on-dev` → `dev`。
3. **改的是 SSOT（判读语言宪法），PR 描述里 @ 行为学同事 review**（Spec B §6 确认清单：松绑范围 + 保留范围 + paradigm md 阈值已确认干净）。

---

## 4. 未完成事项（按优先级）

### 立即：清理一个危险的坏 worktree
- 🔴 **旧坏 worktree 必须删**：`.claude/worktrees/worktree-output-constitution-term-alignment`（分支 `worktree-worktree-output-constitution-term-alignment`，基于**旧 main `529e5c2f`**，落后 dev 38 commit）。
  - 它是 Spec B 第一次实施的产物，但**base 错了**（从 main 切、不是 dev）。改动已被我重做到 `specB-clean-on-dev`，**旧的没用了**。
  - **绝不能 PR 它**：`dev..它` = "删 10472 行"（会回滚 Spec1/Spec2 等已合成果）。
  - 清理：`git worktree remove .claude/worktrees/worktree-output-constitution-term-alignment --force && git branch -D worktree-worktree-output-constitution-term-alignment`
  - **教训记 memory**：EnterWorktree / `git worktree add` 默认从 `origin/HEAD`（=`origin/main`）切，但本项目开发分支是 **dev**。**建 worktree 必须显式 `git worktree add <path> -b <branch> dev`**，否则 base 落后 main 38 commit，PR 卷入大量误删除。

### 高优先级：实施剩余 4 份 spec（A/C/D/E）
- 这 4 份**正交、可并行、各自独立 PR**。实施时**必须从最新 dev 切 worktree**（见上教训）。
- **Spec A（路由）**：lead 把"初次数据判读/洞察"路由给 data-analyst（含 n=1 走 partial）；knowledge-assistant 场景 A 收窄为"只解释已有结论，不从零判读"。核实过 data-analyst 自带 paradigm-knowledge skill，不丢知识。
- **Spec C（seal harness 硬保障）**：⚠️ 改 harness 核心，**红线多**（别加 tool_choice / 别动 max_turns / 别改 resume_prompt 措辞 / 别用 structured output / 兜底不抛异常）。核心 = seal-resume 失败后用已有产出**确定性 auto-seal**（仅 report-writer/chart-maker，code-executor/data-analyst 永不 auto-seal）。
- **Spec D（validate_catalog 路径）**：根因完全确证——plan 内 `/mnt` 虚拟路径字符串不经 bash 替换→OSError→误报 result_file_unreadable。修=新增 `resolve_sandbox_path` helper（用 DEERFLOW_PATH_* env）。**⚠️ env key 拼法必须先实测再写死**。
- **Spec E（图展示）**：**两阶段**，第一阶段必须用前端运行时调试（chrome-devtools/playwright）抓那 8 张图的真实请求 URL 钉死根因，**别跳过调查直接改**。P2 不阻塞。

### 中低优先级
- 重新 dogfood 验证 A/B/C 修复后端到端（尤其"帮我进行数据洞察"是否走 data-analyst、report-writer 是否不再 seal 黑洞）。

---

## 5. 关键发现 / 决策记录

1. **第三轮 dogfood 两个问题是同一条结构缺陷链**（memory `project_2026-06-08_epm_dogfood_routing_and_constitution_leak.md` 全链）：
   - n=1 跳过 data-analyst（Spec2 行为）→ 用户要洞察 → lead 误路由 knowledge-assistant → 它**不受输出宪法约束**产 14 处违禁词 → 污染 report-writer → report-writer 陷入"改写违禁词"地狱 → seal 黑洞。
2. **report-writer 第 1 次 seal 失败是 (a) 叙述黑洞**（gateway.log trace=9876bd7a：captured #1-5 远小于 max_turns，无 reached max_turns，seal-resume did not recover）。**report.md 第 1 次已写好（6151 字节），只差 seal 封条** → 这是 Spec C "用已有产出兜底"的依据。
3. **seal 黑洞已第三次复现**（step2.8 矛盾 / schema 拒 partial / 违宪改写），底层机制"reasoning 当完成、不调 seal 就退出"从没根治 → Spec C 治本。
4. **用户的修复哲学决策**（贯穿 5 spec）：减规则/对齐契约/收窄职责，**不加 if-else 补丁**。
5. **术语松绑边界（用户拍板）**：只松定性行为术语（焦虑样行为/趋近-回避），**保留**绝对阈值（正常范围 X-Y%/典型值）+ 绝对程度（高/低焦虑）。守 CLAUDE.md §9 铁律。
6. **data-analyst 知识来源**：通过 skill 渐进披露（read_file epm.md），**不需要 subagent 调 subagent**——用户原担心"subagent 不能调 subagent 拿不到 Noldus 知识"现状已解决。

---

## 6. 风险与注意事项

- 🔴 **建 worktree 必须显式基于 dev**（`git worktree add <path> -b <name> dev`）。EnterWorktree 和裸 `git worktree add` 都默认 origin/main，会落后 38 commit → PR 卷入误删除。**本会话踩了这个坑两次**（坏 worktree + EnterWorktree 切出来的也是 main base）。
- 🟡 **全量测试 5 个预存在失败**是基线，非新增：`test_chart_maker_config_basic_fields`（dev 本身红）+ deferred_tool_registry_promotion×2 + inspect_gate/paradigm_identification 的 test_async（isolation 污染）。看到这 5 个别归因自己。
- 🟡 **worktree 跑全量需 config.yaml symlink**：`ln -sf /home/wangqiuyang/noldus-insight/packages/agent/config.yaml <worktree>/packages/agent/config.yaml`，否则 ~5 个 config 依赖测试假红。
- 🟡 **接 review/handoff 必须现场核实**（memory `feedback_grill_handoff_must_be_verified`）：本会话纠正了 dogfood 报告 §3.2（root cause 误判）、§3.3（修复位置误判）、Spec1 对 600 的判断（只是第一根因）。别信文字，自己 grep + 解 log。

---

## 7. 下一位 Agent 的第一步建议

1. **读 memory**：`project_2026-06-08_epm_dogfood_routing_and_constitution_leak.md`（全链根因 + 5 spec 索引）+ `feedback_dataanalyst_reportwriter_handoff_status_missing_partial.md`。
2. **清理坏 worktree**（§4 立即项）——这是最该先做的，防止误 PR。
3. **确认 Spec B 状态**：`git -C .claude/worktrees/specB-clean-on-dev log --oneline dev..HEAD` 应只有 `990edf2f`。若用户已推送+合 PR，则 Spec B 收工。
4. **挑一份剩余 spec 实施**（A/C/D 都 ready；C 风险最高需谨慎；D 根因最清晰可先做；E 要先调查）。**从最新 dev 切 worktree**。
5. 每份实施后：red 测试→改→全量回归（认那 5 个基线失败）→ 干净分支（`dev..HEAD` 只含本 spec 改动，无误删除）→ 交用户 PR。

---

## 8. 关键文件清单

| 文件 | 说明 |
|------|------|
| `docs/handoffs/2026-06/2026-06-08-epm-dogfood-findings.md` | 第三轮 dogfood 报告（注意 §3.2/§3.3 根因被本会话纠正） |
| `docs/superpowers/specs/2026-06-08-*.md` | 5 份 spec（A 路由/B 术语/C seal/D 路径/E 图展示） |
| `.claude/worktrees/specB-clean-on-dev/` | Spec B 干净分支 worktree（待推送 PR） |
| `~/.claude/.../memory/project_2026-06-08_epm_dogfood_routing_and_constitution_leak.md` | 全链根因 + 5 spec 索引 |
| `packages/.../subagents/builtins/data_analyst.py` | 自带 paradigm-knowledge skill（Spec A 依据） |
| `packages/.../subagents/executor.py` | seal-resume（735）/ _validate_handoff_emitted（170）（Spec C 改这里） |
| `packages/ethoinsight/.../validate_catalog.py:362` | 路径 bug 点（Spec D） |
| `packages/.../agents/thread_state.py:22` | merge_artifacts 累积 reducer（Spec E 线索） |

---

## milestone 建议

本会话让「EPM dogfood 鲁棒性修复」track 到达 checkpoint：Spec1/Spec2 已合 dev 生效，再诊断出 5 个问题并全部立 spec、Spec B 落地。建议下一 agent 在 5 份 spec 全部合入后，更新/创建 milestone「subagent handoff 鲁棒性 + 判读语言对齐」，摘要：seal 黑洞三次复现的根治路径（prompt→schema→harness 三层）+ 输出宪法 SSOT 对齐 + n=1 路由收窄。
