# 2026-06-02 会话交接 — sync→Gateway→ethoinsight→seal1.5→lead-tool 五连修复全落 dev + 前端现代化 spec 待启动

> **本 handoff 用途**：交接一次超长会话。本会话从「PR #78 merge 冲突收尾」一路做到「ethoinsight 依赖根因修复 + seal 阶段1.5 review/落地 + lead-tool PR #80 review/补测/合并」，**五条修复全部已落 dev 并 push（origin/dev = `2f178e64`）**。最大的真 bug 是 ethoinsight 从 backend 依赖里被 sync 洗掉导致 dogfood 全工具 ModuleNotFoundError，已根治。另有一份「前端上游现代化」spec 已写好但**未 commit、未实施**，是下一个大 track。
>
> **当前状态**：本地 dev == origin/dev == `2f178e64`，working tree 仅剩一批**未跟踪的 docs/spec**（见 §5.1，强烈建议先 commit）。

---

## 0. 一句话现状

dev (`2f178e64`) 已合齐 5 件：PR #78(sync+Gateway) → CLAUDE.md同步规则 → **ethoinsight依赖修复(关键)** → seal阶段1.5 → PR #80(lead探查工具三层防护)。ethoinsight 修好后 dogfood 的 identify/inspect/prep_metric_plan 不再 ModuleNotFoundError、且 inspect 能直读分组字段。

---

## 1. 本会话完成并已落 dev 的工作（按 commit 顺序，全部已 push）

```
2f178e64  Merge PR #80 — lead 探查工具可靠性(三层防护+data_preview+三格式单测)
a5d62256  ├ test(inspect): 补 data_preview 三格式单测 (本会话 review 补的缺口)
72420d32  └ feat(harness): 三层防护 + inspect data_preview (另一 agent 实施)
ab6c3470  fix(seal-robustness): 阶段1.5 ParameterAuditFinding 归一化 + None→{} 兜底
1cfec288  fix(deps): 重新声明 ethoinsight 为 backend workspace 依赖 ★关键★
8b7ff786  docs(CLAUDE): 同步核心规则更新为「全量跟随上游 + surgical 守护」
64c56bfc  Merge PR #78 — DeerFlow 全量 sync + Gateway 迁移
```

### 1.1 ✅ PR #78 收尾（会话开头）
- 接上一会话的 6 个 merge 冲突，全部解完、commit、push，PR #78 已 merge 进 dev。
- 解完后 CI `backend-blocking-io` 一度变红 → 定位到是 ethoinsight 缺失（见 1.3），先用「惰性 import」止血让 CI 过，后根治。

### 1.2 ✅ CLAUDE.md 同步规则更新（`8b7ff786`）
- 把 sync 哲学从「选择性合入」改为「全量跟随上游 + surgical 守护」（2026-06-02 锁定）。纯文档。

### 1.3 ✅ ethoinsight 依赖修复（`1cfec288`）★本会话最重要★
- **根因（git 实证）**：`packages/agent/backend/pyproject.toml` 的 ethoinsight workspace 声明（`fbc2255e` 4-07 加的 3 行）被 **#78 全量跟随 sync(`fa3418ec`) 整文件覆盖洗掉** → `uv sync` 不装 ethoinsight → lead agent(gateway 内嵌 runtime) 调 identify/set_paradigm/prep_metric_plan/inspect 全 `ModuleNotFoundError`。`make dev` 只跑 `uv sync` 不跑 `make install`(那里才有补救的 `uv pip install -e`)，sync 后没人重装。**第 2 次同 vector 复发**。
- **修复**（让 4 环境都靠 uv sync 自动装）：pyproject 重放 fbc2255e 3 行 + uv.lock 重生成 + `deploy.sh` 加 `resolve_ethoinsight_symlink`(cp -rL + trap 还原，修 Docker COPY 跟不进出 context 的 symlink) + 删两个 Makefile 的手工 `uv pip install -e` + **CI `backend-blocking-io` 加 `import ethoinsight` 断言**(防复发) + SOP 加教训5。
- **验证**：后端 3567 passed/0、ethoinsight 库 522/0、import 链全通。
- memory 已存：`feedback_ethoinsight_must_be_backend_workspace_dep`、`feedback_harness_must_import_without_ethoinsight`。

### 1.4 ✅ seal 阶段1.5（`ab6c3470`）
- review 了 worktree `seal-robustness-phase1.5` 的实施（ParameterAuditFinding 加 `model_validator(mode=before)`：used_value=None→"" / observed_distribution 剔非数字；data_analyst step 2.8 删「整类一条」出口统一「每参数一条」）。
- **review 发现并补的缺口**：原 validator 漏了 `observed_distribution=None`（`elif od is not None` 把 None 排除了）→ deepseek 降级填 null 仍崩。补了 `None/缺key 区分处理` + `test_normalize_od_none`。
- **cherry-pick `834a0e3a` 到 dev**（只 3 文件，**不含那个 worktree 里删 ethoinsight 的脏 uv.lock**）+ amend 折入 None→{} 兜底。
- 验证：参数审计+seal 78 passed；全量 3574 passed（唯一 1 failed 是 `test_client_live::test_agent_uses_bash_tool` 撞 intent guardrail，**与本改动无关、是 live 测试**）。

### 1.5 ✅ PR #80 review + 补测（`a5d62256` / 你已 merge）
- review `feat/lead-tool-invocation-reliability` 对 spec `2026-06-02-lead-tool-invocation-reliability-design.md` 的 9 条 DoD：**8.5/9 满足**，架构正确无越界。
- **唯一缺口 DoD-8**：spec 要求「inspect data_preview 三格式单测」但分支零覆盖 → 本会话补 `tests/test_inspect_data_preview.py`（11 测，覆盖 `_build_data_preview_df` 的 xlsx/csv 路径 + `_build_data_preview_txt` 的 EV19 txt 路径，自带 UTF-16-LE fixture），30 passed，push，你已 merge 进 dev。

### 1.6 ✅ 清理
- 删除作废的 `seal-robustness-phase1.5` worktree + 分支（内容已 cherry-pick 进 dev，且是超集）。那个删 ethoinsight 的脏 uv.lock 随之丢弃。

---

## 2. 关键发现（下一 agent 必读，避免重蹈覆辙）

1. **ethoinsight 依赖会被 sync 反复洗掉**（vector：backend/pyproject.toml 是上游也有的文件，全量跟随整文件覆盖）。每次全量 sync 后**必查** `grep -q ethoinsight packages/agent/backend/pyproject.toml`。CI 现已加 import 断言兜底。详见 memory `feedback_ethoinsight_must_be_backend_workspace_dep`。
2. **worktree 从旧 base 分出 + 跑 `uv sync` 会重生成删 ethoinsight 的脏 uv.lock**。这是 seal worktree 那个"33 failed 假象"的真因（它在无 ethoinsight 的 venv 测的）。**worktree cherry-pick/merge 前务必看 commit 是否含 uv.lock、别带进去**。
3. **接 grill/PR review 不能信自述**：本会话两次靠 head-to-head 核实推翻了声称——seal 的"33 failed 无关"实为环境假象、PR #80 的"全量绿"漏了 DoD-8 单测。memory `feedback_grill_handoff_must_be_verified`。
4. **harness 是可发布通用框架，ethoinsight 是领域库**：ethoinsight import 必须惰性（放函数体），不能在 harness 模块顶层 hard-depend。但「声明为 backend 依赖让它装上」与「惰性 import」是正交的两层（import-time 安全 + call-time 可用），都要。
5. **deepseek 降级场景易把文字塞进数字字段 / 填 None**：seal 1.5 的 ParameterAuditFinding 归一化就是治这个。同 pattern 还有 DataQualityWarning。新加 schema 字段时考虑同构的 model_validator 兜底。

---

## 3. dogfood 现状（用户在手动跑 FST）

- ethoinsight 修好后，重启 `make dev` 的 gateway 进程已能 import ethoinsight。用户观察到：inspect_uploaded_file **直接读出实验组/对照组**（= parse_header 能解析 EV19 Treatment 字段了，修复生效强信号，见 memory `feedback_ev19_header_has_treatment_field`）。
- **下一关**：lead 能否走到「问 all-zones / no-zones」——这一步要 catalog resolve 真跑通才出现，是上次卡死的下一关。PR #80 的三层防护（强制先调 identify）也应在此生效。
- **已知小问题（记录，未修，project memory `project_2026-06-02_double_tool_call_observation`）**：lead 在同一 AIMessage 把 identify+inspect 各发两遍（只读幂等无害）。

---

## 4. PR #80 review 发现的两个非阻塞语义点（已合 dev，建议后续处理）

1. **after_model gate 在"二次上传不同范式"漏 enforce**：`uploaded_files` 累积（UploadsMiddleware before_agent 每轮重扫含历史文件）+ 一旦调过 identify 则 `_has_identify_in_history` 永真 → 后续新上传第二个范式文件不会被强制重新 identify。是漏 enforce 非误拦，v0.1 可接受。
   - 文件：`agents/middlewares/paradigm_identification_gate_middleware.py`
2. **`_reminder_counts` 按 run_id 永不清理** → 长驻进程轻微内存累积。可忽略，但加 LRU/上限更干净。

---

## 5. 未完成事项（按优先级）

### 5.1 🔴 高：commit 一批未跟踪的 docs/spec 到 dev（零风险，纯文档，先做）
当前 `git status` 有一批 `??` 文件**从未进 git**，包括本会话写的关键 spec：
- `docs/superpowers/specs/2026-06-02-frontend-upstream-modernization-spec.md`（前端现代化，21KB，§5.2 要用）
- `docs/superpowers/specs/2026-06-02-{deerflow-upstream-sync,lead-tool-invocation-reliability}-design.md`
- `docs/superpowers/specs/2026-06-01-{all-subagent-seal-robustness,data-analyst-pendulum-audit-fix,report-chart-404-fix,tst-paradigm-e2e}-design.md`
- 9 份 `docs/handoffs/2026-06/*.md`（含本 handoff）
- ⚠️ `.env.wecom` 是 `??` 但**不要 commit**（含密钥，应进 .gitignore）

建议：`git add docs/ && git commit -m "docs: 落档本批 spec + handoff"`，**排除 .env.wecom**。

### 5.2 🟡 中：前端上游现代化（spec 已就绪，是下一个大 track）
- spec：`docs/superpowers/specs/2026-06-02-frontend-upstream-modernization-spec.md`（未 commit，见 5.1）
- **方向锁定**：吃上游 infra（消息分组 getMessageGroups / run messages 分页 / 隐藏消息去重 / token-usage 抽象 / UI 原语），**不动 mode 语义**——前端 mode 保 auto/flywheel 两态 + workflow_mode（memory `feedback_frontend_mode_two_state_keep_workflow_mode`）。
- **必须基于最新 dev 做**（它改的前端文件正是 #78 改的）。worktree `frontend-modernization` 已存在（`64c56bfc`，但那是旧 base，要 rebase 到 dev 2f178e64 再做）。
- 最难是 `hooks.ts`（4-5h，mode→flag 映射保 workflow_mode）。A/B/C 分档见 spec §3。
- **能编译≠跑对**：mode 嫁接后必须 dogfood 看 DevTools Network 确认 flywheel 发 `workflow_mode:"manual"`。

### 5.3 🟡 中：seal 阶段 2（worktree 已开）
- worktree `seal-robustness-phase2`（`ab6c3470`，已基于最新 dev）。spec §2.3/§7.3-7.4。
- 做的是 code-executor 产 `_signal_distributions`(periodicity/velocity 逐帧分布) → data-analyst step 2.8 用真分布审计（不再降级跳过）。
- **第一步钉死**：分布在 `dispatcher.py` 还是 `compute_*.py` 算（spec §7.4 标注实施前必确认，2026-06-02 勘探发现 dispatcher 可能不在 catalog 真实流水线上）。

### 5.4 🟢 低：其它在跑的独立 worktree（本会话没碰）
`data-analyst-pendulum-audit-fix` / `report-chart-404-fix` / `tst-paradigm-e2e` / `sprint-6-7-memory-assumptions` — 各自独立 track，状态见各自 handoff。

---

## 6. 建议接手路径（第一步）

1. **先 commit §5.1 的 docs**（纯文档零风险，避免下次又像 ethoinsight 那样丢东西）。注意排除 `.env.wecom`。
2. 读本 handoff + memory（尤其 `feedback_ethoinsight_must_be_backend_workspace_dep` / `feedback_frontend_mode_two_state_keep_workflow_mode`）。
3. 若继续 **dogfood**：直接 `cd packages/agent && make dev`，跑 FST，看是否走到 all-zones/no-zones 反问（§3）。
4. 若做 **前端现代化**：基于最新 dev 开新 worktree，读 spec §3 分档，从 A 档 infra 打底开始（§5.2）。

---

## 7. 风险与注意事项

- ❌ **别 commit `.env.wecom`**（密钥）。
- ❌ **别把任何 worktree 的 uv.lock 带进 PR**（旧 base + uv sync 会删 ethoinsight）。merge/cherry-pick 前 `git show --name-only <commit> | grep uv.lock`。
- ❌ **前端现代化别把 mode 扩成上游四态、别删 workflow_mode**（产品特性，非技术债）。
- ⚠️ `test_client_live.py::test_agent_uses_bash_tool` 全量跑会 fail（live 测试 + 撞 intent guardrail），**不是回归**，正常 CI 不跑 live 测试。
- ⚠️ dev 上若再做全量 DeerFlow sync，**第一件事查 ethoinsight wiring 还在不在**（CI 已有断言兜底，但 SOP 也写了 grep 检查）。

---

## 8. 关键路径/命令速查

- 主仓库 = dev：`/home/wangqiuyang/noldus-insight`（HEAD = origin/dev = `2f178e64`）
- 后端 venv（有 ethoinsight）：`packages/agent/backend/.venv/bin/python`
- 跑后端测试：`cd packages/agent/backend && DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`
- 三方 git 路径：我们 `packages/agent/frontend/X` ↔ `git show deerflow/main:frontend/X` ↔ `git show origin/dev:packages/agent/frontend/X`
- ethoinsight 健康检查：`cd packages/agent/backend && uv run python -c "import ethoinsight; from ethoinsight.catalog.loader import load_common_catalog; load_common_catalog(); print('ok')"`

## milestone 建议
本会话让 **DeerFlow sync→Gateway 迁移→基础设施鲁棒性** 这条 track 到达 checkpoint：完成 standalone→Gateway 内嵌部署迁移（单向阀门）+ 根治 ethoinsight 依赖被 sync 反复洗掉的系统缺陷（CI 断言兜底）+ seal/lead-tool 两个鲁棒性 PR 合入。建议更新/创建 milestone 记录：Gateway 迁移完成 + ethoinsight 依赖 SSOT 化 + sync 防复发机制（前端 protected-files 仍待补，见前端现代化 spec §9 善后）。
