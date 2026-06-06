# 2026-06-03 会话交接 — OFT dogfood 全链路失败：grill 定方案 → 三 PR 实施 → review 发现 3 缺陷 → 全部修复并核实（待 rebase + 合并态终验）

> **本 handoff 用途**：交接 OFT（旷场）dogfood 全链路失败的修复 track。**已闭环**：① 现场核实四层根因（推翻问题文档/前任 handoff 多处假设）；② grill 15+ 轮定出三 PR 方案，写成可分发 spec；③ 三 PR 由独立 agent 实施并推送；④ review 现场核实发现 3 个缺陷（PR-1 必爆 bug / PR-3 绕过漏洞 / PR-2 编号撞），已由 agent 修复，**逐项实跑核实通过**。
> **🟡 唯一剩余**：三 PR 待 **rebase 到最新 dev + 合并态端到端 OFT dogfood 终验** 后才能合 main。**代码层已全绿，缺的是"三改齐备协同"那一道验证**。
>
> **当前状态**：三 PR 分支已推 origin，各带 fix commit。主仓库 HEAD 停在 `fd2830ff`（未动），origin/dev 已到 `6cc13f55`。

---

## 0. 一句话现状

OFT dogfood 因「catalog 要 `in_zone_center_*` 命名列、真实 EV19 导出只有单匿名 `in_zone` 列 → prep_metric_plan 硬失败无 plan → lead 越界绕 plan 硬派 code-executor → code-executor 用错脚本名 + 越权（cp 进 venv/伪造模块）」四层塌方。**已拆三 PR 修复并各自核实通过**：PR-1 地基（匿名区反问状态机 + center_zone 注入 + 缺列硬失败）、PR-2 编排硬约束（plan-gate + lead prompt）、PR-3 越权防御（guardrail 扩 chart-maker + 路径校验）。**剩 rebase + 合并态终验。**

---

## 1. 任务目标（已达成代码层，待终验）

让 OFT dogfood 走通：上传真实旷场数据 → 识别单匿名区 → lead 反问用户「这个区是不是中心区」→ 用户确认 → 算出 `center_time_ratio≈0.10` → 走完 code/data/chart/report。同时堵住 lead 越界绕 plan、code-executor 越权写 venv 两个安全洞。

---

## 2. 关键发现（grill + review 现场核实，下一 agent 必读，别再走回头路）

### 2.1 推翻的假设（核实证伪，别采信旧文档）
1. **`in_zone=1` = 中心区**（前任 handoff 说 in_zone=0 是中心，**记反了**）。几何实测：in_zone=1 时 distance_to_wall=14.3cm、半径=10、占时 10.07%（焦虑样中心回避）。**别采信前任 handoff 的语义判定，要自验几何。**
2. **问题文档「方向 A：catalog 改 in_zone* + 脚本默认裸 in_zone=center」被行为学专家 SSOT 否决**。专家原话（`docs/review-packages/2026-05-12-feedback.md` §OFT+Q2）：「不知道哪一列代表什么分析区，**要问，不要猜。虽然这次猜中了，但是不要猜。**」`metrics/oft.py:_find_center_zone_column` 刻意拒裸 in_zone 当 center 是**对的，不许放宽**。
3. **专家本会话追加裁定**：EV19 导出**极少**出现单匿名区；真遇到 → 反问用户确认该区是什么 → 用户明确则算、**用户不知道则停止让用户弄明白（或回 EthoVision 重标记）再来**，绝不猜。
4. **🔴 问题文档「guardrail 守住底线、venv 干净」核实证伪**：实跑 `ScriptInvocationOnlyProvider`，`cp 进 .venv`/`mkdir 进 site-packages`/`mkdir 伪造模块` 修复前全 ALLOW（file-ops 白名单只看首词不校验路径）；local sandbox `execute_command` 用 `subprocess.run(shell=True)` 直落宿主机、无围栏、PATH 注入真实 venv。**venv 当时干净是侥幸不是防御，这是真实沙箱逃逸面。** 且 **chart-maker 也有 bash 且原本完全不受白名单约束**。

### 2.2 同一地基缺陷波及 3 个范式（不止 OFT）
真实 demo data 实测：**OFT/LDB(明暗箱)/zero_maze(O迷宫) 导出都是单匿名 `in_zone`**（catalog 分别要 `in_zone_center_*`/`in_zone_light*`/`in_zone_open*`，匹配 0 列）。EPM 命名良好（`in_zone_open_arms_center`）不受影响。**PR-1 的匿名区检测是范式无关设计**（裸 in_zone 存在 + 命名 zone pattern 落空 → zone_unnamed），已覆盖三范式。

### 2.3 OFT 匿名区状态机（权威定义，已实施）
```
prep_metric_plan 检测到裸 in_zone 存在但命名 zone 列缺失 → error_code=zone_unnamed
  → lead 读 hint → ask_clarification「这个未命名分析区代表哪里？是中心区吗？」(问角色不问值)
     ├ "是中心区" → lead 写 parameter_overrides={"center_zone":"in_zone"} → 重调 prep → plan 生成 → 算
     └ "不是/不知道" → 停止，告知数据缺明确中心区，请确认/重标记区域后再来（不猜、不用无关区代算）
```
注：EthoVision `分析区中`(in_zone) 恒 `1=在区内`，值语义不歧义，**只声明哪列是中心区（center_zone）**，不声明哪个值。

---

## 3. 三 PR 现状（全部推送，各带 review-fix commit，已逐项核实）

| PR | 分支 | HEAD | 状态 |
|---|---|---|---|
| **PR-1 地基** | `pr-1-zone-semantics-fix` | `bcddc553` | ✅ 实施 + fix 核实通过 |
| **PR-2 编排** | `pr-2-plan-gate-fix` | `05bbf719` | ✅ 实施 + fix 核实通过 |
| **PR-3 越权** | `pr-3-sandbox-escape-fix` | `3517f186` | ✅ 实施 + fix 核实通过 |

**三分支均基于 `fd2830ff`（旧 dev）。origin/dev 现已到 `6cc13f55`（PR #84 data-analyst thinking 合入）。**

### PR-1 改动（`metrics/oft.py` + `catalog/oft.yaml` + `catalog/resolve.py` + `prep_metric_plan_tool.py`）
- `compute_center_time`/`compute_center_distance` 补 `center_zone` 参数透传（消 TypeError）。
- `resolve.py:_detect_anonymous_zone` 区分 zone_unnamed（匿名区，有 override 则放行）vs columns_missing（真缺列，硬失败友好文案）。
- **review fix `bcddc553`**：`center_zone` 从 `paradigm_parameters` **下沉到 5 个 center metric 各自 `metric.parameters`**，删掉 `_compute_parameters_in_use` 无差别注入的 2c 段。**核实**：5 center metric 正确收 override、3 optional 零污染、ethoinsight 全量 **570 passed / 0 failed**。

### PR-2 改动（`guardrails/path_sequence_provider.py` + `lead_agent/prompt.py`）
- `_check_plan_precondition`：派 code-executor 校验 `plan_metrics.json` 非空、派 chart-maker/report-writer 校验 `handoff_code_executor.json` 非空，缺失 → deny 含明确指令。
- prompt 加规则：prep_metric_plan error 处理（zone_unnamed/columns_missing → ask_clarification）+ ethoinsight 脚本是唯一计算途径。
- **review fix `05bbf719`**：编号 6→7(原 seal)→8(prep 失败)→9(脚本唯一)，消除双 7。**核实**：无重复编号；plan gate 实跑正确（无 plan→code-executor DENY、无 handoff→chart/report DENY、有产物→放行、QA 路径不受影响）；14 新测试 passed（含 workspace=None fail-open 覆盖）。

### PR-3 改动（`guardrails/script_invocation_only_provider.py` + `subagents/builtins/code_executor.py`）
- bash 白名单 gate 扩到 chart-maker；file-ops 拆 read-only(放行) vs write(路径校验)；code-executor prompt 正面引导禁猜名→seal failed。
- **review fix `3517f186`**：`_is_path_safe` 用 `os.path.normpath` 消 `..` 后判边界，相对路径先拼 `/mnt/user-data/workspace`。**核实**：3 个相对路径逃逸 payload（`cp ../../../usr`、`mkdir ../../etc/cron.d`、`cp /mnt/user-data/../../etc`）全 DENY、venv 回归 DENY、5 个正常命令(含相对路径)不误伤。

---

## 4. 未完成事项（按优先级）

### 4.0 🟡 最高：三 PR rebase + 合并态端到端终验（唯一剩余）
1. **rebase 三分支到 origin/dev(`6cc13f55`)**。PR #84 是 data-analyst thinking，与三 PR 文件零重叠，rebase 应无冲突；rebase 后各自重跑全量。
2. **合并态终验（关键，不可省）**：把三改合到一处（临时分支或 dev）→ 重启 dev → 跑一次完整 OFT dogfood：上传 `/home/wangqiuyang/DemoData/newdemodata/旷场_小鼠_三点/` 两份 xlsx → lead 反问「是不是中心区」→ 答「是」→ 走完 code/data/chart/report，确认 `center_time_ratio≈0.10` 落盘、4 handoff 齐全、无 TypeError/无越权。
   - **为什么必须合并态验**：PR-1 原始 bug（override+optional 共存 TypeError）恰恰是隔离测试漏掉的那类（memory `feedback_pr_merge_must_run_full_suite_on_shared_logic`，PR #66 血泪）。隔离态各自绿 ≠ 协同正确。
3. 终验通过 → 三 PR 合 main（建议合并顺序 PR-3 安全先 → PR-1 → PR-2，但无强依赖）。

### 4.1 🟢 单列 issue（不在本 track）
- **sandbox 层路径围栏**（spec §4.4）：`local_sandbox.py:execute_command` 加 cwd/路径白名单对所有 bash 生效，防「未来给新 subagent 加 bash 忘上白名单」。属受保护核心 + 上游 deerflow 演进区，单开 issue，不在 bug-fix PR 顺手改核心。

### 4.2 🟢 善后
- 本地残留 3 个 agent 建的 worktree（`.claude/worktrees/pr-{1,2,3}-*`）—— 合并后清理（`git worktree remove`）。
- spec `docs/superpowers/specs/2026-06-03-oft-pipeline-failure-fix-design.md` 在三 PR 分支里已随 `dc933029` 提交；主仓库 `fd2830ff` 显示为 untracked（正常，因主仓库未含那两个 docs commit）。
- 失败 thread `752980d6` workspace 留作回归回放素材，无需清理（venv 已核实干净）。

---

## 5. 关键文件/命令速查

- **spec（权威实施蓝图，三 PR 可分发版）**：`docs/superpowers/specs/2026-06-03-oft-pipeline-failure-fix-design.md`
- **问题文档（原始证据）**：`docs/problems/2026-06-03-oft-dogfood-pipeline-failure-problem-statement.md`
- **SSOT（zone 语义铁律）**：`docs/review-packages/2026-05-12-feedback.md` §OFT + Q2
- **真实 demo data**：`/home/wangqiuyang/DemoData/newdemodata/{旷场_小鼠_三点,明暗箱,O迷宫,高架十字迷宫_小鼠_三点,强迫游泳_大鼠}`
- ethoinsight 全量：`cd packages/ethoinsight && .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`（PR-1 基线 **570 passed**）
- 后端全量：`cd packages/agent/backend && DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/ -q -p no:cacheprovider -o addopts=""`（基线 3612 + 2 pre-existing fail 非回归）
- 重启 dev：`cd packages/agent && make stop && make dev`
- dogfood handoff：`packages/agent/backend/.deer-flow/users/<uid>/threads/<tid>/user-data/workspace/handoff_*.json`

---

## 6. 风险与注意事项

- ⚠️ **别再把根因当「catalog 改 in_zone* + 脚本默认 center」**——专家 SSOT 否决（§2.1.2）。metrics 层拒裸 in_zone 是对的不许放宽。
- ⚠️ **别采信前任 handoff「in_zone=0 是中心」**——记反了，几何证 in_zone=1=中心（§2.1.1）。接 handoff 要自验（memory `feedback_grill_handoff_must_be_verified`）。
- ⚠️ **合并态终验不可省**——隔离态全绿不代表协同正确（PR-1 bug 就是隔离漏掉的）。
- ⚠️ **PR-2/PR-3 改受保护文件**（prompt.py/code_executor.py/guardrails），rebase 时 surgical 保留 Noldus 定制；**改后重启 dev**（system_prompt 在 agent 创建时构建）。
- ⚠️ **deepseek 正面措辞**（CLAUDE.md §6）——prompt 改动一律正面，硬约束交 harness gate。
- ❌ **别给 sandbox execute_command 在本 track 加围栏**——单列 issue（受保护核心 + 上游演进）。
- ❌ **别 commit** `docs/handoffs/2026-06/2026-06-02-seal-robustness-phase2-complete-handoff.md`（pre-existing untracked，非本 track）。

---

## 7. 下一位 Agent 的第一步

1. **先读 memory**：`feedback_oft_single_zone_must_ask_not_guess`（本 track 核心结论 + 几何语义 + 前任误记警示）、`feedback_grill_handoff_must_be_verified`、`feedback_pr_merge_must_run_full_suite_on_shared_logic`、`feedback_parameters_used_must_reflect_actual_resolution_path`。
2. **读 spec** `docs/superpowers/specs/2026-06-03-oft-pipeline-failure-fix-design.md`（含三 PR 完整改动点 + 验收 + §5 命令）。
3. **执行 §4.0**：`git fetch` → 三分支 rebase 到 `6cc13f55` → 合并态重启 dev → 跑 OFT dogfood 终验 → 通过则合 main。
4. **代码 review 已闭环**（三缺陷修复均现场核实通过），**不要重复 review 代码**——直接做合并态终验。

## milestone 建议
本 track 已到 checkpoint（根因澄清 + 三 PR 实施 + review 修复闭环，待终验合并）。建议在 §4.0 合并态终验通过后，创建/更新 milestone `docs/milestone/oft-pipeline-robustness.md`（OFT 单匿名区处理 + 越权防御 + 编排 plan-gate 三道地基），并登记 README。下一 agent 终验通过后执行。
