# 交接：data-analyst 漏 seal 根因取证（playwright + 手段 c 独立复现）—— 与并行 ETHO effort 交叉验证同一根因

> **会话日期**：2026-06-23
> **来源**：用户让我读 follow-up spec（`2026-06-22-data-analyst-residual-thinking-cost-followup-spec.md`）并按其 §二「取证优先」用 playwright 跑 EPM dogfood（`Raw data-EPM-Xuhui-28`）前后端监控取证。
> **产物**：
> - Spec：[docs/superpowers/specs/2026-06-23-data-analyst-seal-args-truncation-forensics-spec.md](../../superpowers/specs/2026-06-23-data-analyst-seal-args-truncation-forensics-spec.md)（取证结论 + 治本方向，194 行）
> - 取证数据：`/tmp/pw-driver/`（repro-data-analyst.py / repro/full/data-analyst-repro000-full.json / ai_messages.json / evidence/）—— **建议入库 `scripts/forensics/` 持久化**
> - 生产代码零改动（临时探针已 `git checkout` 撤除，executor.py git diff 空）
> - 用户的 `make dev` 三服务全程未动（2026/8001/3000）

## ⚠️ 与并行 ETHO effort 的关系（重要，给下个 agent）

本会话期间，**另一批 agent 并行处理 `~/ETHOINSIGHT_BUGS.md` 的 10 个 E2E bug**（写了 etho1/2/3/5/7/9 + ETHO-8 七个 spec，dump 了 prod trace，读了 HarnessX 报告，见 `2026-06-23-ethoinsight-bugs-7specs-prod-trace-harnessx-handoff.md`）。

**两边独立工作收敛到同一根因**：
- 我的路径：接 follow-up spec §二「取证优先」→ playwright dogfood + Workflow 验证 + 手段(c) 复现 → 12:04 spec 落盘「data-analyst seal args 撞 max_tokens=4096 腰斩」。
- ETHO 路径：etho1 spec 作者最初判「data-analyst 收尾漏 call」（基于 prod e6ea7946 lead thread dump）→ 14:08 读了我的 forensics spec + 自己复现 → memory `feedback_etho1_real_root_cause_is_seal_args_truncated_by_max_tokens_4096` 修正为「args 腰斩（非收尾漏 call）」，并明确「etho1 升级 SealGate 对 data-analyst 无效」。

**两个独立 agent + 两个独立复现路径（playwright dogfood + repro-data-analyst.py）都指向 args 腰斩** = 根因确定的强证据。下个 agent 不要再质疑这个根因，直接进治本（提 max_tokens）。

**治本点正交**：
- etho1 spec（report-writer / chart-maker 漏 seal）→ SealGate after_agent 兜底（可重建产物）。**对 data-analyst 明确不兜底**（etho1 §2.2）。
- 我的 forensics spec（data-analyst 漏 seal）→ 提 max_tokens 4096→8192+（结构对症 args 腰斩）。**etho1 解决不了 data-analyst**。
- 两 spec 不冲突，各自负责自己的 subagent。

## 本会话做了什么（取证三步）

### 1. playwright 端到端 dogfood（你指定的 28-subject EPM）
- 装独立 node playwright（`/tmp/pw-driver/`，指向已有 chromium `~/.cache/ms-playwright/chromium-1217/`），复用 `access_token` cookie 登录 `qiuyang.wang@noldus.com`。
- 上传 28 xlsx → 答 2 次 HITL 反问（EV19 模板 FewZones + 分组；列语义 open=open_arms/closed=closed_arms）→ code-executor 成功（140/140，~69s）→ **data-analyst 连续 2 次 FAILED** → lead 降级跳过专业判读。
- persistent monitor（盯 gateway.log 的 SealGate/subagent/token 事件）抓到完整现象时间线：两次派遣都是 5 AI msg + SealGate count=1→2 + handoff MISSING + seal-resume FAILED。

### 2. Workflow adversarial 验证（2 独立 agent 审查）
- 纠正我两处误述：① "seal-resume 失败"实为 lead 的 task 重派（state msg19/21/23 证明）；② 排除"flash 弱推理"等竞争假设。
- 推荐手段(c)（独立复现 subagent）取代有风险的手段(a)（改 executor 探针 + 重启 dev + conftest mock 假绿）。
- 注：Workflow 闭包 bug（`.then(v => ({dim: d.key}))` 里 `d` 不可见）drop 了结果成 null，但 review agent 的 jsonl 里有完整结论，手动提取后照用。**教训**：pipeline stage 闭包要用 originalItem 参数，不捕获外层 stage1 变量。

### 3. 手段(c) 独立复现（拿逐 turn 内容，spec §二 的核心）
- `/tmp/pw-driver/repro-data-analyst.py`：用现成 `SubagentExecutor` + 真实 thread workspace（同一份 handoff_code_executor.json，靠 `set_current_user` contextvar + `provider.acquire(THREAD_ID)` 把 sandbox 指向 dogfood thread）+ flash model + 真实 task prompt 复跑。
- 关键坑：① 循环导入（`deerflow.subagents` ↔ `tools.builtins`）→ 先 `import app.gateway` 破环；② `get_effective_user_id` 读 `user.id` contextvar 不是环境变量 → `set_current_user(_FakeUser{id=USER_ID})`。
- 复现成功，行为与 dogfood 一致（5 AI msg + 2 次 SealGate + handoff MISSING）。
- 临时加 env 门控纯读探针（`DEERFLOW_FORENSICS_FULL_MESSAGES`）dump 完整 final_state messages（含 ToolMessage + invalid_tool_calls），**取证后 `git checkout executor.py` 撤除**。

## 根因（逐字节实证，full-message dump）

`seal_data_analyst_handoff` 的 tool_call arguments 超 `max_tokens=4096` 单 turn 输出预算 → 被腰斩成未终止 JSON（char 1178/2142 处 `Unterminated string`）→ LangChain 判 `invalid_tool_calls` → 不执行 → 无 ToolMessage → SealGate 催回 → 再试再腰斩 → handoff 永不落盘 → FAILED。**两次 seal 内容都正确**（status=completed + 5 条含 Mann-Whitney U/p/Cohen's d 的 key_findings + outliers + method_warnings），纯粹长度撞墙。

**是 06-18 PR#161 的镜像残留**：换 flash 无 thinking 没消除"单 turn 预算 < 判读+seal 体积"根因，只换载体（thinking-truncation → args-truncation）。flash 还把判读推理写进可见纯文本 AIMessage（turn[2][4]），是 v4-pro thinking 过载的结构等价物。

## 治本方向（首选 → 最优，spec §四）

1. **首选**：提 data-analyst model `max_tokens`（4096→8192+）。最小改动、结构对症。验证 = `repro-data-analyst.py` 改 max_tokens 重跑断言不腰斩。
2. 若 8K 不够：invalid-args 抢救式 parse（确定性兜底）/ 分步 seal（骨架+append）。
3. **不推荐**：再加 prompt 提醒（HarnessX Telecom 禁令 + memory 4 次打地鼠）。

## 接手第一步（给下个 agent）

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline -1 dev                                  # 确认 dev 最新
ls docs/superpowers/specs/2026-06-23-data-analyst-seal-args-truncation-forensics-spec.md   # 本会话 spec
# 验证取证产物（还在 /tmp，建议先入库）
ls /tmp/pw-driver/repro-data-analyst.py /tmp/pw-driver/repro/full/data-analyst-repro000-full.json
# 验证 executor.py 干净（探针已撤）
git diff --stat packages/agent/backend/packages/harness/deerflow/subagents/executor.py    # 应空
```

## 待办（未做，等你定）

1. **实施治本**：按 forensics spec §四 方向 1 开工程 spec + PR（提 max_tokens + 复现脚本回归断言 + dogfood 验收）。
2. **取证产物入库**：`/tmp/pw-driver/repro-data-analyst.py` + full dump 入 `scripts/forensics/`（持久化，不依赖 /tmp）。memory `feedback_etho1_real_root_cause...` 第 27 行写"取证产物在另一 agent 处"——入库后可更新该行指向 repo 路径。
3. **etho1 协调**：etho1 PR 合入时确认 data-analyst 路径不受 SealGate after_agent 影响（etho1 §2.2 已明确不兜底 data-analyst，但合入时 grep 确认）。

## milestone 建议

「seal 漏调根因治理」track 的**第 5 类根因**：
- 1-3 类：收尾漏 call（诱因）→ SealGate 结构门（已修）。
- 第 4 类：thinking 撞 turn 内超时（v4-pro）→ 06-18 换 flash（PR#161，部分有效）。
- **第 5 类（本会话）：flash seal args 撞输出预算腰斩（06-18 镜像残留）→ 提 max_tokens（待 PR）**。

checkpoint：「data-analyst 判读层功能性失效 → 取证完成（两独立 agent 交叉验证 args 腰斩根因）→ 治本首选提 max_tokens，待实施」。06-18 thinking-overload 修复标注：**部分有效**（消 thinking-truncation，留 args-truncation 镜像），需本会话 spec 闭环。
