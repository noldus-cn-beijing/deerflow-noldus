# 2026-05-29 Sprint 5.7 合入 + serve.sh watchfiles 修复 + FST dogfood + Sprint 5.8 seal-resume spec — handoff

**会话主体**：5.7 合 dev 核验 → serve.sh watchfiles 卡死修复(make dev 起不来)→ FST E2E dogfood(5.7 真机接住漏调)→ "如何避免 seal 漏调"深挖 → 探针排除 forced tool_choice → Sprint 5.8 seal-resume spec + 9 轮 grill 加固 + ADR-0001 → GitLab remote 取消
**触发分支**：dev
**dev HEAD**：`c586ae09`（本地 = origin/dev，已同步）
**核心 commit 链**：`af40c0ad`(5.7 实施,PR #61) → `fd5815b5`(5.7 merge) → `1b6eb2c7`(5.7 docs) → `c9c99712`(serve.sh 修复) → `c586ae09`(5.8 spec + ADR-0001)
**测试基线**：ethoinsight **504** / agent backend **3133**(5.7 +22) / frontend tsc 0 错
**关键产出**：5.8 spec(grill 加固版,可交付) + ADR-0001(防 forced tool_choice 重提) + serve.sh 修复(已 push) + 1 条 MEMORY(handoff 内容校验待修)

---

## 1. 本次会话做了什么（时间顺序）

### 阶段 1：读 5-28 handoff + SOTA roadmap 第一性讨论
用户让读 `2026-05-28-sprint1-2a-deploy-prep-and-seal-bug-handoff.md`，然后问"sprint 大方向对吗（从第一性原理）"。我对照 PRD（公测标准=6 范式跑得准、demo 不翻车、端到端、lead 自由路由）+ roadmap v2（尼采"主人"哲学）做了批判性分析，指出：**v2 路线图在用 v0.2 的愿景路线图跑 v0.1 工期**，且发现两个真实缺口——golden-cases 为空（0 个专家标注）、OFT/LDB/Zero Maze 3 范式无端到端 dogfood 痕迹。**这个战略判断未落地为行动**（用户没让改 roadmap），仅作为讨论。

### 阶段 2：写 Sprint 5.7 实施文档（前序，已完成）
用户让把已有的 5.7 spec 写成实施文档交独立 agent。我对 dev HEAD 真实代码核验后**修正了原 spec 3 处会踩空的点**（C1 复用现成 resolve_workspace / C2 连字符 vs 下划线命名 / C3 validation helper 异常安全），产出 `docs/superpowers/plans/2026-05-29-sprint-5.7-handoff-emission-validator-impl.md`。

### 阶段 3：5.7 实施完成 + 我核验（关键纪律实践）
新 agent 报 5.7 完成（分支 `sprint-5.7-handoff-validator`，commit `af40c0ad`），**自我报告"3106 passed（基线 3084 +22）"**。我没直接信——隔离 worktree checkout 它的分支亲手跑全量，发现它的 "3084 基线" 是它本地环境的计数误差（环境性 skip 多）。**统一基线下实测：合并后 backend = 3133（3111 + 22），无退化**。用户在远端合并 PR #61，我 rebase 本地 docs commit + push。

### 阶段 4：serve.sh watchfiles 卡死修复（make dev 起不来）
用户 `make dev` 时 Gateway 卡 180s 超时失败、gateway.log 0 字节。我诊断过程**走了两个错误假设**（uv 冷同步、瞬态端口占用）后，用 faulthandler + 隔离实验定位真根因：**`uvicorn --reload` 依赖的 `watchfiles` 库在此机器（Precision 7960 / 内核 6.17 / watchfiles 1.1.1）`watch()` 入口即卡死**（裸调 `watchfiles.watch('app')` 也卡）。修复：`serve.sh` 把 Gateway 热重载改为 opt-in（`GATEWAY_RELOAD=1` 才开，默认关），commit `c9c99712` 已 push。daemon 模式帮用户重启服务成功。

### 阶段 5：FST E2E dogfood（5.7 真机验证 + 整链跑通）
用户上传 `原始数据-Porsolt forced swim test XT190-Trial 1.xlsx`（大鼠 FST，n=1 Drug vs n=1 Saline）跑 E2E。**5.7 完整闭环实测成功**：data-analyst 第一次派遣漏调 seal（输出"Now let me compile and seal"却没调 tool）→ executor 标 FAILED + "terminated without emitting" → lead 自动 retry（带强化提示，未询问用户）→ 第二次补轮成功 → 整链跑通到 report-writer（report.md + 4 图）。thread `439b1405`。data-analyst 还正确识别"immobility 极低=可能检测参数过严" + "n=1 无法统计推断"。

### 阶段 6："如何避免 seal 漏调" + 探针排除死路
用户问如何**减少**漏调（5.7 是兜底，这是降发生率）。讨论升级为"把 seal 做成抽象 feature"。我用 deepseek-v4-pro 真机探针逐一证伪了"看起来更直接"的方案（详见 ADR-0001）：
- **forced tool_choice**：dashscope thinking mode **400 拒绝**该参数；关 thinking 后强制生效**但 args 全空**（产空 handoff 比漏调更糟）
- **structured output**：strip 掉 seal tool 的 runtime 注入参数
- **executor 从 tool_calls 提取代调**：漏调现场 tool_calls 为空，无 payload
关键洞察：**deepseek 在干净/聚焦上下文里不漏调**（探针 Q1 实证），漏调是长上下文末端注意力衰减。且**必须模型无关**（未来换 Qwen3 / vLLM / Fireworks）。

### 阶段 7：Sprint 5.8 seal-resume spec + 9 轮 grill
用户拍板方向："subagent 结束任务时工程化保证 seal 发生"。落地为 **seal-resume 补轮**：executor 标 FAILED 前先就地补一轮（喂完整历史 messages + 聚焦"请调 seal"指令再 astream），补不上才走 5.7。写成 spec 后用户跑 `/grill-with-docs` 加固，9 轮 grill 实质改进了 spec（见 §3）。最后 commit `c586ae09`（spec + ADR-0001）。

### 阶段 8：GitLab remote 取消
用户为双推加了 GitLab 到 origin 的 push URL（导致 origin push 分裂：fetch=GitHub / push=GitLab），push 撞 GitLab 认证墙。用户决定退回 GitHub-only。我删掉 GitLab push URL，origin 恢复 fetch+push 都 GitHub，`c586ae09` 成功 push。**双推未放弃，只是搁置**（等用户配 GitLab SSH key 或 token）。

---

## 2. 当前状态

### 代码 / 测试（dev HEAD = c586ae09，本地=origin/dev）
| 模块 | 状态 |
|---|---|
| Sprint 5.7（handoff emission validator） | ✅ 合 dev（PR #61）+ 22 测试 + **FST dogfood 真机接住一次漏调** |
| serve.sh watchfiles 修复（Gateway reload opt-in） | ✅ 合 dev `c9c99712` + 已 push + 服务实测能起 |
| 全量测试 | ✅ ethoinsight 504 / backend 3133 / frontend tsc 0 |
| Sprint 5.8 spec + ADR-0001 | ✅ 合 dev `c586ae09` + 已 push（**待新 agent 实施**） |

### Sprint 5.8 spec 状态（grill 加固版，可直接交付）
`docs/superpowers/specs/2026-05-29-sprint-5.8-seal-resume-design.md` — **用户正在/即将交给新 agent 实施**。
- 估期 ~3.5 天，改 `subagents/executor.py`（加 `_attempt_seal_resume` helper + 5.7 检查点前插补轮）+ 新建 `tests/test_seal_resume.py`（12 case）
- 所有行号/变量已对 dev HEAD `c9c99712` 核验；最关键的两个地基用探针实证（见 §4）

### 工作区残留（未动，未决定）
- `.env.wecom` — 敏感凭证，**永不 commit**
- 4 个 untracked 老文件（2 个老 handoff + 1 个老 plan + 1 个 b2-b10 handoff）— 用户多次会话未决定是否单独 commit
- `/tmp/probe_*.py` 4 个探针脚本 — 本会话产出，结论已写进 spec/ADR，可删

### 并发注意
- **用户有另一个 agent 在执行 Sprint 2b**（独立分支/worktree）。本会话改过 origin push 配置（删 GitLab URL）— 2b agent 的 git push 现在也回落 GitHub。下个 agent 操作 git 前留意 2b 并发。

---

## 3. 9 轮 grill 锁定的 5.8 设计决策（spec 已固化）

| # | 分叉 | 裁决 | 依据 |
|---|---|---|---|
| 1 | 补轮续什么上下文 | 喂 final_state.messages + 收尾 HumanMessage | **探针 probe_resume.py 实证 B 行为**：不重放历史 tool_call，只响应收尾，args 完整 |
| 2 | 补几次 | **1 次**（`_SEAL_RESUME_MAX_ATTEMPTS=1` 显式常量） | 失败主因系统性非随机，再补无益，交 5.7 重派 |
| 3 | 无分析内容补不补 | 不补（无米下锅守卫：无 AIMessage 返回 None） | 防"强行补→空 handoff" |
| 4 | 补轮后查文件存在 vs 内容 | 接受查存在，**立案后续 fix** | 概率低 + 普遍问题别在此堵；MEMORY 持久化 |
| 5 | 补轮 prompt 措辞 | 保持现状 HumanMessage（实证能跑通，勿优化） | "能跑通不动"；SystemMessage 被 API 限制排除 |
| 6 | 补轮 state 有 thread_data 吗 | 透传成立 + **仍显式重注入**（防御） | **探针 probe_thread_data.py 实证**两种都产文件 |
| 7 | max_turns 加不加 | **删掉 §3.1（不改）** | **查 dogfood jsonl 证伪**：成功仅用 3-4/12 轮，漏调与轮数无关 |
| 9 | 与 5.7 lead retry 打架? | **不打架** | 查 prompt.py:254，AND 触发条件按 COMPLETED/FAILED 天然隔离 |

**grill 的实际产出**：删 1 个负优化（max_turns）+ 补 2 个守卫（无米下锅 / 显式重注入）+ 立案 1 个后续 fix + 标记 1 个 training-data 隐患 + 3 个探针把"靠猜"变"靠证"。

---

## 4. 关键探针结论（本会话核心实证，已写入 spec §0 / ADR-0001）

| 探针 | 验证 | 结论 |
|---|---|---|
| `/tmp/probe_tool_choice.py` | deepseek forced tool_choice | thinking mode **400 拒绝**；且 Q1 证明干净上下文不漏调 |
| `/tmp/probe2.py` | 关 thinking 后 forced tool_choice | 强制生效**但 args 全空** → 陷阱 |
| `/tmp/probe_resume.py` | create_agent 第二次喂历史+收尾指令 | **B 行为**：不重放历史，只响应收尾，args 基于历史结论完整 |
| `/tmp/probe_thread_data.py` | 补轮 state 的 thread_data 透传 | 透传成立 + seal 能写文件 |

**这些结论的价值**：把 forced tool_choice / structured output 这两条"看起来对"的死路用真机证伪，固化进 ADR-0001 防三个月后有人重提。

---

## 5. 下一会话执行清单

### A. Sprint 5.8 实施（用户已/即将交新 agent）
spec 就位可直接做。**实施 agent 注意**：
- TDD 先写 12 个 test case（spec §4.1）再写实现
- `_attempt_seal_resume` 5 个要点：异常安全 + 无米下锅守卫 + 显式重注入 thread_data/sandbox + 补轮次数=1 常量 + 正面措辞（勿动 resume_prompt）
- 保留 5.7 FAILED 路径（补不上的兜底，绝不删）
- dogfood 验收：临时削弱 data_analyst seal 指令复现漏调 → 验证 executor 日志出现 "seal-resume" → handoff 产出 → COMPLETED（非 FAILED）→ revert 临时改动
- 基线 ≥3133

### B. handoff 内容校验后续 fix（已立案，必做）
见 MEMORY `project_2026-05-29_handoff_content_validation_pending` + spec §8 + ADR-0001。新开 sprint：`_validate_handoff_emitted` 增强为按各 subagent 契约查核心字段非空，覆盖正常 + 补轮路径。

### C. GitLab 双推（搁置，等用户）
用户想 origin 同时推 GitHub + GitLab。当前退回 GitHub-only。恢复方法：给 origin 加回 GitHub push URL + GitLab 用 SSH（`git@gitlab.com:noldus-core-team/china-devs/etho-insight.git`，前提是 SSH 公钥已加到 GitLab 账号）或 https + PAT。**GitLab 认证当前未配通**（之前 https 报 token 错误）。

### D. 战略判断（阶段 1，未落地，供参考）
v2 roadmap 是否在用 v0.2 愿景跑 v0.1 工期？两个真实缺口：golden-cases 为空、OFT/LDB/Zero Maze 无端到端 dogfood。用户当时未让改 roadmap。若未来要收敛到公测标准，这是切入点。

### E. 清理（可选）
- `/tmp/probe_*.py` 4 个探针（结论已固化，可删）
- 4 个 untracked 老文件 + `.env.wecom` 的去留（用户长期未决定）

---

## 6. 关键文件地图

| 类别 | 路径 |
|---|---|
| Sprint 5.8 spec（grill 加固，待实施） | `docs/superpowers/specs/2026-05-29-sprint-5.8-seal-resume-design.md` |
| ADR-0001（seal-resume 而非 forced tool_choice） | `docs/adr/0001-seal-resume-not-forced-tool-choice.md` |
| Sprint 5.7 实施文档（已实施） | `docs/superpowers/plans/2026-05-29-sprint-5.7-handoff-emission-validator-impl.md` |
| serve.sh 修复 | `packages/agent/scripts/serve.sh`（GATEWAY_RELOAD opt-in 段，约 150-175 行） |
| 5.7 代码 + 5.8 挂载点 | `packages/agent/backend/packages/harness/deerflow/subagents/executor.py:807-828`（5.7 检查点 = 5.8 补轮插入处） |
| FST dogfood thread | `.deer-flow/users/cd95effa.../threads/439b1405-.../`（4 handoff + report.md + 4 图） |
| 探针脚本 | `/tmp/probe_tool_choice.py` `/tmp/probe2.py` `/tmp/probe_resume.py` `/tmp/probe_thread_data.py` |

---

## 7. milestone 建议

本次会话让 SOTA agent v2 的"防漏调纵深防御" track 推进到：
- **5.7 完整 dogfood-pass**（FST 真机接住一次漏调，端到端跑通）
- **5.8 spec 就绪**（grill 加固 + 3 探针实证 + ADR）
- **防漏调三层防御成形**：5.7 兜底（重派）→ 5.8 补救（补轮）→ 内容校验（已立案）

应更新 milestone：
- `sota-agent-v2`：状态加 "5.7 合 dev + FST dogfood-pass + 5.8 spec 待实施 + ADR-0001"
- 可新建 `seal-handoff-reliability` milestone：记录 seal 漏调问题的完整解决脉络（5.7 检测 → 5.8 补轮 → 内容校验立案 + ADR 决策 + 探针实证），这是个完整的工程化纵深防御案例

---

## 8. 给下一 agent 的友情提示

1. **dev HEAD = `c586ae09`**，本地 = origin/dev，所有改动已 push GitHub
2. **测试基线**：ethoinsight 504 / backend 3133 / frontend tsc 0，任何实施不得退化
3. **用户有 2b agent 并发**：操作 git 前确认无并发 push；origin 已退回 GitHub-only（本会话删了 GitLab push URL）
4. **5.8 实施纪律**：保留 5.7 兜底、勿优化 resume_prompt 措辞、勿重新加 max_turns（已查证否决）、异步路径真测
5. **本会话方法论**：全程"先验事实/探针再设计"——5.7 核验、watchfiles 诊断、forced tool_choice 排除、grill 7 个分叉，都靠实证不靠猜。延续这个纪律
6. **make dev 现在默认无 Gateway 热重载**（watchfiles 在此机器坏）——改 backend 代码后需重启；想要热重载且 watchfiles 修好后用 `GATEWAY_RELOAD=1 make dev`
