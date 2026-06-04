# 2026-06-02 会话交接 — sync PR #77 冲突待解 + Gateway 模式决策 + 3 份 spec 待实施

> **本 handoff 用途**:交接一次超长会话。主线是 **DeerFlow 全量 sync(PR #77)的 review + 善后**,衍生出 **Gateway 模式部署迁移决策**(反复撞到,必须收口)、**3 份待实施 spec**(seal 阶段1.5/2、反幻觉、sync)。最紧急:**PR #77 有 3 个冲突待解,其中卡在 Gateway 模式决策上**。
>
> **当前 dev HEAD**:`3f353458`(含 seal PR #76)。**先 `git fetch && git log --oneline origin/dev -5` 确认是否前进。**

---

## 0. 🔴 最紧急:PR #77 (DeerFlow 全量 sync) 有 3 个冲突待解

**PR**:[#77](https://github.com/noldus-cn-beijing/noldus-insight/pull/77),分支 `worktree-sync-0602`(HEAD `0c895a31`),base `dev`,状态 `CONFLICTING`。
**worktree**:`/home/wangqiuyang/noldus-insight/.claude/worktrees/sync-0602`(已 push,3 commit:`fa3418ec` sync 主体 + `3d5870b7` 善后修复 + `0c895a31` state 更新)。

### sync 本身已就绪(全绿)
- 全量跟随上游 f9b70713→74e3e80c(34 commit),受保护 harness 定制全 surgical 保住
- 测试:`DEER_FLOW_CONFIG_PATH=<repo>/packages/agent/config.yaml` 下全量 = **3254 passed / 0 failed / 30 skipped**
- ⚠️ **跑测试必须给 config**(worktree 无 config.yaml,否则一批 FileNotFoundError 假失败)

### 3 个冲突(本地 `git merge --no-commit --no-ff origin/dev` 复现,看完 `git merge --abort`)
| 文件 | HEAD(sync/上游) | origin/dev | 解法 |
|---|---|---|---|
| `app/gateway/internal_auth.py` | `INTERNAL_AUTH_ENV_VAR` 常量 + `_load_internal_auth_token()`(上游 #3184 重构)| 内联简单版 | **用上游版**(纯改进,dev 侧无定制) |
| `app/channels/manager.py` | `DEFAULT_LANGGRAPH_URL=8001/api`(Gateway 模式)| `_detect_extension`(**Noldus 定制**,EthoVision 文件类型检测)+ URL=2024 | **保 dev 的 `_detect_extension`** + URL **取决于 Gateway 决策**(见 §1) |
| `frontend/.../artifact-file-detail.tsx`(3 块)| HTML preview scroll restoration/base href(**上游新功能**)| `normalizeArtifactImageSrc`(**我们的 img rewrite,report-chart-404 修复 PR #75!**)| **两边合**:既要上游 scroll restoration,又必须保 dev 的 `normalizeArtifactImageSrc`(否则回退 PR #75) |

**核心警告**:sync 分支基线是**旧 dev**(不含 PR #75/#76)。全量 sync 会**回退掉 dev 上比 sync 基线新的 Noldus 改动**——`normalizeArtifactImageSrc`(#75 前端修复)和 `_detect_extension` 都是这种,**冲突解决时必须从 dev 侧保留**。

---

## 1. 🔴 必须收口的决策:Gateway 模式 vs Standard 模式

**这个决策反复撞**(gateway_runtime_cleanup 10 个 skip、manager.py URL 冲突都卡在它),不收口会变成每次 sync 的隐性债。

**背景**:上游已废弃 standalone LangGraph server,全切 **Gateway-embedded runtime**(3 进程,runtime 嵌入 Gateway)。我们仍用 **standard mode**(`serve.sh` 的 `langgraph dev`,4 进程含独立 LangGraph server,`/api/langgraph→2024`)。

**用户 2026-06-02 已锁定方向:跟随上游切 Gateway 模式**。理由:deerflow 是 infra 底座,守着被淘汰的 standalone 模式只会让分叉越来越痛;ECS 部署可随时改(轻),底座架构对齐(重收益);Gateway 模式本身更优(3 进程省资源)。

**但"切 Gateway 模式" = 一次真实部署架构迁移**,涉及:
- `serve.sh`:`langgraph dev` → Gateway embedded runtime 启动
- `Makefile`:删 `dev-pro`/`start-pro` 等 transition target(Gateway 变默认)
- `docker-compose.yaml`:4 服务(frontend/gateway/langgraph/nginx)→ 3 服务(去独立 langgraph)
- `nginx.conf`:`/api/langgraph/*` 从 LangGraph(2024)→ Gateway(8001)
- `manager.py` 的 `DEFAULT_LANGGRAPH_URL` → 8001/api
- **这些部署文件多在 `~/ethoinsight-prod/`(不进 git)+ ECS 实际 compose**
- **必须 dogfood 验证 Gateway 模式端到端跑通**(我们从没在 Gateway 模式跑过)

**待新 agent 决策的分叉**:
- **方案 A(之前倾向)**:PR #77 先解冲突时 manager.py URL 暂用 **2024**(保持 standard),merge sync;Gateway 迁移**单独 spec/PR** 做(改部署 + dogfood + 解除 10 个 gateway_runtime_cleanup skip)。
- **方案 B**:PR #77 直接连 Gateway 迁移一起做(URL 用 8001)——但把全量 sync + 架构迁移捆一个 PR,且没 dogfood 验证就改部署,风险高。
- **用户最后倾向**:"把 gateway 的解决了"——倾向**现在就把 Gateway 模式迁移做掉**(不再拖)。需新 agent 确认是 A 还是 B,我倾向**先 merge sync(A 的冲突解法,URL 2024)+ 紧接着做 Gateway 迁移 spec**,两步但不拖。

---

## 2. 本会话已完成

### 2.1 sync PR #77 review + 善后(见 §0)
- 戳破 sync agent "29 failed 属于预期" 的假结论:真相是 dev 全绿(3329)→ sync 后 27 failed(真回归,sync agent 在缺 config 的污染环境测出假基线)
- 定位 27 个全是"测试断言被上游覆盖的源码行为",已处置(commit `3d5870b7`):
  - 16 个:还原配套测试到 dev 版(`test_lead_agent_model_resolution`/`_training_middleware`/`test_channels` 删 1 用例)
  - 10 个:skip `test_gateway_runtime_cleanup`(Gateway 决策)
  - 1 个:skip `test_runtime_lifecycle_e2e`(setup_agent per-user vs 我们扁平路径)
  - 1 个:skip `test_thread_run_messages_pagination`(测试隔离污染,见 §3）
- `.deerflow-sync-state` 更新到 74e3e80c(commit `0c895a31`)

### 2.2 三份 spec 产出(均在 `docs/superpowers/specs/`)
| spec | 状态 |
|---|---|
| `2026-06-01-all-subagent-seal-robustness-design.md` | 阶段1 已合(PR #76);**阶段 1.5 + 阶段 2 待实施**(§7 有可直接照写的代码骨架) |
| `2026-06-02-lead-tool-invocation-reliability-design.md` | 🆕 **待实施**(反幻觉,见 §4) |
| `2026-06-02-deerflow-upstream-sync-design.md` | 全量跟随策略(本次 sync 依据),已随 PR #77 落地 |

### 2.3 CLAUDE.md 更新(已在 dev `3f353458`)
- 6 范式(加 TST)+ 同步核心规则翻成"全量跟随 + surgical 守护"+ Tier4 去过时

### 2.4 memory 固化(3 条新)
- `feedback_sync_full_follow_upstream_infra`(全量跟随策略 + 3 条配套防护:配套测试隐性受保护/worktree 必给 config/全量会带进断言上游行为的测试)
- `feedback_skill_describing_tool_output_enables_hallucination`(反幻觉根因)

---

## 3. 🟡 待定位:pagination 测试隔离污染

`test_thread_run_messages_pagination.py::test_get_run_hydrates_store_only_run` — sync 引入的测试隔离缺陷(单独跑 passed,全量跑 404)。已 skip 止血。**根因定位文档已备好全部线索**:`docs/problems/2026-06-02-pagination-test-isolation-pollution.md`(在 sync worktree,随 PR #77 merge 进来)。派新 agent 照文档查:大概率是 FastAPI `dependency_overrides[get_run_manager]` 全局泄漏或 RunManager 单例污染。修复后解除 skip。

---

## 4. 🟡 待实施:反幻觉 spec(dogfood 实证根因)

**根因(已钉死)**:lead agent 在 thinking 里把 `ethovision-paradigm-knowledge` skill 的 `forced_swim.md` 候选清单**冒充**成 `identify_ev19_template` 的工具返回值,绕过真实调用直接 ask_clarification。identify 工具全程零调用(training-data + log 双证,thread `9af3ba6d`/`81051535`)。

**三层方案**(spec 详):① skill 瘦身(删工具输出格式描述,釜底抽薪)② prompt 主体化(仿 deerflow `<clarification_system>` 硬指令)③ after_model+jump_to(复刻上游 #2135,sync 后已带入)+ guardrail 拦 ask_clarification 双兜底。+ inspect_uploaded_file 加 data_preview。

**顺序**:等 PR #77 merge 进 dev 后做(它改 prompt.py/agent.py 受保护文件,要干净基线)。层1 skill 瘦身可抢先(纯 markdown)。

---

## 5. 🟡 待实施:seal 阶段 1.5(解锁 dogfood,最紧急的功能修复)

**当前 dogfood 真阻塞**:data-analyst seal 失败 —— step 2.8 跳过出口产出的 `ParameterAuditFinding` 形状违反 schema(`used_value=None` + `observed_distribution={"note":文字}`)。langgraph.log thread `81051535` 03:24:45/03:25:12 ValidationError 实证。

**修法 A+B(spec §7.1/7.2 有完整代码)**:A=prompt 教退化 finding 填合法字段(used_value 填真实参数值/observed_distribution 填 `{}`/说明文字放 suggestion);B=`ParameterAuditFinding` 加 model_validator 归一化兜底(照抄 DataQualityWarning)。**不碰 prompt.py/agent.py,可和 sync 并行**。

---

## 6. 下一位 agent 的第一步建议

1. `git -C /home/wangqiuyang/noldus-insight fetch && git log --oneline origin/dev -5`
2. **先收口 Gateway 决策(§1)**:确认 A(先 merge sync + 单独 Gateway 迁移)还是 B(一起做)。用户倾向"现在解决 Gateway"。
3. **解 PR #77 冲突(§0)**:按表逐块解(internal_auth 用上游 / manager.py 保 _detect_extension + URL 按决策 / artifact 两边合保 normalizeArtifactImageSrc)。解完跑全量(给 config)确认仍 3254 passed,push。
4. 若决策 A:merge PR #77 后,新开 Gateway 迁移 spec/worktree(改部署文件 + dogfood + 解除 gateway_runtime_cleanup skip)。
5. 其余按优先级:seal 阶段1.5(解锁 dogfood)→ 反幻觉 → pagination 定位 → seal 阶段2。

## 7. 关键文件/命令速查
- sync worktree:`.claude/worktrees/sync-0602`,跑测试 `export DEER_FLOW_CONFIG_PATH=<repo>/packages/agent/config.yaml; PYTHONPATH=packages/harness:. <主venv>/python -m pytest tests/ -q -p no:cacheprovider`
- 冲突复现:worktree 里 `git merge --no-commit --no-ff origin/dev`(看完 `git merge --abort`)
- 受保护文件清单:`scripts/sync-deerflow.sh:51`(22 个)
- 部署 SOP:`docs/sop/deploy-via-tar-sop.md`;模式说明:`packages/agent/backend/CLAUDE.md`「Runtime Modes」
- dogfood log:`packages/agent/logs/langgraph.log`(UTC 时间 = 本地 -8h)

## milestone 建议
本会话让 **DeerFlow sync track 到达 checkpoint**:从"选择性合入"翻转为"全量跟随 infra 底座"策略并首次落地(PR #77)。建议更新/创建 sync milestone,记录策略翻转 + 3 条配套防护 + Gateway 模式迁移作为下一里程碑。
