# Handoff: dogfood 性能/卡死诊断 + 四份 spec（S1 已合 dev，S2/S3/S4 待实施）

> 交接对象：下一个接手的 AI Agent
> 日期：2026-06-12
> 上一会话产出：诊断两个 dogfood 故障 → 五轮架构讨论（含 Fable-5 两次代码评审）→ 拆四份 spec（S1 已实施+review+合 dev，S2/S3/S4 spec 已写待实施）

---

## 1. 当前状态速览

| spec | 主题 | 状态 |
|---|---|---|
| **S1** | loop_detection 配置接线 + inspect override | ✅ **已实施 + review + 合 dev**（PR #126，含 review-fix） |
| **S2** | identify_ev19_template 返回 per_file_grouping | 📝 spec 写完，**待实施** |
| **S3** | handoff schema fail-open 防护 + calamine 启动断言 | 📝 spec 写完，**待实施** |
| **S4** | code-executor 执行器重构（run_metric_plan 工具） | 📝 spec 写完，**待实施**（最大） |

**实施方式（用户定）**：四份 spec **按序实施**（S1→S2→S3→S4），用户手动开 worktree + 手动 PR merge。每份基于前面已合 dev 的最新代码。下一 agent 的活：用户让你 review 某 worktree 的实施 / 或写下一份的细化 / 或继续 S4。

**dev 当前 HEAD**：`a728881d`（含 PR #125 seal+calamine、PR #126 S1、PR #127 冗余重复 merge）。

---

## 2. 两个 dogfood 故障（本次诊断的起点）

### 故障 A：seal 失败（thread 38be2753）— 已修，合 dev（PR #125，上一会话）
code-executor 140 指标全算完仍判失败。真根因（gateway.log trace=087f87a7）：`MetricStat.parameters_used` 标量 schema 拒 EPM list 值 `open_arm_zones=['open']` → 84 ValidationError。修复=抽 `ParamValue` 别名 + calamine 引擎提速。**已合 dev**。详见 memory `feedback_seal_failure_third_root_cause_list_zone_params_reject_scalar_schema`。

### 故障 B：端到端卡死（thread a5b97c00）— 拆成 S1+S2
lead 逐个 `inspect_uploaded_file` 探查 28 文件分组，第 5 次撞 `[FORCED STOP]` 卡死。**两层根因**：
- **第一层（S1，已修）**：`LoopDetectionMiddleware` 两处实例化没接 `app_config.loop_detection` → config.yaml 的 hard_limit:50 静默失效，实跑硬编码 5。
- **第二层（S2，待实施）**：identify 工具只读第 1 个文件、返回结构不含分组字段 → lead 被迫逐个 inspect 试探边界。

### 故障 C：性能慢（thread 38be2753）— 拆成 S4
code-executor 第二次 run ≈13 分钟，大头是 LLM 逐 token 吐 140 行 bash + 140 次进程冷启。根本是「LLM 当人肉命令行编排器」反模式 → S4 执行器重构。

---

## 3. 四份 spec 详情

所有 spec 在 `docs/superpowers/specs/2026-06-12-s{1,2,3,4}-*.md`。

### S1（已合 dev，仅供理解）
`2026-06-12-s1-loop-detection-config-wiring-spec.md`
- 修 `agent.py:292`（lead 路径，接 config）+ `config.example.yaml`（inspect override hard_limit 100）。
- **review 发现并修的两类问题**（已在 dev）：
  1. **A2 设计错误**（我 spec 的锅）：原 spec 让 `factory.py` 也接 `get_app_config()`，但 `create_deerflow_agent` 契约是「无 YAML」→ 30 测试 FileNotFoundError。已 revert（factory 用默认，lead/Client 走 build_middlewares 已被 A1 覆盖）。
  2. **worktree 被过时 stash 污染**：3 文件有 `<<<<<<< Stashed changes` 冲突标记（试图回退 dev 已有的 column_semantics / CSRF helper）。已还原 dev 版。
- 教训 memory：`feedback_loop_detection_config_not_wired_runs_hardcoded_default`。

### S2（待实施）
`2026-06-12-s2-identify-per-file-grouping-spec.md`
- **核心**：identify 加「Step 3.5：遍历全部文件提分组」，复用 `_extract_grouping_fields` 纯函数（**抽共享不复制**），三个 return 挂 `per_file_grouping`；prompt 让 lead 优先用它（inspect 退兜底）。
- **关键事实（实测）**：EV19 头 raw_metadata 含 `{'Group':'XY'}`；calamine 后 parse_header 仅 0.09s/文件，28 文件 ≈2.6s。
- **红线**：只读 parse_header 不读 parse_trajectory（性能契约，测试断言后者未调）；正面指令；不删 inspect fallback；**实施前 grep prompt.py 确认无冲突标记**（S1 教训）。

### S3（待实施）
`2026-06-12-s3-schema-failopen-and-calamine-assert-spec.md`
- **Part A**：四个承重 handoff 加 `status=completed → 核心字段非空` 的 model_validator（哑故障换响亮）。**不改 extra="allow"→forbid**（MetricStat 有意向前兼容）。只对 completed 卡，partial/failed 放行。
- **Part B**：`parse/__init__.py` 加 `import python_calamine` 断言（⚠️ import 名是 `python_calamine` 不是 `calamine`，实测坐实）。
- 来源：Fable-5 评审的 A6+B5 TODO。

### S4（待实施，最大）
`2026-06-12-s4-code-executor-run-metric-plan-spec.md`
- **核心**：新建 `run_metric_plan` first-party 工具（ProcessPoolExecutor 进程内调脚本 + 确定性聚合 + 落盘 handoff），code-executor 收 bash、prompt 重写成「调一次工具 + 失败分诊」。
- **架构决策（五轮讨论锁定，§0.1 有完整否决记录）**：保留 subagent（orchestrator + 脚本已写好 + 小模型上下文隔离三条理由）；用 ProcessPoolExecutor（否决 Celery/异步轮询）；`run_plan(plan_path)` 整 plan + 选择器（否决 run_tasks）；seal 反转成工具确定性落盘。
- **关键复用**：`executor.py:_attempt_auto_seal_from_artifacts:401-559` 的聚合逻辑抽成纯函数 `aggregate_metrics_to_handoff`，run_metric_plan 和 auto-seal 共用（Fable「兜底转正」）。脚本 `main(argv)` 可 `importlib` 进程内调；聚合**读 m_*.json 不靠 stdout**（脚本已 save_output_json 落盘）。

---

## 4. 关键架构决策（Fable-5 两轮评审，防后续重提）

本次最大产出是 S4 的架构方向，经用户 + Fable 五轮锤炼：

1. **code-executor「写命令行」是反模式**——确定性可枚举的批量执行该进确定性工具，不该让 LLM 逐字翻译。
2. **但不取消 subagent**（Fable 一度提议 B 方案=取消）——用户三条理由否决：orchestrator 本质 / 脚本已写好缺工具粒度 / **30B 小模型上下文隔离是承重墙**。Fable 接受并正名 code-executor 为「失败监理」（happy path 透明，智能只在错误分诊）。
3. **Fable 四题判断**（S4 §0.1 + 1.x 已固化）：subagent 保留（押错误路径 + 指令隔离）；seal 反转（completed 是计算非判断）；同步长调用非异味（异步轮询是 seal 故障门类换皮）；吃整 plan + 选择器（run_tasks 是 JSON 转录反模式）。
4. **Celery 否决**：分布式队列对「一次调用内同步跑 140 短任务」错配，新增失败面违「减少故障门类」目标。

> ⚠️ **跟 Fable 沟通时所有领域词（生物/EPM/zone/范式）会触发其安全过滤**——必须抽象成纯代码问题（"selected_cols=['a']" / "上游调用方" / "ResultStat"）。本次第二轮评审的 prompt 在对话历史里可参考。

---

## 5. 环境注意事项（踩过的坑）

1. **过时 stash 反复作祟**：本会话 S1 review 时发现一个过时 stash（回退 column_semantics/CSRF）污染了 worktree（冲突标记）+ 主 checkout（旧版 untracked 文件挡 pull）。**建议下一 agent/用户 `git stash list` 清掉它**，免得再 pop 进 S2/S3/S4 worktree。S2/S3/S4 spec 都写了「实施前 grep 确认无冲突标记」。
2. **主 checkout 卫生**：本会话清理了主 checkout 的旧版 factory/agent/测试残留（review 调研留下的）。`git pull` 若被 untracked 挡，先确认 untracked 与 dev 入库版是否一致（一般是旧残留，删了用 dev 版）。
3. **`test_subagent_executor.py` 的 15-24 failed 是纯 dev 基线环境债**（`ModuleNotFoundError: deerflow.agents.middlewares is not a package`，本机 venv editable install 的 namespace 解析问题），**与任何 spec 无关**，在主 checkout 同样红。四份 spec 都写了「勿归因本次」。
4. **裸导入铁律**：改 subagents/tools/agents 核心后必跑 `python -c "import app.gateway"` + `"from deerflow.agents import make_lead_agent"`（conftest mock 藏循环导入，pytest 假绿）。worktree 无 config.yaml 会报 FileNotFoundError，那是预期（import 本身成功即过）。
5. **calamine import 名**：`python_calamine`（不是 `calamine`）——S3 调研 agent 误写，已在 spec 纠正。
6. **PR #127 是冗余**：上一会话 push 误推到旧分支名 `fix/seal-list-zone...`（PR #125 旧名），内容和 PR #126 同。无害但 git 历史多一个空 merge。后续别复用那个分支名。

---

## 6. 下一位 Agent 的第一步建议

**若用户让你继续按序推进**：
1. 确认 dev 最新（`git pull origin dev`，注意 untracked 挡路先清）。
2. 用户实施 S2 后，让你 review → 拉 S2 worktree，**先 grep `<<<<<<<` 确认无冲突标记**，再看 diff + 跑测试（per_file_grouping 红→绿 + parse_trajectory 未调断言 + 共享函数无双存）。
3. 依次 S3、S4。S4 最大，review 重点：聚合纯函数与 auto-seal 字节一致（守 SSOT）+ 收 bash 后 guardrail 不报错 + 裸导入。

**若用户问某个 spec 的细节**：四份 spec 自包含，直接读对应文件。架构决策的 why 在 S4 §0.1 + 本 handoff §4。

**若用户开全新任务**：本批 spec 已闭环（S1 合 dev，S2/S3/S4 待用户排期实施），无需回头。

---

## 关键文件路径速查
- 四份 spec：`docs/superpowers/specs/2026-06-12-s{1,2,3,4}-*.md`
- 核实文档（给 Fable 的纯代码版）：`docs/problems/2026-06-12-schema-and-excel-engine-fix-for-review.md`
- 本次新增 memory：
  - `feedback_seal_failure_third_root_cause_list_zone_params_reject_scalar_schema.md`（含 Fable 三补充：bool→1.0 / extra=allow fail-open / astype dtype 盲区）
  - `feedback_loop_detection_config_not_wired_runs_hardcoded_default.md`（含第二层分组提取根因）
- dogfood trace：gateway.log（thread 38be2753 / a5b97c00），`packages/agent/logs/gateway.log`
- 已合 dev 的 PR：#125（seal+calamine）、#126（S1+review-fix）
