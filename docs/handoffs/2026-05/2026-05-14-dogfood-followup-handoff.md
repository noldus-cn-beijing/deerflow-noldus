# 2026-05-14 Dogfood 修复迭代验证交接

## Batch A/B 修复验证结果（vs Issue #1-8）

| Issue | 检查项 | 期望 | 实测 | 结论 |
|-------|--------|------|------|------|
| #1 | gateway reload 次数 | 0 | 0 (gateway.log grep确认) | ✅ |
| #2 | thinking 400 次数 | 0 | 0 (langgraph.log grep) | ✅ |
| #3 | lead 自写判读 | 0 | 子agent手递含"典型高焦虑""参考范围"，lead透传 | ❌ |
| #3 | lead 编品系 | 不出现 | 0 (grep确认) | ✅ |
| #3 | lead 引常模/金标准 | 不出现 | subagent手递含"金标准"，lead透传 | ❌ |
| #4 | reasoning 自动折叠 | 不发生 | 未观察到折叠（thinking面板保持展开） | ✅ |
| #5 | 阶段播报次数 | ≥4 次 | 1 (archived messages中仅1个emoji匹配) | ❌ |
| #6 | plan.json 输出虚拟路径 | 全虚拟 | True（G5 修复生效，全部 /mnt/user-data/workspace/ 前缀） | ✅ |
| #7 | compute_* 重跑次数 | 1 次/个 | 5个compute各1次 | ✅ |
| #8 | report-writer 被派 | 观察 | 1 次派遣，后被lead取消（用户新消息打断） | ✅ |
| #8 | report-writer 读 catalog YAML | 观察 | 未完成即被取消 | ⚠️ |

### 可自动验证的项（已验证）

- **#1 gateway reload**: make dev 启动后 gateway.log 无 "WatchFiles detected changes" — Task 1 修复生效
- **#10 checkpointer warning**: langgraph.log 不再出现 "Custom checkpointer missing adelete_for_runs" — Task 10 修复生效
- **服务起停正常**: http://localhost:2026 显示登录页面，服务正常

### 需人工 dogfood 验证的项（9项）

剩余 9 项检查清单需要在实际 EPM 分析 pipeline 运行中观察。建议操作：

1. 打开 http://localhost:2026
2. 新建 thread
3. 上传 `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/` 下任意 1 个 Subject 文件
4. 发消息："请分析这个 EPM 单只数据"
5. 回答 lead 的反问
6. 观察完整分析流程
7. 按上述检查清单逐项打勾

## thread 信息
- thread_id: 8ff3be6d-43b5-4724-ab09-60ce23db6f2e
- run_ids: 019e257b-5668-74f0-aa56-2a9432f539f3 (lead parse+catalog), 019e2584-e304-74a0-b8a2-89210048abaf (report-writer, cancelled)
- 数据文件: 轨迹-Elevated Plus Maze XT190-Trial 1-Arena 1-Subject 1.txt
- 开始时间: 2026-05-14 15:51 (CST)
- 结束时间: 2026-05-14 16:11 (CST)

## 10 个 Commit 汇总

| Commit | Task | 描述 |
|--------|------|------|
| `555db882` | Task 1 | fix(reload): exclude .deer-flow/** recursively from uvicorn reload watcher |
| `24715250` | Task 2 | fix(lead): forbid self-written interpretation + unsupported metadata + absolute references |
| `8e53d064` | Task 3 | test(thinking): reproduce Issue #2 thinking-field 400 error in unit tests |
| `5d071e9a` | Task 4 | fix(claude_provider): strip malformed thinking blocks before API request |
| `356c3de9` | Task 5 | fix(lead): upgrade transparency from suggestion to mandatory checklist |
| `658f78a1` | Task 6 | fix(frontend): keep reasoning panel expanded via controlled state wrapper |
| `2eb1532a` | Task 7 | fix(catalog): force virtual paths in metric_plan.json output field |
| `49414a19` | Task 8 | fix(code-executor): forbid re-running compute_* after ls verification |
| _(pending)_ | Task 9 | docs(dogfood): record Batch A/B verification results (本文件) |
| `c745748c` | Task 10 | feat(checkpointer): implement adelete_for_runs for cancelled-run cleanup |

## 4 个新测试文件

| 测试文件 | 测试数 | 结果 |
|---------|--------|------|
| `packages/agent/backend/tests/test_lead_prompt_role_boundaries.py` | 4 | 4 PASS |
| `packages/agent/backend/tests/test_thinking_field_preserved.py` | 13 | 13 PASS |
| `packages/ethoinsight/tests/test_catalog_resolve_paths.py` | 2 | 2 PASS |
| `packages/agent/backend/tests/test_checkpointer_adelete_for_runs.py` | 5 | 5 PASS |

**合计: 24/24 PASS**

## 异常观察

### Issue #3 违规话术（subagent→lead透传）

- code-executor handoff: "低于典型正常水平（20-40%），提示可能存在较高焦虑水平"
- data-analyst handoff: "呈现典型的高焦虑样行为特征"、"远低于正常小鼠的20-40%参考范围"、"EPM 金标准要求每组至少6-8只"
- lead 在用户可见输出中直接透传了以上话术，未做过滤
- 根因：prompt 级约束（commit 24715250）未能阻止 subagent 手递中的违规话术透传；机制层（guardrail）不负责语义审查

### Issue #5 阶段播报未达标

- archived messages 中仅匹配到 1 个 emoji 阶段标记
- 可能原因：播报 emoji 未被 lead 实际使用、或播报格式与 grep pattern 不匹配

### Issue #6 plan.json 路径未虚拟化

- ~~metric_plan.json 中 output 字段为物理路径（/home/wangqiuyang/.../workspace/...）~~
- ~~commit 2eb1532a 的虚拟化修复可能未覆盖本次路径或已被后续改动退化~~
- **2026-05-14 G5 修复复测通过**：新 thread `ab36da5f-2d36-450f-a24e-050350cc517f`，metric_plan.json output 字段全部 `/mnt/user-data/workspace/` 前缀，G5 修复生效（commit `f2a60122`，CLI 改用 env var DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE 兜底虚拟路径）

### Step 6 图表补充请求未完整观测

- 用户发送"需要！补充轨迹图和汇总表格图表"后，lead 取消了正在运行的 report-writer
- 后续 lead 响应未能在 UI 中渲染（页面可能冻结）
- LeadAgentExecutionBoundary 已在早期 bash 调用中成功阻断 1 次（code=lead_execution_boundary.bash_not_allowed）
- G9 确认 workspace 中无 .py 文件——guardrail 有效

## 下一步建议

1. **人工 dogfood** — 完成上述 9 项检查清单，填写 thread 信息和实测结果
2. **如果 checkpoint warning 消失但实际行为有问题** — 回 Task 10 确认 writes 清理逻辑
3. **如果 thinking 400 仍未消除** — 回 Task 4，可能丢字段链路不止一处
4. **后置 Issue #9 / #11 / #12 / #13** — 不在本次修复范围，需独立设计讨论
5. **如果所有 ✅** — 可以 merge 回 dev 分支

## 阶段 1.5 LeadAgentExecutionBoundaryProvider 验证（2026-05-14）

复现 thread b0d3a611 路径：
- 步骤 1-2: 上传 EPM Subject 1 数据 + "请分析这份EPM数据"
- 步骤 3: lead 正常反问 (Gate 1) — "仅有Subject 1的数据，无法进行组间统计分析"
- 步骤 4: 选择 "只有 Subject 1，先看看单个被试的数据质量和轨迹可视化"
- 步骤 5: lead 调用 set_experiment_paradigm → 尝试 bash 命令

grep 验证：
```
$ grep "lead_execution_boundary" packages/agent/logs/langgraph.log | tail -5
Guardrail denied: tool=bash policy=lead_execution_boundary code=lead_execution_boundary.bash_not_allowed
Guardrail denied: tool=bash policy=lead_execution_boundary code=lead_execution_boundary.bash_not_allowed
```

结果：
- 第 1 次 deny: lead 尝试非白名单 bash → GuardrailMiddleware 机制层阻断
- 第 2 次 deny: lead 尝试 `python -c "from ethoinsight.parse import dump_headers; print('ok')"` → deny（不在白名单，不是 python -m ethoinsight.parse.* 格式）
- 系统正常运行，无 import 错误，中间件链初始化成功

判定：**plan 成功** — LeadAgentExecutionBoundaryProvider 在机制层阻断 lead 越权 bash 调用。白名单内命令 (python -m ethoinsight.parse.dump_headers) 正常通过。

## G5 修复复测（2026-05-14）

修复 commit hash: f2a60122

修复后 thread: ab36da5f-2d36-450f-a24e-050350cc517f

metric_plan.json output 字段抽样：
```
/mnt/user-data/workspace/m_open_arm_time_ratio.json
/mnt/user-data/workspace/m_open_arm_time.json
/mnt/user-data/workspace/m_open_arm_entry_count.json
/mnt/user-data/workspace/m_open_arm_entry_ratio.json
/mnt/user-data/workspace/m_total_entry_count.json
```

判定：✅ G5 修复成功（output 字段全部 /mnt/user-data/workspace/ 前缀）

Batch A/B 检查表 G5 行原"实测"列从 "False（仍为物理路径）" 改为 "True"，结论 ❌ 改为 ✅。
