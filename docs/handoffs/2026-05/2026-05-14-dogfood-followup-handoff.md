# 2026-05-14 Dogfood 修复迭代验证交接

## Batch A/B 修复验证结果（vs Issue #1-8）

| Issue | 检查项 | 期望 | 实测 | 结论 |
|-------|--------|------|------|------|
| #1 | gateway reload 次数 | 0 | 0 (gateway.log grep确认) | ✅ |
| #2 | thinking 400 次数 | 0 | _(需人工dogfood)_ | ⏳ |
| #3 | lead 自写判读 | 0 | _(需人工dogfood)_ | ⏳ |
| #3 | lead 编品系 | 不出现 | _(需人工dogfood)_ | ⏳ |
| #3 | lead 引常模/金标准 | 不出现 | _(需人工dogfood)_ | ⏳ |
| #4 | reasoning 自动折叠 | 不发生 | _(需人工dogfood)_ | ⏳ |
| #5 | 阶段播报次数 | ≥4 次 | _(需人工dogfood)_ | ⏳ |
| #6 | plan.json 输出虚拟路径 | 全虚拟 | _(需人工dogfood)_ | ⏳ |
| #7 | compute_* 重跑次数 | 1 次/个 | _(需人工dogfood)_ | ⏳ |
| #8 | report-writer 被派 | 观察 | _(需人工dogfood)_ | ⏳ |
| #8 | report-writer 读 catalog YAML | 观察 | _(需人工dogfood)_ | ⏳ |

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
- thread_id: _(待人工dogfood后填写)_
- run_ids: _(待人工dogfood后填写)_
- 数据文件: _(待人工dogfood后填写)_
- 开始时间: _(待人工dogfood后填写)_
- 结束时间: _(待人工dogfood后填写)_

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

_(人工dogfood后填写发现的任何问题)_

## 下一步建议

1. **人工 dogfood** — 完成上述 9 项检查清单，填写 thread 信息和实测结果
2. **如果 checkpoint warning 消失但实际行为有问题** — 回 Task 10 确认 writes 清理逻辑
3. **如果 thinking 400 仍未消除** — 回 Task 4，可能丢字段链路不止一处
4. **后置 Issue #9 / #11 / #12 / #13** — 不在本次修复范围，需独立设计讨论
5. **如果所有 ✅** — 可以 merge 回 dev 分支
