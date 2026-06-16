# DeerFlow 上游 sync f92a26d5 → 2d5f0787 实施 handoff

**日期**: 2026-06-11
**分支**: `chore/sync-deerflow-2d5f0787` (已 push)
**PR**: https://github.com/noldus-cn-beijing/noldus-insight/pull/new/chore/sync-deerflow-2d5f0787
**spec**: docs/superpowers/specs/2026-06-11-deerflow-sync-f92a26d5-to-2d5f0787-spec.md

## 实施摘要

共合入 **9 个 commit**（spec 列出的 10 个中跳过 1 个）：

### 🟢 全量接受（7 commits）
| Commit | 描述 |
|--------|------|
| `b62c5a7b` | fix(agents): offload blocking filesystem IO off event loop |
| `ba9cc5e9` | fix(gateway): enforce thread ownership on stateless run endpoints |
| `a57d05fe` | fix runtime journal run lifecycle events |
| `ae9e8bc0` | fix(sandbox): missing sandbox.mounts host_path → loud ERROR |
| `b3c2cc42` | fix(agents): require config.yaml in resolve_agent_dir |
| `37337b77` | feat(models): add StepFun reasoning model adapter |
| `167ef451` | feat(memory): token_counting config（非受保护部分） |

### 🟡 Surgical 合入（2 commits）
| Commit | 描述 | 涉及受保护文件 |
|--------|------|---------------|
| `8db16bb3` | fix(config): coerce null config.yaml list sections | `config/app_config.py`（3 处 surgical） |
| `0fb18e36` | refactor(lead-agent): build_middlewares 重命名 | `agent.py` + `client.py` + 6 测试文件 |
| `167ef451` | feat(memory): token_counting config（受保护部分） | `memory/prompt.py`（全量搬运+回植 Noldus 隔离）+ `lead_agent/prompt.py`（3 行） |

### ⏭️ 跳过
- **`16391e35` fix(skills): slash skill activation** — 体量大、触受保护文件多（agent.py + prompt.py）、IM channel 场景与我们 v0.1 web 主路径弱相关。spec 默认选 A 跳过。待独立评估是否需要 IM channel slash 能力。

## 验证结果

- `app.gateway` 裸导入 ✅
- `make_lead_agent` 裸导入 ✅
- `make test`: **3970 passed**, 14 failed（全为已知/预期：2 已知污染 + 8 stateless-runs 需 live server + 3 sandbox mounts Noldus 分歧 + 1 chart_maker_config 既定）

## 关键操作细节

1. **app.py 污染处理**: 初次全量拷贝 `deerflow/main:app.py` 引入了 `2b795265`（auth-disabled）的 import，该 commit 不在 sync 范围。已回退到 dev 版本——167ef451 对 app.py 的改动（tiktoken warm-up 条件化）在 dev 无法直接应用（dev 无 warm-up 调用），留待后续 sync。
2. **test_lead_agent_prompt.py**: 上游新增了大量测试，但我们的 prompt.py 有 Noldus 定制，上游测试期望不匹配。已回退到 dev 版本（6 tests all pass）。
3. **memory/prompt.py surgical**: 先全量搬运上游版本（含 167ef451 的 LOADING sentinel + CJK-aware char estimate + use_tiktoken），再回植 2 处 Noldus 定制：① MEMORY_SYSTEM_PROMPT 末尾的"不要记录会话级事件"块 ② format_memory_for_injection 的 topOfMind/history 隔离逻辑。

## 后续待办

- [ ] 创建 PR 从 `chore/sync-deerflow-2d5f0787` → `dev`
- [ ] 评估是否需要 `16391e35`（slash skill activation for IM channels）
- [ ] app.py 的 tiktoken warm-up 需要单独处理（当前 dev 无 warm-up，167ef451 的条件化改动无法直接应用）
