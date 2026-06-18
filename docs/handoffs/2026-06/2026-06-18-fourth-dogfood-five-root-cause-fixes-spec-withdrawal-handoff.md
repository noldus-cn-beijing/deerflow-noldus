# Handoff: 2026-06-18 — 第四轮 EPM dogfood：5 根因修复 + open_arm_time_ratio spec 撤回 review + docker 镜像源救场

> 交接对象：下一个接手的 AI Agent。
> 会话主线：用户本地复跑第四轮 EPM dogfood（PR#161 thinking-overload 已合后），暴露 **5 个问题**（其中 2 个全新结构根因 + 1 个真回归 + 1 个独立 bug + 1 个公式裁决）。本 agent 逐一诊断 → 修 5 处（带测试）→ 写 1 份公式 spec → /loop 等实施 agent → review（专家撤回 spec、代码零改动）→ 同步撤回标注 → 顺带救场 docker 部署的镜像源限流。
> 当前位置：主仓 `/home/wangqiuyang/noldus-insight`，分支 `dev`，本地 HEAD = `1dca36b0`（用户已 merge origin/dev PR#162 进来）。**本地 dev 领先 origin/dev 8 个 commit，全部未 push**。

---

## 1. 当前状态速览

### 🔴 必须接着做的事（按优先级）

1. **本地 dev 的 8 个 commit 待 push**（守"用户控制 push"约定，本 agent 没 push）：
   - 7 个本会话 commit（见 §2）+ 1 个用户 merge commit `1dca36b0`。
   - `git log origin/dev..HEAD` 可列全。要 push 时 `git push origin dev`。
2. **工作区残留非我产物，别顺手带**（`git status` 的 `M`/`??`）：
   - ` M docs/problems/2026-06-17-...md`（别的 babysit agent 的 D-3 段，#158）
   - ` M packages/agent/backend/uv.lock`（历史遗留噪音，来历不明，往期 handoff 反复警告**别 commit**）
   - 2 个 untracked handoff + `reports/report for june/`（不是本会话产物）
3. **docker base 镜像缓存是临时的**：本会话为救 deploy 预拉了 3 个 base 打本地 tag（见 §5），**清缓存/换机器会失效**。长期修复（换 daemon.json 镜像源）用户选了"先不改"。

---

## 2. 本会话 5 处修复（7 个 commit，全在本地 dev，测试全绿）

| commit | 问题 | 根因 | 修复 |
|---|---|---|---|
| `ec92beb7` | SealGate 无限弹回 | reminder 计数键 `id(runtime)` 每个 after_model 调用都变 → 计数永不累加 → cap 永不触发 | 改 per-instance int（executor 每 run 新建实例，天然 per-run） |
| `9b467fe5` | data-analyst 反复 seal、9 轮空转 11 分钟 | thinking 每轮重演整套判读，与 seal arguments 共享 4096 输出预算 → arguments 腰斩 → tool_call 悬空 → 弹回重演 | model `inherit(v4-pro)` → `deepseek-v4-pro-summary`（无 thinking）；判读判据全在 skill+旁路文件，thinking 纯属浪费 |
| `272b6fdc` | box plot 箱线图 "No data" | dispatcher group_summary 按 subject-name 匹配文件路径 groups → 零交集 → group_summary={} | 加 `file_subjects` 桥接（parse_batch 已返回），第 11 个结构缺口（statistics 路径当初修了、box plot 直调这条路没接） |
| `4ffe7cea` | compute 慢（前几次快） | **真回归**：run_metric_plan 工具化（6-15 commit 63793162）时 `MAX_WORKERS=4` 把 bash-`&` 时代的"≈可用核并行"砍到 4 | `MAX_WORKERS = _available_cpus()`（`sched_getaffinity` 尊重 cgroup，dev 8vCPU→8，不踩容器 cpu_count 高报陷阱，不封顶让 elastic 吃满） |
| `8cdb28b8` | 并行派遣崩 `InvalidUpdateError: At key 'sandbox'` | `SandboxMiddlewareState.sandbox` 裸 LastValue 无 reducer，仅靠 `SandboxAuditMiddleware` 的 ThreadState reducer **碰巧救回**；并行派遣触发一步两写 | 给 `SandboxMiddlewareState.sandbox` 挂 `merge_sandbox`，channel 恒为 BinaryOp 不依赖偶然 |

+ `e67ed87d`（原 spec 执行版）+ `547c0d04`（spec 撤回标注同步，见 §3）。

**验证**：后端全量 4214 passed（仅 3 个**已知 local_sandbox 隔离污染**，与改动零交集）；ethoinsight 全量 938 passed；裸导入两生产入口 0 退出；box plot 修复在第四轮 dogfood thread `fb3ed752` 端到端坐实（箱线图有数据，本 agent 目视确认）。

### ⚠️ sandbox 修复的诚实边界（值得记进 memory）
sandbox 的**精确运行时触发机制未 100% 复现**——4 个简化复现（单 subagent / 并发 gather / 共享 sandbox dict / SandboxMiddleware-only）都不报错。但 git stash 实证：修复前 `SandboxMiddlewareState`-only graph 的 sandbox channel = LastValue、修复后 = BinaryOperatorAggregate。reducer 修复是**结构保证**（channel 不再靠偶然救），也是 LangGraph 报错本身的建议。因无法构造真 red→green 行为测试，只保留**确定性结构守卫**（`test_sandbox_middleware_state_sandbox_has_reducer`，注解带 reducer）——删掉了会被全量测试前序状态污染的 fragile channel-type 字符串断言测试。

---

## 3. open_arm_time_ratio spec —— 写了 → 实施 agent 撤回 → review 闭环

- 用户口头裁定「`open_arm_time_ratio` 改成 open/(open+close) 排除中央区」，本 agent 按"SSOT 级裁决走 spec、不直接改代码"写了 `docs/superpowers/specs/2026-06-18-epm-open-arm-time-ratio-denominator-spec.md`，**前置标注 3 个待确认项**（§3.2 无闭臂列降级 / §3.3 跨范式一致性 / 历史结果重算）。
- 实施 agent（曲若衡专家在场 + Sonnet 4.6）实施时**专家多轮更正后改口**：分母=总帧（含中央区），即**当前实现 `combined.mean()` 是对的，无需改**。否决理由正是本 spec §3.3 标注的跨范式风险（OFT 只有 center 列、数据拓扑不同）。
- 实施分支 `worktree-epm-open-arm-ratio-denominator` 做法正确：**代码零改动**，只给 spec 加 `⛔ 撤回标注`，PR#162 已合 origin/dev。
- 本 agent review 判定**无需改**，并把撤回标注同步回 dev（`547c0d04`），消除分叉。

> **教训（值得记 memory）**：坚持"SSOT 级公式裁决走 spec + 前置标注跨范式风险 + 不按口头裁定直接改代码"的纪律，**正好避免了一次错误的 SSOT 改动**——专家最初说改、实施时改回保持现状，若当时直接改代码现在就要回滚 + 重算所有历史结果 + 改 catalog/skill。

---

## 4. 第四轮 dogfood 验证了哪些修复（thread `fb3ed752`，跑修复后 dev）

- ✅ **data-analyst 换 flash**：一次完成判读+seal，无 9 轮空转。
- ✅ **box plot 桥接**：箱线图正常显示 control vs treatment（目视确认，84KB vs No data 版 56KB）。
- ✅ **sandbox**：这次 lead 走串行（E2E_FULL_ASKVIZ 先 data 后问 viz），未触发并行，但 reducer 修复已就位防未来。
- compute MAX_WORKERS / SealGate：无对照数据，单测+逻辑已验证。

**一个 review 时该盯的小瑕疵（非本轮范围）**：data-analyst gate_signals `outlier_count: 7`，但摘要/handoff 实际只详述 2-3 条精选离群——计数与 findings 数组可能不一致，或把旁路 65 条压成 7 条。判读质量小问题，下轮可查。

---

## 5. docker 部署镜像源救场（deploy 卡 base 镜像）

- **现象**：`make deploy-tar` 反复卡在拉 base 镜像，`short read: expected N bytes but got 0: unexpected EOF`，先 `python:3.12-slim-bookworm` 再 `docker:cli`。
- **根因**：daemon.json 的镜像源 `docker.xuanyuan.me` **限流 HTTP 429**，`docker.1ms.run` 慢/超时。逐个 curl 探针不可靠（`/v2/` 通≠能拉具体镜像，dockerproxy.net 探针 200 但拉 manifest 500）。
- **实测唯一稳定源**：`docker.m.daocloud.io`。
- **已做（非侵入，未改 daemon、未重启 docker）**：从 daocloud 预拉 gateway Dockerfile 的 3 个 base 并打本地 tag：`python:3.12-slim-bookworm`、`docker:cli`、`ghcr.io/astral-sh/uv:0.7.20`（ghcr 直接拉成功）。buildx 实测三个 base 全命中本地缓存、零远程拉。**重跑 `make deploy-tar` 不再卡这三个 base**。
- **长期修复（用户选"先不改"）**：把 daemon.json `registry-mirrors` 的限流 `docker.xuanyuan.me` 换成 `docker.m.daocloud.io`，需 sudo + 重启 docker（中断运行中容器）。
- **docker 位置**：用户已把 data-root 改到 `/data/docker-lib`（daemon.json 已是这个，正常）。

---

## 6. 关键文件指针

- **5 修复改动**：`agents/middlewares/seal_gate_middleware.py`、`subagents/builtins/data_analyst.py`、`sandbox/middleware.py`、`tools/builtins/run_metric_plan_tool.py`（以上 backend harness）、`packages/ethoinsight/ethoinsight/metrics/dispatcher.py`
- **新测试**：`tests/test_dispatcher_group_summary_file_bridge.py`（新建）+ 改 `test_seal_gate_middleware.py` / `test_data_analyst_thinking_enabled.py` / `test_sandbox_middleware.py` / `test_groups_subject_bridge.py`
- **spec（撤回）**：`docs/superpowers/specs/2026-06-18-epm-open-arm-time-ratio-denominator-spec.md`（顶部 ⛔ 撤回标注，原文存档）

---

## 7. milestone 建议

「harness 鲁棒性 / dogfood 根因治理」track 再到 checkpoint：第四轮复跑暴露 5 问题，全部诊断+修复/裁决：
- **新结构根因 ×2**：① data-analyst thinking 重演撞 4096 输出预算（换 flash，与第三轮的 thinking-overload 同源但触发点不同：第三轮撞 50K/900s、第四轮撞 4096 tool_call 腰斩）；② sandbox channel reducer 靠偶然救回（并行派遣暴露）。
- **真回归 ×1**：run_metric_plan 工具化把 compute 并行度从"≈可用核"砍到 4。
- **结构缺口 ×1**：box plot file→subject 桥接（第 11 个，statistics 路径修过、box plot 没接）。
- **SSOT 裁决 ×1**：open_arm_time_ratio 公式，走 spec 避免错误改动。

建议记录：① **第 5/6 类 seal-relevant 根因边界**（thinking 与 tool_call arguments 共享单次响应输出预算；判读类 subagent 用非 thinking 模型）；② **MAX_WORKERS 容器陷阱**（`os.cpu_count()` 容器内高报宿主机核数，并行度用 `sched_getaffinity`）；③ **SSOT 公式裁决纪律的价值**（走 spec + 前置跨范式标注，避免按口头裁定误改）；④ **docker 镜像源救场手法**（预拉打本地 tag，非侵入）。
