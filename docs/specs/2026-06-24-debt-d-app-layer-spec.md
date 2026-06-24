# 债 D：app 层三类缺口（独立可执行 spec）

> **母 spec**：`docs/specs/2026-06-24-deerflow-sync-historical-debt.md` 的「债 D」小节（§D.1–D.4）。本文自包含、可直接交给 agent 执行。
>
> **解锁目标**：采用上游 `tests/test_gateway_services.py`（884/346 行，+34 测试）、`tests/test_uploads_router.py`（766/640 行，+7 测试）、`tests/test_feishu_parser.py`（502/436 行，+3 测试）并通过。
>
> **规模**：中。改 5 个 app 层文件，**无 harness 受保护文件**。可拆 3 个小 PR（D1/D2/D3），也可合一个。
>
> **关键**：`sync-deerflow.sh` 不覆盖 `app/`，本债需**手动同步**上游改动到 app 层。

---

## 〇、给实施 agent 的一句话

把 app 层三类上游接口缺口补齐——D1 sandbox readable、D2 Feishu `user_id` 显式传参、D3 internal auth + checkpoint regenerate——让本地 app 层追平上游，解锁 3 个回退测试文件。

---

## ⚠️ 给执行 agent 的前置须知（务必先读）

1. **路径前缀**：本地 app 层是 `packages/agent/backend/app/`；上游对应去掉 `packages/agent/`，用 `git show deerflow/main:backend/app/<path>` 读取。
2. **commit SHA 不可信，一律以「符号/签名 diff」为准**。别 `git cherry-pick`，看 `git show deerflow/main:<file>` 当前态。
3. **app 层无导入环高危**（harness→app 防火墙，`tests/test_harness_boundary.py` 守），但改完仍建议裸导入确认：`PYTHONPATH=packages/harness:. python -c "import app.gateway"`。
4. **本地已知 baseline 污染**：当前 baseline = **18 红**（test isolation 污染 + 需 live server 的 test_client_live）。开工前记 baseline，判「新红」用对比数字。
5. **测试解释器 + env**（worktree 必带 config env 否则假红）：
   ```bash
   cd packages/agent/backend
   PYTHONPATH=packages/harness:. DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml \
     .venv/bin/python -m pytest tests/test_gateway_services.py tests/test_uploads_router.py tests/test_feishu_parser.py -q
   ```
6. **app 层有 Noldus 定制吗？先 diff**：5 个文件每个先 `diff <(git show deerflow/main:backend/app/<path>) <local>`，识别上游「真正修复」（几行 try/except / 几行参数 / 几行边界检查），手工搬入，保留 Noldus 定制（如有）。

---

## D.1：sandbox readable（`app/gateway/routers/uploads.py`）

### 现状（2026-06-24 已确认）
- 本地有 `_make_file_sandbox_writable`（行 82），**缺** `_make_file_sandbox_readable`。
- 上游有 `_make_file_sandbox_readable`（行 101，加 `S_IRGRP|S_IROTH`），在 sync 路径行 331 调 readable、行 335 调 writable。
- 本地 sync 路径：行 221 `sync_to_sandbox = not _uses_thread_data_mounts(...)`，行 253/273/295 条件分支，行 297 调 `_make_file_sandbox_writable`。

### 步骤
1. 照上游搬入 `_make_file_sandbox_readable`（`git show deerflow/main:backend/app/gateway/routers/uploads.py` 看 body）。
2. **先 diff 本地 vs 上游 sync 路径的循环结构**——本地循环可能与上游不完全一致。在本地对应的文件写入/同步路径，**按上游时序**补 `_make_file_sandbox_readable(file_path)` 调用（本地行 297 `writable` 调用隔壁补 readable）。对齐**调用时序**而非照抄行号。
3. 验收：`git show deerflow/main:backend/tests/test_uploads_router.py > tests/test_uploads_router.py`（上游 766 / 本地 640，含 `test_make_file_sandbox_readable_*`）→ pytest。

---

## D.2：Feishu `user_id` 显式传参（`app/channels/feishu.py` + `app/channels/manager.py`）

### 现状 + 裁决（已调研，结论：**采用上游传参**，非保留本地）
- 本地 `receive_file`（行 311）/`_receive_single_file`（行 334）**无 `user_id` 参数**，内部用 `get_effective_user_id()`（行 373）解析。
- 上游加 `*, user_id: str | None = None`（`receive_file` 行 314、`_receive_single_file` 行 344），body 改 `effective_user_id = user_id or get_effective_user_id()`（行 384），下游 `ensure_thread_dirs` / `sandbox_uploads_dir` / `sandbox_provider.acquire` 全用 `effective_user_id`（行 385/386/415）。
- 上游 `manager.py` 新增 resolver 链：`_effective_owner_user_id`（500）→ `_safe_user_id_for_run`（518）→ `_channel_storage_user_id`（528），调用处 `channel.receive_file(msg, thread_id, user_id=storage_user_id)`（行 1275）。

### 为什么必须采用上游传参（不是冗余）
上游 commit 文档明确这修一个真实 bug：dispatcher 线程的 contextvar 未设时 `get_effective_user_id()` 退化成 `"default"`，文件落进 `users/default/...` 而 agent 从 `users/{真实}/...` 读，**文件对不上**。`user_id` 参数携带的是「消息所属/channel owner 的 user」，与 contextvar 不同源。

### 步骤
1. `feishu.py`：`receive_file` / `_receive_single_file` 加 `*, user_id: str | None = None`（keyword-only，默认 None，向后兼容）。
2. `feishu.py`：body 行 373 `user_id = get_effective_user_id()` 改 `effective_user_id = user_id or get_effective_user_id()`，下游 `ensure_thread_dirs` / `sandbox_uploads_dir` / `sandbox_provider.acquire` 用 `effective_user_id`。
3. `manager.py`：照上游搬 `_channel_storage_user_id(msg)` + 依赖 `_effective_owner_user_id` / `_safe_user_id_for_run`（`git show deerflow/main:backend/app/channels/manager.py` 看 500–552）。调用处 `channel.receive_file(msg, thread_id, user_id=storage_user_id)`（对齐上游行 1275 / 1249 `run_user_id`）。
4. **若 Noldus 当前没有 per-user IM ownership 映射**（`_effective_owner_user_id` 返 None）：`_channel_storage_user_id` 先退化为 `_safe_user_id_for_run(msg.user_id)`（无 owner 分支），保证不退化成 `"default"`。**先 grep 确认 Noldus 有无 ownership 映射机制**，别假设。
5. 验收：`git show deerflow/main:backend/tests/test_feishu_parser.py > tests/test_feishu_parser.py`（上游 502 / 本地 436，含 `test_feishu_receive_file_syncs_sandbox_with_explicit_user_id`）→ pytest。

---

## D.3：internal auth + checkpoint regenerate（`app/gateway/internal_auth.py` + `app/gateway/services.py`）

### 现状（已确认本地全缺，`INTERNAL_SYSTEM_ROLE` 已有无需新增）
| 符号 | 本地 | 上游 |
|---|---|---|
| `INTERNAL_OWNER_USER_ID_HEADER_NAME` | **无** | `internal_auth.py:13` = `"X-DeerFlow-Owner-User-Id"` |
| `INTERNAL_SYSTEM_ROLE` | **已有**（`internal_auth.py:13`=`"internal"`，`services.py:22` 已消费）| `internal_auth.py:15` |
| `get_trusted_internal_owner_user_id(request)` | **无** | `internal_auth.py:46` |
| `create_internal_auth_headers` | `()`（行 26，无参） | `(*, owner_user_id: str \| None = None)`（行 28） |
| `apply_checkpoint_to_run_config` | **无** | `services.py:292`（async），调用点 `services.py:467` |

### 步骤
1. `internal_auth.py`：照上游搬 `INTERNAL_OWNER_USER_ID_HEADER_NAME` 常量 + `get_trusted_internal_owner_user_id(request)`（行 46，校验 `INTERNAL_SYSTEM_ROLE` + 读 header）+ 给 `create_internal_auth_headers` 加 `*, owner_user_id=None`（行 28，非 None 时塞 header）。
2. `services.py`：照上游搬 `apply_checkpoint_to_run_config`（async，行 292）+ 在 start-run 路径（上游行 467）补 `await apply_checkpoint_to_run_config(config, body=body, thread_id=thread_id, request=request)`。**先 diff 本地 `services.py` 该函数上下文**——本地 start-run 可能有 Noldus 定制，对齐插入点而非照抄行号。
3. **grep `create_internal_auth_headers(` 所有调用点**，确认加 `owner_user_id` 默认值后不破现有调用（默认 None 向后兼容）。
4. 验收：`git show deerflow/main:backend/tests/test_gateway_services.py > tests/test_gateway_services.py`（上游 884 / 本地 346，含 `test_apply_checkpoint_to_run_config_*`、`test_start_run_uses_internal_owner_header_*`）→ pytest。

---

## 四、测试（TDD，红→绿）

每个 D 子债采用上游对应测试，跑通：

```bash
cd packages/agent/backend
CONFIG=DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml
for f in test_uploads_router test_feishu_parser test_gateway_services; do
  git show deerflow/main:backend/tests/$f.py > tests/$f.py
  env $CONFIG PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/$f.py -q
done
```

**采用上游测试的统一手法**：跑通后若个别断言依赖 Noldus **没有**的上游专属行为（如 Noldus 没接的 ownership 映射、特定的 checkpoint 持久化后端），surgical 删该断言/测试并 **PR 注明**（守「保留本地定制」铁律，不为迁就上游测试改 Noldus 业务）。

---

## 五、验收标准（债 D 整体）

1. ✅ `PYTHONPATH=packages/harness:. python -c "import app.gateway"` exit 0（app 层改动也要确认 Gateway 可导入）。
2. ✅ `test_uploads_router.py` / `test_feishu_parser.py` / `test_gateway_services.py` 采用上游版后全绿（或 surgical 注明跳过的断言）。
3. ✅ `make test` 全量 ≤ 开工 baseline 红数（无新增红）。
4. ⏳ 可选 manual：D.2 改完若走真实飞书 webhook 路径，确认文件落进 `users/{真实}/` 而非 `users/default/`。

---

## 六、风险与注意事项

1. **app 层有 Noldus 定制**：D.2 feishu/manager、D.3 services 的 start-run 路径可能有 Noldus 定制（IM 接入、降级熔断挂 lead 链等）。**绝不整文件覆盖**，先 diff 识别上游修复点，手工搬入保留定制。
2. **D.2 ownership 映射前提未定**：上游 `_channel_storage_user_id` 依赖 `_effective_owner_user_id`（per-user IM ownership）。**先 grep 确认 Noldus 有无此机制**；无则退化分支（`_safe_user_id_for_run(msg.user_id)`），别凭空假设。
3. **D.3 `create_internal_auth_headers` 是共享函数**：加 `owner_user_id` 参数前 grep 所有调用点（`grep -rn 'create_internal_auth_headers(' app/`），确认默认值 None 向后兼容。
4. **`apply_checkpoint_to_run_config` 依赖持久化层**：本仓库已吃 Tier 4（persistence 齐全），可跟随上游。若该函数依赖 Noldus 确实没有的 checkpoint 后端，surgical 隔离 + 记录。
5. **与债 A/B/C 的关系**：纯 app 层改动，**零文件重叠、零接口依赖，完全可并行**。

---

## 附：核验命令速查（开工前复跑，防代码漂移）

```bash
cd /home/wangqiuyang/noldus-insight
A=packages/agent/backend/app
# D.1
grep -nE '_make_file_sandbox_readable|_make_file_sandbox_writable' $A/gateway/routers/uploads.py
git show deerflow/main:backend/app/gateway/routers/uploads.py | sed -n '101,115p;325,340p'
# D.2
grep -nE 'def receive_file|def _receive_single_file|get_effective_user_id|effective_user_id' $A/channels/feishu.py
grep -nE '_channel_storage_user_id|_effective_owner_user_id|_safe_user_id_for_run' $A/channels/manager.py
git show deerflow/main:backend/app/channels/manager.py | sed -n '500,555p'
# D.3
grep -nE 'INTERNAL_OWNER_USER_ID_HEADER_NAME|get_trusted_internal_owner_user_id|apply_checkpoint_to_run_config' $A/gateway/internal_auth.py $A/gateway/services.py
git show deerflow/main:backend/app/gateway/internal_auth.py | sed -n '10,62p'
git show deerflow/main:backend/app/gateway/services.py | sed -n '292,330p;460,470p'
```
