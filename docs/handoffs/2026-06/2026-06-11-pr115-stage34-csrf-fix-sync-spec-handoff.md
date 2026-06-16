# 交接文档 — 2026-06-11 PR#115 全 4 stage 合入 + 线上 CSRF 403 修复 + 上游 sync spec

> 写给下一位 AI Agent。本次会话完成三件事：①review PR#115 Stage 3/4 并确认已合 dev（四 stage 全闭环）②诊断并修复线上 dev 的 CSRF 403（已 commit+push，**待建 PR**）③写好上游 sync spec（待派 agent 执行）。

---

## 一句话现状

- **PR#115 catalog 概念整合 4 个 stage 全部合入 dev**（dev HEAD `036223da`）。Stage 3（PR#120）、Stage 4（PR#121）本次会话 review 通过，确认已合。**无遗留。**
- **线上 CSRF 403 已修**：根因=csrf_token 是 session cookie 而 access_token 持久（生命周期不对称）。修复在分支 `fix/csrf-cookie-lifetime-asymmetry`（`8f8444d4`，已 push），**待建 PR 合 dev + 部署**。
- **上游 sync spec 已就绪**：`docs/superpowers/specs/2026-06-11-deerflow-sync-f92a26d5-to-2d5f0787-spec.md`，11 个 commit 已分类，**待派 agent 执行**。

---

## 本次会话做了什么

### 1. PR#115 Stage 3 review（已合 dev，PR#120）
- 分支 `worktree-pr115-stage3-complement-zone-concepts`（`b07bdfb7`）→ review 通过 → 已合 dev。
- 内容：补 OFT `border`(binding=None) / LDB `dark`(ParamBinding `dark_zone`) / ZM `closed`(ParamBinding `closed_zones`) 进 `resolved_zone_concepts`。
- **review 验收**：7/7 闸门全过；golden 是真 baseline（dev 上重跑 4 值逐字节相等）；**三个负控全转红**（删 border 填充→存在性红；删 `binding is None: continue`→border-alias 护栏两条红；篡改 golden→等价性红）。789 passed 0 failed。

### 2. PR#115 Stage 4 review（已合 dev，PR#121）
- 分支 `feature/pr115-stage4-generate-concept-menu`（`a5ec778a`）→ review 通过 → 已合 dev。
- 内容：新增 `concept_menu.py` 生成器从 `resolved_zone_concepts` 单点生成两份 `.generated.md`；SKILL.md/answer-mapping.md 删手写表改链接；Makefile 加 `gen-references`。消除「catalog vs skill md」双存（EPM center/OFT corner 该删项已删）。
- **review 验收**（临时 worktree `/tmp/stage4-review-wt` 实跑，已清理）：library 11 passed + harness staleness 9 passed + **重新生成逐字节==已提交** + resolve/catalog 264 passed 零回归 + 裸导入 OK。
- **两个负控全转红**：手改 .generated.md→staleness 红；篡改 oft.yaml 加回 corner→覆盖度+staleness 同时红。staleness+覆盖度双层真承重。
- **PR#115 至此四 stage 全部 review 通过 + 全部合 dev。**

### 3. 线上 CSRF 403 诊断 + 修复（⭐ 重点，待建 PR）
- **现象**：线上 `dev.ethoinsight.com` POST `/api/langgraph/threads/search` + `/threads/{id}/history` 返回 403 `{"detail":"CSRF token missing"}`。用户线索"用一阵后才 403"。
- **真根因（curl 三段实证，非推测）**：`csrf_token` cookie 是 **session cookie**（CSRFMiddleware set 时无 max_age），而 `access_token` 持久（`token_expiry_days`）。生命周期不对称→关标签页/浏览器回收 session cookie 后 access_token 仍在（仍登录），前端 `readCsrfCookie()` 返回 null→POST 不带 `X-CSRF-Token`→403。
  - 实证链：①GET 无 cookie→401（证 auth 开启，**非 auth-disabled**）②POST 无 CSRF→403 ③POST 带任意 csrf cookie+header→403 变 401（证 CSRF 中间件正常只差 token）④线上 set-cookie 头确认 csrf_token **无 Max-Age**=session cookie。
  - **差点误判**：上游 `2b795265 fix: align auth-disabled mode and mock history loading` 改了 csrf_middleware（auth-disabled 跳过 CSRF），标题"mock history loading"很像→差点直接拉上游。被①证伪。详见 MEMORY [[feedback_csrf_403_root_cause_is_session_cookie_lifetime_asymmetry]]。
- **修复**（`packages/agent/backend/app/gateway/csrf_middleware.py`）：抽 `_set_csrf_cookie(response, request)` helper 收口两处 set（auth POST + GET 补发），加 `max_age=token_expiry_days*24*3600 if is_https else None`（与 `routers/auth.py:_set_session_cookie` 的 access_token 对齐）+ `samesite` strict→lax。
- **TDD**：新增 `tests/test_csrf_cookie_lifecycle.py`（5 测试：持久化/lax/secure/http-session/js可读）。负控验证 strict+无max_age 时转红。63 passed + 裸导入两入口无环。
- **状态**：commit `8f8444d4` 在分支 `fix/csrf-cookie-lifetime-asymmetry`，**已 push origin，待建 PR 合 dev**。
- **⚠️ 教训（本次踩了 2 次）**：这个修复在会话中途丢了两次——第一次是负控 `cp /tmp/csrf.bak` 还原序列覆盖、第二次是 PR#121 merge 时 `git checkout` 把未 commit 的工作区改动丢弃。**未 commit 的工作区改动在切分支/merge 时会丢**。这次已立即 commit+push 钉死。

### 4. 上游 sync spec（待派 agent）
- `docs/superpowers/specs/2026-06-11-deerflow-sync-f92a26d5-to-2d5f0787-spec.md`
- 基准 `f92a26d5`（2026-06-09）→ 上游 HEAD `2d5f0787`，11 个 commit，逐个 head-to-head 评估完。

---

## 下一位 Agent 的待办（按优先级）

### P0 — 建 CSRF 修复 PR + 部署
```
分支 fix/csrf-cookie-lifetime-asymmetry (8f8444d4) → PR to dev → 合 → make deploy-tar 部署
```
- **部署后用户需重新登录一次**（拿到新的持久 csrf cookie），之后不再复发——已"半失效"的用户当前 session 不会自动恢复。
- 验证：部署后线上登录，等一会/重开标签页再 POST（如新建对话/搜索），不再 403。
- 也可顺手用 curl 验：`curl -sS -D - .../api/models | grep -i set-cookie` 看 csrf_token 是否带 `Max-Age=` 和 `SameSite=lax`。

### P1 — 派 agent 执行上游 sync
- spec：`docs/superpowers/specs/2026-06-11-deerflow-sync-f92a26d5-to-2d5f0787-spec.md`
- **7 个 🟢 纯安全全量合**（高价值：`8db16bb3` null-list 修复正好修裸导入时见的报错 / `b62c5a7b` event-loop 卸载阻塞IO / `ba9cc5e9` thread 越权防护）
- **2 个 🟡 surgical**：`0fb18e36`（纯机械重命名 `_build_middlewares`→公开，需同步 client.py + 5 测试）、`167ef451`（memory token_counting 离线部署相关，memory/prompt.py 须 head-to-head 保住 Noldus 67 行定制）
- **1 个 🟡 默认跳过待拍板**：`16391e35`（IM channel slash skill 激活，体量大触受保护多、与 web 主路径弱相关）。**spec §7 决策表里这是唯一开放决策**——需用户确认要不要 IM slash 能力。

### P2 — milestone 更新（可选）
- PR#115 catalog 概念整合 track 已闭环（四 stage 全合 dev）。若 `docs/milestone/` 下有对应 track（column-semantics-alignment 或 catalog 概念整合），建议更新：四 stage 全部实现+review+合入；两决策门已由 Fable 闭合。

---

## 关键上下文

### dev 当前状态
- HEAD `036223da`，PR#115 四 stage 全合。工作区干净（只有未跟踪的 docs + CSRF 测试文件，但测试已随 fix 分支 commit）。
- **ethoinsight 测试用 `uv run pytest`**（venv 在 `packages/ethoinsight/.venv`），无裸 `python`。
- **backend 测试用 `source .venv/bin/activate && PYTHONPATH=. python -m pytest`** 或 `make test`。

### CSRF 修复关键文件
- 后端：`packages/agent/backend/app/gateway/csrf_middleware.py`（`_set_csrf_cookie` helper）
- 对齐基准：`packages/agent/backend/app/gateway/routers/auth.py:_set_session_cookie`（access_token 的 max_age + samesite=lax）
- 前端 CSRF 注入（无需改，已正确）：`frontend/src/core/api/fetcher.ts:readCsrfCookie` + `api-client.ts:injectCsrfHeader`
- 测试：`packages/agent/backend/tests/test_csrf_cookie_lifecycle.py`

### 部署形态（CSRF 修复部署相关）
- Gateway-embedded 模式，`make deploy-tar`（本地 build→镜像 tar→ECS docker compose）。
- 线上 ECS：1Panel + OpenResty 443→127.0.0.1:2026 nginx→gateway。CSRF 中间件依赖请求被识别为 https（经 `x-forwarded-proto`）才设 Secure + 持久 cookie——反代层 header 传递正常（curl 实证线上确实 set 了 Secure cookie）。

---

## 风险与注意事项

1. **未 commit 的工作区改动会在切分支/merge 时丢**（本次 CSRF 修复丢了 2 次的教训）。生产修复改完**立即 commit**，别留在工作区。
2. **CSRF 不是上游能拉来修的**——已证伪上游 `2b795265`（线上 auth 开启，auth-disabled 路径不触发）。别在 sync 时误以为拉了上游 CSRF commit 就修了 403。
3. **sync spec 的 16391e35 默认跳过**——它是唯一需用户拍板的 commit。其余 10 个判断已闭合。
4. **受保护文件 sync 必 head-to-head**（绝不整文件覆盖）：`memory/prompt.py`、`app_config.py`、`lead_agent/agent.py`、`lead_agent/prompt.py` 都含 Noldus 定制，2026-06-09 sync 曾被 full-follow 洗掉过（MEMORY [[feedback_sync_nonprotected_files_with_noldus_customization_overwritten]]）。
5. **改 harness 核心后必裸导入两入口**（conftest mock 藏循环导入假绿，CLAUDE.md 铁律）：`PYTHONPATH=. python -c "import app.gateway"` + `python -c "from deerflow.agents import make_lead_agent"`。

---

## 下一位 Agent 的第一步建议

1. 读本文档 + CLAUDE.md「同步核心规则」段。
2. **先做 P0**：`git checkout fix/csrf-cookie-lifetime-asymmetry` → 确认 `8f8444d4` → 用 `gh` 建 PR 到 dev（标题/body 参考 commit message）。问用户是否要立即部署。
3. **再做 P1**：读 sync spec，问用户 16391e35（IM slash skill）的取舍，然后按 spec §3 批次顺序执行（或派专门 agent）。
4. 全程：ethoinsight 用 `uv run`，backend 改完裸导入验证。

---

## milestone 建议

PR#115 catalog 概念整合 track 到达**完成 checkpoint**（四 stage 全实现+review+合 dev，两决策门 Fable 闭合）。若有对应 milestone，更新为：Q1 门控 CNF（Stage1）+ Q3 概念统一模型（Stage2）+ 补集区枚举（Stage3）+ 概念菜单生成消双存（Stage4）全部落地；catalog 现为 zone 概念的完整 SSOT。
