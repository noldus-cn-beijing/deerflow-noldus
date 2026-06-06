# 2026-06-04 文件上传扩容 + zone override 三范式统一 交接文档

> **面向下一个 AI Agent**。本会话完成两个独立 feature（文件上传扩容、zone override 三范式统一修复 review+merge），**核心未完成事项 = O迷宫 dogfood 实跑验证**。仓库当前 HEAD = `69bc2ca5`（origin/dev 同步）。

---

## 1. 当前任务目标

本会话处理了两件独立的事：

1. **文件上传扩容**（已完成并合入 dev）：用户对 EthoVision 文件有"几十份"批量需求，原 `max_files=10` / `max_total_size=100MiB` 不够。
2. **zone override 三范式统一修复**（已 review + merge，**dogfood 待跑**）：2026-06-03 O迷宫 dogfood 全链路失败的根治——把 OFT 的"匿名 in_zone 区 → 用户声明 → override 放行"机制泛化到 OFT/zero_maze/LDB 三范式，且统一成单一 agent 接口 key。

---

## 2. 当前进展

### ✅ Feature A：文件上传扩容（commit `02c0b9e1`，已在 origin/dev）
- `config.yaml` + `config.example.yaml`：`max_files 10→50`、`max_total_size 100→500 MiB`（524288000），单份 50 MiB 不变。
- **prod 配置**：`~/ethoinsight-prod/config.yaml`（不在 git）**原本没有 `uploads:` 段**（一直走硬编码默认 10/100MiB）——已**新增**整个 uploads 段。⚠️ **线上生效仍需 `cd packages/agent && make deploy-tar`**（未执行）。
- 前端 `input-box.tsx`：给 `PromptInput` 传 `maxFiles={50}` + `onError` sonner toast；i18n 加 `inputBox.tooManyFiles`（en/zh，`{max}` 占位）。
- 验证：后端 `test_uploads_router.py` 28 passed；前端 typecheck 干净、lint 零新增（4 个 error 全是 dev 预存在，stash 对比已证）。
- **已知双存点**：前端 `MAX_UPLOAD_FILES=50` 是后端 config 的镜像值，改 `uploads.max_files` 时两处要同步（已在代码注释 + commit message 标注）。

### ✅ Feature B：zone override 三范式统一（PR #89 = commit `25eaebfc`，已合 dev）
- **设计 spec**（自包含，可执行）：`docs/superpowers/specs/2026-06-04-zone-override-unification-three-paradigm-design.md`（本会话产出，**untracked 待提交**）。
- **核心机制**：agent 对三范式只写**统一 key** `parameter_overrides={"anonymous_zone_is": "in_zone"}`，后端按 catalog 元字段 `anonymous_zone_override: {target_param, wrap_list}` 翻译成各范式真实参数：
  - OFT → `center_zone="in_zone"`（str）
  - zero_maze → `open_zones=["in_zone"]`（**list**，wrap_list=true）
  - LDB → `light_zone="in_zone"`（str）
- **改动 11 文件**：schema.py（`AnonymousZoneOverride` dataclass）/ loader.py（解析元字段 + `_parse_param_spec` default 放宽到含 list）/ resolve.py（`_detect_anonymous_zone` 范式无关化 + `_compute_parameters_in_use` 统一 key 翻译）/ oft+zero_maze+ldb.yaml（per-metric zone 参数 + 顶层元字段）/ inspect_uploaded_file_tool.py（`_compute_anonymous_zone_evidence` 注入 txt/xlsx/csv 三路径）/ prep_metric_plan_tool.py（zone_unnamed hint 改统一 key + 正面措辞）/ experiment_context.py（docstring）/ 2 测试文件。
- **review 已亲自核实（非采信实施总结）**：
  - ✅ 端到端实跑真实 O迷宫列结构：裸 in_zone→zone_unnamed；统一 key→`open_zones=['in_zone']` **list**（含 `--parameters-json` 落盘）；OFT 同 key→`center_zone='in_zone'` **str**。
  - ✅ ethoinsight 全量 **598 passed, 69 skipped**（worktree 代码自测）。
  - ✅ 不误伤：EPM 命名列齐全正常出 5 metrics，无 zone_unnamed 误判。
  - ✅ inspect 三路径都注入 `anonymous_zone_evidence` + try/except 降级。
  - ✅ rebase 干净（实施初版 base 落后 dev 138 commit，已 rebase 到 dev HEAD 后 merge）。

---

## 3. 关键发现（已验证，下一个 agent 必读）

1. **🔴 几何实测：O迷宫 `in_zone=1` = 开放臂，不是封闭区**。三组占时 10.7%/14.2%/6.4%（少占时=动物回避的开放臂），角度上占两段相对弧。**2026-06-03 dogfood 中 deepseek 在死循环里诱导用户二选一，用户答了"封闭区"（答反了）。** 这次修复**不需要"取反"**——三范式值语义统一 1=在目标区内。
2. **dogfood 失败真根因**：OFT 的修复（PR-1 的 center_zone）没泛化 + zero_maze/LDB catalog 没声明 zone 参数 → `_compute_parameters_in_use` replace-only 把 LLM 写的 override 静默丢弃 → LLM 脑补 `~in_zone` 取反语法 → 死循环 → 撞 read_file 5 次安全限 FORCED STOP。失败 thread：`.deer-flow/users/cd95effa-.../threads/6dea6ae9-.../`（experiment-context.json 存有脑补的 `open_zone:"~in_zone"`，留作回放）。
3. **zero_maze 用 list 参数 `open_zones`，OFT/LDB 用 str**——这个不一致被后端 `wrap_list` 元字段吸收，**agent 永不感知 list**。这是用户明确拍板的"统一 key、零困扰 agent"方案。
4. **backend 改了工具文件 → dogfood 前必须 `make stop && make dev` 重启**（tool 定义在 agent 创建时构建，不重启跑的是旧代码）。

---

## 4. 未完成事项（按优先级）

### 🔴 高优先级：O迷宫 dogfood 实跑（本会话核心遗留）
单元层已全绿，但 **LLM 行为层未验**——lead 会不会真写统一 key、inspect 吐的 evidence lead 会不会用、反问话术会不会再次诱导用户答错。这些只有真实跑 agent 才看得出。
- **必跑**：O迷宫（`/home/wangqiuyang/DemoData/newdemodata/O迷宫/`）——本次根因 + 最复杂 case。
- **推荐补**：明暗箱（LDB，验 str 路径）+ 高架十字迷宫（EPM，验不误伤对照）。
- 验收点（spec §5.3）：inspect 吐 `anonymous_zone_evidence`（in_zone=1 占时 6-14%）→ prep 报 zone_unnamed → lead 带证据反问 → 用户确认开放臂 → 写 `{"anonymous_zone_is":"in_zone"}` → 翻译成 `open_zones=["in_zone"]` → compute 出真值（open_zone_time_ratio≈0.10）→ 走完 data/chart/report。

### 🟡 中优先级
- **文件上传 prod 生效**：`~/ethoinsight-prod/config.yaml` 已改好，需 `cd packages/agent && make deploy-tar` 才线上生效（未执行）。
- **提交本会话产出的 spec**：`docs/superpowers/specs/2026-06-04-zone-override-unification-three-paradigm-design.md`（untracked）。

### 🟢 低优先级 / 可选
- 两条 memory 待记（用户已确认要记，会话末未落）：① 区域语义判定不可全信用户回答、必须几何实测兜底（`feedback_oft_single_zone_must_ask_not_guess` 的延伸）；② 派 worktree 给 agent 实施、交付前必须 rebase 到最新 dev（这次 base 落后 138 commit 的教训）。

---

## 5. 风险与注意事项

- **⚠️ 工作区有 3 个 modified 文件不属于本会话两个 feature**：`CLAUDE.md`（+SkillOpt 第 14 条）/ `docs/milestone/README.md` / `packages/agent/backend/tests/test_handoff_content_validation.py`（+105 行）。这些是 **SkillOpt 路线 / 并行工作**的脏改动，**不要当成本会话 feature 的一部分去提交或回退**——确认归属后再处理。
- **untracked 文件一堆**（docs/handoffs、docs/plans、docs/superpowers/specs、golden-cases/case-00X）——多数是会话外产出，pull 时若挡路先核对 md5 再决定（本会话就遇到 `test_zone_unnamed_detection_all_paradigms.py` untracked 挡 pull，md5 与远端一致才删的）。
- **不要再走"取反"思路**：几何实测已定论 in_zone=1=目标区。
- **dogfood 前务必重启 dev**，否则验的是旧代码（白跑）。
- 用户硬性铁律（memory）：deepseek 正面措辞（禁用"不要/禁止"）；改共享逻辑必跑全量；SSOT 不双存；dev/prod 行为对齐。

---

## 6. 建议接手路径（下一个 agent 第一步）

1. **重启 dev**：`cd /home/wangqiuyang/noldus-insight/packages/agent && make stop && make dev`
2. **跑 O迷宫 dogfood**：上传 `/home/wangqiuyang/DemoData/newdemodata/O迷宫/` 的 3 个 xlsx，发"帮我分析数据"，盯以下链路：
   - lead 调 inspect → 确认返回含 `anonymous_zone_evidence`，in_zone=1 占时 6-14%
   - prep 报 `zone_unnamed` → lead 带占时证据反问（**不该再诱导用户二选一答封闭区**）
   - 用户答"开放臂" → lead 写 `parameter_overrides={"anonymous_zone_is":"in_zone"}`
   - prep 重调 → plan 里 `open_zones=["in_zone"]`（list）→ compute 出真值
3. **盯 log**：后端 log 看 `zone_unnamed`/`anonymous_zone_is`/`open_zones`；workspace 看 `experiment-context.json` 的 `parameter_overrides` 是否是统一 key（不该再有脑补的 `~in_zone`）。
4. 跑通后按需补 LDB + EPM dogfood。

---

## 7. 关键路径速查
- zone 修复 spec：`docs/superpowers/specs/2026-06-04-zone-override-unification-three-paradigm-design.md`（含 §2 已核实事实表、§5 验收回归、§7 路径速查）
- 翻译逻辑：`packages/ethoinsight/ethoinsight/catalog/resolve.py`（`_detect_anonymous_zone` / `_compute_parameters_in_use` 的 effective_overrides + wrap_list）
- catalog 元字段：三个 yaml 顶层 `anonymous_zone_override: {target_param, wrap_list}`
- inspect 几何证据：`tools/builtins/inspect_uploaded_file_tool.py:_compute_anonymous_zone_evidence`（txt/excel/csv 三注入点）
- prep hint：`tools/builtins/prep_metric_plan_tool.py` 的 `_ERROR_HINTS["zone_unnamed"]`（统一 key 三步指南）
- 真实 demo data：`/home/wangqiuyang/DemoData/newdemodata/{O迷宫,明暗箱,高架十字迷宫_小鼠_三点,旷场_小鼠_三点,强迫游泳_大鼠,悬尾}`
- ethoinsight 全量回归：`cd packages/ethoinsight && .venv/bin/python -m pytest tests/ -q`（基线 598 passed）
- 失败 thread 回放：`.deer-flow/users/cd95effa-.../threads/6dea6ae9-.../`
