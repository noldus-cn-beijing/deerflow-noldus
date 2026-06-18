# Spec：report 图片路径 SSOT 统一 —— 终结 dev↔deploy 反复横跳

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-18
> 性质：结构性根治（SSOT 漂移）。report.md 里图片能否显示，取决于 src 字符串有没有 `/mnt/user-data/` 前缀、有没有前导斜杠——而决定这个的逻辑**分散在前端 2 处 + 后端 3 处共 5 个地方，各自漂移、无单一约定**。每修一端迁就一种形态，另一端就崩——这就是「本地改好部署崩、部署改好本地崩」的反复横跳。
> 关联：
> - SSOT 铁律：memory `feedback_single_source_of_truth`（同一约定绝不多处定义）
> - 根因纪律：memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`（5 个改写点叠在一起 = 每次都在打地鼠）
> - 部署对齐：memory `feedback_dev_prod_behavior_alignment`、`feedback_deploy_compose_per_service_image_tag`（dev/prod 行为差异要查镜像与挂载，但本 bug 真因是路径约定漂移，非镜像）
> - 前序（幻影文件防御，与本 bug 正交但同区）：memory 2026-06-08 批次 `present_files 拒绝磁盘上不存在的文件`

---

## 〇、给实施 agent 的一句话

report.md 图片 404 的真因：**「artifact src 路径的规范形态」没有单一定义**。后端 seal 把 `{{img:}}` 解析成 `mnt/user-data/outputs/X.png`（**无**前导斜杠），前端 markdown img 渲染却用 `src.startsWith("/mnt/")`（**要求有**前导斜杠）判断——不命中就落到 `normalizeArtifactImageSrc` 的 5-case 兜底，case 4 把它砍成 `/outputs/X.png`（**丢了 `mnt/user-data` 前缀**），后端 `resolve_virtual_path` 要求路径以 `mnt/user-data/` 开头 → 拒绝 → 404。**定一个规范形态（建议：report.md 内统一写 `/mnt/user-data/outputs/X.png` 含前导斜杠的虚拟路径），让前端只认这一种、后端只认这一种，删掉所有"猜测/兜底/再normalize"的中间层。**

**核心约束**：规范形态在前后端各只解析一次，不再有"多 case 猜测"。任何不符合规范的 src 一律视为 bug 暴露（响亮失败/可见错误），不静默 normalize。

---

## 一、根因（全链路追踪，dogfood thread `3bcbee10`）

report.md 图片从占位符到浏览器请求，经过 **5 个独立改写/判断点**，每个对"前缀该长什么样"有自己的假设：

| # | 位置 | 它产出/期望的形态 | 文件:行 |
|---|---|---|---|
| 1 | 后端 seal `_load_chart_files_map` | `mnt/user-data/outputs/X.png`（`.lstrip("/")` → **无**前导斜杠） | seal_handoff_tools.py:175 |
| 2 | 后端 seal `_normalize_report_image_paths` | 裸路径改写成 `mnt/user-data/outputs/X.png`（**无**前导斜杠） | seal_handoff_tools.py:246 |
| 3 | 前端 markdown img 分支 1 | `src.startsWith("/mnt/")`（**要求有**前导斜杠 `/`） | markdown-content.tsx:73 |
| 4 | 前端 `normalizeArtifactImageSrc` | 5 个 case 各种猜测；case 4「不以 `/` 开头」→ 砍成 `/outputs/X.png`（**丢前缀**） | utils.ts:37-70 |
| 5 | 后端 `resolve_virtual_path` | 要求路径以 `mnt/user-data/`（无前导斜杠）开头，否则抛 "Path must start with /mnt/user-data" | paths.py:373 |

**故障序列**（当前 dev 形态）：
1. seal（#1/#2）把 report.md 写成 `mnt/user-data/outputs/X.png`（**无**前导斜杠）。
2. 前端 markdown（#3）判 `startsWith("/mnt/")` → `mnt/...` 无斜杠 → **不命中** → 落 #4。
3. `normalizeArtifactImageSrc`（#4）：case 1 需 `/mnt/user-data/`（不命中）；case 3 需 `outputs/`（不命中）；case 4「不以 `/` 开头且非空」→ 命中 → 返回 `/outputs/X.png`（**`mnt/user-data` 前缀被丢弃**）。
4. 前端请求 `/api/threads/{tid}/artifacts/outputs/X.png`。
5. 后端 `resolve_virtual_path`（#5）：`stripped="outputs/X.png"`，要求以 `mnt/user-data/` 开头 → **不符 → ValueError → 400/404**。

**为什么反复横跳**：5 个点没有共同约定。历史上每次「修好」都是改其中一两个点去迁就当时的形态——
- 改 #4 让 case 4 保留前缀 → 但 #1/#2 产出无斜杠，#3 仍不命中，下一种数据形态又崩。
- 改 #5 接受无前缀 `outputs/` → 但破坏路径校验语义（安全风险 + 其他调用方依赖前缀）。
- dev 和 deploy 的 `get_effective_user_id()` 返回不同（deploy 有真实 user → `users/{uid}/threads/...` 布局；dev 可能走 legacy `threads/...`），让"哪条路径能 resolve 成功"在两个环境表现不同，**进一步掩盖**了真因是路径约定漂移而非环境差异。

这是教科书级 `feedback_single_source_of_truth` 违反：**「artifact 路径规范」这一条知识，存在 5 份各自漂移的定义。**

---

## 二、设计：定义唯一规范形态，前后端各只解析一次

### 2.1 规范形态（CANONICAL）

**report.md 内图片路径一律为虚拟绝对路径 `/mnt/user-data/outputs/<name>.png`（含前导斜杠）。** 理由：
- 这是 sandbox 内的真实虚拟路径，与所有其他工具（read_file/write_file/handoff）的路径约定一致，不另造方言。
- 前导斜杠让前端 `startsWith("/mnt/")` 判断稳定命中（#3 已是这个判断）。

### 2.2 后端：seal 把占位符解析为规范形态（唯一产出点）

`_load_chart_files_map` / `_resolve_report_image_placeholders` 产出**带前导斜杠的 `/mnt/user-data/outputs/X.png`**（去掉当前的 `.lstrip("/")`）。`_normalize_report_image_paths` 同样统一到这个形态（带前导斜杠）。**两个函数产出必须字节一致的规范形态**——最好抽一个 `_to_canonical_artifact_path(name) -> "/mnt/user-data/outputs/" + name` 单一函数，两处都调它（SSOT）。

### 2.3 前端：只认规范形态，一次解析（删兜底）

markdown img 渲染（markdown-content.tsx）：
- 命中 `src.startsWith("/mnt/user-data/")` → 走 artifact URL：把 `/mnt` 之后的部分（`/user-data/outputs/X.png`）交给后端（见 §2.4 约定）。
- **不命中规范形态的 src**：不再用 `normalizeArtifactImageSrc` 的 5-case 猜测兜底。改为：① http(s) 外链原样；② 其余视为异常，渲染可见占位（`[图片路径非规范: <src>]`）或原样 `<img>` 让其 404 暴露——**不静默 normalize**（响亮失败，便于发现 report-writer 写错路径）。
- `normalizeArtifactImageSrc` 的 case 2/3/4/5（host 绝对路径 / `outputs/` / 裸名 / `/outputs/`）**全部删除**——这些是历史兜底，正是漂移之源。保留 case 1（`/mnt/user-data/` → 转 artifact 路径）作为唯一规范解析。

### 2.4 前后端交接约定：artifact API path 段

前端把规范 src `/mnt/user-data/outputs/X.png` 转成 artifact 请求时，URL path 段必须是后端 `resolve_virtual_path` 能接受的形态——即 `mnt/user-data/outputs/X.png`（`resolve_thread_virtual_path` 内部 `lstrip("/")`，#5 接受 `mnt/user-data/...`）。

具体：`resolveArtifactURL(src)` 收到 `/mnt/user-data/outputs/X.png` → 拼成 `/api/threads/{tid}/artifacts/mnt/user-data/outputs/X.png`（保留 `mnt/user-data` 前缀，仅去掉最前导斜杠由 URL path 拼接自然处理）。**关键修正**：当前前端有的分支会把前缀剥成 `/outputs/`——删掉，统一保留全前缀，让 #5 的校验直接命中。

> 即：前端送 `mnt/user-data/outputs/X.png`，后端 `resolve_virtual_path` 的 `startsWith(prefix+"/")` 命中（prefix=`mnt/user-data`）→ resolve 成功。链路闭合。

### 2.5 dev/deploy 一致性：user_id 必须一致流经

`resolve_thread_virtual_path` 用 `get_effective_user_id()`。确认 **artifact 读路径与 seal 写路径用同一个 user_id 来源**——否则 dev（无 user / legacy 布局）和 deploy（真实 user / `users/{uid}` 布局）会 resolve 到不同物理目录。这不是本 bug 的主因（主因是前缀漂移），但是「dev/deploy 表现不同」的放大器，需一并核实锁死（见 §四测试 4.4）。

---

## 三、改动清单

### 3.1 `seal_handoff_tools.py` —— 抽 canonical 函数，两处共用

- 新增 `_to_canonical_artifact_path(name: str) -> str`，返回 `f"/mnt/user-data/outputs/{name}"`（带前导斜杠）。
- `_load_chart_files_map`（L175）：`result[Path(f).name] = _to_canonical_artifact_path(Path(f).name)`（去掉 `.lstrip("/")`，统一带斜杠的规范形态；注意 chart_files 里本就是 `/mnt/user-data/outputs/...`，直接规范化即可）。
- `_normalize_report_image_paths`（L246）：`_BAD_IMG_PATH_RE.sub` 的替换目标改为带前导斜杠的 `/mnt/user-data/outputs/\1`，与 canonical 一致。

### 3.2 `utils.ts` —— 删兜底 case，只留规范解析

- `normalizeArtifactImageSrc`：删除 case 2/3/4/5，只保留 http(s) 外链返回 null + case 1（`/mnt/user-data/` → 提取 `/user-data/outputs/X.png` 或直接交全路径）。**或**直接废弃此函数，把规范解析内联到 markdown-content.tsx。
- `resolveArtifactURL` / artifact URL 拼接：确保送给后端的 path 段保留 `mnt/user-data/` 全前缀（不剥成 `/outputs/`）。

### 3.3 `markdown-content.tsx` —— 单一分支 + 响亮失败

- img 渲染：`src.startsWith("/mnt/user-data/")` → artifact URL（保留全前缀）；http(s) → 原样；其余 → 可见占位/原样暴露，不猜测。
- 删除「落 `normalizeArtifactImageSrc` 多 case 兜底」的分支。

### 3.4 `report_writer.py` SKILL（核查）

report-writer 已用 `{{img:<文件名>}}` 占位符（SKILL.md:53/100/233），seal 负责解析——**确认 SKILL 不让 LLM 自己写绝对路径**。L108-111 的正确/错误示例保留（它们已正确引导用占位符）。无需大改，仅核实占位符是唯一图片引用方式。

---

## 四、测试（红→绿坐实，TDD 强制）

### 4.1 后端 seal 产出规范形态

`packages/agent/backend/tests/test_report_image_path_canonical.py`：
1. **`test_seal_resolves_placeholder_to_canonical`**：handoff_chart_maker.json 含 `chart_files=["/mnt/user-data/outputs/plot_box_open_arm.png"]`，report.md 含 `![x]({{img:plot_box_open_arm.png}})`，seal 后断言 report.md 含 **`/mnt/user-data/outputs/plot_box_open_arm.png`**（带前导斜杠的规范形态）。
   - **坐实红**：在改前跑，断言当前产出是 `mnt/user-data/...`（无斜杠）—— 记录改前形态，证明测试真咬。
2. **`test_normalize_image_path_canonical`**：report.md 含裸 `(outputs/X.png)` / `(/mnt/user-data/outputs/X.png)` 多形态，normalize 后全部 → 带斜杠规范形态。
3. **`test_canonical_path_helper_single_source`**：`_load_chart_files_map` 和 `_normalize_report_image_paths` 对同一文件名产出**字节相同**的路径（SSOT 一致性）。

### 4.2 后端 resolve 接受规范形态对应的 API path

4. **`test_resolve_virtual_path_accepts_canonical_artifact`**：`resolve_virtual_path(tid, "mnt/user-data/outputs/X.png")` 成功 resolve（前端送这个形态）；`resolve_virtual_path(tid, "outputs/X.png")`（旧兜底形态，丢前缀）**仍抛 ValueError**（确认我们没靠放宽校验来"修"，而是靠前端送对前缀）。

### 4.3 前端规范解析（若有前端测试框架；无则人工核查 + 类型/lint）

> 前端无测试框架（frontend/CLAUDE.md）。改动后跑 `pnpm check`，并在 PR 描述里贴：规范 src `/mnt/user-data/outputs/X.png` → 最终请求 URL `/api/threads/{tid}/artifacts/mnt/user-data/outputs/X.png` 的手工验证截图/curl。

### 4.4 dev/deploy user_id 一致性

5. **`test_artifact_read_path_matches_seal_write_path`**：模拟 seal 写 outputs 到 `users/{uid}/threads/{tid}/...`，再用 `resolve_thread_virtual_path` + 同一 `get_effective_user_id()` 读，断言命中同一物理文件。**坐实 dev（无 user）与 deploy（有 user）都自洽**——参数化 user_id=None / "real-uid" 两种跑。

### 4.5 全量回归 + 两入口裸导入

改 `seal_handoff_tools.py`（共享）+ 前端：后端 `make test` 全量 + 裸导入两入口；前端 `pnpm check`。

---

## 五、验收标准

1. dogfood EPM 复跑：report.md 里所有图片在前端正常显示（无 404），src 形态为 `/mnt/user-data/outputs/X.png`。
2. **同一份 report.md 在 dev（本地 `make dev`）和 deploy（docker compose）下图片都显示**——这是本 bug 的终极验收，必须两个环境都验。
3. 前端不再有多 case 路径猜测；后端图片路径只由一个 canonical 函数产出。
4. 非规范 src 不被静默 normalize，而是可见暴露（便于未来发现 report-writer 写错）。

---

## 六、风险与注意事项

1. **必须两环境都验**（验收 §五.2）：本 bug 的本质就是「只验一个环境就以为修好」。改完**先本地 `make dev` 验图能显示，再 `make deploy-tar` 部署到 ECS 验同一流程**——任一环境图片 404 都算没修完。memory `feedback_dev_prod_behavior_alignment`。
2. **别靠放宽 `resolve_virtual_path` 校验"修"**：把 #5 改成接受无前缀 `outputs/` 是最诱人的"快修"，但会破坏路径安全语义 + 给其他调用方留坑。正解是让前端送对前缀（§2.4），#5 保持严格。测试 4.2 专门锁这一点。
3. **删前端兜底 case 可能影响其他图片来源**：`normalizeArtifactImageSrc` 也被 present_files content / handoff summary 的图片用（utils.ts 注释提到）。删 case 前 grep 全部调用方，确认它们产出的 src 也已是规范形态或走 http(s)；若有其他来源仍产非规范路径，先把那个来源也改成产规范形态，再删兜底（否则把一个静默问题换成另一个）。**这是本 spec 最大风险点**——`grep -rn "normalizeArtifactImageSrc\|resolveArtifactURL" packages/agent/frontend/src` 列全调用方再动手。
4. **chart_files 本身的形态**：`handoff_chart_maker.json.chart_files` 里是 `/mnt/user-data/outputs/...`（chart-maker SKILL 要求虚拟路径）。canonical 函数对它们是幂等规范化，确认不会二次加前缀。
5. **与 handoff 瘦身 / 并行绘图 spec 正交**：本 spec 只碰图片路径，不碰 handoff 体积、不碰并行。三份可独立合，但都改 `seal_handoff_tools.py` / subagent SKILL，合并时注意 `seal_handoff_tools.py` 的 diff 协调（瘦身 spec 改 task_context 注入、本 spec 改 img 路径解析、并行 spec 删 report_writer bundle——三处不同段，但同文件，rebase 时核对）。
