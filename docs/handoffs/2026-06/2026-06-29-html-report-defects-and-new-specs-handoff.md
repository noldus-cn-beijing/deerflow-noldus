# Handoff：dogfood 复测发现 #234 多缺陷 + 写出 4 份新 spec + git 分叉调和 + spec 命名风格遗留（2026-06-29 续）

> 本会话承接 `2026-06-29-frontend-dogfood-fixes-and-thread-assets-panel-handoff.md`。上一会话写了 4 份 spec（流式性能/删进度轨/HTML报告/gallery三修），**本会话期间它们已被 #231–#234 实施合入 dev**。本会话主线：① 把那 4 份 spec 归档进 git ② 调和本地/远端 git 分叉 ③ 监控用户 dogfood thread `73b41dc3` 复测，发现 #234（HTML 报告）一串新缺陷 ④ 写出 4 份新 spec（修复 + 1 个 feature）并 push。**会话结束时远端又推了 #235，已实施其中「图全坏」那份 spec。**

---

## 〇、一句话现状

- **本地 dev 落后 origin/dev 1 个 commit**（`594006bb` #235，别的 agent 刚推的「HTML 报告图全坏」修复）。**下一个 agent 第一步必须 `git pull --rebase`**（本地工作区只有 3 个历史 untracked 目录，无未提交改动，rebase 安全）。
- 本会话写的 4 份新 spec 已 commit+push（`b6966d32`）。其中**「图全坏」已被 #235 实施**，其余 3 份待实施。
- 用户 dogfood thread `73b41dc3`（user `cd95effa`）后端流水线**健康**（metric 140、chart 113、report.html 已出），但 #234 HTML 报告暴露 3 个缺陷（图/样式/title）。

---

## 一、本会话 git 产出（全已 push 到 dev）

| commit | 内容 | 性质 |
|---|---|---|
| `9d37d031` | 归档上一批 4 份 spec（streaming-render-perf / remove-progress-rails / html-report-format / assets-gallery-fixes） | docs |
| `10b674f8` | 部署修复：`.dockerignore` 排除 `.deer-flow/*.db`（防镜像膨胀 3.4GB+）+ `deploy-via-tar.sh` 迁移 entrypoint 改 bash（脚本 :ro 不能 chmod） | 🟢 部署 |
| `b6966d32` | 本会话新写的 4 份 spec（见 §三） | docs |

> 注：`10b674f8` 那两个部署修复**不是本会话写的**（来历是你/别的会话的本地改动，本会话只是确认有意义后 commit）。

---

## 二、关键过程记录

### 2.1 git 分叉调和（已完成）
本地 dev 曾与 origin/dev 分叉（本地领先 1 = handoff 归档；远端领先 4 = #231–#234 实施）。用 **rebase** 调和（线性历史）：stash 两个部署修复 → rebase origin/dev → pop。无冲突（docs vs 前端代码零重叠）。**经验**：并发实例在动 dev，每次操作前 `git fetch` + `git rev-list --left-right --count` 看分叉。

### 2.2 监控 thread 73b41dc3 复测（核盘结论）
- 用户自己跑 dogfood（EPM 范式，28 subject）。**走磁盘核盘**（e2e 用户跨用户看不到 user `cd95effa` 的 thread）：`packages/agent/backend/.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/73b41dc3-.../user-data/`。
- ✅ 后端健康：metric 140、data-analyst handoff completed（5 findings）、chart 113 png+113 webp、`report.html` 已出（124KB）。#231/#232/#233 落地正常。
- 🔴 发现 #234 HTML 报告 3 缺陷（见 §三）。

---

## 三、本会话写的 4 份新 spec（全在 `docs/superpowers/specs/`）

| spec 文件 | 治什么 | 根因（已坐实） | 状态 |
|---|---|---|---|
| `2026-06-29-fix-html-report-inline-img.md` | HTML 报告内联图全坏 | prompt 让 LLM 写 `<img src="{{img}}">`，seal 却把 `{{img}}` 替换成**整个** `<img>` → 嵌套畸形 `<img src="&lt;img src=...`，图全废 | **✅ 已被 #235 实施**（远端 `594006bb`，用了 spec 方案 A） |
| `2026-06-29-fix-html-report-styling.md` | HTML 报告完全无样式 | `prose` 是**死类**——`@tailwindcss/typography` 根本没装（package.json/node_modules/css 全无）；附带 `<title>` 被 sanitizer 剥标签留内容→裸露在 head | ⬜ 待实施。**用户选「前端 prose 方案」**（v4 用 `@plugin "@tailwindcss/typography"`，别用 v3 config） |
| `2026-06-29-fix-tab-switchback-jank.md` | 切回 tab 长卡顿 | #232 故意 bypass in-flight 大消息缓存；后台渲染冻结+SSE 继续灌大→切回一次性追平整条重渲染 | ⬜ 待实施。**实施前必须先 CDP 实测坐实**（prod build），方案候选 A=committed prefix+live tail |
| `2026-06-29-report-export-formats.md` | 导出 PDF/Word/LaTeX | 当前无导出能力（只能下原始 .html/.md） | ⬜ **新 feature**，依赖图+样式修复先落地；pandoc(Word/LaTeX)+Playwright/WeasyPrint(PDF)，**实施前 spike** 验证转换保真+镜像体积 |

**两个贯穿性发现**（写进了 spec，下个 agent 注意）：
1. **#234 系列 bug 共性 = 漏了「产物可被目标渲染器正确显示」的端到端断言**（只测占位符被替换/套了 prose 类，没测真能显示）。修复必须补这类端到端断言。
2. **`prose` 死类可能波及更广** —— `MarkdownContent`（消息流 + md 报告，report-card.tsx:90）也套 `prose`，typography 没装意味着**可能一直没生效**。styling spec 让实施者一并核实范围。

---

## 四、未完成事项（按优先级）

| # | 事项 | 依赖/状态 |
|---|---|---|
| 1 | **`git pull --rebase`** 同步远端 #235 | 第一步，无依赖 |
| 2 | 实施 `fix-html-report-styling`（前端装 typography + 修 title 裸露 + 核实 markdown prose 范围） | 确定性 bug，独立 |
| 3 | 实施 `fix-tab-switchback-jank`（**先 CDP 实测**再改） | 需 prod build + e2e skill CDP perf |
| 4 | 实施 `report-export-formats`（**先 spike**） | 依赖 #235(图) + styling(样式) 先落地 |
| 5 | **spec 命名风格统一**（用户在意，见 §六） | 待用户拍板是否改名 |

---

## 五、关键陷阱 / 注意事项

1. **本地落后远端 1**：先 pull --rebase，否则又分叉。dev 有并发实例在推。
2. **3 个历史 untracked 目录**（`docs/reports/`、`reports/report for june/`、`scripts/repro/`）**保持原样别提交**（多会话前与用户确认过）。`git add` 永远用精确路径，绝不 `-A`/`.`。
3. **跨用户核盘**：用户自己的 thread（user `cd95effa`）走磁盘 `.deer-flow/users/<uid>/threads/<tid>/user-data/`；e2e 登录用户（`qiuyang.wang@noldus.com` id `e281f251`）看不到。CDP attach 现有浏览器：本机 9222/9223/9229 是 **VS Code Node 调试端口**（非浏览器，列不出 page），别误当浏览器。
4. **改后端 seal/prompt 前**过 HarnessX 三病理；改 tools/builtins/subagents/agents 后跑两裸导入。
5. **#234 styling 修复是 Tailwind v4**：`@plugin "@tailwindcss/typography"` 写在 globals.css，不是 `tailwind.config.js`（v4 不读 config）。装完 `pnpm build` 后 grep 产物 CSS 确认 `.prose` 规则真生成（守 token PR 死类教训）。
6. **流式卡顿别盲改流式核心**（useStream/mergeMessages/dedupe）；先 CDP 实测，守 `feedback_perf_is_efficient_impl_not_visual_downgrade`。

---

## 六、spec 命名风格遗留（用户提出，未决）

用户注意到 `docs/superpowers/specs/` 里命名风格不一致：
- **旧 spec（别人/早期）**：长、信息密，多带 `-spec`/`-fix-spec` 后缀（如 `...-crash-fix-spec.md`、`...-affordance-spec.md`）。
- **本会话 8 份（2026-06-29）**：偏短、**无统一 `-spec` 后缀**，动作词位置自身也不统一（`fix-html-report-...` 前置 vs `assets-gallery-fixes` 后置）。

**原因**：项目 CLAUDE.md 只规定了 handoff 命名，**对 specs 目录无明文规范**；本会话起名时没先 `ls` 对照既有惯例。**建议**：与用户确认一套规范（如 `YYYY-MM-DD-<topic>-spec.md` / `-fix-spec.md`），然后 `git mv` 统一改名（注意其中 6 份已 push，改名要再提交）。**未动手，等用户拍板。**

> 补充澄清：`docs/superpowers/specs/` 只是**项目目录名**带 "superpowers"，与「superpowers skill」无关；本会话写 spec **未用任何 skill**，是 Explore agent 调研 + Write 工具写 markdown。

---

## 七、下一位 Agent 的第一步

1. `git pull --rebase origin dev`（同步 #235；先 `git status` 确认工作区只有 3 个历史 untracked）。`git log --oneline -5` 确认 HEAD 含 `594006bb`。
2. 读本 handoff + 4 份 spec（`docs/superpowers/specs/2026-06-29-fix-*.md` + `report-export-formats.md`）。
3. 若派工实施：优先 `fix-html-report-styling`（确定性、独立）；`fix-tab-switchback-jank` 先 CDP 实测；`report-export-formats` 先 spike 且等样式落地。
4. 若用户先要解决 spec 命名（§六）：与用户定规范 → `git mv` 8 份 → 提交。
5. 验证 #235 是否真修好图：对 thread `73b41dc3` 重 seal 或重跑 dogfood，核 `report.html` 内 `data:image/png;base64,` 子串数==内联图数、无 `&lt;img src=`、前端 ReportCard 图可见（守「代码有修复≠现象消除」）。
