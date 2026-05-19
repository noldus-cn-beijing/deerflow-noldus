# 2026-05-19 上线前 DeerFlow 去品牌 + 浮夸 landing 清理 Handoff

## 状态：✅ 完成

7 个 Step 全部提交，`pnpm typecheck` + `pnpm build` 通过。`pnpm lint` 有 2 errors + 8 warnings — **全部预先存在**（与本 PR 无关，已通过 stash 验证），都在 `input-box.tsx` 和 `threads/hooks.ts` 中。

## 关联文档

- 设计 spec：[docs/superpowers/specs/2026-05-19-pre-launch-deerflow-cleanup-design.md](../../superpowers/specs/2026-05-19-pre-launch-deerflow-cleanup-design.md)
- Plan file：`~/.claude/plans/home-wangqiuyang-noldus-insight-claude-velvet-backus.md`

## 提交清单（7 个 commit, 都在 dev 分支）

| Commit | Step | 概要 |
|---|---|---|
| `a79e1bda` | 1 | 品牌资源：deer.svg → noldus-emblem.svg（auth 页 mask 三处）+ `src/app/icon.svg` 新增（favicon）+ 删除遗留 `public/favicon.ico` |
| 同上 | — | 同时归档了 spec 到 `docs/superpowers/specs/` |
| Step 2 | 2 | 删除整个 `src/components/landing/`（11 个 tsx，~1000+ LOC） |
| Step 3 | 3 | 下线 `[lang]/docs` 和 `[lang]/blog`：删 `src/app/[lang]/` + `src/content/{en,zh}/` + `src/mdx-components.ts`；package.json 卸 nextra + nextra-theme-docs（-115 包） |
| `758e9db9` | 4 | i18n DeerFlow→EthoInsight 全量替换（zh-CN/en-US/types 共 32 处） |
| `d17980a1` | 5 | 删 dead `about.md`；frontend CLAUDE.md/AGENTS.md/README.md 顶部品牌行 |
| `598df4ac` | 6 | next.config.js 删 Pages Router 风格的 i18n 配置（修 build 失败） |
| `c0346093` | 7 | 修复 audit 漏掉的：workspace-nav-menu 4 个 DeerFlow 链接、workspace-container GitHub icon href、api.ts 后端错误 toast、memory 导出文件名、recent-chat-list 的 vercel fallback URL |

## 删除的资源 / 文件

```
packages/agent/frontend/public/images/deer.svg
packages/agent/frontend/public/favicon.ico
packages/agent/frontend/src/components/landing/  （整目录 11 文件）
packages/agent/frontend/src/app/[lang]/          （整目录）
packages/agent/frontend/src/content/{en,zh}/     （整目录所有 MDX）
packages/agent/frontend/src/mdx-components.ts
packages/agent/frontend/src/components/workspace/settings/about.md  （dead，已被 about-content.ts 替代）
```

## 卸载的 npm 依赖

- `nextra` ^4.6.1
- `nextra-theme-docs` ^4.6.1
- 合计 -115 个 transitive packages

## 修改的关键文件

- `src/app/(auth)/login/page.tsx` — mask 用 noldus-emblem
- `src/app/(auth)/setup/page.tsx` — mask 用 noldus-emblem ×2
- `src/app/icon.svg` — **新增**，Next.js 15 约定自动生成 favicon
- `next.config.js` — 删 withNextra 包裹 + 删 Pages Router 的 i18n 字段
- `src/core/i18n/locales/{zh-CN,en-US,types}.ts` — DeerFlow→EthoInsight 全量替换 + 删 `t.home` 子树
- `src/components/workspace/workspace-nav-menu.tsx` — 4 项菜单 href 指向 Noldus
- `src/components/workspace/workspace-container.tsx` — 顶部 GitHub icon href 指向 Noldus
- `src/core/agents/api.ts` — 后端错误 toast 改 EthoInsight
- `src/components/workspace/settings/memory-settings-page.tsx` — 导出 JSON 文件名前缀 ethoinsight-
- `src/components/workspace/recent-chat-list.tsx` — 分享 URL 用 `window.location.origin`，去掉 vercel fallback
- `packages/agent/frontend/{CLAUDE,AGENTS,README}.md` — 顶部品牌行

## 故意保留（按 spec non-goals）

- `package.json` `name: "deer-flow-frontend"` — 内部 fork 标识
- `DEER_FLOW_*` 环境变量（`DEER_FLOW_INTERNAL_LANGGRAPH_BASE_URL`、`DEER_FLOW_INTERNAL_GATEWAY_BASE_URL`、`DEER_FLOW_TRUSTED_ORIGINS`、`DEER_FLOW_AUTH_DISABLED`）— 后端契约
- `localStorage` keys：`deerflow.local-settings`、`deerflow.thread-model.*`、`deerflow.agent-create.save-hint-seen` — 改名会让现有用户丢偏好
- 内部 console 日志前缀 `[deer-flow]` — 开发可见，用户不可见
- 注释里的 `upstream deerflow useUpdateSubtask` 引用 — 工程文档
- `about-content.ts` 中开源致谢提及 DeerFlow 2.0 / ByteDance / 核心作者 — 合法的 attribution
- `public/demo/threads/*/thread.json` 和 `public/demo/threads/*/user-data/outputs/*.html` 中的 DeerFlow signature — 内部 demo fixture，用户不会看到
- 后端 Python 侧的 DeerFlow 引用（subagent 注册、harness 子模块）— 不在本 PR 范围

## 验证结果

- ✅ `pnpm typecheck`: 通过
- ✅ `pnpm build`: 通过（路由表里只有 9 个 app 路由 + icon.svg，无 `[lang]/*` 残留）
- ⚠️  `pnpm lint`: 2 errors + 8 warnings — **全部预先存在**，与本 PR 无关
- ✅ `grep -ri "DeerFlow\|🦌\|deerflow\.tech\|bytedance/deer-flow" src/`: 用户可见品牌零残留，剩余仅 about-content.ts 致谢段（合规）
- ✅ Noldus emblem 与源文件 bytewise 一致
- ✅ 浏览器 tab favicon 现由 `src/app/icon.svg` 自动接管

## 浏览器端到端 dogfood（待跑）

下次启动 `cd packages/agent && make dev` 后建议确认：

1. 访问 `/login` — 背景 mask 显示 N + 叶子（非鹿）
2. 访问 `/` — 直接进 `/workspace`，无白屏
3. 浏览器 tab 显示 Noldus emblem（不是鹿）
4. 工作区右上角齿轮 → 4 个外链都指向 noldus.com.cn 或 support@noldus.com.cn
5. 工作区顶部 GitHub 图标点击跳 noldus.com.cn
6. 设置 → 关于：显示 about-content.ts 里的 Noldus 文案
7. 设置 → 通知 → 发送测试通知：标题显示 "EthoInsight"
8. 访问 `/en/docs`、`/zh/docs`、`/en/blog`、`/zh/blog`：全 404
9. 工作区欢迎页：`welcome.description`（若有页面显示）为 EthoInsight 行为学叙事

## 仓库其他不动事项

- 本会话开始时仓库有一个 **预先存在**的 unmerged 文件 `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`（双方修改的合并冲突）。本次工作完全没有触碰它。使用 `git commit -o <files>` 跳过它来提交。**下一任会话需要单独处理这个冲突。**
- `docs/handoffs/2026-05/2026-05-19-data-flywheel-mode-PR3-handoff.md` 和 `2026-05-19-stepwise-gate-redesign-handoff.md` 是仓库里其他正在进行的工作的未跟踪文档，不在本 PR 范围。

## 下一步建议

1. 跑一次端到端 `make dev` dogfood，确认上面 9 项视觉/行为符合预期
2. 处理后端那个 unmerged `code_executor.py`
3. 后续如果要做正式 landing：spec 已删除 `src/components/landing/`，重新做时从零开始更干净
4. EthoInsight 自己的产品文档另起 PR（`/docs` 路由空缺待补）
