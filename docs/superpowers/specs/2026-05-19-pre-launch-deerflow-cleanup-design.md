# 上线前 DeerFlow 去品牌 + 浮夸 landing 清理（前端）

- **日期**: 2026-05-19
- **关联仓库**: `noldus-insight/packages/agent/frontend/`
- **里程碑**: v0.1 上线前清理
- **配套 spec**（计划批准后由实施 agent 写到）: `docs/superpowers/specs/2026-05-19-pre-launch-deerflow-cleanup-design.md`

---

## Context（为什么做）

EthoInsight 即将上线（v0.1 是 9 月硬指标）。前端是 DeerFlow 的 subtree fork，遗留了上游的品牌名、Logo（鹿）、营销 landing 和公开文档站，对客户演示和正式上线都不合适。

具体问题：

1. **用户可见的 DeerFlow 品牌** — i18n（`zh-CN.ts` / `en-US.ts`）里约 32 处「DeerFlow」+ 🦌 emoji，覆盖欢迎页、设置页、内存说明、通知测试、Skill 提示等。
2. **DeerFlow 鹿 Logo 残留** — `public/images/deer.svg` 在登录/注册页用作 CSS mask（每次登录都看见）；`favicon.ico` 是 Apr 27 二进制鹿图标；浏览器 tab、登录页 mask 都不是 Noldus 品牌。
3. **浮夸 landing 孤儿代码** — `src/components/landing/` 下 11 个文件（hero 动画、case studies、community、whats-new、sandbox 演示、progressive-skills 700 LOC 动画）— `/` 已经 `redirect("/workspace")`，组件实际无路由消费，但占 ~1000+ LOC 让仓库看起来像通用 agent 框架而非行为学研究工具。
4. **公开文档站对外暴露 DeerFlow 营销内容** — `/[lang]/docs` 和 `/[lang]/blog` 通过 nextra 渲染 `src/content/{en,zh}/` 下 MDX，含「why-deerflow」「deploy-your-own-deerflow」等大量上游内容。同时 docs layout 复用 landing Header（硬编码 `github.com/bytedance/deer-flow` 链接 + GitHub star 计数器），是关键的对外品牌泄漏点。

**预期结果**：上线时所有用户可见的品牌一致为 EthoInsight/Noldus；`/` 直接进 workspace 无中间 landing；公开 docs/blog 下线；登录页与浏览器 tab 用 Noldus emblem；仓库瘦身约 ~1500 LOC + 2 个 npm 依赖。

---

## Scope（5 个动作领域）

| # | 领域 | 动作 | 影响 |
|---|---|---|---|
| 1 | **品牌资源** | 三处 `mask-[url(/images/deer.svg)]` → `/images/noldus-emblem.svg`；新增 `src/app/icon.svg` 作 favicon；删除 `public/images/deer.svg` + 旧 `public/favicon.ico` | 2 个 auth 页 + 2 个图标文件 |
| 2 | **Landing 浮夸清理** | 删除整个 `src/components/landing/` 目录 | 11 个 tsx，~1000+ LOC |
| 3 | **Docs/Blog 公开站下线** | 删除 `src/app/[lang]/` 整树 + `src/content/{en,zh}/` 全部 MDX；`next.config.js` 拿掉 nextra；`package.json` 卸载 `nextra` + `nextra-theme-docs`；删除 `src/mdx-components.ts`（若仅 nextra 使用） | 整个 `[lang]/` + `content/` + 2 个 npm 依赖 |
| 4 | **i18n 文案** | `zh-CN.ts` 和 `en-US.ts` 里所有「DeerFlow」「🦌」（约 32 处）改成「EthoInsight」（偶尔「Noldus Insight」）；删 `t.home` 子树 + `types.ts` 对应字段；`welcome.description` + `welcome.createYourOwnSkillDescription` 改成行为学研究员叙事 | 3 个文件 |
| 5 | **杂项收尾** | 删除 `src/components/workspace/settings/about.md`（dead，已被 `about-content.ts` 替代）；更新 frontend `CLAUDE.md` / `AGENTS.md` / `README.md` 顶部品牌行 | 4 个 dev 文档文件 |

---

## Non-Goals（明确不做）

- 不动 `package.json` 的 `name: "deer-flow-frontend"`（内部 fork 标识，不可见）
- 不动 `DEER_FLOW_*` 环境变量名（后端契约）
- 不动 `subagents/builtins/__init__.py` 等后端 subagent 注册
- 不动 `public/demo/threads/*/thread.json` demo fixture
- 不重写 `about-content.ts`（已是 Noldus 文案）
- 不动主题色（`globals.css` 已 Noldus 配色）
- 不重构 chat / settings / memory / artifact 业务模块
- 不动后端 / Python 侧 DeerFlow 引用
- 不重新设计 `/` 的 landing（直接 redirect workspace）
- 不写测试（前端项目目前未配测试框架）

---

## 关键文件清单

### Step 1 — 品牌资源
- 编辑：`src/app/(auth)/login/page.tsx:135`（`/images/deer.svg` → `/images/noldus-emblem.svg`）
- 编辑：`src/app/(auth)/setup/page.tsx:160` + `:231`（两处同上）
- 新增：`src/app/icon.svg`（复制 `public/images/noldus-emblem.svg` 内容；Next.js 15 约定自动作 favicon）
- 删除：`public/images/deer.svg`
- 删除：`public/favicon.ico`（避免与 `app/icon.svg` 双源）

### Step 2 — Landing 删除
- 删除目录：`src/components/landing/`（`hero.tsx` `header.tsx` `footer.tsx` `pulse-grid.tsx` `progressive-skills-animation.tsx` `section.tsx` `sections/case-study-section.tsx` `sections/community-section.tsx` `sections/sandbox-section.tsx` `sections/skills-section.tsx` `sections/whats-new-section.tsx`）
- 不动 `src/app/page.tsx`（已是 `redirect("/workspace")`）

### Step 3 — Docs/Blog 下线
- 删除目录：`src/app/[lang]/`（`docs/layout.tsx` + `docs/[[...mdxPath]]/page.tsx`，nextra 唯一消费者）
- 删除目录：`src/content/en/` 和 `src/content/zh/`
- 编辑：`next.config.js`（移除 nextra plugin / withNextra）
- 编辑：`package.json`（删 `nextra` 和 `nextra-theme-docs`）
- 删除：`src/mdx-components.ts`（先确认仅 nextra 用）
- 执行：`pnpm install` 更新 lockfile

### Step 4 — i18n 文案
- 编辑：`src/core/i18n/locales/zh-CN.ts`（约 16 处）
- 编辑：`src/core/i18n/locales/en-US.ts`（约 16 处）
- 编辑：`src/core/i18n/locales/types.ts`（删 `home: { docs; blog }` 字段）

**文案规则**（按上下文）：

- 一般产品名 → `EthoInsight`
- 「DeerFlow 的潜力 / 后台 / 通知」品牌化语境 → `EthoInsight`
- `welcome.description`（zh）:
  > 欢迎使用 EthoInsight — 上传 EthoVision XT 轨迹数据，AI 助手将自动完成统计分析、专业解读、APA 格式报告生成。
- `welcome.description`（en）:
  > Welcome to EthoInsight — upload EthoVision XT trajectory data, and the AI assistant will automatically run statistics, deliver expert interpretation, and generate APA-formatted reports.
- `welcome.createYourOwnSkillDescription`（两语种）改成研究员视角，去掉「super agent / 网络搜索 / 幻灯片 / 视频 / 播客」等不相关能力宣传。如果该键在新欢迎页里不再用，删键 + 同步删 `welcome.tsx` 引用。
- `notifications.testTitle` "DeerFlow" → "EthoInsight"
- `t.home` 整体删（docs/blog 已下线）

### Step 5 — 杂项收尾
- 删除：`src/components/workspace/settings/about.md`（dead，`about-content.ts:1-4` 注释已说明 Inlined）
- 编辑：`packages/agent/frontend/CLAUDE.md` 顶部「DeerFlow Frontend is a Next.js 16...」改为 EthoInsight 描述（保留下面所有架构说明）
- 编辑：`packages/agent/frontend/AGENTS.md` 同上
- 编辑：`packages/agent/frontend/README.md` 顶部品牌行
- 不动 frontend `CLAUDE.md` 关于 fork 来源的工程说明

---

## 实施顺序（每 Step 独立 commit + 独立验证）

```
Step 1 (5 min):  品牌资源就位
  → 验证：pnpm dev，登录页 mask 显示 N+叶子；浏览器 tab 显示 Noldus logo

Step 2 (5 min):  Landing 整目录删除
  → 验证：pnpm typecheck 通过；访问 / 正常 redirect /workspace

Step 3 (10 min): Docs/Blog 下线
  → 验证：pnpm build 通过；访问 /en/docs 与 /zh/docs 返回 404

Step 4 (15 min): i18n 全量去 DeerFlow
  → 验证：pnpm typecheck 通过；workspace 欢迎页文案变更；
         设置 → 通知测试标题变 EthoInsight

Step 5 (5 min):  收尾文件清理
  → 验证：grep 复扫，确认只剩白名单

Step 6 (10 min): 端到端 dogfooding
  - 登录页（看 Noldus mask + favicon）
  - 进入 workspace → 欢迎页文案 + 新会话
  - 设置 → About 弹窗（确认 about-content.ts 内容）
  - / 直接进 workspace 无白屏
  - /en/docs、/zh/docs、/en/blog、/zh/blog 全 404
```

工时合计 ~50 min。

---

## 验证清单

- [ ] `grep -ri "deerflow\|deer-flow\|🦌" packages/agent/frontend/src/` 只剩白名单（注释、`DEER_FLOW_*` env 读取、demo fixture）
- [ ] `pnpm install && pnpm build && pnpm typecheck && pnpm lint` 四绿
- [ ] 浏览器 tab 显示 Noldus emblem 而非鹿
- [ ] 登录页 + 注册页背景 mask 是 Noldus emblem
- [ ] `/` 进 `/workspace`，无白屏、无 console error
- [ ] `/en/docs`、`/zh/docs`、`/en/blog`、`/zh/blog` 全 404
- [ ] workspace 欢迎页 + 设置页文案是 EthoInsight 行为学研究助手叙事
- [ ] About 弹窗显示 `about-content.ts` 里的 Noldus 文案（未受影响）

---

## 风险与回滚

| 风险 | 缓解 |
|---|---|
| 删 `[lang]/` 后 `home.docs/blog` 等 i18n 键残留导致 typecheck 失败 | Step 4 用 `pnpm typecheck` 兜底；提前删 `types.ts` 字段 |
| 删 nextra 依赖后 `pnpm-lock.yaml` 大改 | Step 3 单独 commit |
| `app/icon.svg` 在旧 Safari 不识别 → favicon 缺失 | Next.js 15 自动生成 ICO fallback；若兼容性问题再补 `app/icon.png`（256×256） |
| 后端某处可能引用前端 `home.docs` 链接？ | 已 grep 确认仅 i18n 内部使用 |

**回滚**：每 Step 独立 commit；任意一步出问题可单独 `git revert`。

---

## 关键依赖文件参考（实施者可直接读取）

- Noldus emblem 源：`/home/wangqiuyang/resources/images/Noldus Emblem.svg`（420×290 viewBox，`#20564E` + `#10DD8B`）
- 已部署：`packages/agent/frontend/public/images/noldus-emblem.svg`（与源 bytewise 一致）
- 主题色：`packages/agent/frontend/src/styles/globals.css:193-308`（已 Noldus 配色）
- `app/page.tsx`：已 `redirect("/workspace")`
- About 真源：`src/components/workspace/settings/about-content.ts`（已 Noldus 文案）
- 当前 favicon：`public/favicon.ico`（4.2KB binary，Apr 27 — 怀疑还是鹿）

---

## 实施 agent 给的额外提示

- 实施时**先把这份 plan 文件内容镜像到** `docs/superpowers/specs/2026-05-19-pre-launch-deerflow-cleanup-design.md`（spec 文档归档位置）然后再开始改代码
- 每 Step 用中文 commit message，按 git 规范简洁描述意图
- 6 Step 全部完成后写 handoff 文档到 `docs/handoffs/2026-05/2026-05-19-pre-launch-cleanup-handoff.md`
