# 前端审美升级交接文档

**日期**:2026-04-30
**交接人**:Claude(本会话)
**接手对象**:下一位 AI Agent / 开发者
**任务状态**:**9 个 Checkpoint 全部完成,待人工肉眼验证 + Lighthouse**

---

## 1. 当前任务目标

对 EthoInsight 前端(`packages/agent/frontend/`)做**纯视觉系统升级**,已完成实施。目标:

- **设计调性**:日式极简 × 构成主义融合,科研工具的克制感
- **底色**:浅绿米色 `oklch(0.972 0.012 145)` (冷调,理论无黄味)
- **品牌色**:Lime `#10DD8B` **仅 2 处显式** (hero CTA + input-box Submit),Forest `#1A4840` 为主文字
- **字体**:OPPO Sans 4.0 自托管 (latin 40KB + zh 3.6MB woff2)
- **质感**:三层系统 (纸纹 SVG body::before + 玻璃白 glass-card + 实白 #FFF)
- **范围**:landing + workspace + ai-elements + settings 全覆盖,dark mode 本轮不动

---

## 2. 当前进展

### ✅ 已完成 — 9/9 Checkpoints,7 commits

```
a4b8aab9 前端: 清理 ogl 依赖 + 全量视觉升级完成
42422c17 前端: button 加 brand variant + settings 适配 + OPPO Sans 声明
9fead071 前端: ai-elements 切 light Shiki 主题 + reasoning 视觉调整
77eb788e 前端: workspace 视觉切到浅绿米色 + 玻璃态 + 克制品牌色
61388c64 前端: landing 切 light 主题 + Galaxy 替换为 PulseGrid
147e365f 前端: 重写设计 token (浅绿米底 + OPPO Sans + 三层质感系统)
e5223f41 前端: 添加 OPPO Sans 4.0 字体资源 (latin + zh 子集)
```

### 改动的核心文件 (39 个)

| 文件 | 改动 |
|------|------|
| `src/styles/globals.css` | 完整重写:color token、OPPO Sans @font-face、纸纹 body::before、玻璃态 glass-card、type scale、radius |
| `public/fonts/OPPOSans-latin.woff2` | 新建,40KB,Latin 子集 (原可变字体) |
| `public/fonts/OPPOSans-zh.woff2` | 新建,3.6MB,CJK static wght=400 子集 |
| `public/fonts/LICENSE.txt` | OPPO Sans 许可证副本 |
| `src/app/page.tsx` | `bg-[#0a0a0a]` → `bg-background` |
| `src/components/landing/hero.tsx` | Galaxy+FlickeringGrid → PulseGrid,文字 white→foreground,CTA variant="brand" |
| `src/components/landing/pulse-grid.tsx` | 新建,纯 SVG 点阵 40×24 脉冲动画 |
| `src/components/landing/header.tsx` | 深色底→`bg-background/85 backdrop-blur-sm`,gradient pink→Lime/Forest |
| `src/components/landing/footer.tsx` | 深底→Forest deep `#1A4840` |
| `src/components/landing/section.tsx` | title gradient white/gray→纯 `text-foreground` |
| `src/components/landing/sections/community-section.tsx` | 移除 AuroraText,直接纯文字 |
| `src/components/landing/sections/skills-section.tsx` | progressive-skills lazy split,bg-white/2→移除 |
| `src/components/landing/sections/sandbox-section.tsx` | zinc-* → token (border-border/bg-secondary/text-muted-foreground) |
| `src/components/landing/sections/case-study-section.tsx` | 暗底渐变→bg-background/90,text-white→foreground |
| `src/components/landing/sections/whats-new-section.tsx` | COLOR #0a0a0a→#F4F1EA |
| `src/components/landing/progressive-skills-animation.tsx` | 全量暗色→token 替换 (zinc→muted-foreground,white→foreground,border-zinc→border-border) |
| `src/components/ui/button.tsx` | 新增 brand variant (`bg-brand text-brand-foreground hover:bg-brand-hover`) |
| `src/components/ui/word-rotate.tsx` | AuroraText 金黄→冷绿 `["#20564E","#1A4840","#3A6E66"]` |
| `src/app/workspace/chats/[thread_id]/page.tsx` | h-12→h-14 |
| `src/components/workspace/welcome.tsx` | 移除 AuroraText,display-md size |
| `src/components/workspace/agent-welcome.tsx` | 已用 token,自动适配 |
| `src/components/workspace/input-box.tsx` | rounded-2xl→rounded-3xl,加 glass-card,Submit brand 色 |
| `src/components/workspace/messages/markdown-content.tsx` | 链接 underline-offset-4 hover:text-brand |
| `src/components/workspace/streaming-indicator.tsx` | dot bg-[#a3a1a1]→bg-foreground/50 |
| `src/components/ai-elements/message.tsx` | 用户气泡 bg-secondary→glass-card,rounded-lg→rounded-2xl,px-5 py-3.5 |
| `src/components/ai-elements/code-block.tsx` | 已用 one-light/one-dark-pro,自动适配 |
| `src/components/ai-elements/reasoning.tsx` | 已用 token,自动适配 |
| `src/components/workspace/settings/about-settings-page.tsx` | 新增 OPPO Sans 许可证声明 |
| `src/components/workspace/settings/appearance-settings-page.tsx` | border-slate-200→border-border 等 |
| `package.json` | 移除 ogl 依赖 |
| 其他 sidebar/artifacts/thread-title/token-usage 等 | 已用 token,自动适配 |

### 保留不动 (源码留存)

- `src/components/ui/galaxy.jsx` + `galaxy.css` — 不再被任何文件 import
- `src/components/ui/flickering-grid.tsx` — 不再被任何文件 import
- `src/components/ui/aurora-text.tsx` — 在 3 处调用方已去掉包裹
- `.dark` 块 — 本轮不动

### ⏳ 待人工验证 (无浏览器环境)

1. **底色视觉校对** (CP4):基线 `oklch(0.972 0.012 145)` 是冷绿米理论值,需浏览器实测确认不发黄
2. **字体渲染**:DevTools → Network → Fonts 确认两个 woff2 200;检查中英文字形是 OPPO Sans 非系统字体
3. **对比度实测** (DevTools Computed → Contrast):
   - text-primary `#1A4840` on bg ≈ 13.5:1 (理论 ✅)
   - muted-foreground `#50615C` on bg ≈ 5.7:1 (理论 ✅)
   - Lime `#1A4840` on `#10DD8B` ≈ 8.5:1 (理论 ✅)
4. **Lighthouse** (landing mobile): `zh.woff2` 3.6MB 是主要瓶颈,perf 可能达不到 85
5. **功能完整性**:发消息→上传 EthoVision→切 thread→settings→artifact 预览
6. **响应式**:375/768/1280px 三档不破版

---

## 3. 关键上下文

### 设计 token (已锁定)

```
底色:--background: oklch(0.972 0.012 145)  (#EFF2EA 附近)
文字:--foreground: #1A4840 (Forest deep)
次级:--muted-foreground: #50615C
品牌:--brand: #10DD8B (Lime,仅 CTA)
边框:--border: oklch(0.20 0.04 168 / 0.12)
卡片:--card: oklch(1 0 0 / 0.88) (玻璃白)
侧栏:--sidebar: oklch(0.962 0.014 145) (第二层纸)
```

### 字体

- OPPO Sans 4.0 Variable Font (wght 100-700)
- latin.woff2 40KB (原可变字体,全字重)
- zh.woff2 3.6MB (static wght=400, GB2312 Level 1 ~3755 汉字)
- 许可证声明在 about-settings-page

### 关键约束

- 包管理器 `pnpm`,Node 22+
- **不要用** `make dev` (端口 2026,全栈),**用** `pnpm dev` (端口 3000,Turbopack)
- `components/ui/` 和 `ai-elements/` 是 ESLint-ignored,改前确认
- 工作目录:`/home/wangqiuyang/noldus-insight/packages/agent/frontend/`

---

## 4. 已知风险

| 风险 | 严重度 | 应对 |
|------|--------|------|
| zh.woff2 3.6MB | 高 | 当前 static wght=400;可进一步缩减字符集或拆分 wght 分包;Lighthouse 可能不达标 |
| 底色未经肉眼校 | 中 | 冷绿米理论安全,但需实测确认;若偏黄:hue→150-160°,chroma→0.008-0.010 |
| progressive-skills 24KB | 低 | 已 lazy split,仅滚动到 section 才加载 |
| ogl 已移除 | ✅ | galaxy.jsx 保留但不被 import,ogl 已从 deps 移除 |

---

## 5. 未完成事项

### P0 — 阻塞

无。

### P1 — 人工验证 (需要浏览器)

- [ ] 底色并排校对 (与 claude.ai 对比)
- [ ] 字体加载确认 (DevTools Network → Fonts)
- [ ] 对比度 DevTools 实测
- [ ] Lighthouse 跑分
- [ ] 功能完整性回放

### P2 — 可选优化

- [ ] zh.woff2 缩减 (当前 3.6MB → 目标 < 1MB)
- [ ] 全站 dark mode token 升级 (本轮明确不做)

---

## 6. 建议接手路径

1. **验证基础可用**:
   ```bash
   cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
   pnpm dev
   # 浏览器 http://localhost:3000
   ```
2. **做人工肉眼验证** (见 §5 P1 清单)
3. **若底色偏黄**:改 `globals.css` 中 `--background` 的 hue 到 150-160°,chroma 降到 0.008-0.010
4. **若对比度不达标**:改 `--muted-foreground` 从 `#50615C` 到 `#3F4F4A`
5. **若 Lighthouse 不达标**:先缩小 zh.woff2 字符集

---

## 7. 附录:重要资源链接

| 资源 | 路径 |
|------|------|
| 设计 plan | `~/.claude/plans/spicy-giggling-star.md` |
| 实施手册 (含附录 C) | `~/.claude/plans/spicy-giggling-star-impl.md` |
| OPPO Sans 字体 | `packages/agent/frontend/public/fonts/` |
| 前端工作目录 | `/home/wangqiuyang/noldus-insight/packages/agent/frontend/` |
| Git commits | `dev` 分支,最近 7 个 commit |
