# 实施 spec：dark mode 一键切换按钮 + `.dark` token 补齐（2026-06-29）

> 实施级文档，供 agent 照此直接写代码。**前置认知修正**：dark mode 基础设施已基本建好，本 spec 只补两个真实缺口。**不实施 replay**（另立）。

---

## 〇、背景修正（务必先读）

里程碑 [frontend-generative-ux-phase0.md:64-66](../../milestone/frontend-generative-ux-phase0.md) 写「dark mode 推迟到 Phase 2，`:root` 只覆盖 light」—— **此说法已过期**。实测 dev 现状：

| 已建好 | 证据 |
|---|---|
| next-themes 已装 `^0.4.6` | `packages/agent/frontend/package.json` |
| ThemeProvider 已配 class 策略 | [layout.tsx:22](../../../packages/agent/frontend/src/app/layout.tsx) `<ThemeProvider attribute="class" enableSystem disableTransitionOnChange>` |
| `.dark` token 块已存在（globals.css:383-444，~40 个变量含 background/foreground/card/sidebar/chart/shadow 等全套） | [globals.css](../../../packages/agent/frontend/src/styles/globals.css) |
| `@custom-variant dark` Tailwind v4 钩子 | globals.css:49 |
| 设置面板已有 系统/浅/深 三选一选择器（带预览卡） | [appearance-settings-page.tsx](../../../packages/agent/frontend/src/components/workspace/settings/appearance-settings-page.tsx) |
| i18n 三模式文案齐全 | en-US.ts / zh-CN.ts `appearance` 段 |

**真实缺口只有两个**：

1. **没有一键切换按钮** —— 现在切主题必须翻进「设置 → 外观」对话框；sidebar / header 无任何 sun/moon 切换按钮（已 grep 确认 `setTheme` 仅出现在 appearance 面板、code-editor、sonner，无独立 toggle）。
2. **`.dark` 块缺 8 个 token** —— `--shadow-float` + 7 个 `--stage-*` 进度轨色未在 `.dark` 覆盖，暗色下沿用浅色值（progress-track 阶段色偏亮/不协调）。

---

## 一、目标与验收

### 目标
workspace 内提供一个 sun/moon 一键切换按钮，点击在 **亮 ↔ 暗 两态循环**；补齐 `.dark` 块缺失 token，使暗色下进度轨与悬浮阴影协调。

### 验收标准（DoD）
1. workspace（非落地页）sidebar 顶部 logo 行出现 sun/moon 图标按钮；点击在 light/dark 间切换，**立即生效无刷新**，刷新后保持（next-themes localStorage 持久化）。
2. **落地页 `/` 仍强制暗色不变**（不动 `theme-provider.tsx` 的 `forcedTheme`）。
3. 暗色下进度轨 7 个 stage 色 + 悬浮输入框阴影使用 `.dark` 专属值（非浅色回退）。
4. `pnpm check`（lint + typecheck）通过，无新增告警。
5. Playwright 截图核验：workspace 亮态、暗态各一张；点击按钮切换后状态确实翻转（`<html>` 的 `class` 含/不含 `dark`）。

### 已拍板决策（不再询问）
| 决策点 | 结论 |
|---|---|
| 切换态 | 亮 ↔ 暗 两态循环（**忽略 system**）；设置面板仍保留 system/light/dark 三选一不动 |
| 落地页 | 保留 `forcedTheme="dark"`，按钮只在 workspace 生效 |
| 按钮位置 | sidebar 顶部 logo 行（`workspace-header.tsx` 展开态），与 `SidebarTrigger` 同排 |
| 折叠态 | **不显示**切换按钮（避免与 hover-trigger 逻辑打架；折叠态那行已是 logo↔SidebarTrigger 的 hover 切换，空间挤） |
| token 补齐 | 补 7 个 `--stage-*` + `--shadow-float` 的 `.dark` 变体 |
| 验收 | Playwright 截图核两态（不强制 vitest 单测——纯视觉/交互改动） |

---

## 二、改动清单（3 处文件 + 1 个测试）

### 改动 1：新建一键切换组件 + 接入 header

**新建** `packages/agent/frontend/src/components/workspace/theme-toggle.tsx`：

```tsx
"use client";

import { MoonIcon, SunIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { useI18n } from "@/core/i18n/hooks";

export function ThemeToggle({ className }: { className?: string }) {
  const { resolvedTheme, setTheme } = useTheme();
  const { t } = useI18n();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // 防 hydration 闪烁：未 mount 时渲染占位（与现有 workspace-nav-menu 的 mounted 模式一致）
  if (!mounted) {
    return (
      <Button
        variant="ghost"
        size="icon"
        className={className}
        aria-hidden
        tabIndex={-1}
      />
    );
  }

  const isDark = resolvedTheme === "dark";
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      className={className}
      aria-label={t.workspace.toggleTheme}
      title={t.workspace.toggleTheme}
      onClick={() => setTheme(isDark ? "light" : "dark")}
    >
      {isDark ? <SunIcon className="size-4" /> : <MoonIcon className="size-4" />}
    </Button>
  );
}
```

实现要点（agent 必须遵守）：
- 用 **`resolvedTheme`** 判当前态（不是 `theme`）：若用户当前是 `system`，`theme==="system"` 但 `resolvedTheme` 给出实际亮/暗，点击后写入显式 `light`/`dark`，符合「两态循环」语义。
- **`mounted` guard 必须有**：next-themes SSR 时拿不到主题，直接渲染图标会 hydration mismatch / 闪烁。占位 Button 保持布局尺寸不跳。
- 图标语义：**当前暗→显示 SunIcon（点击回亮）**，当前亮→显示 MoonIcon（点击进暗）。即「图标表示点击后去的方向」，与多数产品一致。
- `Button` 是 shadcn copy-in primitive（`components/ui/button.tsx`，可用 `variant`/`size`，**勿改其源**）。

**接入** [workspace-header.tsx](../../../packages/agent/frontend/src/components/workspace/workspace-header.tsx) —— 只改**展开态**分支（`state !== "collapsed"`，约 50-62 行的 `<div className="flex items-center justify-between gap-2">` 块）。当前结构：

```tsx
<div className="flex items-center justify-between gap-2">
  {/* logo (NoldusWordmark) */}
  <SidebarTrigger />
</div>
```

改为在 `SidebarTrigger` 前插入 `ThemeToggle`，二者归到右侧一组：

```tsx
<div className="flex items-center justify-between gap-2">
  {/* logo (NoldusWordmark) 原样不动 */}
  <div className="flex items-center gap-1">
    <ThemeToggle />
    <SidebarTrigger />
  </div>
</div>
```

- 顶部 import 加 `import { ThemeToggle } from "./theme-toggle";`（遵守该文件的 import 排序：internal `@/` 组 → parent/sibling 组；`./theme-toggle` 属 sibling，与现有 `./github-icon` 同组——实际本文件无 sibling import，新建 sibling 组）。
- **折叠态分支（`state === "collapsed"`）不加** ThemeToggle，原样保留。

### 改动 2：补齐 `.dark` 块 token（globals.css）

在 [globals.css](../../../packages/agent/frontend/src/styles/globals.css) 的 `.dark` 块内、紧接 `--shadow-modal`（globals.css:420）与 shadow-rest 群（421）之间，或 shadow 群尾部，插入以下 8 行：

```css
  /* 悬浮输入框阴影：暗色下加深扩散（对齐 :root --shadow-float 的三层结构） */
  --shadow-float: 0 1px 2px rgba(0, 0, 0, 0.30),
                  0 8px 24px -8px rgba(0, 0, 0, 0.45),
                  0 24px 48px -16px rgba(0, 0, 0, 0.40);
  /* 进度轨 stage 色：暗色下整体提亮度（L 0.60→0.70 档）保对比，色相/chroma 沿用 :root */
  --stage-upload:    oklch(0.72 0.08 200);
  --stage-paradigm:  oklch(0.70 0.10 175);
  --stage-align:     oklch(0.70 0.12 155);
  --stage-compute:   oklch(0.72 0.11 140);
  --stage-qc:        oklch(0.76 0.12 110);
  --stage-interpret: oklch(0.74 0.12  85);
  --stage-report:    oklch(0.72 0.12  60);
```

依据（agent 不要凭空改值）：
- `:root` 原值（globals.css:371-377）L≈0.60-0.66、`--shadow-float`（334-336）三层 rgba 0.04/0.08/0.06。
- 暗色 stage：保持**色相（第三位）与 chroma（第二位）不变**，只把亮度 L 抬到 0.70-0.76 档（深底上需更亮才有同等可读性），与 `.dark` 块既有 chart 色（L 0.49-0.77）量级一致。
- 暗色 shadow-float：rgba alpha 加深到 0.30/0.45/0.40，对齐 `.dark` 块既有 `--shadow-rest/raised/overlap`（alpha 0.30-0.50 档，globals.css:421-424）的风格。
- **不新增 token、不改 `:root` 原值、不动 `.dark` 块其它变量**（守 surgical：只补缺失项）。

### 改动 3：i18n 新增按钮文案（强类型，三处同步）

i18n 是强类型链（types.ts 定义 → en-US.ts / zh-CN.ts 实现），缺一处 `pnpm typecheck` 报错。新增 key `workspace.toggleTheme`：

1. **types.ts** —— `workspace` 段加 `toggleTheme: string;`
   （定位锚点用 **`settingsAndMore`**——它是 `workspace` 段独有 key：`grep -n "settingsAndMore" src/core/i18n/locales/types.ts`。注意 `newChat` 不是好锚点，它在 `sidebar` 段、且多段重复出现。）
2. **en-US.ts** —— `workspace` 段加 `toggleTheme: "Toggle theme",`
3. **zh-CN.ts** —— `workspace` 段加 `toggleTheme: "切换主题",`

> ⚠️ key 放 `workspace.*`（不是 `settings.appearance.*`）——因为按钮在 workspace header、与 appearance 设置面板无关；放对命名空间避免语义漂移。`grep -rn "t.workspace.settingsAndMore" src/` 可见 workspace 段的现有 key 用法范式。

### 测试：Playwright 截图核两态

新建 e2e 截图核验（用项目已有的 `noldus-insight-e2e` skill / Playwright 设施，**不引 browser-use**，守 memory `feedback_e2e_testing_deterministic_playwright_not_llm_browser_use`）：

1. 起 dev（或对 prod build，性能不在本测范围，dev 即可）→ 进任一 workspace 聊天页。
2. 断言初始 `<html>` class 状态（取决于用户 localStorage，首次应跟随 system / 默认）。
3. 点击 sidebar 顶部 sun/moon 按钮 → 断言 `<html class>` 翻转（含 `dark` ↔ 不含）。
4. 各态截图存档，肉眼核进度轨 stage 色不偏白、输入框阴影协调。

---

## 三、不做什么（YAGNI / 边界）

- ❌ 不动 `theme-provider.tsx` 的落地页 `forcedTheme="dark"` 锁定。
- ❌ 不动设置面板三选一选择器（system/light/dark 仍可用）。
- ❌ 折叠态 sidebar 不加按钮。
- ❌ 不补 replay（另立 spec）。
- ❌ 不新增 dark 切换动画/过渡（`disableTransitionOnChange` 已在 layout 设定，保持）。
- ❌ 不改 `Button` / 其它 shadcn primitive 源。

---

## 四、风险 / 注意事项

1. **hydration 闪烁**：next-themes 的老问题。`mounted` guard 是硬要求；漏了会在暗色下白闪一帧 + 控制台 hydration mismatch 警告。
2. **`resolvedTheme` vs `theme`**：必须用 `resolvedTheme` 判当前显示态，否则用户处于 `system` 时点击行为不对（`theme==="system"` 无法判亮暗）。
3. **i18n 强类型**：三文件同步，否则 typecheck 红。
4. **图标方向语义**：暗态显示 Sun（去亮）、亮态显示 Moon（去暗）——别写反。
5. **OKLCH 值**：本 spec 给的暗色 stage 值是基于 L 抬升的推算，**实施后建议 Playwright 截图肉眼核**；若某 stage 仍偏暗/偏艳，可在 L±0.04 内微调，色相 chroma 不动。
6. **改动后验证**：`pnpm check` 必过；前端无后端 import 闭环风险（纯 client 组件），但若 dev 跑起来，确认 workspace 与落地页都正常。

---

## 五、实施顺序建议

1. 改动 2（token，纯 CSS，零依赖，先落）→ 改动 3（i18n 三文件）→ 改动 1（组件 + header 接入，依赖 i18n key 存在否则 typecheck 红）。
2. `pnpm check`。
3. dev 起 → Playwright 截图核两态 + 点击切换。
4. 精确路径 commit（4 文件 + 1 测试），中文 message。
