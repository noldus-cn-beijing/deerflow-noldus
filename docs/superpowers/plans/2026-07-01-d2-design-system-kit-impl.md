# D2 设计系统 kit 业务组件层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `workspace/kit/` 建 4 个统一业务原语（StatusCard / StatusBadge+AccentBar / EmptyState / LoadingState），所有对话业务卡片改成消费它们——让 workspace 新 UI 走库、不再各卡自造状态色/accent bar/空态。

**Architecture:** 三层设计系统：底层 `ui/`（shadcn copy-in，只改 token 保 API）+ `ai-elements/`（registry-pulled，只 wrap 不改）→ 中层 `kit/`（D2 新建的 ethoinsight 业务原语，**只引 D1 token**）→ 上层 workspace 卡片（消费 kit）。视觉一致靠 D1 的 globals.css token 三层同步变样，kit 不立第二份视觉标准。

**Tech Stack:** Next.js 16, React 19, TypeScript 5.8, Tailwind v4, vitest + @testing-library/react。前端 `packages/agent/frontend/`（deerflow 子树外）。

## Global Constraints

- **依赖 D1（已落地 #257）**：kit 原语**只引 D1 token**（`--color-status-*` via `bg-status-*`、`--ease-brand-out`、`--radius-*`、`--shadow-float`、`--dur-*`），**无自定义色值/间距硬编码**。
- **状态色板绝不红绿对立**：用 D1 色盲安全 token（守 memory `feedback_frontend_design_japanese_minimal_motion_craft`）。
- **不改 `ai-elements/` 结构**（re-pull 会覆盖）——只在 kit/ 里 wrap。
- **不改 `ui/` 任何导出/API**——只能改 token（守 CLAUDE.md copy-in 规则）。
- **accent = 左侧细竖条，非整卡变色**（日式克制）。
- **color-not-only 三件套**：状态永远色 + 图标 + 文字同时在，绝不只靠颜色。
- **不动后端**、不动 `settings/`/`agents/` 等非对话面板功能逻辑（迁移触及时只顺手收敛视觉）。
- **已知 baseline**：`utils.test.ts` 2 个 isStreaming 红是 pre-existing，非本轮回归（memory `project_frontend_vitest_already_set_up_2_red_streaming_tests`）。
- 测试：`cd packages/agent/frontend && npx vitest run <path>`。提交前 `pnpm check`（lint+type）须 0。
- **catastrophic forgetting 自检**：改共享卡片前 grep 所有消费点，全量跑 vitest，不只跑新测。

---

### Task 1: `StatusBadge` + `AccentBar`（状态色原语 — SSOT）

**Files:**
- Create: `packages/agent/frontend/src/components/workspace/kit/status-badge.tsx`
- Test: `packages/agent/frontend/src/components/workspace/kit/status-badge.test.tsx`

**Interfaces:**
- Produces:
  - `type Status = "success" | "warning" | "danger" | "info"` (exported)
  - `StatusBadge: (props: { status: Status; label?: string; size?: "sm" | "md" }) => JSX.Element` — 三件套：图标 + `text-status-*` 文字 + label。
  - `AccentBar: (props: { status: Status }) => JSX.Element` — 左侧细竖条 `bg-status-*`。
  - `STATUS_ICON: Record<Status, LucideIcon>` (exported, so cards reuse the same icon per status)。

> **这是状态视觉的唯一来源**。StatusCard 的 accent bar 复用 `AccentBar`；所有卡片的状态色复用这里。`status → bg-status-*` 是从 `decision-card.tsx:54 toneFor` 里重复的映射收敛出来的 SSOT。

- [ ] **Step 1: Write the failing test**

```tsx
// status-badge.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AccentBar, StatusBadge } from "./status-badge";

describe("StatusBadge (color-not-only 三件套)", () => {
  it("renders icon + colored text + label for each status", () => {
    const { container, rerender } = render(<StatusBadge status="danger" label="失败" />);
    expect(screen.getByText("失败")).toBeInTheDocument();
    // color: uses the danger status token class (not a raw hex).
    expect(container.querySelector('[class*="text-status-danger"]')).not.toBeNull();
    // icon: an svg is present (the third leg of color-not-only).
    expect(container.querySelector("svg")).not.toBeNull();

    rerender(<StatusBadge status="success" label="完成" />);
    expect(container.querySelector('[class*="text-status-success"]')).not.toBeNull();
  });
});

describe("AccentBar (细竖条，非整卡变色)", () => {
  it("renders a thin status-colored bar", () => {
    const { container } = render(<AccentBar status="warning" />);
    const bar = container.firstChild as HTMLElement;
    expect(bar.className).toMatch(/bg-status-warning/);
    // thin: has a small width class (w-1 / w-0.5 / w-[3px]), not a full-card bg.
    expect(bar.className).toMatch(/w-/);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/kit/status-badge.test.tsx`
Expected: FAIL — cannot resolve `./status-badge`.

- [ ] **Step 3: Write minimal implementation**

```tsx
// status-badge.tsx
import { AlertTriangleIcon, CheckCircle2Icon, InfoIcon, XCircleIcon, type LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export type Status = "success" | "warning" | "danger" | "info";

/** Icon per status — SSOT so every card shows the same glyph (color-not-only). */
export const STATUS_ICON: Record<Status, LucideIcon> = {
  success: CheckCircle2Icon,
  warning: AlertTriangleIcon,
  danger: XCircleIcon,
  info: InfoIcon,
};

// status → D1 status token utility (text/bg). NEVER a raw hex; NEVER red-green only.
const TEXT_CLASS: Record<Status, string> = {
  success: "text-status-success",
  warning: "text-status-warning",
  danger: "text-status-danger",
  info: "text-status-info",
};
const BAR_CLASS: Record<Status, string> = {
  success: "bg-status-success",
  warning: "bg-status-warning",
  danger: "bg-status-danger",
  info: "bg-status-info",
};

export function StatusBadge({ status, label, size = "md" }: { status: Status; label?: string; size?: "sm" | "md" }) {
  const Icon = STATUS_ICON[status];
  return (
    <span className={cn("inline-flex items-center gap-1", TEXT_CLASS[status], size === "sm" ? "text-xs" : "text-sm")}>
      <Icon className={size === "sm" ? "size-3.5" : "size-4"} aria-hidden />
      {label != null && <span className="font-medium">{label}</span>}
    </span>
  );
}

/** Thin left accent bar — status color as a slim vertical strip, never a full-card fill. */
export function AccentBar({ status }: { status: Status }) {
  return <div className={cn("w-1 shrink-0 self-stretch rounded-full", BAR_CLASS[status])} aria-hidden />;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/kit/status-badge.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/frontend/src/components/workspace/kit/status-badge.tsx packages/agent/frontend/src/components/workspace/kit/status-badge.test.tsx
git commit -m "feat(kit): D2 Task1 — StatusBadge + AccentBar (status-color SSOT)"
```

---

### Task 2: `StatusCard`（卡片壳）

**Files:**
- Create: `packages/agent/frontend/src/components/workspace/kit/status-card.tsx`
- Test: `packages/agent/frontend/src/components/workspace/kit/status-card.test.tsx`

**Interfaces:**
- Consumes: `AccentBar`, `Status` (Task 1).
- Produces: `StatusCard: (props: { status: Status; title?: ReactNode; children?: ReactNode; pulse?: boolean; className?: string }) => JSX.Element` — 壳 = 左 `AccentBar` + 内容区；`--shadow-float` + `--radius-lg` + 间距 token；`pulse` 时用 D1 `animate-pulse-warm`。

- [ ] **Step 1: Write the failing test**

```tsx
// status-card.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusCard } from "./status-card";

describe("StatusCard", () => {
  it("renders an accent bar (status color) + title + children, not a full-card fill", () => {
    const { container } = render(
      <StatusCard status="warning" title="分析已暂停">
        <p>正文</p>
      </StatusCard>,
    );
    expect(screen.getByText("分析已暂停")).toBeInTheDocument();
    expect(screen.getByText("正文")).toBeInTheDocument();
    // accent bar present (thin status strip)
    expect(container.querySelector('[class*="bg-status-warning"]')).not.toBeNull();
    // shell uses radius/shadow tokens, not raw values
    const shell = container.firstChild as HTMLElement;
    expect(shell.className).toMatch(/rounded-|shadow-/);
  });

  it("applies pulse animation only when pulse=true", () => {
    const { container, rerender } = render(<StatusCard status="danger" pulse title="x" />);
    expect(container.innerHTML).toMatch(/pulse-warm/);
    rerender(<StatusCard status="danger" title="x" />);
    expect(container.innerHTML).not.toMatch(/pulse-warm/);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/kit/status-card.test.tsx`
Expected: FAIL — cannot resolve `./status-card`.

- [ ] **Step 3: Write minimal implementation**

```tsx
// status-card.tsx
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

import { AccentBar, type Status } from "./status-badge";

export function StatusCard({
  status,
  title,
  children,
  pulse = false,
  className,
}: {
  status: Status;
  title?: ReactNode;
  children?: ReactNode;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex gap-3 rounded-lg bg-background p-3 shadow-float",
        pulse && "animate-pulse-warm",
        className,
      )}
    >
      <AccentBar status={status} />
      <div className="min-w-0 flex-1">
        {title != null && <div className="text-sm font-medium">{title}</div>}
        {children}
      </div>
    </div>
  );
}
```

> If `shadow-float` or `animate-pulse-warm` utility is not emitted, verify it exists in `src/styles/globals.css` (D1 token `--shadow-float`, `animate-pulse-warm`). Both were confirmed present in D1 #257.

- [ ] **Step 4: Run to verify it passes**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/kit/status-card.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/frontend/src/components/workspace/kit/status-card.tsx packages/agent/frontend/src/components/workspace/kit/status-card.test.tsx
git commit -m "feat(kit): D2 Task2 — StatusCard shell (accent bar + shadow/radius tokens)"
```

---

### Task 3: `EmptyState`

**Files:**
- Create: `packages/agent/frontend/src/components/workspace/kit/empty-state.tsx`
- Test: `packages/agent/frontend/src/components/workspace/kit/empty-state.test.tsx`

**Interfaces:**
- Produces: `EmptyState: (props: { icon?: LucideIcon; title: string; description?: string; action?: ReactNode }) => JSX.Element`.

- [ ] **Step 1: Write the failing test**

```tsx
// empty-state.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { EmptyState } from "./empty-state";

describe("EmptyState", () => {
  it("renders title, optional description, and optional action", () => {
    render(<EmptyState title="暂无产物" description="分析完成后在此显示" action={<button>刷新</button>} />);
    expect(screen.getByText("暂无产物")).toBeInTheDocument();
    expect(screen.getByText("分析完成后在此显示")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "刷新" })).toBeInTheDocument();
  });

  it("renders without description/action", () => {
    render(<EmptyState title="空" />);
    expect(screen.getByText("空")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/kit/empty-state.test.tsx`
Expected: FAIL — cannot resolve `./empty-state`.

- [ ] **Step 3: Write minimal implementation**

```tsx
// empty-state.tsx
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-2 py-8 text-center">
      {Icon != null && <Icon className="text-muted-foreground size-6" aria-hidden />}
      <p className="text-sm font-medium">{title}</p>
      {description != null && <p className="text-muted-foreground text-xs">{description}</p>}
      {action != null && <div className="mt-1">{action}</div>}
    </div>
  );
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/kit/empty-state.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/frontend/src/components/workspace/kit/empty-state.tsx packages/agent/frontend/src/components/workspace/kit/empty-state.test.tsx
git commit -m "feat(kit): D2 Task3 — EmptyState"
```

---

### Task 4: `LoadingState`

**Files:**
- Create: `packages/agent/frontend/src/components/workspace/kit/loading-state.tsx`
- Test: `packages/agent/frontend/src/components/workspace/kit/loading-state.test.tsx`

**Interfaces:**
- Produces: `LoadingState: (props: { variant: "spinner" | "skeleton" | "dots"; label?: string }) => JSX.Element` — 入场 ease-out 非 linear（D1 `--ease-brand-out`），克制不抢焦点。

- [ ] **Step 1: Write the failing test**

```tsx
// loading-state.test.tsx
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { LoadingState } from "./loading-state";

describe("LoadingState", () => {
  it.each(["spinner", "skeleton", "dots"] as const)("renders %s variant", (variant) => {
    const { container } = render(<LoadingState variant={variant} />);
    expect(container.firstChild).not.toBeNull();
    expect(container.querySelector(`[data-variant="${variant}"]`)).not.toBeNull();
  });

  it("renders optional label", () => {
    const { getByText } = render(<LoadingState variant="spinner" label="加载中" />);
    expect(getByText("加载中")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/kit/loading-state.test.tsx`
Expected: FAIL — cannot resolve `./loading-state`.

- [ ] **Step 3: Write minimal implementation**

```tsx
// loading-state.tsx
import { Loader2Icon } from "lucide-react";

import { cn } from "@/lib/utils";

export function LoadingState({ variant, label }: { variant: "spinner" | "skeleton" | "dots"; label?: string }) {
  return (
    <div data-variant={variant} className="text-muted-foreground flex items-center gap-2 text-sm">
      {variant === "spinner" && <Loader2Icon className="size-4 animate-spin" aria-hidden />}
      {variant === "skeleton" && <span className={cn("h-4 w-24 rounded bg-muted", "animate-skeleton-entrance")} aria-hidden />}
      {variant === "dots" && <span className="animate-suggestion-in inline-flex gap-0.5" aria-hidden>•••</span>}
      {label != null && <span>{label}</span>}
    </div>
  );
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/kit/loading-state.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/frontend/src/components/workspace/kit/loading-state.tsx packages/agent/frontend/src/components/workspace/kit/loading-state.test.tsx
git commit -m "feat(kit): D2 Task4 — LoadingState (3 variants, ease-out entrance)"
```

---

### Task 5: 迁移批 1 — 核心对话卡片消费 kit

**Files:**
- Modify: `packages/agent/frontend/src/components/workspace/messages/decision-card.tsx` (replace inline `toneFor` accent/soft with `AccentBar`/`StatusCard` + `STATUS_ICON`)
- Modify: `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx`
- Modify: `packages/agent/frontend/src/components/workspace/messages/quality-warning-banner.tsx`
- Test: existing `decision-card.test.tsx` / `subtask-card` / relevant tests must stay green (no new test file required; this is a refactor to consume kit).

**Interfaces:**
- Consumes: `StatusCard`, `StatusBadge`, `AccentBar`, `Status`, `STATUS_ICON` (Tasks 1-2).

> **This is a shared-component refactor — grep consumers + full vitest before/after (catastrophic-forgetting guard).** The migration maps each card's local status notion onto the kit `Status` type; behavior (which icon/color per state, pulse) must not change.

- [ ] **Step 1: Baseline — grep consumers + run full vitest to record green set**

Run: `cd packages/agent/frontend && npx vitest run 2>&1 | tail -20`
Record the pass/fail counts (expect the known 2 pre-existing `utils.test.ts` reds). This is your no-regression baseline.

- [ ] **Step 2: Migrate `decision-card.tsx` to consume kit**

Replace the `toneFor` return shape so `accent`/`soft` come from the kit `Status` instead of inline `bg-status-*` strings, and render the card shell via `StatusCard`. Map: `risk_confirmation → "danger"`, `suggestion → "info"`, `approach_choice → "warning"`, default → `"warning"`. Keep the exact same icon per type by passing through the existing `icon` (or switch to `STATUS_ICON[status]` only if the icon matches — decision-card uses type-specific icons like `ShieldAlertIcon`, so keep its own `icon` field; only the color/bar goes through kit).

Concretely: change `toneFor` to return `{ status: Status; icon: LucideIcon; pulse: boolean; titleKey: ... }` (drop `accent`/`soft` raw strings), and in the JSX replace the hand-rolled accent bar `<div className={tone.accent} .../>` with `<AccentBar status={tone.status} />` (or wrap the whole card in `<StatusCard status={tone.status} pulse={tone.pulse}>`), and any `text-status-*` title color with `StatusBadge`'s token via the same `status`.

- [ ] **Step 3: Migrate `subtask-card.tsx` and `quality-warning-banner.tsx` the same way**

For `quality-warning-banner.tsx`: its `warningStyle` maps severity → `text-red-600`/`text-orange-600`/`text-yellow-600`/`text-blue-600` (raw Tailwind colors, NOT D1 tokens — a real inconsistency). Map severity → kit `Status` (`critical+blocks → "danger"`, `critical → "danger"`, `warning → "warning"`, `info → "info"`) and render via `StatusBadge`/`StatusCard` so colors go through D1 tokens. (Note: the L4-3 `w.code` exposure is a SEPARATE findings fix — do NOT fold it in here; this task is visual consolidation only.)

- [ ] **Step 4: Run the three cards' tests + full vitest (no regression)**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/messages/ && npx vitest run 2>&1 | tail -20`
Expected: the three cards' tests PASS; full-suite counts match Step 1 baseline (only the known 2 reds).

- [ ] **Step 5: `pnpm check` + commit**

```bash
cd packages/agent/frontend && pnpm check
```
Expected: 0 errors.

```bash
git add packages/agent/frontend/src/components/workspace/messages/decision-card.tsx packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx packages/agent/frontend/src/components/workspace/messages/quality-warning-banner.tsx
git commit -m "refactor(kit): D2 Task5 — batch1 core cards consume kit (decision/subtask/quality-warning)"
```

---

### Task 6: 迁移批 2 — 产物/列表/空态消费 kit

**Files:**
- Modify: `packages/agent/frontend/src/components/workspace/artifacts/thread-assets-panel.tsx` (empty state → `EmptyState`)
- Modify: `packages/agent/frontend/src/components/workspace/artifacts/gallery/artifact-gallery.tsx` (status/failed banner → `StatusBadge`/`StatusCard`)
- Modify: `packages/agent/frontend/src/components/workspace/todo-list.tsx`
- Modify: `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` (empty/status)

**Interfaces:**
- Consumes: `EmptyState`, `StatusBadge`, `StatusCard` (Tasks 1-3).

- [ ] **Step 1: Baseline full vitest** (same as Task 5 Step 1 — record counts).

- [ ] **Step 2: Migrate empty states → `EmptyState`**

In `thread-assets-panel.tsx`, replace the inline empty `<p className="text-muted-foreground text-sm">{t.gallery.assetsEmpty}</p>` (line ~95) with `<EmptyState title={t.gallery.assetsEmpty} />`. Do the same for `todo-list` / `message-group` empty branches.

- [ ] **Step 3: Migrate `artifact-gallery.tsx` failed-banner → kit**

The failed-count banner (line ~103, `failedCount > 0`) renders a status-colored alert — route its color/icon through `StatusBadge`/`StatusCard` (`status="danger"`). **Do NOT touch the 6 dogfood invariants** (chart_type grouping, per_subject collapse, aggregate section, lightbox/compare/ZIP) — only the visual status wrapper.

- [ ] **Step 4: Full vitest + gallery tests (no regression)**

Run: `cd packages/agent/frontend && npx vitest run src/components/workspace/artifacts/ && npx vitest run 2>&1 | tail -20`
Expected: gallery/panel tests PASS; full-suite matches baseline.

- [ ] **Step 5: `pnpm check` + commit**

```bash
cd packages/agent/frontend && pnpm check
git add packages/agent/frontend/src/components/workspace/artifacts/thread-assets-panel.tsx packages/agent/frontend/src/components/workspace/artifacts/gallery/artifact-gallery.tsx packages/agent/frontend/src/components/workspace/todo-list.tsx packages/agent/frontend/src/components/workspace/messages/message-group.tsx
git commit -m "refactor(kit): D2 Task6 — batch2 artifacts/list/empty consume kit"
```

---

### Task 7: 迁移批 3（边角）+ 全量收敛验证

**Files:**
- Modify: 剩余 accent/status/空态用法 — grep-driven: `attachment-chip` / `message-attachments` / `input-box` / `workspace-nav-menu` 等对话业务卡片里绕过 kit 的用法。
- Test: full vitest green (minus baseline 2).

**Interfaces:**
- Consumes: all 4 kit primitives.

- [ ] **Step 1: Grep for bypass sites**

Run:
```bash
cd packages/agent/frontend && grep -rnE "bg-status-|text-status-|border-l-.*status|text-(red|orange|yellow|green|blue)-[0-9]" src/components/workspace/ --include=*.tsx | grep -v "/kit/"
```
This lists every remaining hand-rolled status color in **对话业务卡片**. Migrate each to the kit primitive (StatusBadge/AccentBar/StatusCard). Exempt (list them in the commit body): `settings/`, `agents/` non-conversation panels — not force-migrated.

- [ ] **Step 2: Migrate the edge-case sites** (one component at a time, keeping behavior identical).

- [ ] **Step 3: Verify 全量收敛 — grep proves no bypass in conversation cards**

Run the same grep from Step 1. Expected: remaining hits are ONLY in `/kit/` (the SSOT) or the explicitly-exempt `settings/`/`agents/` panels. List the exemptions.

- [ ] **Step 4: kit-only-token grep (no hardcoded values in kit)**

Run:
```bash
cd packages/agent/frontend && grep -rnE "#[0-9a-fA-F]{3,6}|text-(red|orange|yellow|green|blue)-[0-9]" src/components/workspace/kit/
```
Expected: NO hits (kit only references D1 `bg-status-*`/`text-status-*` token utilities, never raw hex or Tailwind palette colors).

- [ ] **Step 5: Full suite + check + commit**

Run: `cd packages/agent/frontend && npx vitest run 2>&1 | tail -20 && pnpm check`
Expected: full-suite green except known 2 baseline reds; `pnpm check` 0.

```bash
git add packages/agent/frontend/src/components/workspace/
git commit -m "refactor(kit): D2 Task7 — batch3 edge cases + full convergence (grep-verified)"
```

---

## Self-Review

**Spec coverage:**
- 4 kit 原语齐全各带 vitest → Tasks 1-4 ✅ (StatusBadge/AccentBar, StatusCard, EmptyState, LoadingState)
- 批 1-3 迁移 → Tasks 5-7 ✅
- grep 证对话卡片不再自造 + 列豁免面板 → Task 7 Step 3 ✅
- sync 兼容（未改 ai-elements 结构 / ui 导出）→ Global Constraints + 迁移只碰 workspace 卡片 ✅
- kit 只引 D1 token 无硬编码 → Task 7 Step 4 grep ✅
- 现有 vitest 不回归（排除 2 红 baseline）→ 每迁移 Task 有 baseline+回归步骤 ✅

**Placeholder scan:** 原语 Task（1-4）全含完整代码。迁移 Task（5-7）是 refactor，给了精确文件+映射规则+grep 命令而非新代码块（迁移本质是"把现有 inline 映射换成 kit 调用"，规则已具体到每个 status 映射）。无 TBD/TODO。

**Type consistency:** `Status` type 在 Task 1 定义，Tasks 2/5/6/7 一致消费；`StatusBadge`/`AccentBar`/`StatusCard`/`EmptyState`/`LoadingState` 签名 Task 1-4 定义、迁移 Task 一致引用；`STATUS_ICON` Task 1 导出、Task 5 决策卡按需复用。

**Scope note:** L4-3 (`quality-warning code` 暴露) 和 L4-4 (token 指示器) 是 findings 修复，**明确不在本 plan**（Task 5 Step 3 显式排除）——避免与 findings 批次撞车。
