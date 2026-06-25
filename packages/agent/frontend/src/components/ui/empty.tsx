import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

function Empty({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty"
      className={cn(
        "flex min-w-0 flex-1 flex-col items-center justify-center gap-6 rounded-lg border-dashed p-6 text-center text-balance md:p-12",
        className,
      )}
      {...props}
    />
  );
}

function EmptyHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty-header"
      className={cn(
        "flex max-w-sm flex-col items-center gap-2 text-center",
        className,
      )}
      {...props}
    />
  );
}

/**
 * EmptyMedia — 空状态图标/插画槽（Phase0#6 §4.4 第一落点重做）。
 *
 * 工艺纪律（Mœbius 只借语法、不借戏服，§4.5 五铁律）：
 *  - ligne claire：清晰细线、等宽、闭合，无排线/无渐变。
 *  - 平涂色：clay-soft 暖中性底 + Forest 同源描边圈，无渐变（渐变=廉价衍生味）。
 *  - 边角克制：rounded-lg（卡片档），不无脑 rounded-3xl（§1.4）。
 *  - 负空间：图标置于柔色圆盘中央，留白表达“孤独身影在虚空”（§4.1）。
 *
 * 仍是通用槽：调用方可塞任意 lucide 图标做 children（守 spec#3 §3.1 规则2
 * “图标家族唯一=lucide”）。variant=icon 走 ligne claire 圆盘；需更完整的
 * no-data 插画用 <EmptyIllustration />（下方）。
 */
const emptyMediaVariants = cva(
  "flex shrink-0 items-center justify-center mb-2 [&_svg]:pointer-events-none [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        // default：透明底，仅做尺寸/对齐容器（原行为保留）。
        default: "bg-transparent",
        // icon：ligne claire 圆盘 — clay-soft 平涂底 + Forest 同源细描边圈。
        // 图标默认走 size-icon-md（§2.2 icon token，20px），统一不随手。
        icon: "bg-accent-clay-soft text-foreground flex size-12 shrink-0 items-center justify-center rounded-lg ring-1 ring-foreground/15 [&_svg:not([class*='size-'])]:size-icon-md",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

function EmptyMedia({
  className,
  variant = "default",
  ...props
}: React.ComponentProps<"div"> & VariantProps<typeof emptyMediaVariants>) {
  return (
    <div
      data-slot="empty-icon"
      data-variant={variant}
      className={cn(emptyMediaVariants({ variant, className }))}
      {...props}
    />
  );
}

/**
 * EmptyIllustration — Mœbius 语法空状态插画（Phase0#6 §4.4② 第一落点）。
 *
 * 一幅“孤独身影在虚空”的领域小品：一片细线叶子 + 零散轨迹点，置于 clay-soft
 * 柔底之上。纯语法零戏服（§4.5）：ligne claire 等宽闭合线、平涂填色、无渐变、
 * 画领域物体（叶子/轨迹，非科幻母题）、锁品牌 palette（Forest 线 + clay 点缀）。
 *
 * 装饰性图示：默认 aria-hidden（不向 AT 暴露）；语义由 EmptyTitle/EmptyDescription
 * 承载。传 title 则作为 SVG 的 accessible name（<title>）。
 */
function EmptyIllustration({
  className,
  title,
  ...props
}: React.ComponentProps<"svg"> & { title?: string }) {
  return (
    <svg
      data-slot="empty-illustration"
      viewBox="0 0 120 96"
      role="img"
      aria-hidden={title ? undefined : true}
      className={cn("mb-2 h-[6rem] w-[7.5rem] shrink-0", className)}
      {...props}
    >
      {title ? <title>{title}</title> : null}
      {/* 柔和 clay-soft 圆形负空间底（平涂，无渐变） */}
      <circle cx="60" cy="48" r="40" fill="var(--accent-clay-soft)" />
      {/* ligne claire 叶脉：Forest 同源绿黑、等宽闭合路径，无排线 */}
      <path
        d="M60 24c-10 8-16 18-16 28 0 8 6 14 16 14s16-6 16-14c0-10-6-20-16-28z"
        fill="none"
        stroke="var(--foreground)"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="M60 28v34M60 40c-4 2-7 5-9 8M60 40c4 2 7 5 9 8M60 52c-3 1.5-5 3-7 5M60 52c3 1.5 5 3 7 5"
        fill="none"
        stroke="var(--foreground)"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      {/* 零散 clay 轨迹点：暖色的一点呼吸，不大面积铺（§4.3 纪律） */}
      <circle cx="34" cy="64" r="2.2" fill="var(--accent-clay)" />
      <circle cx="88" cy="60" r="1.8" fill="var(--accent-clay)" />
      <circle cx="80" cy="72" r="1.4" fill="var(--accent-clay)" />
    </svg>
  );
}

function EmptyTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty-title"
      className={cn("text-lg font-medium tracking-tight", className)}
      {...props}
    />
  );
}

function EmptyDescription({ className, ...props }: React.ComponentProps<"p">) {
  return (
    <div
      data-slot="empty-description"
      className={cn(
        "text-muted-foreground [&>a:hover]:text-primary text-sm/relaxed [&>a]:underline [&>a]:underline-offset-4",
        className,
      )}
      {...props}
    />
  );
}

function EmptyContent({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty-content"
      className={cn(
        "flex w-full max-w-sm min-w-0 flex-col items-center gap-4 text-sm text-balance",
        className,
      )}
      {...props}
    />
  );
}

export {
  Empty,
  EmptyHeader,
  EmptyTitle,
  EmptyDescription,
  EmptyContent,
  EmptyMedia,
  EmptyIllustration,
};
