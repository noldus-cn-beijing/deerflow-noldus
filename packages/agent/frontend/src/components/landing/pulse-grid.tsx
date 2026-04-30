"use client";

import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";

interface PulseGridProps {
  className?: string;
  /** 网格点数:水平 × 垂直 */
  cols?: number;
  rows?: number;
  /** 每次脉冲间隔(ms) */
  interval?: number;
  /** 静态点的不透明度(基线) */
  baseOpacity?: number;
  /** 脉冲峰值不透明度 */
  peakOpacity?: number;
}

/**
 * Pulse Grid — 替代 Galaxy 的克制工作指示动画
 *
 * 设计意图:构成主义的几何点阵 + 日式呼吸感。
 * 每隔 interval 从中心向外扩散一圈不透明度涟漪,暗示"系统在思考",
 * 但不抢主标题。reduced-motion 时只渲染静态点阵。
 *
 * 纯 SVG + CSS,无 Canvas / WebGL / OGL,bundle 影响 <2KB。
 */
export function PulseGrid({
  className,
  cols = 40,
  rows = 24,
  interval = 2400,
  baseOpacity = 0.18,
  peakOpacity = 0.45,
}: PulseGridProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    // reduced-motion: 不启动脉冲循环
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (mq.matches) return;

    const dots = svg.querySelectorAll<SVGCircleElement>("circle[data-dot]");
    if (!dots.length) return;

    const cx = (cols - 1) / 2;
    const cy = (rows - 1) / 2;
    const maxDist = Math.sqrt(cx * cx + cy * cy);

    let timer: number | null = null;
    let mounted = true;

    const pulse = () => {
      if (!mounted) return;
      const now = Date.now();
      dots.forEach((dot) => {
        const dx = Number(dot.dataset.x) - cx;
        const dy = Number(dot.dataset.y) - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const delay = (dist / maxDist) * 800;
        window.setTimeout(() => {
          if (!mounted) return;
          dot.style.transition = "opacity 600ms ease-out";
          dot.style.opacity = String(peakOpacity);
          window.setTimeout(() => {
            if (!mounted) return;
            dot.style.transition = "opacity 1200ms ease-in";
            dot.style.opacity = String(baseOpacity);
          }, 600);
        }, delay);
        void now;
      });
    };

    pulse();
    timer = window.setInterval(pulse, interval);

    return () => {
      mounted = false;
      if (timer !== null) window.clearInterval(timer);
    };
  }, [cols, rows, interval, baseOpacity, peakOpacity]);

  const dots: { x: number; y: number }[] = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      dots.push({ x, y });
    }
  }

  const cellW = 100 / cols;
  const cellH = 100 / rows;

  return (
    <svg
      ref={svgRef}
      className={cn("pointer-events-none absolute inset-0 h-full w-full", className)}
      viewBox={`0 0 100 ${(rows / cols) * 100}`}
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      {dots.map(({ x, y }) => (
        <circle
          key={`${x}-${y}`}
          data-dot
          data-x={x}
          data-y={y}
          cx={x * cellW + cellW / 2}
          cy={y * cellH + cellH / 2}
          r={0.18}
          fill="#1A4840"
          style={{ opacity: baseOpacity }}
        />
      ))}
    </svg>
  );
}

export default PulseGrid;
