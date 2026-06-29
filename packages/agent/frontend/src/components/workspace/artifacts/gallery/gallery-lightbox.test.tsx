// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { GalleryLightbox } from "@/components/workspace/artifacts/gallery/gallery-lightbox";
import type { ArtifactMeta } from "@/core/artifacts/types";
import { renderWithProviders, screen } from "@/test/test-utils";

/**
 * spec 2026-06-29-assets-gallery-fixes 问题1
 *
 * lightbox 画框溢出看不到下载按钮的根因：外框 flex-col 有 max-h 但无 overflow/min-h-0，
 * 按钮行(~40px)+图(可达 80vh)+caption 之和可超 90vh → 上下等量溢出视口、顶部按钮行被推到
 * 视口上方够不到。
 *
 * 修法契约（结构 token，防回归）：
 *   1. 顶部按钮行 shrink-0——永不被图区挤掉，下载/外链/关闭始终可见。
 *   2. 图区 flex-1 + min-h-0 + overflow-auto——占剩余空间、按需滚动、能收缩。
 *   3. <img> max-h-full + max-w-full——图随框自适应，不再用固定 80vh 撑超外框。
 *   4. caption shrink-0。
 *
 * jsdom 无布局，无法测真实溢出；断言这些 layout token 落在 DOM 上 = 修法被实施且不退化。
 */

function makeImageMeta(path = "/img/plot_box.png"): ArtifactMeta {
  return { path, kind: "chart", chart_id: "box", metric: "open_arm_time_ratio" };
}

// 空函数占位（满足 lint no-empty-function：显式返回 undefined）。
const noop = () => undefined;

describe("GalleryLightbox layout contract (问题1: 画框溢出看不到下载按钮)", () => {
  it("button row is shrink-0 so download/close stay visible (never squeezed off)", () => {
    const meta = makeImageMeta();
    renderWithProviders(
      <GalleryLightbox
        open
        meta={meta}
        allItems={[meta]}
        threadId="t1"
        onClose={noop}
        onNavigate={noop}
      />,
    );

    const download = screen.getByLabelText(/download/i);
    // 按钮行是下载按钮的父容器（flex + shrink-0）。
    const buttonRow = download.parentElement;
    expect(buttonRow?.className ?? "").toMatch(/shrink-0/);
  });

  it("image area takes remaining space and can scroll (flex-1 + min-h-0 + overflow-auto)", () => {
    const meta = makeImageMeta();
    const { container } = renderWithProviders(
      <GalleryLightbox
        open
        meta={meta}
        allItems={[meta]}
        threadId="t1"
        onClose={noop}
        onNavigate={noop}
      />,
    );

    const img = container.querySelector("img");
    expect(img).toBeTruthy();
    // 图区的直接父容器是 img（+ 左右翻页按钮）所在的滚动区。
    const imageArea = img!.parentElement;
    const cls = imageArea?.className ?? "";
    expect(cls).toMatch(/flex-1/);
    expect(cls).toMatch(/min-h-0/);
    expect(cls).toMatch(/overflow-auto/);
  });

  it("image uses max-h-full/max-w-full (fits the frame, not a fixed 80vh)", () => {
    const meta = makeImageMeta();
    const { container } = renderWithProviders(
      <GalleryLightbox
        open
        meta={meta}
        allItems={[meta]}
        threadId="t1"
        onClose={noop}
        onNavigate={noop}
      />,
    );

    const img = container.querySelector("img");
    const cls = img?.className ?? "";
    expect(cls).toMatch(/max-h-full/);
    expect(cls).toMatch(/max-w-full/);
    // 关键：不再用固定 80vh 撑超外框（这是原 bug 的直接标志）。
    expect(cls).not.toMatch(/max-h-\[80vh\]/);
  });
});
