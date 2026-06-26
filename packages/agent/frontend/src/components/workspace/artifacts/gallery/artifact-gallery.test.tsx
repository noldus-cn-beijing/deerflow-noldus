// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { ArtifactGallery } from "@/components/workspace/artifacts/gallery/artifact-gallery";
import type { ArtifactMeta } from "@/core/artifacts/types";
import { renderWithProviders, screen, userEvent } from "@/test/test-utils";

/**
 * spec 2026-06-26-gallery-per-subject-collapse-affordance
 *
 * per_subject 分组默认折叠是对的（防图墙），但折叠入口必须明显可点——E2E agent
 * 滚动 8 次都没发现要点开，误判「112 张不渲染」。这里断言「可发现性 affordance」
 * 而非默认折叠态本身：
 *   1. 折叠态文案带动作提示（不再是裸 `Per-subject (N)`），让用户一眼明白可展开。
 *   2. 折叠态有可点 affordance（role=button + aria-expanded + 视觉 token）。
 *   3. 点击翻转折叠↔展开态（保持默认折叠防图墙；虚拟化缩略图渲染靠真机/E2E 验）。
 */

function perSubjectMeta(n: number): ArtifactMeta[] {
  return Array.from({ length: n }, (_, i) => ({
    path: `/charts/per_subject_${i}.png`,
    kind: "chart",
    output_mode: "per_subject",
    chart_id: `ps_${i}`,
  }));
}

/** 定位 per_subject 折叠入口按钮（带 aria-expanded 的那个）。 */
function perSubjectToggle() {
  return screen
    .getAllByRole("button")
    .find((b) => b.getAttribute("aria-expanded") !== null);
}

describe("ArtifactGallery per-subject collapse affordance", () => {
  it("collapsed toggle surfaces an explicit click-to-expand hint (not bare label)", () => {
    const metas = perSubjectMeta(112);
    renderWithProviders(
      <ArtifactGallery artifacts={metas} threadId="t1" />,
    );

    const toggle = perSubjectToggle();
    expect(toggle).toBeTruthy();
    // 默认折叠。
    expect(toggle!.getAttribute("aria-expanded")).toBe("false");
    // 计数仍在，让用户知道有多少张。
    expect(toggle!.textContent ?? "").toContain("112");
    // 关键：文案带「展开/Expand」动作词，不再是裸 "Per-subject (112)"。
    // E2E 误判根因 = 折叠态不像可点击区，故断言出现明确的动作提示。
    expect(/expand|show|view|展开|查看/i.test(toggle!.textContent ?? "")).toBe(true);
  });

  it("collapsed toggle carries a visible clickable affordance (border/shadow token)", () => {
    renderWithProviders(
      <ArtifactGallery artifacts={perSubjectMeta(3)} threadId="t1" />,
    );

    const toggle = perSubjectToggle()!;
    // 折叠态是显式可点区，不是裸文字。断言 hover/可点 affordance 的视觉 token
    // 落在 className 上（border / shadow-rest / hover 之一）。
    const cls = toggle.className;
    expect(
      /border|shadow-rest|hover:bg|rounded/.test(cls),
      `toggle className 缺可点 affordance token: "${cls}"`,
    ).toBe(true);
  });

  it("stays collapsed by default and toggles to expanded on click (no layout dependency)", async () => {
    // GalleryGrid 用 @tanstack/react-virtual 虚拟化，jsdom 无布局时 getVirtualItems()
    // 返回空（缩略图不挂载）——这是 jsdom 已知限制（见 vitest.setup.ts 注释），不能
    // 用 getByAltText 断言缩略图。故此处只断言「折叠↔展开」的状态契约：
    // 默认折叠 → 点一下 → 翻转成展开态，且展开态文案切到「收起」动作提示。
    renderWithProviders(
      <ArtifactGallery artifacts={perSubjectMeta(2)} threadId="t1" />,
    );

    const toggle = perSubjectToggle()!;
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    // 折叠态文案是「展开」动作提示。
    expect(/expand|展开/i.test(toggle.textContent ?? "")).toBe(true);

    await userEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    // 翻转后文案切到「收起」动作提示，证明状态确实翻转。
    expect(/collapse|收起/i.test(toggle.textContent ?? "")).toBe(true);
  });
});
