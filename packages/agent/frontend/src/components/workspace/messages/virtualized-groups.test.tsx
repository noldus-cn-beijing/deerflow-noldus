// @vitest-environment jsdom
import type { StickToBottomContext } from "use-stick-to-bottom";
import { describe, expect, it } from "vitest";

import {
  VIRTUALIZATION_THRESHOLD,
  VirtualizedGroups,
} from "@/components/workspace/messages/virtualized-groups";

/**
 * Phase0#7 Step 4 — message-list windowing.
 *
 * Note on testability: `@tanstack/react-virtual` measures real DOM geometry
 * (scrollElement clientHeight, ResizeObserver, getBoundingClientRect) which
 * jsdom does not compute, so true windowing (mounted < total) cannot be
 * asserted here. That is verified manually via the DevTools Elements panel
 * (spec §五: "50+ 组滚动时 DOM 节点恒定"). These tests cover what IS
 * deterministic: the threshold constant, that the component renders without
 * crashing for both small and large lists, and that it gracefully handles a
 * null scroll context (StickToBottom hasn't mounted yet).
 */

function makeScrollContext(viewportHeight: number) {
  const el = {
    clientHeight: viewportHeight,
    scrollTop: 0,
    scrollHeight: viewportHeight * 4,
    getBoundingClientRect: () => ({ height: viewportHeight }),
  } as unknown as HTMLElement;
  return {
    scrollRef: { current: el },
    contentRef: { current: null },
  } as unknown as StickToBottomContext;
}

describe("VirtualizedGroups (Phase0#7 Step 4)", () => {
  it("exports a non-trivial threshold so small lists skip virtualization", () => {
    expect(VIRTUALIZATION_THRESHOLD).toBeGreaterThanOrEqual(10);
  });

  it("renders without crashing for a large list with a scroll context", async () => {
    const { render } = await import("@testing-library/react");

    const N = VIRTUALIZATION_THRESHOLD + 50;
    const groups = Array.from({ length: N }, (_, i) => (
      <div key={i} data-testid={`group-${i}`} style={{ height: 100 }}>
        group {i}
      </div>
    ));

    // Must not throw — the virtualizer tolerates a jsdom scroll element with
    // stubbed geometry and renders whatever slice it computes (possibly empty
    // in jsdom, but the component must stay mounted and stable).
    const { unmount } = render(
      <VirtualizedGroups
        groups={groups}
        scrollContext={makeScrollContext(600)}
        paddingBottomPx={160}
      >
        <div data-testid="trailing" />
      </VirtualizedGroups>,
    );
    unmount();
  });

  it("renders trailing children regardless of windowing", async () => {
    const { render } = await import("@testing-library/react");

    const groups = [
      <div key="a" data-testid="group-a" style={{ height: 100 }}>
        a
      </div>,
    ];

    const { getByTestId, unmount } = render(
      <VirtualizedGroups
        groups={groups}
        scrollContext={makeScrollContext(600)}
        paddingBottomPx={0}
      >
        <div data-testid="trailing" />
      </VirtualizedGroups>,
    );
    // The streaming indicator + bottom padding (trailing nodes) must render
    // outside the windowed region so they are always present.
    expect(getByTestId("trailing")).toBeInTheDocument();
    unmount();
  });

  it("does not throw when the scroll context is null (pre-mount)", async () => {
    const { render } = await import("@testing-library/react");

    const groups = [
      <div key="a" data-testid="group-a" style={{ height: 100 }}>
        a
      </div>,
    ];

    // StickToBottom populates contextRef.current after mount; the first render
    // passes null. The component must handle this without throwing.
    const { unmount } = render(
      <VirtualizedGroups
        groups={groups}
        scrollContext={null}
        paddingBottomPx={0}
      />,
    );
    unmount();
  });
});
