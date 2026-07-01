// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { LoadingState } from "./loading-state";

afterEach(() => cleanup());

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
