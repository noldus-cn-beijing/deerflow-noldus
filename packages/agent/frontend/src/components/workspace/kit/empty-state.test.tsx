// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { EmptyState } from "./empty-state";

afterEach(() => cleanup());

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
