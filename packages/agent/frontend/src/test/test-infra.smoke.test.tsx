// @vitest-environment jsdom
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

// Smoke test: confirm the jsdom + @testing-library/react + jest-dom setup
// added by Phase0#7 works end to end (env docblock, setup file matchers).
describe("test infra smoke", () => {
  it("renders and matches jest-dom matcher", () => {
    render(<div>hello perf</div>);
    expect(screen.getByText("hello perf")).toBeInTheDocument();
  });
});
