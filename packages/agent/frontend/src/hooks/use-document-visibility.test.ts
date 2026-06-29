// @vitest-environment jsdom
import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useDocumentVisibility } from "./use-document-visibility";

describe("useDocumentVisibility", () => {
  beforeEach(() => {
    // Reset to visible between tests.
    Object.defineProperty(document, "hidden", {
      value: false,
      configurable: true,
      writable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("defaults to visible (false) on mount", () => {
    const { result } = renderHook(() => useDocumentVisibility());
    expect(result.current).toBe(false);
  });

  it("returns true after the tab becomes hidden", () => {
    const { result } = renderHook(() => useDocumentVisibility());
    expect(result.current).toBe(false);

    act(() => {
      Object.defineProperty(document, "hidden", { value: true, configurable: true });
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(result.current).toBe(true);
  });

  it("returns false again after the tab becomes visible (switchback)", () => {
    const { result } = renderHook(() => useDocumentVisibility());

    act(() => {
      Object.defineProperty(document, "hidden", { value: true, configurable: true });
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(result.current).toBe(true);

    act(() => {
      Object.defineProperty(document, "hidden", { value: false, configurable: true });
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(result.current).toBe(false);
  });

  it("removes the visibilitychange listener on unmount", () => {
    const removeSpy = vi.spyOn(document, "removeEventListener");
    const { unmount } = renderHook(() => useDocumentVisibility());
    unmount();
    expect(removeSpy).toHaveBeenCalledWith("visibilitychange", expect.any(Function));
  });
});
