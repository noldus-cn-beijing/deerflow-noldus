import { describe, expect, it } from "vitest";

import {
  getStreamErrorMessage,
  isRunNotOnThisWorkerError,
  isTerminalRunCancelError,
} from "./stream-error";

describe("getStreamErrorMessage", () => {
  it("returns a string error directly", () => {
    expect(getStreamErrorMessage("something went wrong")).toBe(
      "something went wrong",
    );
  });

  it("returns the message from an Error object", () => {
    expect(getStreamErrorMessage(new Error("timeout"))).toBe("timeout");
  });

  it("returns the message property from a plain object", () => {
    expect(getStreamErrorMessage({ message: "failed" })).toBe("failed");
  });

  it("returns the nested error.message from an object", () => {
    expect(
      getStreamErrorMessage({ error: new Error("nested failure") }),
    ).toBe("nested failure");
  });

  it("returns the nested error string from an object", () => {
    expect(getStreamErrorMessage({ error: "nested string" })).toBe(
      "nested string",
    );
  });

  it('falls back to "Request failed." for unrecognized shapes', () => {
    expect(getStreamErrorMessage(null)).toBe("Request failed.");
    expect(getStreamErrorMessage(undefined)).toBe("Request failed.");
    expect(getStreamErrorMessage({})).toBe("Request failed.");
  });

  it("returns empty string errors as-is (edge case)", () => {
    // If someone passes an empty string, .trim() is falsy so it falls through
    expect(getStreamErrorMessage("   ")).toBe("Request failed.");
  });
});

describe("isRunNotOnThisWorkerError", () => {
  it("matches the server detail string (positive)", () => {
    expect(
      isRunNotOnThisWorkerError(
        "Run 7fb55505-1234-abcd is not active on this worker and cannot be streamed",
      ),
    ).toBe(true);
  });

  it("matches a bare detail string with the key phrase", () => {
    expect(
      isRunNotOnThisWorkerError(
        "is not active on this worker and cannot be streamed",
      ),
    ).toBe(true);
  });

  it("matches error object with status 409 and 'worker' in message", () => {
    expect(
      isRunNotOnThisWorkerError({
        status: 409,
        message: "Run is not active on this worker",
      }),
    ).toBe(true);
  });

  it("matches error object with statusCode 409 and 'worker' keyword", () => {
    expect(
      isRunNotOnThisWorkerError({
        statusCode: 409,
        message: "Some worker-related error",
      }),
    ).toBe(true);
  });

  it("matches an Error with status 409 and 'worker' in message", () => {
    const err = new Error(
      "Run 7fb55505 is not active on this worker and cannot be streamed",
    ) as Error & { status?: number };
    err.status = 409;
    expect(isRunNotOnThisWorkerError(err)).toBe(true);
  });

  // --- Negative cases: must NOT match real errors ---

  it("does NOT match a generic 500 error", () => {
    expect(
      isRunNotOnThisWorkerError({ status: 500, message: "Internal server error" }),
    ).toBe(false);
  });

  it("does NOT match a network timeout error", () => {
    expect(isRunNotOnThisWorkerError(new Error("Request timeout"))).toBe(false);
    expect(
      isRunNotOnThisWorkerError(new TypeError("Failed to fetch")),
    ).toBe(false);
  });

  it("does NOT match a 409 without 'worker' in the message", () => {
    expect(
      isRunNotOnThisWorkerError({ status: 409, message: "Conflict" }),
    ).toBe(false);
  });

  it("does NOT match 'worker' keyword without 409 and without the key phrase", () => {
    expect(
      isRunNotOnThisWorkerError({
        status: 500,
        message: "A worker process crashed",
      }),
    ).toBe(false);
  });

  it("does NOT match a 404 not-found error", () => {
    expect(isRunNotOnThisWorkerError(new Error("Not found"))).toBe(false);
  });

  it("does NOT match null / undefined / empty object", () => {
    expect(isRunNotOnThisWorkerError(null)).toBe(false);
    expect(isRunNotOnThisWorkerError(undefined)).toBe(false);
    expect(isRunNotOnThisWorkerError({})).toBe(false);
  });
});

describe("isTerminalRunCancelError", () => {
  // Spec 2026-06-30 §四 修法 C: a cancel POSTed against an already-terminal run
  // returns 409 "is not cancellable (status: <terminal>)". This is the symptom of
  // the stale-reconnect spin (the user pressed pause, which stop()'d the dead run),
  // NOT a user-actionable failure — we classify it so the toast stays suppressed.

  it("matches the exact server detail string for a success run", () => {
    expect(
      isTerminalRunCancelError(
        'Run 57d06e5b-0492-4ca7-b32b-8f2cc40fe987 is not cancellable (status: success)',
      ),
    ).toBe(true);
  });

  it("matches the bare key phrase regardless of run id", () => {
    expect(isTerminalRunCancelError("is not cancellable (status: success)")).toBe(
      true,
    );
    expect(isTerminalRunCancelError("is not cancellable (status: error)")).toBe(
      true,
    );
    expect(isTerminalRunCancelError("is not cancellable (status: timeout)")).toBe(
      true,
    );
    expect(
      isTerminalRunCancelError("is not cancellable (status: cancelled)"),
    ).toBe(true);
  });

  it("matches an error object carrying status 409 + the key phrase", () => {
    expect(
      isTerminalRunCancelError({
        status: 409,
        message: "Run 57d06e5b is not cancellable (status: success)",
      }),
    ).toBe(true);
  });

  it("matches an Error instance carrying the key phrase", () => {
    const err = new Error(
      "HTTP 409: Run 57d06e5b is not cancellable (status: success)",
    );
    expect(isTerminalRunCancelError(err)).toBe(true);
  });

  it("matches a wrapped object whose nested .error holds the detail string", () => {
    expect(
      isTerminalRunCancelError({
        error: "is not cancellable (status: success)",
      }),
    ).toBe(true);
  });

  // --- Negative cases: must NOT match other errors ---

  it("does NOT match a generic 409 without the key phrase", () => {
    expect(isTerminalRunCancelError({ status: 409, message: "Conflict" })).toBe(
      false,
    );
  });

  it("does NOT match a worker-bridge 409 (that is the other classifier's job)", () => {
    expect(
      isTerminalRunCancelError(
        "Run 7fb55505 is not active on this worker and cannot be streamed",
      ),
    ).toBe(false);
  });

  it("does NOT match a 'running' run (still cancellable, a real failure path)", () => {
    // A cancel against a 'running' run would not normally 409; if it did, the
    // status word proves the run was live and the error is genuinely actionable.
    expect(
      isTerminalRunCancelError("is not cancellable (status: running)"),
    ).toBe(false);
  });

  it("does NOT match a 'pending' run", () => {
    expect(
      isTerminalRunCancelError("is not cancellable (status: pending)"),
    ).toBe(false);
  });

  it("does NOT match a network error or a 500", () => {
    expect(isTerminalRunCancelError(new TypeError("Failed to fetch"))).toBe(
      false,
    );
    expect(
      isTerminalRunCancelError({ status: 500, message: "Internal server error" }),
    ).toBe(false);
  });

  it("does NOT match null / undefined / empty object", () => {
    expect(isTerminalRunCancelError(null)).toBe(false);
    expect(isTerminalRunCancelError(undefined)).toBe(false);
    expect(isTerminalRunCancelError({})).toBe(false);
  });
});
