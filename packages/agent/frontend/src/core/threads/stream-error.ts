/**
 * Stream error classification helpers.
 *
 * These are intentionally free of React / hook dependencies so they can be
 * imported and unit-tested without a DOM or component tree.
 */

/**
 * Best-effort extraction of a human-readable message from any error shape
 * the LangGraph SDK or the network layer may surface.
 */
export function getStreamErrorMessage(error: unknown): string {
  if (typeof error === "string" && error.trim()) {
    return error;
  }
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  if (typeof error === "object" && error !== null) {
    const message = Reflect.get(error, "message");
    if (typeof message === "string" && message.trim()) {
      return message;
    }
    const nestedError = Reflect.get(error, "error");
    if (nestedError instanceof Error && nestedError.message.trim()) {
      return nestedError.message;
    }
    if (typeof nestedError === "string" && nestedError.trim()) {
      return nestedError;
    }
  }
  return "Request failed.";
}

/**
 * True when the error is the Gateway's "run is active on another worker"
 * 409 — emitted by thread_runs.py join_run / stream_existing_run when a
 * store_only RunRecord is reached (multi-worker + in-memory StreamBridge).
 *
 * Switching back to a still-running thread makes the SDK (reconnectOnMount)
 * POST /stream; nginx may route it to a worker that doesn't hold the run's
 * in-memory bridge, yielding this 409. The thread's persisted content is
 * already rendered via fetchStateHistory, so this is NOT a user-facing
 * failure — we suppress the error toast for it.
 */
export function isRunNotOnThisWorkerError(error: unknown): boolean {
  const msg = getStreamErrorMessage(error);
  // Match the server detail string (thread_runs.py:268,308). Also tolerate a
  // status code on the error object if the SDK surfaces one.
  const status =
    typeof error === "object" && error !== null
      ? Reflect.get(error, "status") ?? Reflect.get(error, "statusCode")
      : undefined;
  return (
    msg.includes("is not active on this worker") ||
    (status === 409 && msg.toLowerCase().includes("worker"))
  );
}

/**
 * Terminal-run statuses the backend reports as `is not cancellable (status: X)`.
 * Kept as a Set so the membership test in isTerminalRunCancelError stays exact
 * (a bare substring "running"/"pending" must NOT match — those runs are live and
 * a cancel against them, if it ever 409'd, would be a genuinely actionable error).
 */
const TERMINAL_RUN_STATUSES = new Set([
  "success",
  "error",
  "timeout",
  "cancelled",
]);

/**
 * True when the error is the backend's 409 for POSTing a cancel against a run
 * that is already in a terminal state — "Run <id> is not cancellable
 * (status: success|error|timeout|cancelled)".
 *
 * This is the symptom of the crash-reconnect stale-run spin (spec
 * 2026-06-30): after a browser crash the SDK's `reconnectOnMount` re-joins the
 * last run id it persisted in sessionStorage; when the user hits "pause" the
 * SDK `stop()` POSTs a cancel against that same (already-successful) run id and
 * the backend correctly 409s. Cancelling a finished run is not a user-actionable
 * failure — we classify it so onError can suppress the toast. This is purely a
 * toast-suppression classifier; it does NOT fix the underlying spin (修法 A/B do).
 */
export function isTerminalRunCancelError(error: unknown): boolean {
  const msg = getStreamErrorMessage(error);
  // "is not cancellable (status: <terminal>)" — capture the parenthesised status
  // word and require it to be one of the terminal statuses so a hypothetical
  // "is not cancellable (status: running)" never matches.
  const statusRe = /is not cancellable \(status: ([a-z]+)\)/i;
  const statusMatch = statusRe.exec(msg);
  const captured = statusMatch?.[1];
  if (captured && TERMINAL_RUN_STATUSES.has(captured.toLowerCase())) {
    return true;
  }
  return false;
}
