/* App-wide event log.

   A single in-memory ring buffer that records what the app is doing —
   reading passages, sending chat turns, model errors, uncaught exceptions —
   so the whole flow is visible in one place (the Log tab of the Library
   modal). It is intentionally process-local: it is a live activity feed,
   not a persisted record (the Given-Log already persists messages).

   Anything in the app can append with `logEvent`. The Log panel subscribes
   for live updates. `installGlobalCapture` also routes uncaught errors and
   `console.error` here, so a crash like the LibraryModal one shows up in the
   feed instead of only the browser console. */

const MAX_EVENTS = 400;
let events = [];
let seq = 0;
const subscribers = new Set();

const notify = () => { for (const fn of subscribers) fn(events); };

/* level: "info" | "ok" | "warn" | "error"
   source: short tag for where it came from ("ingest", "chat", "model", …)
   lines: optional array of strings — expandable detail rendered verbatim,
          used to carry EO operator notation (the ops a step applied) and
          the facts recalled/learned on a memory turn. */
export function logEvent(level, source, message, detail, lines) {
  const event = {
    id: `e${++seq}`,
    ts: Date.now(),
    level: level || "info",
    source: source || "app",
    message: String(message ?? ""),
    detail: detail == null ? "" : String(detail),
    lines: Array.isArray(lines) && lines.length ? lines.map(String) : null,
  };
  events = events.concat(event).slice(-MAX_EVENTS);
  notify();
  return event;
}

export const getEvents = () => events;

export function clearEvents() {
  events = [];
  notify();
}

export function subscribeEvents(fn) {
  subscribers.add(fn);
  return () => subscribers.delete(fn);
}

/* Route uncaught errors and console.error into the feed. Safe to call more
   than once — it only patches on the first call. */
let captureInstalled = false;
export function installGlobalCapture() {
  if (captureInstalled || typeof window === "undefined") return;
  captureInstalled = true;

  window.addEventListener("error", (e) => {
    logEvent("error", "window", e.message || "Uncaught error",
      e.error?.stack || `${e.filename || ""}:${e.lineno || ""}`);
  });
  window.addEventListener("unhandledrejection", (e) => {
    const reason = e.reason;
    logEvent("error", "promise", reason?.message || String(reason || "Unhandled rejection"),
      reason?.stack || "");
  });

  const original = console.error.bind(console);
  console.error = (...args) => {
    try {
      const msg = args.map(a => (a instanceof Error ? a.message : typeof a === "string" ? a : JSON.stringify(a))).join(" ");
      const stack = args.find(a => a instanceof Error)?.stack || "";
      logEvent("error", "console", msg, stack);
    } catch { /* never let logging break the app */ }
    original(...args);
  };
}
