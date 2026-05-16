// Auto-router with a DEF -> EVA -> REC learning loop.
// All state lives in localStorage; nothing is sent to a server.

export const AUTO_MODEL = "__auto__";

export const INTENTS = {
  code:      { id: "code",      label: "Code",      icon: "⌨", color: "#a78bfa" },
  reasoning: { id: "reasoning", label: "Reasoning", icon: "🧠", color: "#f59e0b" },
  writing:   { id: "writing",   label: "Writing",   icon: "✍", color: "#34d399" },
  quick:     { id: "quick",     label: "Quick",     icon: "⚡", color: "#60a5fa" },
  general:   { id: "general",   label: "General",   icon: "💬", color: "#9ca3af" },
};

// Tie-break preference order, highest first.
export const INTENT_ORDER = ["code", "reasoning", "writing", "quick", "general"];

// Starting-state priority table. Each entry substring-matches an installed
// model name (case-insensitive). The learning loop overwrites this over time.
export const DEFAULT_PRIORITY = {
  code:      ["qwen2.5-coder", "qwen3-coder", "devstral", "codestral", "deepseek-coder", "qwen3", "qwen2.5", "llama3.1", "phi4-mini"],
  reasoning: ["deepseek-r1", "qwen3", "phi4", "llama3.1", "gemma", "phi4-mini"],
  writing:   ["llama3.1", "qwen3", "mistral", "gemma", "phi4", "phi4-mini"],
  quick:     ["phi4-mini", "gemma:2b", "llama3.2", "qwen2.5:0.5b", "phi4", "llama3.1"],
  general:   ["llama3.1", "qwen3", "mistral", "gemma", "phi4", "phi4-mini"],
};

// ── Storage ──────────────────────────────────────────────────────────────
const LOG_KEY = "llm-router-log";
const WEIGHTS_KEY = "llm-router-weights";
const PREFS_KEY = "llm-router-prefs";
const LOG_MAX_AGE_DAYS = 30;
const REC_INTERVAL = 10;

const loadJSON = (key, fallback) => {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
};
const saveJSON = (key, value) => {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* quota */ }
};

export const loadLog = () => {
  const log = loadJSON(LOG_KEY, []);
  return Array.isArray(log) ? log : [];
};
export const saveLog = (log) => saveJSON(LOG_KEY, log);

export const loadWeights = () => loadJSON(WEIGHTS_KEY, null);
export const saveWeights = (weights) => saveJSON(WEIGHTS_KEY, weights);

export const loadPrefs = () => loadJSON(PREFS_KEY, { autoMode: false });
export const savePrefs = (prefs) => saveJSON(PREFS_KEY, prefs);

export const resetRouter = () => {
  try {
    localStorage.removeItem(LOG_KEY);
    localStorage.removeItem(WEIGHTS_KEY);
  } catch { /* ignore */ }
};

export const appendLog = (entry) => {
  const log = loadLog();
  log.push(entry);
  saveLog(log);
};

const updateLogEntry = (id, updater) => {
  const log = loadLog();
  const idx = log.findIndex((e) => e.id === id);
  if (idx === -1) return;
  log[idx] = updater(log[idx]);
  saveLog(log);
};

// Append an EVA signal to a routing entry and mark it evaluated.
export const appendSignal = (routingId, signal) => {
  updateLogEntry(routingId, (e) => ({
    ...e,
    signals: [...(e.signals || []), signal],
    evaluated: true,
  }));
};

export const recordAlternateModel = (routingId, altModel) => {
  updateLogEntry(routingId, (e) => ({ ...e, alternateModel: altModel }));
};

export const recordFailure = (routingId, failedModel) => {
  updateLogEntry(routingId, (e) => ({
    ...e,
    failures: [...(e.failures || []), failedModel],
  }));
};

// ── Identity helpers ─────────────────────────────────────────────────────
export const uuid = () => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return `r-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
};

export const hashPrompt = async (text) => {
  try {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return Array.from(new Uint8Array(buf))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("")
      .slice(0, 8);
  } catch {
    let h = 0;
    for (let i = 0; i < text.length; i++) h = (Math.imul(31, h) + text.charCodeAt(i)) | 0;
    return Math.abs(h).toString(16).padStart(8, "0").slice(0, 8);
  }
};

// ── Intent classification ────────────────────────────────────────────────
const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const CODE_PATTERNS = [
  "python", "javascript", "typescript", "html", "css", "react", "sql", "bash", "rust", "go", "golang",
  "code", "script", "function", "class", "import", "export", "def", "const", "let", "var",
  "debug", "refactor", "fix bug", "fix error", "compile", "build", "deploy",
  "api", "endpoint", "server", "database", "query", "schema", "migration",
  "regex", "algorithm", "data structure", "loop", "array", "object", "json", "xml",
  "playwright", "selenium", "scraper", "crawl", "xpath", "selector",
  "git", "commit", "merge", "branch", "pull request", "repo",
  "npm", "pip", "ollama", "curl", "wget", "chmod", "mkdir", "grep",
];

const REASONING_PATTERNS = [
  "explain", "why", "how does", "analyze", "compare", "contrast", "evaluate",
  "logic", "proof", "theorem", "hypothesis", "therefore", "implies", "because",
  "step-by-step", "step by step", "break down", "walk through", "reasoning",
  "math", "calculate", "equation", "formula", "statistic", "probability",
  "pros and cons", "trade-off", "tradeoff", "advantage", "disadvantage",
  "what if", "scenario", "assumption", "given that", "suppose",
  "audit", "reconcile", "discrepancy", "oversight", "compliance", "budget",
  "legal", "policy", "regulation", "statute", "ordinance", "amendment",
];

const WRITING_PATTERNS = [
  "write", "draft", "compose", "essay", "article", "blog", "story", "letter",
  "edit", "rewrite", "reword", "rephrase", "proofread", "revise", "polish",
  "tone", "voice", "style", "narrative", "creative", "fiction", "dialogue",
  "email", "message", "memo", "report", "proposal", "pitch", "press release",
  "headline", "subtitle", "caption", "tagline", "slogan", "copy",
  "poem", "haiku", "song", "lyric", "verse", "chapter", "scene",
  "substack", "newsletter", "publish", "column", "op-ed",
];

const QUICK_BREVITY = ["tldr", "quick", "brief", "short answer", "one word", "yes or no"];

const countMatches = (text, patterns) => {
  let n = 0;
  for (const p of patterns) {
    if (new RegExp(`\\b${escapeRegex(p)}\\b`, "i").test(text)) n++;
  }
  return n;
};

const quickScore = (text) => {
  let s = 0;
  const trimLen = text.trim().length;
  if (/^\s*(what|who|when|where|how much|how many)\b/i.test(text) && trimLen < 60) s++;
  if (/^\s*(is|are|was|were|do|does|did|can|will)\b/i.test(text) && trimLen < 50) s++;
  if (/\b(define|meaning|translate|convert|spell|pronounce)\b/i.test(text)) s++;
  if (trimLen < 30) s++;
  for (const b of QUICK_BREVITY) {
    if (new RegExp(`\\b${escapeRegex(b)}\\b`, "i").test(text)) s++;
  }
  return s;
};

const modelNames = (installed) =>
  (installed || []).map((m) => (typeof m === "string" ? m : m.name)).filter(Boolean);

const pickWinner = (scores, installed, weights) => {
  const max = Math.max(...INTENT_ORDER.map((i) => scores[i]));
  if (max <= 0) return "general";
  const tied = INTENT_ORDER.filter((i) => scores[i] === max);
  if (tied.length === 1) return tied[0];
  const names = modelNames(installed);
  for (const intent of tied) {
    const priority = (weights && weights[intent]) || DEFAULT_PRIORITY[intent] || [];
    const top = priority[0];
    if (top && names.some((n) => n.toLowerCase().includes(top.toLowerCase()))) return intent;
  }
  return tied[0]; // INTENT_ORDER already encodes the preferred order
};

// Classify a prompt into an intent. Confidence = raw match count for the winner.
export const classifyIntent = (prompt, installed, weights) => {
  const full = (prompt || "").trim();
  if (!full) return null;
  const text = full.slice(0, 500); // only the head carries intent signal

  const scores = { code: 0, reasoning: 0, writing: 0, quick: 0, general: 0 };
  scores.code = countMatches(text, CODE_PATTERNS);
  if (/```/.test(text)) scores.code += 3;
  if ((text.match(/[{}();=]/g) || []).length >= 3) scores.code += 1;
  scores.reasoning = countMatches(text, REASONING_PATTERNS);
  scores.writing = countMatches(text, WRITING_PATTERNS);
  scores.quick = quickScore(text);

  // Very short prompt with no other signal -> force Quick.
  if (text.length < 15 && scores.code === 0 && scores.reasoning === 0 && scores.writing === 0) {
    return { intent: "quick", confidence: Math.max(1, scores.quick), scores };
  }

  const intent = pickWinner(scores, installed, weights);
  return { intent, confidence: scores[intent], scores };
};

// ── Model routing ────────────────────────────────────────────────────────
export const routeModel = (intent, installed, weights) => {
  const names = modelNames(installed);
  if (!names.length) return { model: null, candidates: [] };

  const priority = (weights && weights[intent]) || DEFAULT_PRIORITY[intent] || [];
  const candidates = [];
  for (const pref of priority) {
    const match = names.find((n) => n.toLowerCase().includes(pref.toLowerCase()));
    if (match && !candidates.includes(match)) candidates.push(match);
  }
  for (const n of names) if (!candidates.includes(n)) candidates.push(n);
  // First entry that matched the table, else first installed model.
  return { model: candidates[0], candidates: candidates.slice(0, 5) };
};

// ── Implicit EVA signals ─────────────────────────────────────────────────
const REJECTION_RE = /\b(no|nope|that'?s wrong|that is wrong|try again|not what i asked|use a different model|wrong)\b/i;

// Inspect prior auto-routed messages in a thread when a new prompt is sent.
// Logs implicit signals and returns [{ routingId, action }] for messages
// that just became evaluated.
export const processImplicitSignals = (thread, newUserText, autoMode) => {
  const results = [];
  if (!thread || !Array.isArray(thread.messages)) return results;
  const shortFollowUp = newUserText.trim().length < 60;

  for (const m of thread.messages) {
    if (m.role !== "assistant" || !m.routing || m.routing.evalDone) continue;
    const userMsgsAfter = thread.messages.filter(
      (x) => x.role === "user" && x.createdAt > m.createdAt
    ).length;

    let signal = null;
    if (userMsgsAfter === 0) {
      // The prompt being sent now immediately follows this response.
      if (shortFollowUp && REJECTION_RE.test(newUserText)) {
        signal = { type: "implicit", value: -0.3, action: "rejection-followup", model: m.routing.model };
      } else if (!autoMode) {
        signal = { type: "implicit", value: -0.2, action: "model-switch", model: m.routing.model };
      }
    } else if (userMsgsAfter >= 2) {
      // Two messages passed with no negative signal -> the choice held up.
      signal = { type: "implicit", value: 0.2, action: "continued-normally", model: m.routing.model };
    }

    if (signal) {
      appendSignal(m.routing.id, signal);
      results.push({ routingId: m.routing.id, action: signal.action });
    }
  }
  return results;
};

// ── REC: recalculate routing weights from the evaluation log ─────────────
export const runREC = (installed) => {
  const names = modelNames(installed);
  const cutoff = Date.now() - LOG_MAX_AGE_DAYS * 86400000;

  // Prune entries older than 30 days.
  let log = loadLog().filter((e) => {
    const t = new Date(e.timestamp).getTime();
    return Number.isFinite(t) ? t >= cutoff : true;
  });
  saveLog(log);

  // Accumulate time-decayed scores per (intent, model).
  const pairScores = {}; // intent -> { model: score }
  let totalEvaluations = 0;
  for (const e of log) {
    if (!e.evaluated || !e.signals || !e.signals.length) continue;
    totalEvaluations++;
    const t = new Date(e.timestamp).getTime();
    const days = Number.isFinite(t) ? Math.max(0, (Date.now() - t) / 86400000) : 0;
    const decay = Math.pow(0.95, days);
    pairScores[e.intent] = pairScores[e.intent] || {};
    for (const s of e.signals) {
      const target = s.model || e.modelChosen;
      if (!target) continue;
      pairScores[e.intent][target] = (pairScores[e.intent][target] || 0) + s.value * decay;
    }
  }

  const weights = {};
  for (const intent of INTENT_ORDER) {
    const def = DEFAULT_PRIORITY[intent] || [];
    const learned = pairScores[intent] || {};
    // Prune learned models that are no longer installed (if we know the list).
    const entries = Object.entries(learned).filter(
      ([model]) => !names.length || names.some((n) => n === model || n.toLowerCase().includes(model.toLowerCase()))
    );
    const positive = entries.filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).map(([m]) => m);
    const negative = entries.filter(([, v]) => v < 0).sort((a, b) => b[1] - a[1]).map(([m]) => m);
    const scored = new Set([...positive, ...negative]);

    const merged = [...positive];
    // Unscored defaults keep their original order.
    for (const d of def) {
      const isScored = [...scored].some((m) => m.toLowerCase().includes(d.toLowerCase()));
      if (!isScored && !merged.includes(d)) merged.push(d);
    }
    for (const m of negative) if (!merged.includes(m)) merged.push(m);
    weights[intent] = merged;
  }

  weights.lastUpdated = new Date().toISOString();
  weights.totalEvaluations = totalEvaluations;
  saveWeights(weights);
  return weights;
};

// Decide whether enough new evaluations have accrued to warrant a REC pass.
export const shouldRunREC = (weights, log) => {
  const evaluated = (log || loadLog()).filter((e) => e.evaluated).length;
  const lastTotal = (weights && weights.totalEvaluations) || 0;
  return evaluated > 0 && evaluated - lastTotal >= REC_INTERVAL;
};

// ── Stats for the Optimize tab ───────────────────────────────────────────
export const routerStats = (log) => {
  const entries = log || loadLog();
  const byIntent = { code: 0, reasoning: 0, writing: 0, quick: 0, general: 0 };
  const confidence = { high: 0, low: 0, zero: 0 };
  let evaluated = 0;
  let satisfied = 0;

  for (const e of entries) {
    if (byIntent[e.intent] != null) byIntent[e.intent]++;
    if (e.confidence >= 3) confidence.high++;
    else if (e.confidence >= 1) confidence.low++;
    else confidence.zero++;
    if (e.evaluated && e.signals && e.signals.length) {
      evaluated++;
      const net = e.signals.reduce((acc, s) => acc + (s.value || 0), 0);
      if (net > 0) satisfied++;
    }
  }

  return {
    total: entries.length,
    byIntent,
    confidence,
    evaluated,
    satisfaction: evaluated ? satisfied / evaluated : null,
  };
};
