import { useState, useEffect, useRef, useMemo } from "react";
import Markdown from "./Markdown.jsx";
import { getBrowserEngine } from "./webllm.js";
import {
  AUTO_MODEL, INTENTS,
  classifyIntent, routeModel, hashPrompt, uuid,
  loadLog, loadWeights, loadPrefs, savePrefs,
  appendLog, appendSignal, recordAlternateModel, recordFailure,
  processImplicitSignals, runREC, shouldRunREC,
} from "./router.js";
import {
  MEMORY_SYSTEM, INGEST_SYSTEM, EXTRACT_SYSTEM, MUTATE_SYSTEM,
  emptyMemory, cloneMemory, memoryStats, mergeMemory, splitSentences, batchSentences,
  signal, reach, buildDossier, buildPosition, buildLibrary, buildRoster,
  collectSpans, dossierHashOf, makeGiven, appendGiven,
  WALK_SCHEMA, MUTATE_SCHEMA,
  parseWalk, applyWalk,
  detectMutationTriggers, buildMutateUser, parseMutate, makeMutation, applyMutation,
} from "./memory.js";
import { loadLibrary, saveLibrary, docStats } from "./library.js";

const mono = `'SF Mono','Menlo','Consolas',monospace`;
const sans = `-apple-system,system-ui,sans-serif`;
const C = {
  bg: "#0b0b0f", s1: "#131318", s2: "#1b1b22", s3: "#232330",
  border: "#282838", text: "#d4d4e4", dim: "#65657e", accent: "#6e56cf",
  green: "#30a46c", red: "#e5484d", orange: "#f76b15",
};

const LS_KEY = "llmanager.chats.v2";
const QUANT_KEY = "llmanager.quantize.v1";
const MODE_KEY = "llmanager.chatmode.v1";
const MEM_MODEL_KEY = "llmanager.memorymodel.v1";

/* Context quantization: cap the history so Ollama re-processes a smaller
   prompt. Keeps the most recent messages and truncates very long ones. */
const HISTORY_MSG_LIMIT = 6;
const MSG_CHAR_CAP = 2000;
const quantizeHistory = (history) =>
  history.slice(-HISTORY_MSG_LIMIT).map(m =>
    m.content.length > MSG_CHAR_CAP
      ? { ...m, content: m.content.slice(0, MSG_CHAR_CAP) + " …[truncated]" }
      : m);

/* Reasoning models (DeepSeek-R1, etc.) emit chain-of-thought either in a
   separate `thinking` field or wrapped in <think>…</think> inside content.
   Split it out so it never leaks into the answer or the history. */
const splitReasoning = (raw, thinking = "") => {
  let reasoning = thinking, answer = raw;
  const open = raw.indexOf("<think>");
  if (open !== -1) {
    const close = raw.indexOf("</think>");
    if (close !== -1) {
      reasoning = (reasoning + "\n" + raw.slice(open + 7, close)).trim();
      answer = (raw.slice(0, open) + raw.slice(close + 8)).trim();
    } else {
      reasoning = (reasoning + "\n" + raw.slice(open + 7)).trim();
      answer = raw.slice(0, open).trim();
    }
  }
  return { reasoning: reasoning.trim(), answer };
};

const Icon = ({ name, size = 14 }) => {
  const s = { width: size, height: size, fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round", strokeLinejoin: "round" };
  switch (name) {
    case "plus":    return <svg viewBox="0 0 24 24" {...s}><path d="M12 5v14M5 12h14" /></svg>;
    case "search":  return <svg viewBox="0 0 24 24" {...s}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>;
    case "send":    return <svg viewBox="0 0 24 24" {...s}><path d="M5 12h14M13 6l6 6-6 6" /></svg>;
    case "stop":    return <svg viewBox="0 0 24 24" {...s} fill="currentColor" stroke="none"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>;
    case "copy":    return <svg viewBox="0 0 24 24" {...s}><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h9" /></svg>;
    case "refresh": return <svg viewBox="0 0 24 24" {...s}><path d="M3 12a9 9 0 0 1 15.5-6.3L21 8M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.3L3 16M3 21v-5h5" /></svg>;
    case "trash":   return <svg viewBox="0 0 24 24" {...s}><path d="M4 7h16M9 7V4h6v3M6 7l1 13a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-13" /></svg>;
    case "branch":  return <svg viewBox="0 0 24 24" {...s}><line x1="6" y1="3" x2="6" y2="15" /><circle cx="18" cy="6" r="3" /><circle cx="6" cy="18" r="3" /><path d="M18 9a9 9 0 0 1-9 9" /></svg>;
    case "chev":    return <svg viewBox="0 0 24 24" {...s}><path d="m6 9 6 6 6-6" /></svg>;
    case "chat":    return <svg viewBox="0 0 24 24" {...s}><path d="M21 12a8 8 0 0 1-11.5 7.2L4 21l1.8-5.5A8 8 0 1 1 21 12Z" /></svg>;
    case "memory":  return <svg viewBox="0 0 24 24" {...s}><path d="M12 5a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 3 4 3 3 0 0 0 5 0 3 3 0 0 0 3-4 3 3 0 0 0-2-5 3 3 0 0 0-3-3Z" /><path d="M12 5v12" /></svg>;
    case "book":    return <svg viewBox="0 0 24 24" {...s}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z" /></svg>;
    case "download":return <svg viewBox="0 0 24 24" {...s}><path d="M12 3v12M7 10l5 5 5-5M5 21h14" /></svg>;
    case "doc":     return <svg viewBox="0 0 24 24" {...s}><path d="M14 3v5h5M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" /></svg>;
    case "x":       return <svg viewBox="0 0 24 24" {...s}><path d="M18 6 6 18M6 6l12 12" /></svg>;
    case "eye":     return <svg viewBox="0 0 24 24" {...s}><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z" /><circle cx="12" cy="12" r="3" /></svg>;
    default: return null;
  }
};

/* ── Routing pill — intent badge on Auto-routed replies ── */
function RoutingPill({ routing }) {
  const intent = INTENTS[routing.intent] || INTENTS.general;
  const high = routing.confidence >= 3;
  return (
    <span title={`confidence: ${routing.confidence} ${high ? "(high)" : "(low)"}`} style={{
      display: "inline-flex", alignItems: "center", gap: 5, fontSize: 10, fontFamily: mono,
      padding: "2px 8px", borderRadius: 99, background: intent.color + "22", color: intent.color, fontWeight: 600,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: 99, boxSizing: "border-box",
        background: high ? intent.color : "transparent", border: `1.5px solid ${intent.color}` }} />
      {intent.icon} {intent.label} → {routing.model}
    </span>
  );
}

/* ── Mode toggle — per-chat Regular / Memory switch ── */
const MODE_TIPS = {
  regular: "Regular mode: the full conversation history is sent to the model every turn.",
  memory: "Memory mode: every turn is distilled into a per-chat graph and fed back as a fixed-size context block — the prompt never grows with the conversation.",
};
function ModeToggle({ mode, onChange, disabled }) {
  return (
    <div style={{ display: "flex", background: C.s1, border: `1px solid ${C.border}`, borderRadius: 7, padding: 2, gap: 2, flexShrink: 0 }}>
      {[["regular", "Regular", "chat"], ["memory", "Memory", "memory"]].map(([val, label, icon]) => {
        const on = mode === val;
        return (
          <button key={val} onClick={() => !disabled && !on && onChange(val)} disabled={disabled} title={MODE_TIPS[val]}
            style={{
              display: "flex", alignItems: "center", gap: 5, padding: "4px 9px", fontSize: 10.5, fontFamily: mono,
              fontWeight: 600, borderRadius: 5, border: "none", cursor: disabled || on ? "default" : "pointer",
              background: on ? C.accent : "transparent", color: on ? "#fff" : C.dim,
            }}>
            <Icon name={icon} size={11} /> {label}
          </button>
        );
      })}
    </div>
  );
}

/* ── Memory pill — shows what a Memory-mode reply drew on ── */
function MemoryPill({ mem }) {
  const txt = `${mem.used} recalled${mem.learned != null ? ` · +${mem.learned} learned` : ""}`;
  return (
    <span title="Memory mode: facts recalled from this chat's projected graph, and new facts read back from the exchange."
      style={{
        display: "inline-flex", alignItems: "center", gap: 4, fontSize: 10, fontFamily: mono,
        padding: "2px 8px", borderRadius: 99, background: C.accent + "22", color: C.accent, fontWeight: 600,
      }}>
      <Icon name="memory" size={10} /> {txt}
    </span>
  );
}

/* ── Mutation pill — a pending MUTATE action awaiting user consent ── */
const MUT_LABEL = { FORK: "Fork", MERGE: "Merge", CORRECT: "Correct", RECLASSIFY: "Reclassify" };
function MutationPill({ mut, onAccept, onDismiss }) {
  const btn = (bg, color) => ({
    fontSize: 10.5, fontFamily: mono, padding: "3px 10px", borderRadius: 6,
    border: `1px solid ${color}55`, cursor: "pointer", background: bg, color, fontWeight: 600,
  });
  return (
    <div title={`MUTATE — flagged by ${mut.trigger || "trigger"}${mut.triggerDetail ? ": " + mut.triggerDetail : ""}`}
      style={{
        display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", margin: "6px 0 12px",
        padding: "8px 11px", borderRadius: 9, background: C.orange + "12",
        border: `1px solid ${C.orange}44`, fontSize: 11.5, fontFamily: mono,
      }}>
      <span style={{ fontWeight: 700, color: C.orange, textTransform: "uppercase", letterSpacing: 0.5 }}>
        {MUT_LABEL[mut.action] || mut.action}
      </span>
      <span style={{ flex: 1, minWidth: 140, color: C.text, lineHeight: 1.5 }}>{mut.reason || "Graph change proposed."}</span>
      <button onClick={onAccept} style={btn(C.green + "22", C.green)}>Accept</button>
      <button onClick={onDismiss} style={btn("transparent", C.dim)}>Dismiss</button>
    </div>
  );
}

function FbBtn({ onClick, children }) {
  return (
    <button onClick={onClick} style={{
      fontSize: 11, fontFamily: mono, padding: "3px 9px", borderRadius: 6, border: `1px solid ${C.border}`,
      cursor: "pointer", background: "transparent", color: C.dim,
    }}>{children}</button>
  );
}

const bucketLabel = (ts) => {
  const now = new Date();
  const d = new Date(ts);
  const sameDay = (a, b) => a.toDateString() === b.toDateString();
  const yest = new Date(now); yest.setDate(now.getDate() - 1);
  if (sameDay(d, now)) return "Today";
  if (sameDay(d, yest)) return "Yesterday";
  const diff = Math.floor((now - d) / 86400000);
  if (diff < 7) return "Previous 7 days";
  if (diff < 30) return "Previous 30 days";
  return "Older";
};

const loadConvos = () => {
  try {
    const raw = localStorage.getItem(LS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
};

/* ── Sidebar ── */
function Sidebar({ convos, activeId, query, onSearch, onNew, onSelect, onDelete }) {
  const buckets = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q ? convos.filter(c => c.title.toLowerCase().includes(q)) : convos;
    const groups = {};
    [...filtered].sort((a, b) => b.updatedAt - a.updatedAt).forEach(c => {
      (groups[bucketLabel(c.updatedAt)] ||= []).push(c);
    });
    return groups;
  }, [convos, query]);
  const empty = Object.keys(buckets).length === 0;

  return (
    <aside style={{ width: 260, flexShrink: 0, background: C.s1, borderRight: `1px solid ${C.border}`, display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ padding: "12px 12px 6px", display: "flex", flexDirection: "column", gap: 8 }}>
        <button onClick={onNew} style={{
          display: "flex", alignItems: "center", justifyContent: "center", gap: 8, padding: "9px 12px",
          background: C.accent, color: "#fff", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 12.5, fontWeight: 600,
        }}>
          <Icon name="plus" size={14} /> New chat
        </button>
        <div style={{ position: "relative" }}>
          <span style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: C.dim, display: "flex" }}>
            <Icon name="search" size={13} />
          </span>
          <input
            value={query} onChange={e => onSearch(e.target.value)} placeholder="Search chats"
            style={{ width: "100%", padding: "8px 10px 8px 32px", background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12, outline: "none", boxSizing: "border-box" }}
          />
        </div>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 6px 16px" }}>
        {empty && (
          <div style={{ padding: "24px 14px", fontSize: 11.5, color: C.dim, textAlign: "center", lineHeight: 1.6 }}>
            {query ? `No chats match "${query}".` : "No chats yet. Start one above."}
          </div>
        )}
        {Object.entries(buckets).map(([label, list]) => (
          <div key={label} style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 10, fontFamily: mono, color: C.dim, textTransform: "uppercase", letterSpacing: 0.6, padding: "6px 10px 4px" }}>{label}</div>
            {list.map(c => (
              <ConvoRow key={c.id} convo={c} active={c.id === activeId} onSelect={() => onSelect(c.id)} onDelete={() => onDelete(c.id)} />
            ))}
          </div>
        ))}
      </div>
    </aside>
  );
}

function ConvoRow({ convo, active, onSelect, onDelete }) {
  const [hover, setHover] = useState(false);
  return (
    <div
      onClick={onSelect}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", borderRadius: 7, cursor: "pointer",
        background: active ? C.s2 : hover ? "rgba(255,255,255,.025)" : "transparent",
      }}>
      <span style={{ flexShrink: 0, color: C.dim, display: "flex" }}><Icon name="chat" size={12} /></span>
      <div style={{ flex: 1, fontSize: 12.5, color: active ? C.text : "#a8a8c0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{convo.title}</div>
      {hover && (
        <button onClick={e => { e.stopPropagation(); onDelete(); }} title="Delete chat" style={{
          background: "transparent", border: "none", color: C.dim, cursor: "pointer", padding: 2, display: "flex",
        }}>
          <Icon name="trash" size={12} />
        </button>
      )}
    </div>
  );
}

/* ── Model picker (opens upward, sits in composer) ── */
function ModelPicker({ value, models, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const h = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  const disabled = models.length === 0;
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button onClick={() => !disabled && setOpen(o => !o)} disabled={disabled} style={{
        display: "flex", alignItems: "center", gap: 7, padding: "5px 9px", background: C.s2,
        border: `1px solid ${C.border}`, borderRadius: 7, cursor: disabled ? "default" : "pointer",
        fontFamily: mono, fontSize: 11, color: disabled ? C.dim : C.text, maxWidth: 240,
      }}>
        <span style={{ width: 6, height: 6, borderRadius: 99, background: disabled ? C.dim : C.accent, flexShrink: 0,
          animation: value === AUTO_MODEL ? "llm-pulse 1.4s ease-in-out infinite" : "none" }} />
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {value === AUTO_MODEL ? "⚡ Auto" : (value || "no models")}
        </span>
        {!disabled && <span style={{ color: C.dim, display: "flex" }}><Icon name="chev" size={12} /></span>}
      </button>
      {open && (
        <div style={{
          position: "absolute", bottom: "calc(100% + 6px)", left: 0, minWidth: 240, maxHeight: 280, overflowY: "auto", zIndex: 30,
          background: C.s2, border: `1px solid ${C.border}`, borderRadius: 10, boxShadow: "0 12px 32px rgba(0,0,0,.5)", padding: 4,
        }}>
          <div style={{ padding: "8px 10px 6px", fontSize: 10, fontFamily: mono, color: C.dim, textTransform: "uppercase", letterSpacing: 0.6 }}>Select model</div>
          {models.map(m => (
            <button key={m} onClick={() => { onChange(m); setOpen(false); }} style={{
              width: "100%", textAlign: "left", display: "flex", alignItems: "center", gap: 10, padding: "8px 10px",
              background: m === value ? "rgba(110,86,207,.18)" : "transparent", border: "none", borderRadius: 6,
              cursor: "pointer", fontFamily: mono, fontSize: 12, color: C.text,
            }}>
              <span style={{ width: 7, height: 7, borderRadius: 99, background: C.accent, flexShrink: 0,
                animation: m === AUTO_MODEL ? "llm-pulse 1.4s ease-in-out infinite" : "none" }} />
              <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {m === AUTO_MODEL ? "⚡ Auto — route per message" : m}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Reasoning (chain-of-thought) panel ── */
function Reasoning({ text, streaming }) {
  const [open, setOpen] = useState(false);
  if (!text) return null;
  return (
    <div style={{ marginBottom: 8 }}>
      <button onClick={() => setOpen(o => !o)} style={{
        display: "flex", alignItems: "center", gap: 6, padding: "4px 9px",
        background: C.s1, border: `1px solid ${C.border}`, borderRadius: 6,
        cursor: "pointer", fontFamily: mono, fontSize: 10.5, color: C.dim,
      }}>
        <span style={{ display: "flex", transform: open ? "none" : "rotate(-90deg)" }}><Icon name="chev" size={11} /></span>
        {streaming ? "Thinking…" : "Reasoning"}
      </button>
      {open && (
        <div style={{
          marginTop: 6, padding: "8px 12px", background: C.s1, border: `1px solid ${C.border}`,
          borderRadius: 8, fontSize: 12.5, lineHeight: 1.55, color: C.dim, whiteSpace: "pre-wrap", wordBreak: "break-word",
        }}>{text}</div>
      )}
    </div>
  );
}

/* ── Prompt audit panel — the exact messages this reply was given, plus the
   underlying evidence spans the projected dossier was built from. The model
   sees the distilled hypotheses; the spans expose where they came from. ── */
function PromptView({ prompt, spans, dossierHash }) {
  const [open, setOpen] = useState(false);
  if (!prompt || !prompt.length) return null;
  const chars = prompt.reduce((n, m) => n + (m.content || "").length, 0);
  return (
    <div style={{ marginTop: 8 }}>
      <button onClick={() => setOpen(o => !o)} style={{
        display: "flex", alignItems: "center", gap: 6, padding: "4px 9px",
        background: C.s1, border: `1px solid ${C.border}`, borderRadius: 6,
        cursor: "pointer", fontFamily: mono, fontSize: 10.5, color: C.dim,
      }}>
        <Icon name="eye" size={11} />
        {open ? "Hide prompt" : "View prompt sent"}
        <span style={{ color: C.dim }}>· {prompt.length} msg · {chars} chars{spans?.length ? ` · ${spans.length} spans` : ""}</span>
      </button>
      {open && (
        <div style={{
          marginTop: 6, padding: "8px 12px", background: C.s1, border: `1px solid ${C.border}`,
          borderRadius: 8,
        }}>
          {prompt.map((m, i) => (
            <div key={i} style={{ marginBottom: 10 }}>
              <div style={{ fontFamily: mono, fontSize: 9.5, fontWeight: 700, letterSpacing: 0.5,
                textTransform: "uppercase", color: C.accent, marginBottom: 3 }}>{m.role}</div>
              <div style={{ fontFamily: mono, fontSize: 11.5, lineHeight: 1.6, color: C.dim,
                whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{m.content}</div>
            </div>
          ))}
          {spans && spans.length > 0 && (
            <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 8 }}>
              <div style={{ fontFamily: mono, fontSize: 9.5, fontWeight: 700, letterSpacing: 0.5,
                textTransform: "uppercase", color: C.accent, marginBottom: 5 }}>
                Source spans{dossierHash ? ` · ${dossierHash}` : ""}
              </div>
              {spans.map((s, i) => (
                <div key={i} style={{ fontFamily: mono, fontSize: 11, lineHeight: 1.55, color: C.dim,
                  marginBottom: 4, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                  <span style={{ color: C.accent }}>{s.source || "—"}</span>
                  {" "}<span style={{ color: C.dim, opacity: 0.7 }}>[{s.entity}]</span> {s.text}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Message bubble ── */
function MessageBubble({ msg, prevModel, onCopy, copied, onRerun, onFork, busy, installed, onFeedback, onWrongModel, wrongModelFor, setWrongModelFor, userMsgsAfter }) {
  if (msg.role === "user") {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", padding: "6px 0" }}>
        <div style={{
          maxWidth: "78%", padding: "10px 14px", borderRadius: 14, borderTopRightRadius: 4,
          background: C.s2, border: `1px solid ${C.border}`, fontSize: 14, lineHeight: 1.55, whiteSpace: "pre-wrap", wordBreak: "break-word",
        }}>{msg.content}</div>
        <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
          <IconBtn onClick={() => onFork(msg.id)} disabled={busy} title="Fork a new chat from here" icon="branch" />
        </div>
      </div>
    );
  }
  const switched = prevModel && prevModel !== msg.model;
  const showFeedback = msg.routing && !msg.routing.evalDone && !msg.routing.feedback
    && !msg.streaming && !msg.error && msg.content
    && installed.length > 1 && userMsgsAfter < 2;
  return (
    <div style={{ padding: "10px 0 14px" }}>
      {switched && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 0 8px", fontSize: 10.5, fontFamily: mono, color: C.dim }}>
          <span style={{ flex: 1, height: 1, background: C.border }} />
          model switched · {prevModel} → {msg.model}
          <span style={{ flex: 1, height: 1, background: C.border }} />
        </div>
      )}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, flexWrap: "wrap" }}>
        <span style={{ width: 7, height: 7, borderRadius: 99, background: msg.error ? C.red : C.accent }} />
        <span style={{ fontFamily: mono, fontSize: 11, fontWeight: 600, color: C.text }}>{msg.model}</span>
        {msg.routing && <RoutingPill routing={msg.routing} />}
        {msg.mem && <MemoryPill mem={msg.mem} />}
        {msg.elapsed && <span style={{ fontFamily: mono, fontSize: 10.5, color: C.dim }}>· {msg.elapsed}s</span>}
        {msg.tokens ? <span style={{ fontFamily: mono, fontSize: 10.5, color: C.dim }}>· {msg.tokens} tok</span> : null}
      </div>
      <div style={{ paddingLeft: 15, borderLeft: `1.5px solid ${C.border}` }}>
        {msg.error ? (
          <div style={{ fontSize: 13, color: C.red, whiteSpace: "pre-wrap" }}>{msg.content}</div>
        ) : (
          <>
            <Reasoning text={msg.reasoning} streaming={msg.streaming && !msg.content} />
            {msg.content && <Markdown text={msg.content} style={{ fontSize: 14 }} />}
            {msg.streaming && (
              <span style={{ display: "inline-block", width: 7, height: 14, marginLeft: 1, background: C.accent, verticalAlign: "text-bottom", animation: "llm-cursor-blink 1s step-start infinite" }} />
            )}
          </>
        )}
      </div>
      {!msg.streaming && !msg.error && (
        <div style={{ display: "flex", gap: 4, paddingLeft: 13, marginTop: 8 }}>
          <IconBtn onClick={() => onCopy(msg.content, msg.id)} active={copied === msg.id} title={copied === msg.id ? "Copied" : "Copy"} icon="copy" />
          <IconBtn onClick={() => onRerun(msg.id)} disabled={busy} title="Re-run" icon="refresh" />
          <IconBtn onClick={() => onFork(msg.id)} disabled={busy} title="Fork a new chat from here" icon="branch" />
        </div>
      )}
      {showFeedback && (
        <div style={{ display: "flex", gap: 6, paddingLeft: 13, marginTop: 8, alignItems: "center" }}>
          <FbBtn onClick={() => onFeedback(msg.id, "up")}>👍</FbBtn>
          <FbBtn onClick={() => onFeedback(msg.id, "down")}>👎</FbBtn>
          {wrongModelFor === msg.id ? (
            <select autoFocus defaultValue="" onChange={e => e.target.value && onWrongModel(msg.id, e.target.value)}
              onBlur={() => setWrongModelFor(null)}
              style={{ fontSize: 11, fontFamily: mono, padding: "3px 6px", borderRadius: 6, background: C.s1, color: C.text, border: `1px solid ${C.border}` }}>
              <option value="">pick correct model…</option>
              {installed.filter(m => m.name !== msg.routing.model).map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
          ) : (
            <FbBtn onClick={() => setWrongModelFor(msg.id)}>🔄 Wrong model</FbBtn>
          )}
        </div>
      )}
      {msg.routing && msg.routing.feedback && (
        <div style={{ fontSize: 10, fontFamily: mono, color: C.dim, paddingLeft: 13, marginTop: 8 }}>
          feedback recorded · {msg.routing.feedback === "up" ? "👍 helpful" : "👎 not helpful"}
        </div>
      )}
      {!msg.streaming && msg.prompt && (
        <div style={{ paddingLeft: 13 }}><PromptView prompt={msg.prompt} spans={msg.spans} dossierHash={msg.dossierHash} /></div>
      )}
    </div>
  );
}

function IconBtn({ onClick, active, disabled, title, icon }) {
  return (
    <button onClick={onClick} disabled={disabled} title={title} style={{
      display: "flex", alignItems: "center", justifyContent: "center", padding: 6, background: "transparent",
      border: "1px solid transparent", borderRadius: 6, cursor: disabled ? "default" : "pointer",
      opacity: disabled ? 0.4 : 1, color: active ? C.green : C.dim,
    }}>
      <Icon name={icon} size={13} />
    </button>
  );
}

/* ── Composer ── */
function Composer({ value, setValue, model, models, setModel, onSend, onStop, busy, isReply, quantize, setQuantize, mode, memModel, memModels, setMemModel }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = "auto";
      ref.current.style.height = Math.min(ref.current.scrollHeight, 200) + "px";
    }
  }, [value]);
  const canSend = !busy && value.trim() && model;
  return (
    <div style={{ padding: "8px 24px 18px", background: `linear-gradient(to bottom, transparent, ${C.bg} 30%)` }}>
      <div style={{ maxWidth: 820, margin: "0 auto", background: C.s1, border: `1px solid ${C.border}`, borderRadius: 16, padding: 8, boxShadow: "0 10px 28px rgba(0,0,0,.25)" }}>
        <textarea
          ref={ref}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (canSend) onSend(); } }}
          placeholder={isReply ? "Reply…  Enter to send, Shift+Enter for newline" : "Start a chat…  Enter to send, Shift+Enter for newline"}
          rows={1}
          style={{ width: "100%", padding: "10px 12px 4px", background: "transparent", color: C.text, border: "none", outline: "none", resize: "none", fontSize: 14, lineHeight: 1.55, fontFamily: sans, boxSizing: "border-box" }}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 6px 4px" }}>
          <ModelPicker value={model} models={models} onChange={setModel} />
          {mode === "memory" ? (
            <>
              <span
                title="Memory mode is on — each turn sends a fixed-size prompt (system + recalled facts + position marker) instead of the conversation history. The model above answers you (READ); the background model below runs EXTRACT, INGEST and MUTATE."
                style={{
                  display: "flex", alignItems: "center", gap: 7, padding: "5px 9px",
                  background: C.accent + "18", border: `1px solid ${C.accent}44`, borderRadius: 7,
                  fontFamily: mono, fontSize: 11, color: C.accent,
                }}>
                <Icon name="memory" size={12} /> Memory
              </span>
              {memModels.length > 0 && (
                <span title="Background model — runs EXTRACT, INGEST and MUTATE. Defaults to the chat model above."
                  style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: mono, fontSize: 10.5, color: C.dim }}>
                  bg
                  <ModelPicker
                    value={memModel || "(chat model)"}
                    models={["(chat model)", ...memModels]}
                    onChange={v => setMemModel(v === "(chat model)" ? "" : v)}
                  />
                </span>
              )}
            </>
          ) : (
            <button
              onClick={() => setQuantize(q => !q)}
              title={`Quantize context — send only the last ${HISTORY_MSG_LIMIT} messages, trimmed, so Ollama re-processes a smaller prompt (faster, less memory). Older context is dropped.`}
              style={{
                display: "flex", alignItems: "center", gap: 7, padding: "5px 9px",
                background: quantize ? "rgba(110,86,207,.18)" : C.s2,
                border: `1px solid ${quantize ? C.accent : C.border}`, borderRadius: 7,
                cursor: "pointer", fontFamily: mono, fontSize: 11, color: quantize ? C.text : C.dim,
              }}>
              <span style={{ width: 6, height: 6, borderRadius: 99, background: quantize ? C.accent : C.dim, flexShrink: 0 }} />
              Quantize
            </button>
          )}
          <div style={{ flex: 1 }} />
          {busy ? (
            <button onClick={onStop} style={{
              display: "flex", alignItems: "center", gap: 6, padding: "7px 14px", borderRadius: 8, border: "none",
              background: C.s3, color: C.text, fontSize: 12.5, fontWeight: 600, cursor: "pointer",
            }}>
              <Icon name="stop" size={11} /> Stop
            </button>
          ) : (
            <button onClick={() => canSend && onSend()} disabled={!canSend} style={{
              display: "flex", alignItems: "center", gap: 6, padding: "7px 14px", borderRadius: 8, border: "none",
              background: canSend ? C.accent : C.s3, color: canSend ? "#fff" : C.dim,
              fontSize: 12.5, fontWeight: 600, cursor: canSend ? "pointer" : "default",
            }}>
              <Icon name="send" size={13} /> Send
            </button>
          )}
        </div>
      </div>
      <div style={{ maxWidth: 820, margin: "8px auto 0", textAlign: "center", fontSize: 10.5, color: C.dim }}>
        Running locally on Ollama · conversations are saved in this browser only.
      </div>
    </div>
  );
}

/* ── Header action button ── */
function HeaderBtn({ icon, label, onClick, disabled, badge }) {
  return (
    <button onClick={onClick} disabled={disabled} title={label} style={{
      display: "flex", alignItems: "center", gap: 6, padding: "5px 10px", flexShrink: 0,
      background: C.s1, border: `1px solid ${C.border}`, borderRadius: 7,
      fontFamily: mono, fontSize: 11, color: disabled ? C.dim : C.text,
      cursor: disabled ? "default" : "pointer", opacity: disabled ? 0.5 : 1,
    }}>
      <Icon name={icon} size={12} /> {label}
      {badge != null && (
        <span style={{ background: C.accent, color: "#fff", borderRadius: 99, padding: "0 5px",
          fontSize: 9.5, fontWeight: 700 }}>{badge}</span>
      )}
    </button>
  );
}

/* ── Per-passage read trace — one row of the reading walk ── */
function ChunkTrace({ chunk }) {
  const [open, setOpen] = useState(false);
  const stColor = chunk.status === "done" ? C.green
    : chunk.status === "error" ? C.red
    : chunk.status === "reading" ? C.orange : C.dim;
  const ner = chunk.signal?.ner || { names: [], dates: [], numbers: [] };
  const kws = chunk.signal?.keywords || [];
  const label = chunk.status === "done" ? `+${chunk.applied} ops`
    : chunk.status === "reading" ? "reading…"
    : chunk.status === "error" ? "error" : "queued";
  const opLine = (e) => {
    if (e.op === "SIG") return `+ site ${e.canonical || e.id} (${e.kind || "thing"})`;
    if (e.op === "DEF") return `~ revise ${e.id || e.canonical}`;
    if (e.op === "CON") return `→ ${e.from} —${e.relation || e.type || "related to"}→ ${e.to}`;
    if (e.op === "REC") return `⟳ rename ${e.id} → "${e.canonical}"`;
    return JSON.stringify(e);
  };
  return (
    <div style={{ border: `1px solid ${C.border}`, borderRadius: 8, marginBottom: 6, background: C.bg }}>
      <button onClick={() => setOpen(o => !o)} style={{
        display: "flex", alignItems: "center", gap: 8, width: "100%", padding: "8px 10px",
        background: "transparent", border: "none", cursor: "pointer", color: C.text,
        fontFamily: mono, fontSize: 11,
      }}>
        <span style={{ display: "flex", transform: open ? "none" : "rotate(-90deg)" }}>
          <Icon name="chev" size={11} />
        </span>
        <span style={{ width: 7, height: 7, borderRadius: 99, background: stColor, flexShrink: 0 }} />
        <span>Passage {chunk.index + 1} · {chunk.chars} chars</span>
        <span style={{ flex: 1 }} />
        <span style={{ color: stColor }}>{label}</span>
      </button>
      {open && (
        <div style={{ padding: "0 10px 10px", fontFamily: mono, fontSize: 10.5, color: C.dim, lineHeight: 1.7 }}>
          <div><span style={{ color: C.text }}>Scanned names:</span> {ner.names.join(", ") || "—"}</div>
          <div><span style={{ color: C.text }}>Keywords:</span> {kws.slice(0, 14).join(", ") || "—"}</div>
          {chunk.ops?.length > 0 && (
            <div style={{ marginTop: 6 }}>
              <span style={{ color: C.text }}>Operations applied to the graph:</span>
              {chunk.ops.map((e, i) => (
                <div key={i} style={{ paddingLeft: 10, color: C.accent, wordBreak: "break-word" }}>
                  {opLine(e)}
                </div>
              ))}
            </div>
          )}
          {chunk.status === "error" && chunk.rawOutput && (
            <div style={{ marginTop: 6, color: C.red }}>{chunk.rawOutput}</div>
          )}
          {chunk.text && (
            <details style={{ marginTop: 6 }}>
              <summary style={{ cursor: "pointer", color: C.text }}>Source text</summary>
              <div style={{ whiteSpace: "pre-wrap", marginTop: 4, wordBreak: "break-word" }}>{chunk.text}</div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Library & reading modal ── */
function LibraryModal({ open, onClose, library, activeConvo, canIngest,
                        ingestRunning, ingestTrace, onIngest, onToggleDoc, onRemoveDoc }) {
  const [text, setText] = useState("");
  const [title, setTitle] = useState("");
  const [source, setSource] = useState("");
  const fileRef = useRef(null);
  if (!open) return null;

  const loadFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      setText(String(reader.result || ""));
      setSource(f.name);
      if (!title) setTitle(f.name.replace(/\.[^.]+$/, ""));
    };
    reader.readAsText(f);
    e.target.value = "";
  };

  const ingest = () => {
    if (!text.trim() || ingestRunning || !canIngest) return;
    onIngest(text, title.trim(), source);
  };

  const done = ingestTrace.filter(t => t.status !== "pending" && t.status !== "reading").length;
  const learned = ingestTrace.reduce((n, t) => n + (t.applied || 0), 0);
  const attached = new Set(activeConvo?.docs || []);
  const btn = (active) => ({
    fontSize: 11, fontFamily: mono, fontWeight: 600, padding: "5px 12px", borderRadius: 6,
    border: "none", cursor: "pointer", background: active ? C.accent : C.s2,
    color: active ? "#fff" : C.dim,
  });

  return (
    <div onClick={() => !ingestRunning && onClose()} style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,.62)", zIndex: 50,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        width: "min(720px, 100%)", maxHeight: "88vh", display: "flex", flexDirection: "column",
        background: C.s1, border: `1px solid ${C.border}`, borderRadius: 14,
        boxShadow: "0 20px 56px rgba(0,0,0,.55)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "14px 18px",
          borderBottom: `1px solid ${C.border}` }}>
          <Icon name="book" size={15} />
          <div style={{ fontSize: 13.5, fontWeight: 700, flex: 1 }}>Library &amp; Reading</div>
          <button onClick={onClose} disabled={ingestRunning} style={{ background: "transparent",
            border: "none", color: C.dim, cursor: ingestRunning ? "default" : "pointer", display: "flex" }}>
            <Icon name="x" size={16} />
          </button>
        </div>

        <div style={{ padding: "14px 18px", overflowY: "auto" }}>
          {/* ── Read new content ── */}
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Read new content</div>
          <div style={{ fontSize: 11.5, color: C.dim, lineHeight: 1.6, marginBottom: 10 }}>
            Paste or upload any text. It is split into sentences, grouped into passages
            of a few sentences, and read by the model one passage at a time — each passage
            against the sites already found, so the graph resolves rather than duplicates.
            The document is added to the library and opted in to the current chat. Every
            passage and its operations are shown below as it is read.
          </div>
          <input value={title} onChange={e => setTitle(e.target.value)} placeholder="Document title (optional)"
            disabled={ingestRunning}
            style={{ width: "100%", padding: "8px 11px", marginBottom: 8, background: C.bg, color: C.text,
              border: `1px solid ${C.border}`, borderRadius: 7, fontSize: 12, fontFamily: mono,
              boxSizing: "border-box", outline: "none" }} />
          <textarea value={text} onChange={e => setText(e.target.value)} placeholder="Paste text to read…"
            disabled={ingestRunning}
            style={{ width: "100%", minHeight: 110, maxHeight: 220, resize: "vertical", padding: "10px 12px",
              background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 8,
              fontSize: 12, fontFamily: mono, boxSizing: "border-box", outline: "none" }} />
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
            <input ref={fileRef} type="file" accept=".txt,.md,.markdown,.csv,.json,text/*"
              onChange={loadFile} style={{ display: "none" }} />
            <button onClick={() => fileRef.current?.click()} disabled={ingestRunning} style={btn(false)}>
              Upload file
            </button>
            <span style={{ fontSize: 10.5, fontFamily: mono, color: C.dim }}>
              {text.length} chars{source ? ` · ${source}` : ""}
            </span>
            <div style={{ flex: 1 }} />
            <button onClick={ingest} disabled={ingestRunning || !text.trim() || !canIngest}
              style={{ ...btn(true), opacity: (ingestRunning || !text.trim() || !canIngest) ? 0.5 : 1 }}>
              {ingestRunning ? "Reading…" : "Read into memory"}
            </button>
          </div>
          {!canIngest && (
            <div style={{ fontSize: 10.5, color: C.orange, marginTop: 6 }}>
              No model available — pull one from Settings → Models first.
            </div>
          )}

          {ingestTrace.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6,
                fontFamily: mono, fontSize: 11, color: C.dim }}>
                <span>{done}/{ingestTrace.length} passages read</span>
                <span>·</span>
                <span style={{ color: C.accent }}>+{learned} facts learned</span>
              </div>
              <div style={{ height: 5, background: C.s3, borderRadius: 3, overflow: "hidden", marginBottom: 10 }}>
                <div style={{ height: "100%", background: C.accent, transition: "width .2s",
                  width: `${ingestTrace.length ? (done / ingestTrace.length) * 100 : 0}%` }} />
              </div>
              {ingestTrace.map(t => <ChunkTrace key={t.index} chunk={t} />)}
            </div>
          )}

          {/* ── Library list ── */}
          <div style={{ fontSize: 12, fontWeight: 700, margin: "20px 0 6px" }}>
            Library · {library.length} document{library.length === 1 ? "" : "s"}
          </div>
          <div style={{ fontSize: 11.5, color: C.dim, lineHeight: 1.6, marginBottom: 10 }}>
            {activeConvo
              ? "Toggle a document to opt this chat in or out. Opted-in documents are merged into the chat's memory and projected into every prompt."
              : "Open or start a chat to opt documents in to it."}
          </div>
          {library.length === 0 ? (
            <div style={{ fontSize: 11.5, color: C.dim, padding: "14px 0", textAlign: "center" }}>
              Nothing read yet. Read some content above to build your first document.
            </div>
          ) : library.map(doc => {
            const st = docStats(doc);
            const on = attached.has(doc.id);
            return (
              <div key={doc.id} style={{ display: "flex", alignItems: "center", gap: 10,
                padding: "9px 11px", marginBottom: 6, background: C.bg,
                border: `1px solid ${on ? C.accent + "66" : C.border}`, borderRadius: 8 }}>
                <Icon name="doc" size={14} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden",
                    textOverflow: "ellipsis" }}>{doc.title}</div>
                  <div style={{ fontSize: 10, fontFamily: mono, color: C.dim }}>
                    {st.entities} sites · {st.edges} links · {doc.passages} passages
                  </div>
                </div>
                <button onClick={() => onToggleDoc(doc.id)} disabled={!activeConvo} style={btn(on)}>
                  {on ? "✓ in chat" : "opt in"}
                </button>
                <button onClick={() => onRemoveDoc(doc.id)} title="Remove from library" style={{
                  background: "transparent", border: "none", color: C.dim, cursor: "pointer", display: "flex" }}>
                  <Icon name="trash" size={13} />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ── Main chat ── */
export default function Chat({ ollamaUrl, installed, ollamaUp, provider = "ollama" }) {
  const isBrowser = provider === "browser";
  const [convos, setConvos] = useState(loadConvos);
  const [activeId, setActiveId] = useState(null);
  const [query, setQuery] = useState("");
  const [draft, setDraft] = useState("");
  const [composerModel, setComposerModel] = useState(() => (loadPrefs().autoMode ? AUTO_MODEL : ""));
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(null);
  const [quantize, setQuantize] = useState(() => localStorage.getItem(QUANT_KEY) === "1");
  const [mode, setMode] = useState(() => localStorage.getItem(MODE_KEY) === "memory" ? "memory" : "regular");
  const [memModel, setMemModel] = useState(() => localStorage.getItem(MEM_MODEL_KEY) || "");
  const [wrongModelFor, setWrongModelFor] = useState(null);
  const [library, setLibrary] = useState(loadLibrary);
  const [libOpen, setLibOpen] = useState(false);
  const [ingestRunning, setIngestRunning] = useState(false);
  const [ingestTrace, setIngestTrace] = useState([]);
  const abortRef = useRef(null);
  const scrollRef = useRef(null);

  const modelNames = useMemo(() => installed.map(m => m.name), [installed]);
  const active = convos.find(c => c.id === activeId) || null;
  const messages = active?.messages || [];

  /* Default composer model once models load */
  useEffect(() => {
    if (!composerModel && modelNames.length) setComposerModel(modelNames[0]);
  }, [modelNames, composerModel]);

  /* Persist Auto-mode on/off */
  useEffect(() => {
    const prefs = loadPrefs();
    const autoMode = composerModel === AUTO_MODEL;
    if (composerModel && prefs.autoMode !== autoMode) savePrefs({ ...prefs, autoMode });
  }, [composerModel]);

  const maybeREC = () => { if (shouldRunREC(loadWeights(), loadLog())) runREC(installed); };

  /* Persist (debounced so token streaming doesn't thrash localStorage) */
  useEffect(() => {
    const id = setTimeout(() => {
      try { localStorage.setItem(LS_KEY, JSON.stringify(convos)); } catch { /* quota — ignore */ }
    }, 400);
    return () => clearTimeout(id);
  }, [convos]);

  useEffect(() => {
    try { localStorage.setItem(QUANT_KEY, quantize ? "1" : "0"); } catch { /* ignore */ }
  }, [quantize]);

  useEffect(() => {
    try { localStorage.setItem(MODE_KEY, mode); } catch { /* ignore */ }
  }, [mode]);

  useEffect(() => {
    try { localStorage.setItem(MEM_MODEL_KEY, memModel); } catch { /* ignore */ }
  }, [memModel]);

  /* The background model for EXTRACT/INGEST/MUTATE — the configured memory
     model when set and installed, otherwise a fallback chat model. */
  const backgroundModel = (fallback) =>
    (memModel && modelNames.includes(memModel)) ? memModel : fallback;

  /* Persist the document library (debounced, same policy as chats). */
  useEffect(() => {
    const id = setTimeout(() => saveLibrary(library), 400);
    return () => clearTimeout(id);
  }, [library]);

  /* The library documents a chat has opted in to. */
  const attachedDocs = (convo) => (convo?.docs || [])
    .map(id => library.find(d => d.id === id))
    .filter(Boolean);

  /* Project a chat's own memory together with every opted-in library
     document into a single graph (entities/edges/defs only). */
  const combinedMemory = (convo) =>
    mergeMemory(convo?.memory, ...attachedDocs(convo).map(d => d.memory));

  /* The memory-mode system prompt: instructions + recalled context +
     library overview + position marker. Empty sections are dropped so the
     prompt stays as small as the chat allows. */
  const memorySystemPrompt = (convo, sig) => {
    const projected = combinedMemory(convo);
    const entities = reach(sig, projected);
    const dossier = buildDossier(entities, projected);
    const parts = [
      MEMORY_SYSTEM,
      dossier,
      buildLibrary(attachedDocs(convo)),
      buildPosition((convo?.memory || emptyMemory()).lastTurn),
    ].filter(Boolean);
    return {
      content: parts.join("\n\n"),
      used: entities.length,
      entities,
      dossierHash: dossierHashOf(dossier),
      spans: collectSpans(entities, projected),
    };
  };

  /* Switch the active chat's mode (or just the default for the next new chat).
     Memory data is kept even when switching back to Regular. */
  const changeMode = (m) => {
    setMode(m);
    if (activeId) {
      setConvos(prev => prev.map(c => c.id === activeId
        ? { ...c, mode: m, memory: m === "memory" ? (c.memory || emptyMemory()) : c.memory }
        : c));
    }
  };

  /* Auto-scroll */
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, activeId, busy]);

  const copy = (t, id) => { navigator.clipboard?.writeText(t); setCopied(id); setTimeout(() => setCopied(null), 1200); };

  const newChat = () => { setActiveId(null); setDraft(""); };

  const deleteConvo = (id) => {
    setConvos(prev => prev.filter(c => c.id !== id));
    if (id === activeId) setActiveId(null);
  };

  /* Fork: branch a new conversation off the message at msgId, carrying
     every message up to and including it. The original is left untouched. */
  const forkConvo = (msgId) => {
    if (!active || busy) return;
    const idx = active.messages.findIndex(m => m.id === msgId);
    if (idx < 0) return;
    const messages = active.messages.slice(0, idx + 1).map(m => ({ ...m, streaming: false }));
    const baseTitle = active.title.replace(/ \(fork\)$/, "");
    const trimmed = baseTitle.length > 40 ? baseTitle.slice(0, 40) + "…" : baseTitle;
    const newId = "c" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    setConvos(prev => [{
      id: newId, title: trimmed + " (fork)",
      model: active.model, updatedAt: Date.now(), messages,
      mode: active.mode || "regular",
      memory: active.memory ? cloneMemory(active.memory) : undefined,
    }, ...prev]);
    setActiveId(newId);
    setMode(active.mode || "regular");
    setDraft("");
  };

  const selectConvo = (id) => {
    setActiveId(id);
    const c = convos.find(x => x.id === id);
    const lastModel = c?.messages.filter(m => m.role === "assistant").pop()?.model || c?.model;
    if (lastModel && modelNames.includes(lastModel)) setComposerModel(lastModel);
    setMode(c?.mode === "memory" ? "memory" : "regular");
  };

  /* Stream a chat completion from Ollama; calls onToken(answer, reasoning)
     with the full text so far. Reasoning output is kept separate. */
  const streamChat = async (model, apiMessages, onToken, signal) => {
    if (isBrowser) {
      const engine = await getBrowserEngine(model, report => {
        const pct = Math.round((report.progress || 0) * 100);
        onToken("", `Loading ${model} — ${report.text || `${pct}%`}`);
      });
      const chunks = await engine.chat.completions.create({
        messages: apiMessages, stream: true, stream_options: { include_usage: true },
      });
      let raw = "", usage = {};
      for await (const chunk of chunks) {
        if (signal?.aborted) {
          try { engine.interruptGenerate(); } catch { /* ignore */ }
          throw Object.assign(new Error("aborted"), { name: "AbortError" });
        }
        const delta = chunk.choices?.[0]?.delta?.content || "";
        if (delta) {
          raw += delta;
          const { reasoning, answer } = splitReasoning(raw, "");
          onToken(answer, reasoning);
        }
        if (chunk.usage) usage = { tokens: chunk.usage.completion_tokens };
      }
      const { reasoning, answer } = splitReasoning(raw, "");
      return { content: answer, reasoning, usage };
    }
    const r = await fetch(`${ollamaUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: apiMessages, stream: true }),
      signal,
    });
    if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", raw = "", thinking = "", usage = {};
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let j;
        try { j = JSON.parse(line); } catch { continue; }
        if (j.error) throw new Error(j.error);
        const msg = j.message || {};
        if (msg.thinking) thinking += msg.thinking;
        if (msg.content) raw += msg.content;
        if (msg.thinking || msg.content) {
          const { reasoning, answer } = splitReasoning(raw, thinking);
          onToken(answer, reasoning);
        }
        if (j.done) usage = { tokens: j.eval_count };
      }
    }
    const { reasoning, answer } = splitReasoning(raw, thinking);
    return { content: answer, reasoning, usage };
  };

  /* Non-streaming chat call — used for the background memory calls. When a
     `format` JSON schema is given, Ollama constrains generation to it via
     constrained decoding, so the EXTRACT/INGEST/MUTATE output is guaranteed
     well-formed JSON. (The in-browser runtime is left unconstrained — the
     walk parser still tolerates fenced or noisy output.) */
  const chatOnce = async (model, apiMessages, format) => {
    if (isBrowser) {
      const engine = await getBrowserEngine(model);
      const reply = await engine.chat.completions.create({
        messages: apiMessages, temperature: 0,
      });
      return reply.choices?.[0]?.message?.content || "";
    }
    const r = await fetch(`${ollamaUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model, messages: apiMessages, stream: false, options: { temperature: 0 },
        ...(format ? { format } : {}),
      }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    return j.message?.content || "";
  };

  /* Patch a single message inside a convo (patch may be an object or a fn) */
  const patchMsg = (convoId, msgId, patch) => {
    setConvos(prev => prev.map(c => c.id === convoId
      ? { ...c, messages: c.messages.map(m => m.id === msgId
          ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m) }
      : c));
  };

  /* Run a model turn into an existing placeholder assistant message.
     opts.candidates enables Auto-mode fallback: if a model errors, the next
     candidate is tried once. opts.routingId logs failures to the routing log. */
  const runTurn = async (convoId, msgId, model, apiMessages, opts = {}) => {
    const { candidates, routingId } = opts;
    const attempts = candidates && candidates.length ? candidates.slice(0, 2) : [model];
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const t0 = Date.now();
    let lastErr = null;
    for (let i = 0; i < attempts.length; i++) {
      const useModel = attempts[i];
      if (i > 0) {
        patchMsg(convoId, msgId, m => ({
          model: useModel, content: "", reasoning: undefined,
          routing: m.routing ? { ...m.routing, model: useModel } : m.routing,
        }));
      }
      try {
        const { content, reasoning, usage } = await streamChat(useModel, apiMessages, (answer, think) => {
          patchMsg(convoId, msgId, { content: answer, reasoning: think });
        }, ctrl.signal);
        patchMsg(convoId, msgId, {
          content, reasoning, streaming: false, error: false, model: useModel,
          elapsed: ((Date.now() - t0) / 1000).toFixed(1), tokens: usage.tokens,
        });
        setConvos(prev => prev.map(c => c.id === convoId ? { ...c, model: useModel, updatedAt: Date.now() } : c));
        abortRef.current = null;
        return content;
      } catch (e) {
        if (e.name === "AbortError") {
          patchMsg(convoId, msgId, { streaming: false });
          abortRef.current = null;
          return null;
        }
        lastErr = e;
        if (routingId) recordFailure(routingId, useModel);
      }
    }
    patchMsg(convoId, msgId, {
      streaming: false, error: true,
      content: isBrowser
        ? `In-browser model failed: ${lastErr?.message}\n\nThe model may be too large for this device's WebGPU memory — try a smaller model from Settings → Models.`
        : `Could not reach Ollama: ${lastErr?.message}\n\nMake sure Ollama is running and that this page's origin is allowed — see Settings → Connection.`,
    });
    abortRef.current = null;
    return null;
  };

  /* The MUTATE call — background, fired only when a mechanical trigger flags
     an ambiguity. Produces exactly one action; FORK/MERGE/CORRECT/RECLASSIFY
     land as a pending mutation needing user consent (auto-commit tier PROMPT),
     NONE is logged silently. `snapshot` is the post-Extract graph. */
  const runMutate = async (convoId, msgId, trigger, model, snapshot) => {
    let mut = null;
    try {
      const out = await chatOnce(model, [
        { role: "system", content: MUTATE_SYSTEM },
        { role: "user", content: buildMutateUser(snapshot || emptyMemory(), trigger) },
      ], MUTATE_SCHEMA);
      const parsed = parseMutate(out);
      if (parsed) mut = makeMutation(parsed, { trigger, msgId });
    } catch { /* fail silently — the turn already succeeded */ }
    if (!mut) return;
    setConvos(prev => prev.map(c => {
      if (c.id !== convoId) return c;
      const memory = cloneMemory(c.memory);
      memory.mutations.push(mut);
      return { ...c, memory };
    }));
  };

  /* The EXTRACT call — background, reads a completed turn into the chat's
     graph. The exchange is labelled with its Given-Log ids; both messages are
     appended to the Given-Log. After applying, mechanical triggers are checked
     and a MUTATE call is fired for each ambiguity found. */
  const runExtract = async (convoId, msgId, userGiven, modelGiven, model, memCtx) => {
    let ops = [];
    try {
      const convo = convos.find(c => c.id === convoId);
      const out = await chatOnce(model, [
        { role: "system", content: EXTRACT_SYSTEM },
        { role: "user", content: `${buildRoster(convo?.memory)}\n\nEXCHANGE:\n`
          + `User [${userGiven.id}]: "${userGiven.text.slice(0, 4000)}"\n`
          + `Model [${modelGiven.id}]: "${modelGiven.text.slice(0, 4000)}"` },
      ], WALK_SCHEMA);
      ops = parseWalk(out);
    } catch { /* fail silently — the turn already succeeded */ }

    let learned = 0, triggers = [], snapshot = null;
    setConvos(prev => prev.map(c => {
      if (c.id !== convoId) return c;
      const memory = cloneMemory(c.memory);
      memory.givenLog.push({ ...userGiven }, { ...modelGiven });
      const res = applyWalk(memory, ops, { source: userGiven.id });
      learned = res.applied;
      memory.lastTurn = {
        entities: (memCtx.entities || []).map(e => e.canonical),
        topic: (memCtx.sig?.keywords || []).slice(0, 3).join(" "),
        userMessage: userGiven.text.slice(0, 100),
      };
      triggers = detectMutationTriggers({
        memory, userMessage: userGiven.text, modelResponse: modelGiven.text, ambigs: res.ambigs,
      });
      snapshot = memory;
      return { ...c, memory };
    }));
    patchMsg(convoId, msgId, m => ({ mem: { ...(m.mem || {}), learned } }));
    for (const t of triggers) runMutate(convoId, msgId, t, model, snapshot);
  };

  /* Read a block of text into a library document with a stateful walk: split
     it into sentences, group them into passages, and read each passage with
     the roster of sites already found in hand — so the model resolves and
     enriches sites instead of duplicating them. The document is added to the
     library and opted in to the current chat. The per-passage trace is live. */
  const runIngest = async (rawText, title, source) => {
    const text = (rawText || "").trim();
    if (!text || ingestRunning) return;
    const model = backgroundModel(composerModel === AUTO_MODEL ? modelNames[0] : composerModel);
    if (!model) return;

    // A document needs a Memory-mode chat to belong to — make or adopt one.
    let convoId = activeId;
    const cur = activeId ? convos.find(c => c.id === activeId) : null;
    if (!cur) {
      convoId = "c" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
      setConvos(prev => [{
        id: convoId, title: title || ("Library · " + text.slice(0, 32)),
        model, updatedAt: Date.now(), messages: [], mode: "memory",
        memory: emptyMemory(), docs: [],
      }, ...prev]);
      setActiveId(convoId);
      setMode("memory");
    } else if (cur.mode !== "memory") {
      setConvos(prev => prev.map(c => c.id === convoId
        ? { ...c, mode: "memory", memory: c.memory || emptyMemory() } : c));
      setMode("memory");
    }

    const docId = "d" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const passages = batchSentences(splitSentences(text));
    const trace = passages.map((c, i) => ({
      index: i, chars: c.length, text: c, signal: signal(c),
      status: "pending", rawOutput: "", ops: [], applied: 0,
    }));
    const sync = () => setIngestTrace(trace.map(t => ({ ...t })));
    setIngestRunning(true);
    sync();

    const docMem = emptyMemory();
    const docGiven = appendGiven(docMem, makeGiven({
      agent: "system:ingest", text: title || source || "pasted text",
    }));
    const allAmbigs = [];
    for (let i = 0; i < passages.length; i++) {
      trace[i].status = "reading"; sync();
      try {
        const out = await chatOnce(model, [
          { role: "system", content: INGEST_SYSTEM },
          { role: "user", content: `${buildRoster(docMem)}\n\nPASSAGE:\n${passages[i]}` },
        ], WALK_SCHEMA);
        const ops = parseWalk(out);
        const res = applyWalk(docMem, ops, { source: docGiven.id });
        trace[i].applied = res.applied;
        trace[i].ambigs = res.ambigs;
        allAmbigs.push(...res.ambigs);
        trace[i].rawOutput = out;
        trace[i].ops = ops;
        trace[i].status = "done";
      } catch (e) {
        trace[i].status = "error";
        trace[i].rawOutput = String(e?.message || e);
      }
      sync();
    }

    // Each AMBIG the ingest emitted fires a MUTATE call for a clean decision.
    const docMutations = [];
    for (const a of allAmbigs) {
      const trigger = { kind: "ambig", target: a.candidate, name: a.name, detail: a.span || a.name };
      try {
        const out = await chatOnce(model, [
          { role: "system", content: MUTATE_SYSTEM },
          { role: "user", content: buildMutateUser(docMem, trigger) },
        ], MUTATE_SCHEMA);
        const parsed = parseMutate(out);
        if (parsed) {
          const mut = makeMutation(parsed, { trigger });
          mut.docId = docId;
          docMutations.push(mut);
        }
      } catch { /* fail silently — the document was still read */ }
    }
    docMem.mutations.push(...docMutations);

    const learned = trace.reduce((n, t) => n + (t.applied || 0), 0);
    const doc = {
      id: docId,
      title: title || source || (text.slice(0, 40) + (text.length > 40 ? "…" : "")),
      source: source || "pasted text",
      addedAt: new Date().toISOString(),
      chars: text.length, passages: passages.length, learned,
      text,
      memory: docMem,
      trace: trace.map(t => ({
        index: t.index, chars: t.chars, status: t.status, applied: t.applied,
        signal: { names: t.signal.ner.names, dates: t.signal.ner.dates, keywords: t.signal.keywords },
        ops: t.ops, ambigs: t.ambigs || [], rawOutput: t.rawOutput,
      })),
    };
    setLibrary(prev => [doc, ...prev]);
    const pendingDocMuts = docMutations.filter(m => m.status === "pending");
    setConvos(prev => prev.map(c => {
      if (c.id !== convoId) return c;
      const memory = cloneMemory(c.memory);
      memory.mutations.push(...pendingDocMuts);
      return { ...c, docs: [...(c.docs || []), docId], memory, updatedAt: Date.now() };
    }));
    setIngestRunning(false);
  };

  /* Opt the active chat in or out of a library document. */
  const toggleDoc = (docId) => {
    if (!activeId) return;
    setConvos(prev => prev.map(c => {
      if (c.id !== activeId) return c;
      const has = (c.docs || []).includes(docId);
      return {
        ...c, mode: "memory", memory: c.memory || emptyMemory(), updatedAt: Date.now(),
        docs: has ? c.docs.filter(d => d !== docId) : [...(c.docs || []), docId],
      };
    }));
    if (mode !== "memory") setMode("memory");
  };

  /* Drop a document from the library and from every chat that used it. */
  const removeDoc = (docId) => {
    setLibrary(prev => prev.filter(d => d.id !== docId));
    setConvos(prev => prev.map(c => (c.docs || []).includes(docId)
      ? { ...c, docs: c.docs.filter(d => d !== docId) } : c));
  };

  /* Export a chat as an auditable JSON file: every message with the exact
     prompt it was given, the chat's memory graph, and the full read trace
     of each opted-in library document. */
  const exportConvo = (c) => {
    if (!c) return;
    const attached = (c.docs || []).map(id => library.find(d => d.id === id)).filter(Boolean);
    const data = {
      app: "LLManager", schema: "chat-audit-1", exportedAt: new Date().toISOString(),
      chat: { id: c.id, title: c.title, mode: c.mode || "regular", model: c.model, updatedAt: c.updatedAt },
      messages: c.messages.map(m => ({
        id: m.id, role: m.role, model: m.model, content: m.content,
        reasoning: m.reasoning || undefined, elapsed: m.elapsed, tokens: m.tokens,
        routing: m.routing || undefined, memory: m.mem || undefined,
        promptSent: m.prompt || undefined,
        givenId: m.givenId || undefined,
        dossierHash: m.dossierHash || undefined,
        spans: m.spans || undefined,
      })),
      memory: c.memory || undefined,
      library: attached.map(d => ({
        id: d.id, title: d.title, source: d.source, addedAt: d.addedAt,
        chars: d.chars, passages: d.passages, learned: d.learned,
        text: d.text, memory: d.memory, readTrace: d.trace,
      })),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `llmanager-${(c.title || "chat").replace(/[^a-z0-9]+/gi, "-").slice(0, 40)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const send = async () => {
    const text = draft.trim();
    if (!text || !composerModel || busy) return;
    const autoMode = composerModel === AUTO_MODEL;
    if (autoMode && !modelNames.length) return;
    setBusy(true);
    setDraft("");

    let convoId = activeId;
    let history = [];
    const existing = convoId ? convos.find(c => c.id === convoId) : null;

    // EVA: implicit signals about prior Auto-routed replies in this convo.
    let evalDoneIds = new Set();
    if (existing) {
      const results = processImplicitSignals({ messages: existing.messages }, text, autoMode);
      evalDoneIds = new Set(results.map(r => r.routingId));
      history = existing.messages.filter(m => !m.error).map(m => ({ role: m.role, content: m.content }));
    }

    // DEF: classify the prompt and commit to a model when in Auto mode.
    let routing = null, candidates = null, chosenModel = composerModel;
    if (autoMode) {
      const weights = loadWeights();
      const cls = classifyIntent(text, installed, weights);
      const r = routeModel(cls.intent, installed, weights);
      candidates = r.candidates;
      chosenModel = r.model;
      routing = { id: uuid(), intent: cls.intent, confidence: cls.confidence, model: chosenModel, candidates };
      appendLog({
        id: routing.id, timestamp: new Date().toISOString(), convoId: convoId || null,
        promptHash: await hashPrompt(text), promptLength: text.length,
        intent: cls.intent, confidence: cls.confidence, modelChosen: chosenModel,
        candidates, evaluated: false, signals: [], alternateModel: null,
      });
    }

    if (!convoId) {
      convoId = "c" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
      setConvos(prev => [{
        id: convoId, title: text.slice(0, 48) + (text.length > 48 ? "…" : ""),
        model: chosenModel, updatedAt: Date.now(), messages: [],
        mode, memory: mode === "memory" ? emptyMemory() : undefined,
      }, ...prev]);
      setActiveId(convoId);
    }

    // Build the API messages. Memory mode replaces the growing history with a
    // fixed-size prompt: system + projected dossier + one-turn position marker.
    // Every turn is projected and every turn is read back into the graph — the
    // prompt stays minimal regardless of how much the chat has covered.
    let apiMessages, memCtx = null, memBadge;
    if (mode === "memory") {
      const sig = signal(text);
      const sys = memorySystemPrompt(existing, sig);
      apiMessages = [
        { role: "system", content: sys.content },
        { role: "user", content: text },
      ];
      memCtx = { sig, entities: sys.entities, dossierHash: sys.dossierHash, spans: sys.spans };
      memBadge = { used: sys.used };
    } else {
      const apiHistory = quantize ? quantizeHistory(history) : history;
      apiMessages = [...apiHistory, { role: "user", content: text }];
    }

    const userMsg = { id: "u" + Date.now(), role: "user", content: text };
    const aId = "a" + Date.now();
    const placeholder = {
      id: aId, role: "assistant", model: chosenModel, content: "", streaming: true,
      routing, mem: memBadge, prompt: apiMessages,
      dossierHash: memCtx?.dossierHash, spans: memCtx?.spans,
    };
    setConvos(prev => prev.map(c => {
      if (c.id !== convoId) return c;
      const base = evalDoneIds.size
        ? c.messages.map(m => (m.routing && evalDoneIds.has(m.routing.id))
            ? { ...m, routing: { ...m.routing, evalDone: true } } : m)
        : c.messages;
      return { ...c, messages: [...base, userMsg, placeholder], updatedAt: Date.now() };
    }));

    const finalContent = await runTurn(convoId, aId, chosenModel, apiMessages, { candidates, routingId: routing?.id });
    setBusy(false);
    maybeREC();

    // Background: read every memory-mode turn back into the chat's graph.
    // Both messages are recorded in the Given-Log with content hashes; the
    // model entry carries the dossier projection that produced it.
    if (mode === "memory" && memCtx && finalContent) {
      const userGiven = makeGiven({ agent: "user", text });
      const modelGiven = makeGiven({
        agent: "model:" + chosenModel, text: finalContent,
        dossierHash: memCtx.dossierHash, spans: memCtx.spans,
      });
      patchMsg(convoId, aId, { givenId: modelGiven.id });
      runExtract(convoId, aId, userGiven, modelGiven, backgroundModel(chosenModel), memCtx);
    }
  };

  /* Re-run an assistant turn. forcedModel set => "Wrong model" pick;
     otherwise a plain regenerate (a weak negative EVA signal in Auto mode). */
  const rerun = async (msgId, forcedModel) => {
    if (!active || busy) return;
    const idx = active.messages.findIndex(m => m.id === msgId);
    if (idx < 1) return;
    const oldMsg = active.messages[idx];
    setBusy(true);

    const history = active.messages.slice(0, idx)
      .filter(m => !m.error)
      .map(m => ({ role: m.role, content: m.content }));
    const autoMode = !forcedModel && composerModel === AUTO_MODEL;

    if (!forcedModel && oldMsg.routing && !oldMsg.routing.evalDone) {
      appendSignal(oldMsg.routing.id, { type: "implicit", value: -0.3, action: "regenerate", model: oldMsg.routing.model });
    }

    let routing = null, candidates = null, useModel = forcedModel || composerModel;
    if (autoMode) {
      const userMsg = [...active.messages.slice(0, idx)].reverse().find(m => m.role === "user");
      const promptText = userMsg?.content || "";
      const weights = loadWeights();
      const cls = classifyIntent(promptText, installed, weights);
      const r = routeModel(cls.intent, installed, weights);
      candidates = r.candidates;
      useModel = r.model;
      routing = { id: uuid(), intent: cls.intent, confidence: cls.confidence, model: useModel, candidates };
      appendLog({
        id: routing.id, timestamp: new Date().toISOString(), convoId: active.id,
        promptHash: await hashPrompt(promptText), promptLength: promptText.length,
        intent: cls.intent, confidence: cls.confidence, modelChosen: useModel,
        candidates, evaluated: false, signals: [], alternateModel: null,
      });
    }

    // Fresh routing for an Auto re-run; otherwise keep the old pill but settle it.
    const newRouting = autoMode
      ? routing
      : (oldMsg.routing ? { ...oldMsg.routing, evalDone: true, model: useModel } : undefined);

    // Memory mode: rebuild the fixed-size prompt instead of replaying history.
    // A re-run only regenerates the reply — it does not re-extract memory.
    let apiMessages, memBadge = oldMsg.mem;
    if (active.mode === "memory") {
      const promptMsg = [...active.messages.slice(0, idx)].reverse().find(m => m.role === "user");
      const promptText = promptMsg?.content || "";
      const sys = memorySystemPrompt(active, signal(promptText));
      apiMessages = [
        { role: "system", content: sys.content },
        { role: "user", content: promptText },
      ];
      memBadge = { used: sys.used };
    } else {
      apiMessages = quantize ? quantizeHistory(history) : history;
    }

    patchMsg(active.id, msgId, {
      model: useModel, content: "", reasoning: undefined, streaming: true, error: false,
      elapsed: undefined, tokens: undefined, routing: newRouting, mem: memBadge, prompt: apiMessages,
    });
    await runTurn(active.id, msgId, useModel, apiMessages, { candidates, routingId: routing?.id });
    setBusy(false);
    maybeREC();
  };

  /* EVA: explicit feedback on an Auto-routed reply */
  const handleFeedback = (msgId, action) => {
    if (!active) return;
    const r = active.messages.find(m => m.id === msgId)?.routing;
    if (!r || r.evalDone) return;
    if (action === "up") appendSignal(r.id, { type: "explicit", value: 1.0, action: "thumbs-up", model: r.model });
    else if (action === "down") appendSignal(r.id, { type: "explicit", value: -0.5, action: "thumbs-down", model: r.model });
    patchMsg(active.id, msgId, m => ({ routing: { ...m.routing, evalDone: true, feedback: action } }));
    maybeREC();
  };

  /* Accept or dismiss a pending MUTATE action. Accepting applies it to the
     graph it belongs to — the chat's own memory, or a library document's
     graph when the mutation came from ingest. */
  const resolveMutation = (mutId, accept) => {
    if (!active) return;
    const mut = (active.memory?.mutations || []).find(m => m.id === mutId);
    if (!mut) return;
    if (accept && mut.docId) {
      setLibrary(prev => prev.map(d => {
        if (d.id !== mut.docId) return d;
        const memory = cloneMemory(d.memory);
        applyMutation(memory, mut);
        return { ...d, memory };
      }));
    }
    setConvos(prev => prev.map(c => {
      if (c.id !== active.id) return c;
      const memory = cloneMemory(c.memory);
      if (accept && !mut.docId) applyMutation(memory, mut);
      memory.mutations = memory.mutations.map(m =>
        m.id === mutId ? { ...m, status: accept ? "accepted" : "dismissed" } : m);
      return { ...c, memory };
    }));
  };

  const handleWrongModel = (msgId, altModel) => {
    setWrongModelFor(null);
    if (!active || !altModel) return;
    const r = active.messages.find(m => m.id === msgId)?.routing;
    if (!r || r.evalDone) return;
    appendSignal(r.id, { type: "explicit", value: -1.0, action: "wrong-model", model: r.model });
    appendSignal(r.id, { type: "explicit", value: 0.5, action: "wrong-model-alt", model: altModel });
    recordAlternateModel(r.id, altModel);
    rerun(msgId, altModel);
  };

  const stop = () => abortRef.current?.abort();

  const headerModel = active
    ? (active.messages.filter(m => m.role === "assistant").pop()?.model || active.model)
    : composerModel;

  const offline = ollamaUp === false || ollamaUp === "cors";

  const memStats = mode === "memory" ? memoryStats(combinedMemory(active)) : null;
  const docCount = active?.docs?.length || 0;

  /* Pending MUTATE actions awaiting consent. Those tied to a message render
     inline beneath it; the rest (e.g. from ingest) render after the thread. */
  const pendingMuts = (active?.memory?.mutations || []).filter(m => m.status === "pending");
  const orphanMuts = pendingMuts.filter(m => !m.msgId || !messages.some(x => x.id === m.msgId));

  return (
    <div style={{ display: "flex", height: "100%", width: "100%", background: C.bg, fontFamily: sans, color: C.text }}>
      <Sidebar
        convos={convos} activeId={activeId} query={query}
        onSearch={setQuery} onNew={newChat} onSelect={selectConvo} onDelete={deleteConvo}
      />
      <main style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <header style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 22px", borderBottom: `1px solid ${C.border}`, minHeight: 54 }}>
          <div style={{ flex: 1, minWidth: 0, fontSize: 13.5, fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {active ? active.title : "New chat"}
          </div>
          {active && (
            <span style={{ fontSize: 10.5, fontFamily: mono, color: C.dim, flexShrink: 0 }}>{active.messages.length} messages</span>
          )}
          <ModeToggle mode={mode} onChange={changeMode} disabled={busy} />
          {memStats && (
            <span title={`Projected memory: ${memStats.entities} entities, ${memStats.edges} connections, ${memStats.defs} facts`
              + (docCount ? ` — including ${docCount} library document${docCount === 1 ? "" : "s"}.` : ".")}
              style={{ display: "flex", alignItems: "center", gap: 5, padding: "5px 10px", background: C.accent + "18", border: `1px solid ${C.accent}44`, borderRadius: 7, fontFamily: mono, fontSize: 11, color: C.accent, flexShrink: 0 }}>
              <Icon name="memory" size={11} /> {memStats.entities} remembered{docCount ? ` · ${docCount} doc${docCount === 1 ? "" : "s"}` : ""}
            </span>
          )}
          <HeaderBtn icon="book" label="Library" onClick={() => setLibOpen(true)}
            badge={library.length || undefined} />
          {active && <HeaderBtn icon="download" label="Export" onClick={() => exportConvo(active)} />}
          {headerModel && (
            <span style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 10px", background: C.s1, border: `1px solid ${C.border}`, borderRadius: 7, fontFamily: mono, fontSize: 11.5, flexShrink: 0 }}>
              <span style={{ width: 6, height: 6, borderRadius: 99, background: C.accent,
                animation: headerModel === AUTO_MODEL ? "llm-pulse 1.4s ease-in-out infinite" : "none" }} />
              {headerModel === AUTO_MODEL ? "⚡ Auto" : headerModel}
            </span>
          )}
        </header>

        {offline && (
          <div style={{ margin: "12px 24px 0", padding: "10px 14px", background: C.red + "12", border: `1px solid ${C.red}40`, borderRadius: 8, fontSize: 12, color: C.red }}>
            {ollamaUp === "cors"
              ? "Ollama is running but is blocking this page. Open Settings → Connection to fix the allowed origins."
              : "Ollama is not reachable. Open Settings → Connection for setup steps."}
          </div>
        )}

        <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "16px 24px 8px", minHeight: 0 }}>
          <div style={{ maxWidth: 820, margin: "0 auto" }}>
            {messages.length === 0 ? (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "50vh", textAlign: "center" }}>
                <div style={{ maxWidth: 420 }}>
                  <div style={{ display: "inline-flex", padding: 14, borderRadius: 99, background: "rgba(110,86,207,.12)", color: C.accent, marginBottom: 14 }}>
                    <Icon name="chat" size={22} />
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 600, marginBottom: 8 }}>Start a new chat</div>
                  <div style={{ fontSize: 13, color: C.dim, lineHeight: 1.6 }}>
                    {!modelNames.length
                      ? "No models installed yet. Pull one from Settings → Models first."
                      : mode === "memory"
                        ? "Memory mode is on. Every turn is distilled into a per-chat graph and replayed as a fixed-size context block, so the prompt never grows with the conversation. Use Library in the header to read documents into memory. Switch back to Regular in the header."
                        : "Pick a model below and ask anything. You can switch models mid-conversation — each reply is labelled with the model that produced it. Try Memory mode in the header for a chat whose prompt never grows."}
                  </div>
                </div>
              </div>
            ) : (
              messages.map((m, i) => {
                const prevAi = [...messages.slice(0, i)].reverse().find(x => x.role === "assistant");
                const userMsgsAfter = messages.slice(i + 1).filter(x => x.role === "user").length;
                const msgMuts = pendingMuts.filter(x => x.msgId === m.id);
                return (
                  <div key={m.id || i}>
                    <MessageBubble
                      msg={m} prevModel={prevAi?.model}
                      onCopy={copy} copied={copied} onRerun={rerun} onFork={forkConvo} busy={busy}
                      installed={installed} userMsgsAfter={userMsgsAfter}
                      onFeedback={handleFeedback} onWrongModel={handleWrongModel}
                      wrongModelFor={wrongModelFor} setWrongModelFor={setWrongModelFor}
                    />
                    {msgMuts.map(mut => (
                      <MutationPill key={mut.id} mut={mut}
                        onAccept={() => resolveMutation(mut.id, true)}
                        onDismiss={() => resolveMutation(mut.id, false)} />
                    ))}
                  </div>
                );
              })
            )}
            {orphanMuts.map(mut => (
              <MutationPill key={mut.id} mut={mut}
                onAccept={() => resolveMutation(mut.id, true)}
                onDismiss={() => resolveMutation(mut.id, false)} />
            ))}
          </div>
        </div>

        <Composer
          value={draft} setValue={setDraft}
          model={composerModel} models={modelNames.length ? [AUTO_MODEL, ...modelNames] : []} setModel={setComposerModel}
          onSend={send} onStop={stop} busy={busy}
          isReply={messages.length > 0}
          quantize={quantize} setQuantize={setQuantize}
          mode={mode}
          memModel={memModel} memModels={modelNames} setMemModel={setMemModel}
        />
      </main>
      <LibraryModal
        open={libOpen} onClose={() => setLibOpen(false)}
        library={library} activeConvo={active} canIngest={modelNames.length > 0}
        ingestRunning={ingestRunning} ingestTrace={ingestTrace}
        onIngest={runIngest} onToggleDoc={toggleDoc} onRemoveDoc={removeDoc}
      />
    </div>
  );
}
