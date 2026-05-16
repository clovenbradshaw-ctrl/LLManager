import { useState, useEffect, useRef, useMemo } from "react";
import Markdown from "./Markdown.jsx";
import {
  AUTO_MODEL, INTENTS,
  classifyIntent, routeModel, hashPrompt, uuid,
  loadLog, loadWeights, loadPrefs, savePrefs,
  appendLog, appendSignal, recordAlternateModel, recordFailure,
  processImplicitSignals, runREC, shouldRunREC,
} from "./router.js";

const mono = `'SF Mono','Menlo','Consolas',monospace`;
const sans = `-apple-system,system-ui,sans-serif`;
const C = {
  bg: "#0b0b0f", s1: "#131318", s2: "#1b1b22", s3: "#232330",
  border: "#282838", text: "#d4d4e4", dim: "#65657e", accent: "#6e56cf",
  green: "#30a46c", red: "#e5484d", orange: "#f76b15",
};

const LS_KEY = "llmanager.chats.v1";

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
        {msg.elapsed && <span style={{ fontFamily: mono, fontSize: 10.5, color: C.dim }}>· {msg.elapsed}s</span>}
        {msg.tokens ? <span style={{ fontFamily: mono, fontSize: 10.5, color: C.dim }}>· {msg.tokens} tok</span> : null}
      </div>
      <div style={{ paddingLeft: 15, borderLeft: `1.5px solid ${C.border}` }}>
        {msg.error ? (
          <div style={{ fontSize: 13, color: C.red, whiteSpace: "pre-wrap" }}>{msg.content}</div>
        ) : (
          <>
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
function Composer({ value, setValue, model, models, setModel, onSend, onStop, busy, isReply }) {
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

/* ── Main chat ── */
export default function Chat({ ollamaUrl, installed, ollamaUp }) {
  const [convos, setConvos] = useState(loadConvos);
  const [activeId, setActiveId] = useState(null);
  const [query, setQuery] = useState("");
  const [draft, setDraft] = useState("");
  const [composerModel, setComposerModel] = useState(() => (loadPrefs().autoMode ? AUTO_MODEL : ""));
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(null);
  const [wrongModelFor, setWrongModelFor] = useState(null);
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
    }, ...prev]);
    setActiveId(newId);
    setDraft("");
  };

  const selectConvo = (id) => {
    setActiveId(id);
    const c = convos.find(x => x.id === id);
    const lastModel = c?.messages.filter(m => m.role === "assistant").pop()?.model || c?.model;
    if (lastModel && modelNames.includes(lastModel)) setComposerModel(lastModel);
  };

  /* Stream a chat completion from Ollama; calls onToken with the full text so far */
  const streamChat = async (model, apiMessages, onToken, signal) => {
    const r = await fetch(`${ollamaUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: apiMessages, stream: true }),
      signal,
    });
    if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", full = "", usage = {};
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
        if (j.message?.content) { full += j.message.content; onToken(full); }
        if (j.done) usage = { tokens: j.eval_count };
      }
    }
    return { content: full, usage };
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
          model: useModel, content: "",
          routing: m.routing ? { ...m.routing, model: useModel } : m.routing,
        }));
      }
      try {
        const { content, usage } = await streamChat(useModel, apiMessages, (full) => {
          patchMsg(convoId, msgId, { content: full });
        }, ctrl.signal);
        patchMsg(convoId, msgId, {
          content, streaming: false, error: false, model: useModel,
          elapsed: ((Date.now() - t0) / 1000).toFixed(1), tokens: usage.tokens,
        });
        setConvos(prev => prev.map(c => c.id === convoId ? { ...c, model: useModel, updatedAt: Date.now() } : c));
        abortRef.current = null;
        return;
      } catch (e) {
        if (e.name === "AbortError") {
          patchMsg(convoId, msgId, { streaming: false });
          abortRef.current = null;
          return;
        }
        lastErr = e;
        if (routingId) recordFailure(routingId, useModel);
      }
    }
    patchMsg(convoId, msgId, {
      streaming: false, error: true,
      content: `Could not reach Ollama: ${lastErr?.message}\n\nMake sure Ollama is running and that this page's origin is allowed — see the Status tab.`,
    });
    abortRef.current = null;
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
      }, ...prev]);
      setActiveId(convoId);
    }

    const userMsg = { id: "u" + Date.now(), role: "user", content: text };
    const aId = "a" + Date.now();
    const placeholder = { id: aId, role: "assistant", model: chosenModel, content: "", streaming: true, routing };
    setConvos(prev => prev.map(c => {
      if (c.id !== convoId) return c;
      const base = evalDoneIds.size
        ? c.messages.map(m => (m.routing && evalDoneIds.has(m.routing.id))
            ? { ...m, routing: { ...m.routing, evalDone: true } } : m)
        : c.messages;
      return { ...c, messages: [...base, userMsg, placeholder], updatedAt: Date.now() };
    }));

    await runTurn(convoId, aId, chosenModel, [...history, { role: "user", content: text }], { candidates, routingId: routing?.id });
    setBusy(false);
    maybeREC();
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

    patchMsg(active.id, msgId, {
      model: useModel, content: "", streaming: true, error: false,
      elapsed: undefined, tokens: undefined, routing: newRouting,
    });
    await runTurn(active.id, msgId, useModel, history, { candidates, routingId: routing?.id });
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
              ? "Ollama is running but is blocking this page. Open the Status tab to fix the allowed origins."
              : "Ollama is not reachable. Open the Status tab for setup steps."}
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
                    {modelNames.length
                      ? "Pick a model below and ask anything. You can switch models mid-conversation — each reply is labelled with the model that produced it."
                      : "No models installed yet. Pull one from the Models tab first."}
                  </div>
                </div>
              </div>
            ) : (
              messages.map((m, i) => {
                const prevAi = [...messages.slice(0, i)].reverse().find(x => x.role === "assistant");
                const userMsgsAfter = messages.slice(i + 1).filter(x => x.role === "user").length;
                return (
                  <MessageBubble
                    key={m.id || i} msg={m} prevModel={prevAi?.model}
                    onCopy={copy} copied={copied} onRerun={rerun} onFork={forkConvo} busy={busy}
                    installed={installed} userMsgsAfter={userMsgsAfter}
                    onFeedback={handleFeedback} onWrongModel={handleWrongModel}
                    wrongModelFor={wrongModelFor} setWrongModelFor={setWrongModelFor}
                  />
                );
              })
            )}
          </div>
        </div>

        <Composer
          value={draft} setValue={setDraft}
          model={composerModel} models={modelNames.length ? [AUTO_MODEL, ...modelNames] : []} setModel={setComposerModel}
          onSend={send} onStop={stop} busy={busy}
          isReply={messages.length > 0}
        />
      </main>
    </div>
  );
}
