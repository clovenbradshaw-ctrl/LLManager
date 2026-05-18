import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  OPERATORS, OP_COLORS, processText, emptyGraph, appendToGraph,
  buildDossier, runSecondPass, reclassifyFlags, GROUNDED_SYSTEM,
  askAboutEntity, foldFrames,
} from "./eo-classifier.js";
import { getBrowserEngine } from "./webllm.js";

/* Chat Mode 2.0 — the EO Classifier as a chat.

   The library-and-folders ingest flow is replaced by the chat itself:
   paste material and it is cleaned, split into clauses, classified against
   the 27 EO centroids and folded into a cumulative in-memory graph.

   Answering is graph-grounded, not conversation-grounded: a question is
   never answered by replaying the chat or the raw pasted text. It is
   answered from a dossier — entities, claims and passages retrieved from
   the structured graph for that specific question — and nothing else. */

const MODEL_KEY = "llmanager.chat2.model";
const AUTO = "__auto__";

/* Ollama model base names in rough quality order — used by the Auto picker. */
const AUTO_PREFS = [
  "qwen3:30b-a3b", "qwen2.5:14b", "phi3:medium", "qwen3:8b", "deepseek-r1:8b",
  "llama3.1:8b", "gemma2:9b", "qwen2.5:7b", "mistral", "phi3:mini",
  "llama3.2:3b", "gemma2:2b",
];

/* Pick the strongest available model: prefer Ollama over in-browser, then
   rank by AUTO_PREFS. An unranked model sorts last but still beats nothing. */
function autoPick(options) {
  if (options.length === 0) return null;
  const ollama = options.filter(o => !o.isBrowser);
  const pool = ollama.length ? ollama : options;
  const rank = o => {
    const base = o.name.split(":")[0];
    const i = AUTO_PREFS.findIndex(p => o.name === p || p.split(":")[0] === base);
    return i < 0 ? AUTO_PREFS.length : i;
  };
  return [...pool].sort((a, b) => rank(a) - rank(b))[0];
}

/* ── Point-based fold routing ──

   An entity question can be answered globally (the dossier path) or as a
   situated fold AT A POINT in the text. The fold path is taken when the
   question names a known entity and either pins a clause window ("up to
   clause 40") or asks something interpretive ("what does X seem to be"). */

const INTERPRETIVE = /\b(seem|seems|seemed|think|thinks|thinking|thought|feel|feels|feeling|really|actually|become|becomes|becoming|became|change|changes|changing|changed|arc|impression|portrayed|come across|like at this point|so far)\b/i;

/* Parse an explicit clause window — "up to clause 40", "at clause 12", "c12".
   Returns a 0-based clause index, or null when no window is named. */
function parseClauseWindow(text) {
  const m = text.match(/\bclause\s*c?(\d+)/i) || text.match(/\bc(\d+)\b/i);
  if (!m) return null;
  const n = parseInt(m[1], 10);
  return Number.isFinite(n) && n > 0 ? n - 1 : null;
}

/* Find the single known entity a question is about. Matches the full entity
   name or any distinctive token of it; the longest match wins. */
function findFocusEntity(graph, text) {
  const q = text.toLowerCase();
  let best = null;
  for (const [key, entity] of Object.entries(graph.entities)) {
    const name = entity.name.toLowerCase();
    const candidates = [name, ...name.split(/\s+/).filter(t => t.length > 2)];
    for (const cand of candidates) {
      const re = new RegExp(`\\b${cand.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
      if (re.test(q) && (!best || cand.length > best.matchLen)) {
        best = { key, name: entity.name, matchLen: cand.length };
      }
    }
  }
  return best;
}

/* Ungrounded chat — used while no source has been added to the graph. */
const CHAT_SYSTEM =
  "You are a helpful, concise assistant. Answer the user's question directly. "
  + "No sources have been added yet, so answer from general knowledge. When it "
  + "would help, mention that the user can add a source — a text file or pasted "
  + "text — to get answers grounded in their own material.";

/* Small models love to fall into a loop — emitting the same phrase over
   and over until the token budget runs out. Detect that by scanning the
   tail of the text for a short unit that repeats back-to-back, so the
   stream can be cut off instead of replaying nonsense. */
function looksLooping(text) {
  if (text.length < 120) return false;
  const tail = text.slice(-800);
  for (let unit = 3; unit <= 160 && unit * 4 <= tail.length; unit++) {
    const seg = tail.slice(-unit);
    if (!seg.trim()) continue;
    let reps = 1, pos = tail.length - unit;
    while (pos - unit >= 0 && tail.slice(pos - unit, pos) === seg) {
      reps++; pos -= unit;
      if (reps >= 4) return true;
    }
  }
  return false;
}

/* Once a loop is cut off, the tail is still N copies of the repeated
   unit — strip the extra copies so only one clean instance remains. */
function trimLoopTail(text) {
  for (let unit = 3; unit <= 160 && unit * 4 <= text.length; unit++) {
    const seg = text.slice(-unit);
    if (!seg.trim()) continue;
    let reps = 1, pos = text.length - unit;
    while (pos - unit >= 0 && text.slice(pos - unit, pos) === seg) {
      reps++; pos -= unit;
    }
    if (reps >= 4) return text.slice(0, pos + unit).trimEnd();
  }
  return text;
}

function ordinal(n) {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

const MAX_TOKENS = 1024;

/* A chat post longer than this is material, not a question — it is
   ingested as a source so the classifier can build a graph from it and
   answers can be grounded in it, rather than stuffed into one prompt. */
const LONG_POST_CHARS = 2500;

/* Unified generation over both runtimes. Pass `onToken` to stream (it is
   called with the full text so far); omit it for a blocking call. A `format`
   JSON schema constrains Ollama output for the deep-read calls. */
async function generate({ model, isBrowser, ollamaUrl, system, user, format, onToken }) {
  const messages = [
    { role: "system", content: system },
    { role: "user", content: user },
  ];

  if (isBrowser) {
    const engine = await getBrowserEngine(model, report => {
      if (onToken) {
        const pct = Math.round((report.progress || 0) * 100);
        onToken(null, `Loading ${model} — ${report.text || pct + "%"}`);
      }
    });
    const params = {
      temperature: format ? 0 : 0.3,
      frequency_penalty: 0.6, presence_penalty: 0.3,
      max_tokens: MAX_TOKENS,
    };
    if (onToken) {
      const chunks = await engine.chat.completions.create({
        messages, stream: true, ...params,
      });
      let raw = "";
      for await (const ch of chunks) {
        const d = ch.choices?.[0]?.delta?.content || "";
        if (d) {
          raw += d;
          onToken(raw, null);
          if (looksLooping(raw)) {
            await engine.interruptGenerate();
            raw = trimLoopTail(raw);
            break;
          }
        }
      }
      return raw;
    }
    const reply = await engine.chat.completions.create({ messages, ...params });
    return reply.choices?.[0]?.message?.content || "";
  }

  // Ollama
  const body = {
    model, messages, stream: !!onToken, keep_alive: "30m",
    options: {
      temperature: format ? 0 : 0.3, num_ctx: 8192,
      repeat_penalty: 1.3, num_predict: MAX_TOKENS,
    },
  };
  if (format) body.format = format;
  const r = await fetch(`${ollamaUrl}/api/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`Ollama HTTP ${r.status} — is the server running and the model installed?`);
  if (!onToken) {
    const data = await r.json();
    return data.message?.content || "";
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "", raw = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      let j;
      try { j = JSON.parse(line); } catch { continue; }
      if (j.error) throw new Error(j.error);
      const c = j.message?.content || "";
      if (c) { raw += c; onToken(raw, null); }
    }
    if (looksLooping(raw)) { await reader.cancel(); raw = trimLoopTail(raw); break; }
  }
  return raw;
}

/* Split a model reply into its hidden <think> reasoning and the visible
   answer. Reasoning models (Qwen3 and friends) emit a <think>…</think>
   block that should not appear inline in the answer. Handles partial
   streaming: an unclosed <think> means the model is still reasoning. */
function splitThinking(text) {
  if (!text) return { thinking: "", answer: "", reasoning: false };
  let thinking = "", answer = text, reasoning = false;
  const open = answer.indexOf("<think>");
  if (open !== -1) {
    const close = answer.indexOf("</think>", open);
    if (close === -1) {
      thinking = answer.slice(open + 7);
      answer = answer.slice(0, open);
      reasoning = true;
    } else {
      thinking = answer.slice(open + 7, close);
      answer = answer.slice(0, open) + answer.slice(close + 8);
    }
  }
  return { thinking: thinking.trim(), answer: answer.trim(), reasoning };
}

const mono = `'SF Mono','Menlo','Consolas',monospace`;
const C = {
  bg: "#0b0b0f", s1: "#131318", s2: "#1b1b22", s3: "#232330",
  border: "#282838", text: "#d4d4e4", dim: "#65657e", accent: "#6e56cf",
  green: "#30a46c", red: "#e5484d", orange: "#f76b15", amber: "#fbbf24",
};

let msgSeq = 0;
const nextId = () => `m${++msgSeq}`;

/* ── Operator distribution bar for an analysis card ── */
function OpSummary({ opCounts, total, inertCount }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "10px 0" }}>
      {Object.keys(OPERATORS).map(op => {
        const n = opCounts[op] || 0;
        const pct = total > 0 ? Math.round((n / total) * 100) : 0;
        return (
          <div key={op} style={{
            padding: "5px 9px", background: C.s2, borderRadius: "0 4px 4px 0",
            borderLeft: `3px solid ${OP_COLORS[op]}`, fontSize: 11, fontFamily: mono,
            opacity: n === 0 ? 0.4 : 1,
          }}>
            <span style={{ color: "#fff", fontWeight: 600 }}>{op}</span>{" "}
            <span style={{ color: C.dim }}>{n} ({pct}%)</span>
          </div>
        );
      })}
      {inertCount > 0 && (
        <div style={{
          padding: "5px 9px", background: C.s2, borderRadius: "0 4px 4px 0",
          borderLeft: "3px solid #333", fontSize: 11, fontFamily: mono, color: C.dim,
        }}>INERT {inertCount} not classified</div>
      )}
    </div>
  );
}

/* ── A single clause's classification rows. Salience drives the block's
   opacity — a low-salience clause fades, and lifts to full on hover. ── */
function ClauseCard({ ci, rows }) {
  const [hover, setHover] = useState(false);
  const head = rows[0];
  const wholeRow = rows.find(r =>
    r.rawType === "clause" || r.rawType === "revived" || r.rawType === "inert");
  const sal = (wholeRow && wholeRow.salience != null) ? wholeRow.salience
    : (head && head.salience != null ? head.salience : 0.5);
  const opacity = hover ? 1 : Math.max(0.2, Math.min(1, sal * 2.5));

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        background: C.s2, border: `1px solid ${C.border}`, borderRadius: 6,
        marginBottom: 8, overflow: "hidden", opacity, transition: "opacity 0.2s",
      }}>
      <div style={{
        padding: "6px 10px", fontSize: 9, textTransform: "uppercase",
        letterSpacing: "0.1em", color: C.dim, borderBottom: `1px solid ${C.border}`,
      }}>Clause {Number(ci) + 1}</div>
      <div style={{
        padding: "8px 10px", fontSize: 13, lineHeight: 1.5, color: C.text,
        borderBottom: `1px solid ${C.border}`,
      }}>{head.clause}</div>
      <div style={{ padding: 8 }}>
        {rows.map((r, i) => {
          const isWholeClause = r.rawType === "clause";
          const isInert = r.rawType === "inert";
          const isRevived = r.rawType === "revived";
          const isWhole = isWholeClause || isInert || isRevived;
          const opColor = OP_COLORS[r.operator.name] || "#666";
          return (
            <div key={i} style={{
              padding: 6, marginBottom: 4, borderRadius: 4,
              borderLeft: isRevived ? `2px solid ${C.green}` : undefined,
              fontStyle: isInert ? "italic" : undefined,
              opacity: isInert ? 0.5 : 1,
              background: isRevived ? "rgba(48,164,108,0.07)"
                : isWholeClause ? "rgba(110,86,207,0.07)" : "rgba(255,255,255,0.02)",
            }}>
              <div style={{ fontSize: 11, marginBottom: isWhole && !isInert ? 4 : 0 }}>
                {isInert
                  ? <em style={{ color: C.dim }}>inert ({(r.salience || 0).toFixed(2)})</em>
                  : isRevived
                    ? <em style={{ color: C.green }}>revived ({(r.salience || 0).toFixed(2)})</em>
                    : isWholeClause
                      ? <em style={{ color: C.dim }}>whole clause</em>
                      : <><strong style={{ color: "#fff" }}>{r.entity}</strong>
                        <span style={{ color: C.dim }}> → "{r.value}"</span></>}
                {!isWhole && <span style={{
                  fontSize: 9, marginLeft: 6, padding: "1px 5px", borderRadius: 3,
                  background: C.border, color: C.dim, textTransform: "uppercase",
                }}>{r.rawType}</span>}
              </div>
              {!isInert && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", fontFamily: mono }}>
                  <span style={{ fontSize: 10, padding: "1px 7px", borderRadius: 3, border: `1px solid ${opColor}`, color: opColor }}>{r.operator.name}</span>
                  <span style={{ fontSize: 10, padding: "1px 7px", borderRadius: 3, border: `1px solid ${C.border}`, color: C.green }}>{r.terrain.name}</span>
                  <span style={{ fontSize: 10, padding: "1px 7px", borderRadius: 3, border: `1px solid ${C.border}`, color: C.dim }}>{r.stance.name}</span>
                  <span style={{ fontSize: 10, color: C.dim, fontStyle: "italic" }}>
                    {r.operator.name}({r.terrain.name}, {r.stance.name})
                  </span>
                </div>
              )}
              {isWhole && r.needsReading && r.flagReasons && r.flagReasons.length > 0 && (
                <div style={{ marginTop: 4, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", fontSize: 10 }}>
                  <span style={{ color: C.amber }}>⚑</span>
                  {r.flagReasons.map((reason, k) => (
                    <span key={k} style={{
                      padding: "1px 5px", borderRadius: 3, fontSize: 9,
                      background: "rgba(251,191,36,0.12)", color: C.amber,
                      textTransform: "uppercase", letterSpacing: "0.05em",
                    }}>{reason}</span>
                  ))}
                  <span style={{ color: C.dim, fontStyle: "italic" }}>
                    {r.mechanicalOp} — flagged for deep read
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Per-clause classification detail (collapsible) ── */
function ClauseDetail({ results }) {
  const byClause = {};
  for (const r of results) {
    (byClause[r.clauseIndex] = byClause[r.clauseIndex] || []).push(r);
  }
  return (
    <div style={{ marginTop: 8 }}>
      {Object.keys(byClause).map(ci => (
        <ClauseCard key={ci} ci={ci} rows={byClause[ci]} />
      ))}
    </div>
  );
}

/* ── Hypothesis register frames ── */
function FramePanel({ register }) {
  if (!register || register.frames.length === 0) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: C.dim, marginBottom: 6 }}>
        Hypothesis register — {register.frames.length} frame{register.frames.length !== 1 ? "s" : ""}
      </div>
      {register.frames.map(f => {
        const barColor = f.source === "llm" ? C.accent : (f.drift || 0) > 0.12 ? C.amber : C.green;
        const pct = Math.round((f.strength || 0) * 100);
        return (
          <div key={f.id} style={{
            padding: "7px 9px", background: C.s2, border: `1px solid ${C.border}`,
            borderRadius: 4, marginBottom: 5, fontSize: 12,
          }}>
            <div style={{ fontSize: 10, color: C.amber, fontWeight: 600 }}>
              {f.id} {f.source === "llm" ? "· LLM" : ""}{" "}
              {f.triggers ? `· ${f.triggers.map(t => t.type).join("+")}` : f.trigger ? `· ${f.trigger}` : ""}
            </div>
            <div style={{ margin: "3px 0", color: C.text }}>{f.text}</div>
            <div style={{ fontSize: 10, color: C.dim }}>
              born c{f.generatedAt + 1} · confirmed {f.confirmedBy.length}× · strength {pct}%
            </div>
            <div style={{ height: 4, background: C.border, borderRadius: 2, marginTop: 4, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${pct}%`, background: barColor }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Flagged clauses — where the surface and the function diverge ── */
function FlagPanel({ register }) {
  const flags = register.flags || [];
  if (flags.length === 0) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: C.amber, marginBottom: 6 }}>
        Flagged clauses — {flags.length} · surface ≠ function
      </div>
      {flags.map((f, i) => {
        const reclassified = f.functionalOp && f.functionalOp !== f.mechanicalOp;
        const opColor = OP_COLORS[f.functionalOp] || OP_COLORS[f.mechanicalOp] || "#666";
        return (
          <div key={i} style={{
            padding: "7px 9px", background: "rgba(251,191,36,0.06)",
            border: `1px solid ${reclassified ? opColor : "rgba(251,191,36,0.4)"}`,
            borderLeft: `3px solid ${reclassified ? opColor : C.amber}`,
            borderRadius: 4, marginBottom: 5, fontSize: 12,
          }}>
            <div style={{ fontSize: 10, color: C.amber, fontFamily: mono }}>
              c{f.clauseIndex + 1} · flagged: {f.reason}
            </div>
            <div style={{ margin: "3px 0", color: C.text, fontStyle: "italic" }}>
              "{f.text.length > 140 ? f.text.slice(0, 140) + "…" : f.text}"
            </div>
            <div style={{ fontSize: 11, fontFamily: mono }}>
              {reclassified ? (
                <span>
                  <s style={{ color: C.dim }}>{f.mechanicalOp}</s>
                  <span style={{ color: C.dim }}> → </span>
                  <strong style={{ color: opColor }}>{f.functionalOp}</strong>
                  <span style={{ color: C.text }}>: {f.functionalReason}</span>
                </span>
              ) : f.functionalOp ? (
                <span style={{ color: C.green }}>
                  {f.mechanicalOp} holds — confirmed by reading
                  {f.functionalReason ? `: ${f.functionalReason}` : ""}
                </span>
              ) : (
                <span style={{ color: C.dim }}>
                  {f.mechanicalOp} — the shadow flagged this; run a deep read to interpret
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── A single analysis card (assistant turn for ingested material) ── */
function AnalysisCard({ msg, onDeepRead, deepReadBusy, llmReady }) {
  const [showDetail, setShowDetail] = useState(false);
  const { stats, opCounts, results, register, triggerCount } = msg;
  const flagCount = register.flags?.length || 0;
  const canDeepRead = (triggerCount > 0 || flagCount > 0) && !msg.deepReadDone;

  return (
    <div style={{
      background: C.s1, border: `1px solid ${C.border}`, borderRadius: 10,
      padding: 14, fontFamily: mono,
    }}>
      <div style={{ fontSize: 11, color: C.green, fontWeight: 600, marginBottom: 2 }}>
        ◆ Source read into the graph
      </div>
      {msg.provenance && (
        <div style={{ fontSize: 10, color: C.dim, marginBottom: 4 }}>
          from: <span style={{ color: C.text }}>{msg.provenance}</span>
        </div>
      )}
      <div style={{ fontSize: 12, color: C.text }}>
        {stats.newClauses} clause{stats.newClauses !== 1 ? "s" : ""} ·{" "}
        {stats.newClaims} claim{stats.newClaims !== 1 ? "s" : ""} ·{" "}
        {stats.newEntities} new entit{stats.newEntities !== 1 ? "ies" : "y"} ·{" "}
        {register.frames.length} frame{register.frames.length !== 1 ? "s" : ""}
        {stats.inertCount > 0 && (
          <span style={{ color: C.dim }}> · {stats.inertCount} inert</span>
        )}
        {stats.revivedCount > 0 && (
          <span style={{ color: C.green }}> · {stats.revivedCount} revived</span>
        )}
        {flagCount > 0 && (
          <span style={{ color: C.amber }}> · {flagCount} flagged</span>
        )}
      </div>

      <OpSummary opCounts={opCounts} total={stats.clauseRows} inertCount={stats.inertCount} />

      {stats.entityNames.length > 0 && (
        <div style={{ fontSize: 12, color: C.text, marginBottom: 8 }}>
          <span style={{ color: C.dim }}>Entities: </span>
          {stats.entityNames.join(", ")}
        </div>
      )}

      <FramePanel register={register} />
      <FlagPanel register={register} />

      <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
        <button onClick={() => setShowDetail(s => !s)} style={btn(false)}>
          {showDetail ? "Hide" : "Show"} per-clause analysis
        </button>
        {canDeepRead && (
          <button
            onClick={() => onDeepRead(msg.id)}
            disabled={deepReadBusy || !llmReady}
            title={!llmReady ? "No model available"
              : `${triggerCount} trigger point${triggerCount !== 1 ? "s" : ""}, ${flagCount} flagged clause${flagCount !== 1 ? "s" : ""}`}
            style={btn(true, deepReadBusy || !llmReady)}
          >
            {deepReadBusy ? "Reading…"
              : `Deep read (${triggerCount} trigger${triggerCount !== 1 ? "s" : ""}` +
                (flagCount > 0 ? `, ${flagCount} flag${flagCount !== 1 ? "s" : ""}` : "") + ")"}
          </button>
        )}
        {msg.deepReadDone && (
          <span style={{ fontSize: 11, color: C.accent, alignSelf: "center" }}>
            ⬡ deep read complete
          </span>
        )}
      </div>

      {showDetail && <ClauseDetail results={results} />}
    </div>
  );
}

/* ── Situated folds — the per-site readings behind a point-based answer ── */
function FoldColumn({ msg }) {
  return (
    <div>
      <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.08em", color: C.green, marginBottom: 6, fontFamily: mono }}>
        Situated folds{Number.isFinite(msg.foldAt) ? ` · up to c${msg.foldAt + 1}` : ""}
      </div>
      {msg.spans.length === 0 && (
        <div style={{ fontSize: 11, color: C.dim, fontStyle: "italic" }}>
          Too few claims to fold — answered from raw evidence.
        </div>
      )}
      {msg.spans.map((s, i) => (
        <div key={i} style={{
          padding: "6px 9px", borderLeft: `3px solid ${C.green}`,
          background: C.s2, borderRadius: "0 3px 3px 0", marginBottom: 5,
        }}>
          <div style={{ fontSize: 9, color: C.dim, fontFamily: mono }}>
            @{s.site} · {s.claimCount} claim{s.claimCount !== 1 ? "s" : ""}
          </div>
          <div style={{ fontSize: 11.5, color: C.text, lineHeight: 1.45, marginTop: 2 }}>
            {s.text}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── A grounded answer card ── */
function AnswerCard({ msg }) {
  const [showThink, setShowThink] = useState(false);
  const { thinking, answer, reasoning } = splitThinking(msg.text);
  const thinkingOnly = reasoning && !answer;
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: msg.spans.length ? (msg.foldMode ? "1fr 280px" : "1fr 220px") : "1fr",
      gap: 12, alignItems: "start",
    }}>
      <div style={{
        fontSize: 14, lineHeight: 1.65, color: C.text, whiteSpace: "pre-wrap",
      }}>
        {msg.foldMode && msg.foldEntity && (
          <div style={{
            fontSize: 10, fontFamily: mono, color: C.green, marginBottom: 6,
          }}>
            ◆ {msg.foldEntity} folded at clause {msg.foldAt + 1}
          </div>
        )}
        {thinking && (
          <div style={{ marginBottom: answer ? 8 : 0 }}>
            <button
              onClick={() => setShowThink(s => !s)}
              style={{
                fontSize: 10, fontFamily: mono, color: C.dim,
                background: "none", border: "none", padding: 0, cursor: "pointer",
              }}
            >
              {showThink ? "▾" : "▸"} {msg.streaming && thinkingOnly ? "Thinking…" : "Reasoning"}
            </button>
            {showThink && (
              <div style={{
                marginTop: 4, padding: "6px 8px", fontSize: 12, lineHeight: 1.55,
                color: C.dim, background: C.s2, borderRadius: 4,
                borderLeft: `2px solid ${C.border}`,
              }}>
                {thinking}
              </div>
            )}
          </div>
        )}
        {answer || (msg.streaming
          ? <span style={{ color: C.dim }}>{thinkingOnly ? "" : "Generating…"}</span>
          : "")}
        {msg.streaming && (answer || thinkingOnly) && <span style={{ color: C.accent }}>▍</span>}
      </div>
      {msg.foldMode ? (msg.spans.length >= 0 && <FoldColumn msg={msg} />) : (
      msg.spans.length > 0 && (
        <div>
          <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.08em", color: C.dim, marginBottom: 6, fontFamily: mono }}>
            Retrieved context
          </div>
          {msg.spans.map((s, i) => {
            const opColor = OP_COLORS[s.operator] || "#555";
            return (
              <div key={i} style={{
                padding: "5px 8px", borderLeft: `3px solid ${opColor}`,
                background: C.s2, borderRadius: "0 3px 3px 0", marginBottom: 4,
              }}>
                <div style={{ fontSize: 9, color: C.dim, fontFamily: mono }}>
                  {s.entity} · {s.operator}
                </div>
                <div style={{ fontSize: 11, fontStyle: "italic", color: C.text, lineHeight: 1.4 }}>
                  "{s.text.slice(0, 110)}"
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

const selectStyle = {
  background: C.s1, border: `1px solid ${C.border}`, color: C.text,
  fontSize: 11, fontFamily: mono, padding: "5px 7px", borderRadius: 6,
  maxWidth: 168,
};

function btn(primary, disabled) {
  return {
    padding: "5px 12px", fontSize: 11, fontFamily: mono, fontWeight: 600,
    borderRadius: 6, cursor: disabled ? "not-allowed" : "pointer",
    border: `1px solid ${primary ? C.accent : C.border}`,
    background: primary ? C.accent : "transparent",
    color: primary ? "#fff" : C.dim, opacity: disabled ? 0.4 : 1,
  };
}

export default function Chat2({ ollamaUrl, ollamaModels, browserModels }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("Ask anything below. Add sources to ground the answers in your own material.");
  const [busy, setBusy] = useState(false);
  const [deepReadBusy, setDeepReadBusy] = useState(false);
  const [model, setModel] = useState(() => {
    try { return localStorage.getItem(MODEL_KEY) || AUTO; } catch { return AUTO; }
  });
  // Source composer — paste text with a provenance label, or attach files.
  const [showSources, setShowSources] = useState(false);
  const [sourceText, setSourceText] = useState("");
  const [sourceProv, setSourceProv] = useState("");
  const fileRef = useRef(null);

  const graphRef = useRef(emptyGraph());
  const clauseBaseRef = useRef(0);
  const postSeqRef = useRef(0); // ordinal for chat posts ingested as sources
  const scrollRef = useRef(null);
  const [graphTick, setGraphTick] = useState(0); // forces stat re-render

  // Every Ollama model and every loaded in-browser model — the same rosters
  // the main Chat picker offers.
  const modelOptions = useMemo(() => [
    ...(ollamaModels || []).map(m => ({ name: m.name, isBrowser: false, label: m.name })),
    ...(browserModels || []).map(m => ({ name: m.name, isBrowser: true, label: `${m.name} · in-browser` })),
  ], [ollamaModels, browserModels]);

  // Keep the selection valid as the rosters change — fall back to Auto if a
  // pinned model disappears.
  useEffect(() => {
    if (model !== AUTO && modelOptions.length > 0
        && !modelOptions.some(o => o.name === model)) {
      setModel(AUTO);
    }
  }, [modelOptions]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (model) { try { localStorage.setItem(MODEL_KEY, model); } catch { /* ignore */ } }
  }, [model]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const autoMode = model === AUTO;
  const selectedModel = autoMode
    ? autoPick(modelOptions)
    : modelOptions.find(o => o.name === model) || null;
  const llmReady = !!selectedModel;
  const graphHasContent = Object.keys(graphRef.current.entities).length > 0
    || graphRef.current.clauses.length > 0;

  const ingestMaterial = useCallback(async (text, provenance) => {
    setStatus("Loading embedding model (first run downloads ~25 MB)…");
    const { results, clauses, register } = await processText(text, setStatus, graphRef.current);

    if (clauses.length === 0) {
      setMessages(m => [...m, {
        id: nextId(), role: "assistant", kind: "note",
        text: "Nothing extractable in that source — no clauses found.",
      }]);
      setStatus("Ready.");
      return;
    }

    const base = clauseBaseRef.current;
    const added = await appendToGraph(graphRef.current, results, base, register);
    clauseBaseRef.current = base + clauses.length;
    setGraphTick(t => t + 1);

    // Operator distribution over whole-clause rows.
    const clauseRows = results.filter(r => r.rawType === "clause" || r.rawType === "revived");
    const opCounts = {};
    for (const r of clauseRows) opCounts[r.operator.name] = (opCounts[r.operator.name] || 0) + 1;
    const inertCount = results.filter(r => r.rawType === "inert").length;
    const revivedCount = results.filter(r => r.rawType === "revived").length;
    const entityNames = [...new Set(results
      .filter(r => r.rawType !== "inert" && r.rawType !== "clause" && r.rawType !== "revived"
        && r.entity && !r.entity.startsWith("("))
      .map(r => r.entity))];

    setMessages(m => [...m, {
      id: nextId(), role: "assistant", kind: "analysis",
      provenance: provenance || "pasted text",
      stats: { ...added, clauseRows: clauseRows.length, inertCount, revivedCount, entityNames },
      opCounts, results, clauses, register, clauseBase: base,
      triggerCount: register.triggerPoints?.length || 0,
      deepReadDone: false,
    }]);
    setStatus(`Graph: ${Object.keys(graphRef.current.entities).length} entities, ${graphRef.current.claims.length} claims, ${graphRef.current.clauses.length} clauses.`);
  }, []);

  const answerQuestion = useCallback(async (text) => {
    if (!selectedModel) {
      throw new Error("No model available. Start Ollama, or load an in-browser model from the Chat or Settings tab.");
    }

    // ── Point-based fold path ──
    // A question about one known entity, pinned to a clause window or asking
    // something interpretive, is answered from situated folds at that point —
    // not the global dossier. The fold at clause N is what a reader holds in
    // mind at clause N: provisional, committed, superseded as the text moves.
    if (graphHasContent) {
      const focus = findFocusEntity(graphRef.current, text);
      const clauseWindow = parseClauseWindow(text);
      if (focus && (clauseWindow !== null || INTERPRETIVE.test(text))) {
        const lastClause = graphRef.current.clauses.reduce(
          (mx, c) => Math.max(mx, c.index), 0);
        const upTo = clauseWindow !== null ? clauseWindow : lastClause;
        const foldLlm = (sys, user) => generate({
          model: selectedModel.name, isBrowser: selectedModel.isBrowser, ollamaUrl,
          system: sys, user,
        });
        const result = await askAboutEntity(
          graphRef.current, focus.key, text, upTo, foldLlm, setStatus);
        if (result) {
          setMessages(m => [...m, {
            id: nextId(), role: "assistant", kind: "answer",
            text: result.answer || "(the model returned an empty response)",
            spans: result.folds.map(f => ({
              entity: f.site, operator: "fold", site: f.site,
              text: f.text, claimCount: f.claimCount,
            })),
            streaming: false, foldMode: true, foldAt: result.atClause,
            foldEntity: result.entity,
          }]);
          setStatus(`Answered from ${result.folds.length} situated fold`
            + `${result.folds.length !== 1 ? "s" : ""} of ${result.entity}`
            + ` up to clause ${result.atClause + 1}.`);
          return;
        }
        // No claims for this entity in the window — fall through to the dossier.
      }
    }

    // Short follow-ups ("why?", "and her brother?") carry little signal of
    // their own — fold in the previous question so retrieval still lands on
    // the right region.
    const prevQ = [...messages].reverse().find(m => m.role === "user" && m.kind === "question");
    const prevA = [...messages].reverse().find(m => m.role === "assistant" && m.kind === "answer");

    // Light continuity: the previous exchange only, marked explicitly as a
    // reference for resolving pronouns — never treated as a source.
    let convo = "";
    if (prevQ && prevA && prevA.text) {
      convo = "[EARLIER EXCHANGE — for resolving references only, NOT a source]\n"
        + `Q: ${prevQ.text}\nA: ${prevA.text.slice(0, 400)}\n[/EARLIER EXCHANGE]\n\n`;
    }

    let system, prompt;
    const uniqueSpans = [];

    if (graphHasContent) {
      setStatus("Retrieving context from the graph…");
      const retrievalQuery = (text.split(/\s+/).length < 5 && prevQ)
        ? `${prevQ.text} ${text}` : text;

      // The dossier IS the context — entities, claims and passages pulled from
      // the structured graph for this question. The chat history and the raw
      // source text are never sent wholesale.
      const { ctx, docs, spans } = await buildDossier(graphRef.current, retrievalQuery);

      // Dedup spans for the evidence column.
      const seen = new Set();
      for (const s of spans) {
        const k = s.text.slice(0, 60);
        if (seen.has(k)) continue;
        seen.add(k);
        uniqueSpans.push(s);
      }

      system = GROUNDED_SYSTEM;
      prompt = `${ctx}\n\n${docs}\n\n`
        + `[NOTE] The numbered [DOCS] passages are listed in the order they `
        + `were received in this conversation — lower numbers came first.\n\n`
        + `${convo}QUESTION: ${text}`;
    } else {
      // No sources yet — answer as an ordinary assistant.
      system = CHAT_SYSTEM;
      prompt = `${convo}QUESTION: ${text}`;
    }

    const answerId = nextId();
    setMessages(m => [...m, {
      id: answerId, role: "assistant", kind: "answer",
      text: "", spans: uniqueSpans.slice(0, 12), streaming: true,
      systemPrompt: system, userPrompt: prompt,
      model: selectedModel.name, grounded: graphHasContent,
    }]);
    setStatus(graphHasContent ? "Generating grounded answer…" : "Generating answer…");

    let final = "";
    await generate({
      model: selectedModel.name, isBrowser: selectedModel.isBrowser, ollamaUrl,
      system, user: prompt,
      onToken: (raw, loading) => {
        if (loading != null) { setStatus(loading); return; }
        final = raw;
        setMessages(m => m.map(x => x.id === answerId ? { ...x, text: raw } : x));
      },
    });

    const hasAnswer = splitThinking(final).answer || final.trim();
    setMessages(m => m.map(x => x.id === answerId
      ? { ...x, text: hasAnswer ? final.trim() : "(the model returned an empty response)", streaming: false }
      : x));
    setStatus("Ready.");
  }, [messages, selectedModel, ollamaUrl, graphHasContent]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || busy) return;

    setInput("");
    setBusy(true);
    try {
      if (text.length > LONG_POST_CHARS) {
        // Too long to be a question — treat the paste as a source. It is
        // ingested into the graph so later questions are grounded in it.
        // The ordinal records where in the conversation it arrived.
        const seq = ++postSeqRef.current;
        const prov = `chat post #${seq}`;
        setMessages(m => [...m, { id: nextId(), role: "user", kind: "source",
          text, provenance: prov }]);
        setMessages(m => [...m, { id: nextId(), role: "assistant", kind: "note",
          text: `That was long, so it was added as a source — "${prov}", `
            + `the ${ordinal(seq)} post received in this conversation. `
            + `Ask a question about it below.` }]);
        await ingestMaterial(text, prov);
      } else {
        setMessages(m => [...m, { id: nextId(), role: "user", kind: "question", text }]);
        await answerQuestion(text);
      }
    } catch (e) {
      setMessages(m => [...m, { id: nextId(), role: "assistant", kind: "error",
        text: e?.message || String(e) }]);
      setStatus("Error — see the message above.");
    } finally {
      setBusy(false);
    }
  }, [input, busy, answerQuestion, ingestMaterial]);

  // ── Sources — ingest a file or pasted text, tagged with its provenance ──
  const addSource = useCallback(async (text, provenance) => {
    const clean = (text || "").trim();
    if (!clean) return;
    const prov = (provenance || "").trim() || "pasted text";
    setMessages(m => [...m, { id: nextId(), role: "user", kind: "source", text: clean, provenance: prov }]);
    setBusy(true);
    try {
      await ingestMaterial(clean, prov);
    } catch (e) {
      setMessages(m => [...m, { id: nextId(), role: "assistant", kind: "error",
        text: e?.message || String(e) }]);
      setStatus("Error — see the message above.");
    } finally {
      setBusy(false);
    }
  }, [ingestMaterial]);

  const handlePasteSource = useCallback(async () => {
    if (!sourceText.trim() || busy) return;
    const text = sourceText;
    const prov = sourceProv;
    setSourceText("");
    setSourceProv("");
    await addSource(text, prov);
  }, [sourceText, sourceProv, busy, addSource]);

  const handleFiles = useCallback(async (fileList) => {
    const files = [...(fileList || [])];
    if (fileRef.current) fileRef.current.value = "";
    for (const file of files) {
      try {
        const text = await file.text();
        if (text.trim()) await addSource(text, file.name);
        else setMessages(m => [...m, { id: nextId(), role: "assistant", kind: "note",
          text: `${file.name} is empty — nothing to read.` }]);
      } catch (e) {
        setMessages(m => [...m, { id: nextId(), role: "assistant", kind: "error",
          text: `Could not read ${file.name}: ${e?.message || e}` }]);
      }
    }
  }, [addSource]);

  const handleDeepRead = useCallback(async (msgId) => {
    const msg = messages.find(m => m.id === msgId);
    if (!msg || deepReadBusy) return;
    if (!selectedModel) {
      setStatus("No model available for the deep read — load one in the Chat or Settings tab.");
      return;
    }
    setDeepReadBusy(true);
    try {
      const llm = (sys, user) => generate({
        model: selectedModel.name, isBrowser: selectedModel.isBrowser, ollamaUrl,
        system: sys, user,
      });
      await runSecondPass(msg.register, llm, setStatus);
      // Functional reclassification of the flagged clauses — the LLM names
      // the operator that is actually functioning where surface and function
      // diverged. The result is written back into the cumulative graph.
      const changed = await reclassifyFlags(
        msg.register, msg.clauses, graphRef.current, msg.clauseBase, llm, setStatus);
      // The deep read added LLM frames to the register — re-fold them into the
      // graph so point-based folds see the active reading they produced.
      foldFrames(graphRef.current, msg.register, msg.clauseBase);
      setGraphTick(t => t + 1);
      setMessages(m => m.map(x => x.id === msgId
        ? { ...x, register: { ...msg.register }, deepReadDone: true } : x));
      const flagCount = msg.register.flags?.length || 0;
      setStatus(flagCount > 0
        ? `Deep read complete — ${changed.length}/${flagCount} flagged clause${flagCount !== 1 ? "s" : ""} reclassified.`
        : "Deep read complete — LLM hypotheses added to the register.");
    } catch (e) {
      setStatus(`Deep read failed: ${e?.message || e}`);
    } finally {
      setDeepReadBusy(false);
    }
  }, [messages, deepReadBusy, selectedModel, ollamaUrl]);

  /* Export the whole session — graph, entity definitions, every ingestion
     pass, and each exchange with the exact system + user prompt sent to the
     model — as a plain-text log the user can download. */
  const exportLog = useCallback(() => {
    const gr = graphRef.current;
    const L = [];
    const rule = (c = "─") => c.repeat(64);

    L.push("LLM MANAGER — CHAT 2.0 SESSION LOG");
    L.push(`Exported   ${new Date().toISOString()}`);
    L.push(`Model      ${selectedModel ? selectedModel.name + (selectedModel.isBrowser ? " (in-browser)" : "") : "—"}`);
    L.push(`Messages   ${messages.length}`);
    L.push("");

    L.push(rule("═"));
    L.push("GRAPH — ENTITY DEFINITIONS");
    L.push(rule("═"));
    L.push(`${Object.keys(gr.entities).length} entities · ${gr.claims.length} claims · ${gr.clauses.length} clauses`);
    L.push("");
    for (const e of Object.values(gr.entities)) {
      L.push(`◆ ${e.name}  [terrain: ${e.terrain}]`);
      for (const c of e.claims) {
        L.push(`    ${c.notation}  ${c.rawType}: ${c.value}`);
        L.push(`      "${c.span}"  (clause ${c.clauseIndex})`);
      }
      for (const ed of e.edges || []) L.push(`    → ${ed.type}: ${ed.to}`);
      L.push("");
    }
    if (gr.clauses.length) {
      L.push(rule());
      L.push("CLAUSES");
      L.push(rule());
      for (const c of gr.clauses) {
        const op = c.functionalOp || c.mechanicalOp || "—";
        L.push(`[${c.index}] (${op}${c.needsReading ? ", flagged" : ""}`
          + `${c.salience != null ? `, salience ${c.salience.toFixed(2)}` : ""})`);
        L.push(`    ${c.text}`);
      }
      L.push("");
    }

    L.push(rule("═"));
    L.push("TRANSCRIPT");
    L.push(rule("═"));
    for (const m of messages) {
      L.push("");
      if (m.role === "user" && m.kind === "question") {
        L.push(rule());
        L.push("QUESTION");
        L.push(rule());
        L.push(m.text);
      } else if (m.role === "user" && m.kind === "source") {
        L.push(rule());
        L.push(`SOURCE · ${m.provenance || "pasted text"}`);
        L.push(rule());
        L.push(m.text);
      } else if (m.kind === "analysis") {
        const s = m.stats || {};
        L.push(rule());
        L.push(`INGESTION · ${m.provenance}`);
        L.push(rule());
        L.push(`+${s.newEntities || 0} entities, +${s.newClaims || 0} claims, +${s.newClauses || 0} clauses`);
        L.push(`clause rows: ${s.clauseRows}, inert: ${s.inertCount}, revived: ${s.revivedCount}`);
        L.push(`operators: ${Object.entries(m.opCounts || {}).map(([k, v]) => `${k}×${v}`).join(", ") || "—"}`);
        if (s.entityNames?.length) L.push(`entities found: ${s.entityNames.join(", ")}`);
        L.push("");
        L.push("clause classification:");
        for (const r of m.results || []) {
          if (r.rawType === "clause" || r.rawType === "revived" || r.rawType === "inert") {
            L.push(`  [${m.clauseBase + r.clauseIndex}] ${r.rawType}`
              + `${r.operator ? ` · ${r.operator.name}` : ""}`
              + `${r.terrain ? `(${r.terrain.name}, ${r.stance?.name})` : ""}`);
            L.push(`      ${(r.clause || "").trim()}`);
          } else {
            L.push(`  · ${r.entity} — ${r.rawType}: ${r.value}`
              + `  [${r.operator?.name}(${r.terrain?.name}, ${r.stance?.name})]`);
          }
        }
      } else if (m.kind === "answer") {
        L.push(rule());
        L.push("ANSWER"
          + `${m.grounded ? " · grounded" : ""}`
          + `${m.foldMode ? " · point-fold" : ""}`
          + `${m.model ? ` · ${m.model}` : ""}`);
        L.push(rule());
        if (m.foldMode && m.foldEntity) {
          L.push(`[FOLD] ${m.foldEntity} folded at clause ${m.foldAt + 1}`);
          L.push("");
        }
        if (m.systemPrompt) {
          L.push("[SYSTEM PROMPT]");
          L.push(m.systemPrompt);
          L.push("");
        }
        if (m.userPrompt) {
          L.push("[USER PROMPT — exact dossier sent to the model]");
          L.push(m.userPrompt);
          L.push("");
        }
        const { thinking, answer } = splitThinking(m.text || "");
        if (thinking) {
          L.push("[MODEL REASONING]");
          L.push(thinking);
          L.push("");
        }
        L.push("[ANSWER]");
        L.push(answer || m.text || "");
        if (m.spans?.length) {
          L.push("");
          L.push(m.foldMode ? "[SITUATED FOLDS]" : "[RETRIEVED CONTEXT]");
          for (const s of m.spans) {
            L.push(`  · ${s.entity} / ${s.operator}: "${s.text}"`);
          }
        }
      } else if (m.kind === "note") {
        L.push(`NOTE: ${m.text}`);
      } else if (m.kind === "error") {
        L.push(`ERROR: ${m.text}`);
      }
    }

    const blob = new Blob([L.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat2-log-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [messages, selectedModel]);

  const g = graphRef.current;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: C.bg, color: C.text }}>
      {/* Header */}
      <div style={{ padding: "12px 18px", borderBottom: `1px solid ${C.border}`, flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
          <div style={{ fontSize: 14, fontWeight: 700 }}>
            Chat 2.0 <span style={{ color: C.dim, fontWeight: 400, fontSize: 12 }}>· EO Classifier</span>
          </div>
          <button
            onClick={exportLog}
            disabled={messages.length === 0}
            style={btn(false, messages.length === 0)}
            title="Download a full log — graph, ingestion, prompts and answers"
          >
            ⤓ Export log
          </button>
        </div>
        <div style={{ fontSize: 11, fontFamily: mono, color: C.dim, marginTop: 3 }}>
          {graphTick >= 0 && (
            <>{Object.keys(g.entities).length} entities · {g.claims.length} claims · {g.clauses.length} clauses · </>
          )}
          {llmReady
            ? <span style={{ color: C.green }}>
                {autoMode ? "auto · " : ""}answering with {selectedModel.name}{selectedModel.isBrowser ? " (in-browser)" : ""}
              </span>
            : <span style={{ color: C.orange }}>no model available — load one in the Chat or Settings tab</span>}
        </div>
      </div>

      {/* Status */}
      <div style={{ padding: "6px 18px", fontSize: 11, fontFamily: mono, color: C.amber, flexShrink: 0,
        borderBottom: `1px solid ${C.border}` }}>
        {status}
      </div>

      {/* Messages */}
      <div ref={scrollRef} style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: 18 }}>
        {messages.length === 0 && (
          <div style={{ color: C.dim, fontSize: 13, lineHeight: 1.6, maxWidth: 620 }}>
            <p style={{ marginBottom: 8 }}>
              This is the 2.0 chat. <strong style={{ color: C.text }}>Just start asking</strong> —
              type a question below and it is answered right away.
            </p>
            <p style={{ marginBottom: 8 }}>
              <strong style={{ color: C.text }}>Add sources</strong> with the “+ Add source” button:
              attach any text file, or paste text and label where it came from. Each source is
              cleaned, split into clauses, and classified against the 27 EO centroids
              (operator × terrain × stance), building up a graph of entities and claims.
            </p>
            <p>
              Once you have added sources, your questions are answered <strong style={{ color: C.text }}>grounded</strong>{" "}
              in them — with the retrieved passages shown alongside the answer. Pick a model below,
              or leave it on <strong style={{ color: C.text }}>Auto</strong> to use the strongest one available.
            </p>
          </div>
        )}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {messages.map(m => {
            if (m.role === "user") {
              const isQ = m.kind === "question";
              return (
                <div key={m.id} style={{ alignSelf: "flex-end", maxWidth: "82%" }}>
                  <div style={{ fontSize: 9, fontFamily: mono, color: C.dim, textAlign: "right", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                    {isQ ? "Question" : `Source · ${m.provenance || "pasted text"}`}
                  </div>
                  <div style={{
                    background: isQ ? C.accent : C.s3,
                    color: isQ ? "#fff" : C.text,
                    padding: "9px 13px", borderRadius: 10, fontSize: 13, lineHeight: 1.5,
                    whiteSpace: "pre-wrap", wordBreak: "break-word",
                  }}>
                    {m.text.length > 600 ? m.text.slice(0, 600) + "\n…" : m.text}
                  </div>
                </div>
              );
            }
            return (
              <div key={m.id} style={{ alignSelf: "flex-start", width: "100%", maxWidth: 760 }}>
                {m.kind === "analysis" && (
                  <AnalysisCard msg={m} onDeepRead={handleDeepRead}
                    deepReadBusy={deepReadBusy} llmReady={llmReady} />
                )}
                {m.kind === "answer" && <AnswerCard msg={m} />}
                {m.kind === "note" && (
                  <div style={{ fontSize: 12, color: C.dim, fontStyle: "italic", padding: "4px 0" }}>{m.text}</div>
                )}
                {m.kind === "error" && (
                  <div style={{ fontSize: 12, color: C.red, padding: "8px 12px",
                    background: "rgba(229,72,77,0.08)", borderRadius: 8, border: `1px solid rgba(229,72,77,0.3)` }}>
                    {m.text}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Composer */}
      <div style={{ borderTop: `1px solid ${C.border}`, padding: 14, flexShrink: 0 }}>
        {showSources && (
          <div style={{
            background: C.s1, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: 12, marginBottom: 10,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <div style={{ fontSize: 11, fontFamily: mono, color: C.text, fontWeight: 600 }}>Add a source</div>
              <button onClick={() => setShowSources(false)} style={{
                background: "none", border: "none", color: C.dim, cursor: "pointer", fontSize: 13,
              }}>✕</button>
            </div>
            <div style={{ fontSize: 11, color: C.dim, lineHeight: 1.5, marginBottom: 9 }}>
              Attach a text file, or paste text and label where it came from. Each source is read
              into the graph and grounds your answers.
            </div>
            <input
              ref={fileRef}
              type="file"
              multiple
              accept=".txt,.md,.markdown,.csv,.tsv,.json,.log,.rtf,.html,.htm,.xml,.yaml,.yml,text/*"
              onChange={e => handleFiles(e.target.files)}
              disabled={busy}
              style={{ fontSize: 11, fontFamily: mono, color: C.dim, marginBottom: 10, display: "block" }}
            />
            <textarea
              value={sourceText}
              onChange={e => setSourceText(e.target.value)}
              placeholder="…or paste source text here"
              rows={3}
              style={{
                width: "100%", boxSizing: "border-box", resize: "vertical", background: C.bg,
                border: `1px solid ${C.border}`, borderRadius: 6, color: C.text,
                fontSize: 12, lineHeight: 1.5, padding: 8, fontFamily: "inherit", marginBottom: 8,
              }}
            />
            <div style={{ display: "flex", gap: 8 }}>
              <input
                value={sourceProv}
                onChange={e => setSourceProv(e.target.value)}
                placeholder="Provenance — e.g. NYT article, meeting notes, my email"
                style={{
                  flex: 1, background: C.bg, border: `1px solid ${C.border}`, borderRadius: 6,
                  color: C.text, fontSize: 12, padding: "7px 9px", fontFamily: mono,
                }}
              />
              <button
                onClick={handlePasteSource}
                disabled={busy || !sourceText.trim()}
                style={btn(true, busy || !sourceText.trim())}
              >
                {busy ? "Reading…" : "Add source"}
              </button>
            </div>
          </div>
        )}
        <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); handleSend(); }
            }}
            placeholder="Ask a question… (Cmd/Ctrl+Enter to send)"
            rows={3}
            style={{
              flex: 1, resize: "vertical", minHeight: 60, background: C.s1,
              border: `1px solid ${C.border}`, borderRadius: 8, color: C.text,
              fontSize: 13, lineHeight: 1.5, padding: 10, fontFamily: "inherit",
            }}
          />
          <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 168 }}>
            <button
              onClick={() => setShowSources(s => !s)}
              style={btn(showSources)}
            >
              {showSources ? "Hide sources" : "+ Add source"}
            </button>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              title="Model used for answers and the deep read"
              style={selectStyle}
            >
              <option value={AUTO}>
                Auto{selectedModel ? ` · ${selectedModel.name}` : ""}
              </option>
              {modelOptions.map(o => (
                <option key={o.name} value={o.name}>{o.label}</option>
              ))}
            </select>
            <button onClick={handleSend} disabled={busy || !input.trim()} style={{
              padding: "9px 18px", fontSize: 12, fontWeight: 600, borderRadius: 8,
              border: "none", cursor: busy || !input.trim() ? "not-allowed" : "pointer",
              background: C.accent, color: "#fff", opacity: busy || !input.trim() ? 0.4 : 1,
            }}>{busy ? "Working…" : "Send"}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
