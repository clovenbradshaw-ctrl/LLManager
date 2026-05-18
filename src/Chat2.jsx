import { useState, useEffect, useRef, useCallback } from "react";
import {
  OPERATORS, OP_COLORS, processText, emptyGraph, appendToGraph,
  buildDossier, runSecondPass, reclassifyFlags, looksLikeQuestion, GROUNDED_SYSTEM,
} from "./eo-classifier.js";
import { initRouter, callModel } from "./model-router.js";

/* Chat Mode 2.0 — the EO Classifier as a chat.

   The library-and-folders ingest flow is replaced by the chat itself:
   paste material and it is cleaned, split into clauses, classified against
   the 27 EO centroids and folded into a cumulative in-memory graph. Ask a
   question and it is answered, grounded, from that graph. */

const mono = `'SF Mono','Menlo','Consolas',monospace`;
const C = {
  bg: "#0b0b0f", s1: "#131318", s2: "#1b1b22", s3: "#232330",
  border: "#282838", text: "#d4d4e4", dim: "#65657e", accent: "#6e56cf",
  green: "#30a46c", red: "#e5484d", orange: "#f76b15", amber: "#fbbf24",
};

let msgSeq = 0;
const nextId = () => `m${++msgSeq}`;

/* ── Operator distribution bar for an analysis card ── */
function OpSummary({ opCounts, total, chromeCount }) {
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
      {chromeCount > 0 && (
        <div style={{
          padding: "5px 9px", background: C.s2, borderRadius: "0 4px 4px 0",
          borderLeft: "3px solid #333", fontSize: 11, fontFamily: mono, color: C.dim,
        }}>CHROME {chromeCount} skipped</div>
      )}
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
      {Object.keys(byClause).map(ci => {
        const rows = byClause[ci];
        const head = rows[0];
        return (
          <div key={ci} style={{
            background: C.s2, border: `1px solid ${C.border}`, borderRadius: 6,
            marginBottom: 8, overflow: "hidden",
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
                if (r.rawType === "chrome") {
                  return <div key={i} style={{ fontSize: 11, fontStyle: "italic", color: C.dim, padding: 4 }}>chrome — skipped</div>;
                }
                const opColor = OP_COLORS[r.operator.name] || "#666";
                const whole = r.rawType === "clause";
                return (
                  <div key={i} style={{
                    padding: 6, marginBottom: 4, borderRadius: 4,
                    background: whole ? "rgba(110,86,207,0.07)" : "rgba(255,255,255,0.02)",
                  }}>
                    <div style={{ fontSize: 11, marginBottom: 4 }}>
                      {whole ? <em style={{ color: C.dim }}>whole clause</em>
                        : <><strong style={{ color: "#fff" }}>{r.entity}</strong>
                          <span style={{ color: C.dim }}> → "{r.value}"</span></>}
                      {whole && r.needsReading && (
                        <span style={{
                          fontSize: 9, marginLeft: 6, padding: "1px 5px", borderRadius: 3,
                          background: "rgba(251,191,36,0.15)", color: C.amber,
                          border: `1px solid rgba(251,191,36,0.4)`,
                        }}>flagged · {r.flagReason}</span>
                      )}
                      {!whole && <span style={{
                        fontSize: 9, marginLeft: 6, padding: "1px 5px", borderRadius: 3,
                        background: C.border, color: C.dim, textTransform: "uppercase",
                      }}>{r.rawType}</span>}
                    </div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", fontFamily: mono }}>
                      <span style={{ fontSize: 10, padding: "1px 7px", borderRadius: 3, border: `1px solid ${opColor}`, color: opColor }}>{r.operator.name}</span>
                      <span style={{ fontSize: 10, padding: "1px 7px", borderRadius: 3, border: `1px solid ${C.border}`, color: C.green }}>{r.terrain.name}</span>
                      <span style={{ fontSize: 10, padding: "1px 7px", borderRadius: 3, border: `1px solid ${C.border}`, color: C.dim }}>{r.stance.name}</span>
                      <span style={{ fontSize: 10, color: C.dim, fontStyle: "italic" }}>
                        {r.operator.name}({r.terrain.name}, {r.stance.name})
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
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
              {f.id} {f.source === "llm" ? "· LLM" : ""} {f.trigger ? `· ${f.trigger}` : ""}
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
        ◆ Material read into the graph
      </div>
      <div style={{ fontSize: 12, color: C.text }}>
        {stats.newClauses} clause{stats.newClauses !== 1 ? "s" : ""} ·{" "}
        {stats.newClaims} claim{stats.newClaims !== 1 ? "s" : ""} ·{" "}
        {stats.newEntities} new entit{stats.newEntities !== 1 ? "ies" : "y"} ·{" "}
        {register.frames.length} frame{register.frames.length !== 1 ? "s" : ""}
        {flagCount > 0 && (
          <span style={{ color: C.amber }}> · {flagCount} flagged</span>
        )}
      </div>

      <OpSummary opCounts={opCounts} total={stats.clauseRows} chromeCount={stats.chromeCount} />

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

/* ── A grounded answer card ── */
function AnswerCard({ msg }) {
  return (
    <div style={{
      display: "grid", gridTemplateColumns: msg.spans.length ? "1fr 220px" : "1fr",
      gap: 12, alignItems: "start",
    }}>
      <div style={{
        fontSize: 14, lineHeight: 1.65, color: C.text, whiteSpace: "pre-wrap",
      }}>{msg.text}</div>
      {msg.spans.length > 0 && (
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
      )}
    </div>
  );
}

function btn(primary, disabled) {
  return {
    padding: "5px 12px", fontSize: 11, fontFamily: mono, fontWeight: 600,
    borderRadius: 6, cursor: disabled ? "not-allowed" : "pointer",
    border: `1px solid ${primary ? C.accent : C.border}`,
    background: primary ? C.accent : "transparent",
    color: primary ? "#fff" : C.dim, opacity: disabled ? 0.4 : 1,
  };
}

export default function Chat2({ ollamaUrl, ollamaUp }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sendMode, setSendMode] = useState("auto"); // auto | material | question
  const [status, setStatus] = useState("Paste any text below — it is read straight into the graph.");
  const [busy, setBusy] = useState(false);
  const [deepReadBusy, setDeepReadBusy] = useState(false);
  const [router, setRouter] = useState(null);

  const graphRef = useRef(emptyGraph());
  const clauseBaseRef = useRef(0);
  const scrollRef = useRef(null);
  const [graphTick, setGraphTick] = useState(0); // forces stat re-render

  useEffect(() => {
    let alive = true;
    initRouter(ollamaUrl).then(r => { if (alive) setRouter(r); }).catch(() => {});
    return () => { alive = false; };
  }, [ollamaUrl, ollamaUp]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const llmReady = !!router?.assignments?.write;
  const graphHasContent = Object.keys(graphRef.current.entities).length > 0
    || graphRef.current.clauses.length > 0;

  const askLLM = useCallback(async (assignment, system, user, options) => {
    if (!assignment) throw new Error("No model available — start Ollama or load an in-browser model, then reopen this tab.");
    return callModel(assignment, system, user, options);
  }, []);

  const ingestMaterial = useCallback(async (text) => {
    setStatus("Loading embedding model (first run downloads ~25 MB)…");
    const { results, clauses, register } = await processText(text, setStatus);

    if (clauses.length === 0) {
      setMessages(m => [...m, {
        id: nextId(), role: "assistant", kind: "note",
        text: "Nothing extractable in that text — no clauses found.",
      }]);
      setStatus("Ready.");
      return;
    }

    const base = clauseBaseRef.current;
    const added = await appendToGraph(graphRef.current, results, base);
    clauseBaseRef.current = base + clauses.length;
    setGraphTick(t => t + 1);

    // Operator distribution over whole-clause rows.
    const clauseRows = results.filter(r => r.rawType === "clause");
    const opCounts = {};
    for (const r of clauseRows) opCounts[r.operator.name] = (opCounts[r.operator.name] || 0) + 1;
    const chromeCount = results.filter(r => r.rawType === "chrome").length;
    const entityNames = [...new Set(results
      .filter(r => r.rawType !== "chrome" && r.rawType !== "clause" && r.entity && !r.entity.startsWith("("))
      .map(r => r.entity))];

    setMessages(m => [...m, {
      id: nextId(), role: "assistant", kind: "analysis",
      stats: { ...added, clauseRows: clauseRows.length, chromeCount, entityNames },
      opCounts, results, clauses, register, clauseBase: base,
      triggerCount: register.triggerPoints?.length || 0,
      deepReadDone: false,
    }]);
    setStatus(`Graph: ${Object.keys(graphRef.current.entities).length} entities, ${graphRef.current.claims.length} claims, ${graphRef.current.clauses.length} clauses.`);
  }, []);

  const answerQuestion = useCallback(async (text) => {
    setStatus("Retrieving context from the graph…");
    const { ctx, docs, spans } = await buildDossier(graphRef.current, text);
    const a = router?.assignments?.write;
    setStatus("Generating grounded answer…");
    const prompt = `${ctx}\n\n${docs}\n\n${text}`;
    const answer = await askLLM(a, GROUNDED_SYSTEM, prompt, { temperature: 0.3, numCtx: 8192, maxTokens: 1024 });

    // Dedup spans for the evidence column.
    const seen = new Set();
    const uniqueSpans = [];
    for (const s of spans) {
      const k = s.text.slice(0, 60);
      if (seen.has(k)) continue;
      seen.add(k);
      uniqueSpans.push(s);
    }

    setMessages(m => [...m, {
      id: nextId(), role: "assistant", kind: "answer",
      text: String(answer).trim() || "(empty response)",
      spans: uniqueSpans.slice(0, 12),
    }]);
    setStatus("Ready.");
  }, [router, askLLM]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || busy) return;

    let asQuestion;
    if (sendMode === "material") asQuestion = false;
    else if (sendMode === "question") asQuestion = true;
    else asQuestion = looksLikeQuestion(text, graphHasContent);

    if (asQuestion && !graphHasContent) {
      setMessages(m => [...m, { id: nextId(), role: "assistant", kind: "note",
        text: "The graph is empty — paste some material first, then ask questions about it." }]);
      return;
    }

    setMessages(m => [...m, {
      id: nextId(), role: "user", kind: asQuestion ? "question" : "material", text,
    }]);
    setInput("");
    setBusy(true);
    try {
      if (asQuestion) await answerQuestion(text);
      else await ingestMaterial(text);
    } catch (e) {
      setMessages(m => [...m, { id: nextId(), role: "assistant", kind: "error",
        text: e?.message || String(e) }]);
      setStatus("Error — see the message above.");
    } finally {
      setBusy(false);
    }
  }, [input, busy, sendMode, graphHasContent, answerQuestion, ingestMaterial]);

  const handleDeepRead = useCallback(async (msgId) => {
    const msg = messages.find(m => m.id === msgId);
    if (!msg || deepReadBusy) return;
    const a = router?.assignments?.hypothesis || router?.assignments?.read;
    if (!a) {
      setStatus("No model available for the deep read.");
      return;
    }
    setDeepReadBusy(true);
    try {
      await runSecondPass(msg.register, (sys, user) =>
        callModel(a, sys, user, { temperature: 0.4, maxTokens: 300 }), setStatus);
      // Functional reclassification of the flagged clauses — the LLM names
      // the operator that is actually functioning where surface and function
      // diverged. The result is written back into the cumulative graph.
      const changed = await reclassifyFlags(
        msg.register, msg.clauses, graphRef.current, msg.clauseBase,
        (sys, user) => callModel(a, sys, user, { temperature: 0.2, maxTokens: 200 }),
        setStatus);
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
  }, [messages, deepReadBusy, router]);

  const g = graphRef.current;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: C.bg, color: C.text }}>
      {/* Header */}
      <div style={{ padding: "12px 18px", borderBottom: `1px solid ${C.border}`, flexShrink: 0 }}>
        <div style={{ fontSize: 14, fontWeight: 700 }}>
          Chat 2.0 <span style={{ color: C.dim, fontWeight: 400, fontSize: 12 }}>· EO Classifier</span>
        </div>
        <div style={{ fontSize: 11, fontFamily: mono, color: C.dim, marginTop: 3 }}>
          {graphTick >= 0 && (
            <>{Object.keys(g.entities).length} entities · {g.claims.length} claims · {g.clauses.length} clauses · </>
          )}
          {llmReady
            ? <span style={{ color: C.green }}>model ready</span>
            : <span style={{ color: C.orange }}>no model — ingest works, answers need Ollama/in-browser</span>}
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
              This is the 2.0 chat. There are no folders or library to load — whatever you
              paste here is read straight into a knowledge graph.
            </p>
            <p style={{ marginBottom: 8 }}>
              <strong style={{ color: C.text }}>Paste material</strong> — an article, notes, a transcript —
              and it is cleaned, split into clauses, and each clause is classified against the 27 EO
              centroids (operator × terrain × stance), building up a graph of entities and claims.
            </p>
            <p>
              <strong style={{ color: C.text }}>Ask a question</strong> and it is answered, grounded,
              from everything you have pasted so far. Questions are detected automatically; use the
              selector to force a mode.
            </p>
          </div>
        )}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {messages.map(m => {
            if (m.role === "user") {
              return (
                <div key={m.id} style={{ alignSelf: "flex-end", maxWidth: "82%" }}>
                  <div style={{ fontSize: 9, fontFamily: mono, color: C.dim, textAlign: "right", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                    {m.kind === "question" ? "Question" : "Material"}
                  </div>
                  <div style={{
                    background: m.kind === "question" ? C.accent : C.s3,
                    color: m.kind === "question" ? "#fff" : C.text,
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
        <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); handleSend(); }
            }}
            placeholder="Paste material to read into the graph, or ask a question about it… (Cmd/Ctrl+Enter to send)"
            rows={3}
            style={{
              flex: 1, resize: "vertical", minHeight: 60, background: C.s1,
              border: `1px solid ${C.border}`, borderRadius: 8, color: C.text,
              fontSize: 13, lineHeight: 1.5, padding: 10, fontFamily: "inherit",
            }}
          />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <select value={sendMode} onChange={e => setSendMode(e.target.value)} style={{
              background: C.s1, border: `1px solid ${C.border}`, color: C.text,
              fontSize: 11, fontFamily: mono, padding: "5px 7px", borderRadius: 6,
            }}>
              <option value="auto">Auto-detect</option>
              <option value="material">As material</option>
              <option value="question">As question</option>
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
