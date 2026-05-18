import { useState, useMemo, useRef, useEffect } from "react";

/* Grounded-answer renderer — the model interprets (terse), the code presents
   evidence (verbatim, beside). The spans are first-class: every span is built
   by code from retrieval, never minted by the model. Ported from a UX mockup. */

const CIRCLES = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫"];
const circle = (n) => CIRCLES[n - 1] || `(${n})`;

/* Epistemic status — how much processing has been done on this span */
const STATUS = {
  walked:  { color: "#30a46c", bg: "rgba(48,164,108,.13)",  bd: "rgba(48,164,108,.40)",  label: "walked",  desc: "fully read, provenanced" },
  walking: { color: "#f59e0b", bg: "rgba(245,158,11,.13)",  bd: "rgba(245,158,11,.40)",  label: "walking", desc: "partially processed" },
  sig:     { color: "#8a8aa6", bg: "rgba(138,138,166,.12)", bd: "rgba(138,138,166,.40)", label: "sig",     desc: "scanned, impressionistic" },
};

/* Citation glyph (①②③) — inline in response text, drives hover state.
   Focusable and clickable so it works on touch and via keyboard, not just hover. */
function Cite({ n, hovered, onHover, status = "walked" }) {
  const active = hovered === n;
  const s = STATUS[status] || STATUS.walked;
  return (
    <span
      role="button"
      tabIndex={0}
      aria-label={`source ${n}`}
      onMouseEnter={() => onHover(n)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(n)}
      onBlur={() => onHover(null)}
      onClick={() => onHover(active ? null : n)}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onHover(active ? null : n); } }}
      data-cite={n}
      style={{
        display: "inline-block",
        marginLeft: 1, marginRight: 1,
        fontFamily: "var(--mono)",
        fontWeight: 600,
        fontSize: "0.95em",
        color: active ? "#0b0b0f" : s.color,
        background: active ? s.color : "transparent",
        borderRadius: 3,
        padding: active ? "0 3px" : "0",
        lineHeight: 1.4,
        cursor: "pointer",
        transition: "all .12s ease",
        verticalAlign: "baseline",
        textDecoration: active ? "none" : "underline",
        textDecorationColor: s.bd,
        textDecorationThickness: 1,
        textUnderlineOffset: 3,
        outline: "none",
      }}>
      {circle(n)}
    </span>
  );
}

/* A run of response text, optionally followed by one or more citation glyphs */
function Run({ run, hovered, onHover, statusOf }) {
  if (run.ungrounded) {
    const active = hovered === "u";
    return (
      <span
        onMouseEnter={() => onHover("u")}
        onMouseLeave={() => onHover(null)}
        title="No source — model interpolation"
        style={{
          color: active ? "#e5484d" : "var(--dim)",
          fontStyle: "italic",
          background: active ? "rgba(229,72,77,.06)" : "transparent",
          borderBottom: "1px dashed rgba(229,72,77,.35)",
          padding: "0 1px",
          transition: "color .12s ease, background .12s ease",
        }}
      >{run.text}</span>
    );
  }
  return (
    <span>
      {run.text}
      {(run.cites || []).map(n => (
        <Cite key={n} n={n} hovered={hovered} onHover={onHover} status={statusOf(n)} />
      ))}
    </span>
  );
}

/* Span card in the evidence column */
function SpanCard({ span, hovered, onHover }) {
  const active = hovered === span.index;
  const s = STATUS[span.status] || STATUS.walked;
  const isHypothesis = !!span.isHypothesis;

  return (
    <div
      onMouseEnter={() => onHover(span.index)}
      onMouseLeave={() => onHover(null)}
      data-index={span.index}
      style={{
        position: "relative",
        display: "grid",
        gridTemplateColumns: "auto 1fr",
        gap: 10,
        padding: "10px 12px 10px 14px",
        background: active ? "var(--s2)" : "var(--s1)",
        borderRadius: "0 8px 8px 0",
        borderLeft: `3px solid ${s.color}`,
        boxShadow: active ? `inset 1px 0 0 ${s.color}, 0 4px 14px rgba(0,0,0,.22)` : "none",
        transition: "all .15s ease",
        cursor: "default",
      }}>
      <div style={{
        fontFamily: "var(--mono)",
        fontSize: 18,
        lineHeight: 1.2,
        color: s.color,
        fontWeight: 600,
        flexShrink: 0,
        userSelect: "none",
      }}>{circle(span.index)}</div>

      <div style={{ minWidth: 0 }}>
        {isHypothesis ? (
          <>
            <div style={{
              fontSize: 9.5, fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: 0.6,
              color: s.color, marginBottom: 3, opacity: 0.8,
            }}>hypothesis</div>
            <div style={{
              fontFamily: "Iowan Old Style, Georgia, serif",
              fontStyle: "italic",
              fontSize: 13.5,
              lineHeight: 1.5,
              color: "#e0c878",
            }}>{span.text}</div>
          </>
        ) : (
          <blockquote style={{
            margin: 0, padding: 0, border: "none",
            fontFamily: "Iowan Old Style, Georgia, serif",
            fontStyle: "italic",
            fontSize: 13.5,
            lineHeight: 1.5,
            color: "var(--text)",
          }}>
            <span style={{ color: s.color, marginRight: 2 }}>“</span>
            {span.text}
            <span style={{ color: s.color, marginLeft: 2 }}>”</span>
          </blockquote>
        )}

        <div style={{
          display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6,
          marginTop: 7, fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--dim)",
        }}>
          {span.source && <span>{span.source}</span>}
          {span.field && <><span style={{ opacity: 0.4 }}>·</span><span>{span.field}</span></>}
          {span.passageIndex != null && <><span style={{ opacity: 0.4 }}>·</span><span>passage {span.passageIndex + 1}</span></>}
          <span style={{ flex: 1 }} />
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 4,
            padding: "1px 7px", borderRadius: 99,
            background: s.bg, color: s.color, border: `1px solid ${s.bd}`,
            fontSize: 9.5, fontWeight: 600, letterSpacing: 0.3,
            textTransform: "uppercase",
          }} title={s.desc}>
            <span style={{ width: 5, height: 5, borderRadius: 99, background: s.color }} />
            {s.label}
          </span>
        </div>
      </div>
    </div>
  );
}

/* Main component. `runs` is the model's interpretation (split into cited
   segments); `spans` is the code-built evidence. */
export default function GroundedAnswer({ runs, spans }) {
  const [hovered, setHovered] = useState(null);
  const rootRef = useRef(null);
  const [stacked, setStacked] = useState(false);

  /* Below ~720px of available width, stack the two columns. */
  useEffect(() => {
    const el = rootRef.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(([e]) => setStacked(e.contentRect.width < 720));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const safeSpans = spans || [];
  const safeRuns = runs || [];

  const statusOf = useMemo(() => {
    const map = {};
    safeSpans.forEach(s => { map[s.index] = s.status || "walked"; });
    return (n) => map[n] || "walked";
  }, [safeSpans]);

  const docs = useMemo(() => {
    const counts = {};
    safeSpans.forEach(s => {
      const k = s.source || "unknown";
      counts[k] = (counts[k] || 0) + 1;
    });
    return Object.entries(counts);
  }, [safeSpans]);

  const statusCounts = useMemo(() => {
    const c = { walked: 0, walking: 0, sig: 0 };
    safeSpans.forEach(s => { c[s.status || "walked"] = (c[s.status || "walked"] || 0) + 1; });
    return c;
  }, [safeSpans]);

  const sectionHeader = (label, extra) => (
    <div style={{
      display: "flex", alignItems: "center", gap: 7, marginBottom: 10,
      fontSize: 10, fontFamily: "var(--mono)", color: "var(--dim)",
      textTransform: "uppercase", letterSpacing: 0.6,
    }}>
      {label}
      <span style={{ color: "var(--s3)" }}>·</span>
      <span>{extra}</span>
    </div>
  );

  const responseColumn = (
    <div>
      {sectionHeader("Response", "synthesis")}
      <div style={{
        fontFamily: "Iowan Old Style, Georgia, 'Times New Roman', serif",
        fontSize: 15,
        lineHeight: 1.75,
        color: "var(--text)",
      }}>
        {safeRuns.map((r, i) => (
          <Run key={i} run={r} hovered={hovered} onHover={setHovered} statusOf={statusOf} />
        ))}
      </div>
    </div>
  );

  const evidenceColumn = (
    <div>
      {sectionHeader("Source evidence", `${safeSpans.length} span${safeSpans.length !== 1 ? "s" : ""}`)}

      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {safeSpans.map(span => (
          <SpanCard key={span.index} span={span} hovered={hovered} onHover={setHovered} />
        ))}
      </div>

      {docs.length > 0 && (
        <div style={{
          marginTop: 12, padding: "8px 12px",
          display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10,
          fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--dim)",
          background: "rgba(255,255,255,.015)",
          border: "1px dashed var(--border)", borderRadius: 6,
        }}>
          <span>
            from{" "}
            {docs.map(([d, n], i) => (
              <span key={d}>
                <span style={{ color: "var(--text)" }}>{d}</span>
                {n > 1 && <span> ({n})</span>}
                {i < docs.length - 1 ? ", " : ""}
              </span>
            ))}
          </span>
          <span style={{ flex: 1 }} />
          {Object.entries(statusCounts).filter(([, n]) => n > 0).map(([k, n]) => {
            const s = STATUS[k];
            return (
              <span key={k} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: 99, background: s.color }} />
                {n} {s.label}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );

  return (
    <div
      ref={rootRef}
      style={stacked ? undefined : {
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)",
        gap: 24,
        alignItems: "start",
      }}>
      {responseColumn}
      {stacked && <div style={{ height: 18 }} />}
      {evidenceColumn}
    </div>
  );
}
