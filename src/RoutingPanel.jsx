import { useState, useMemo, useCallback } from "react";
import {
  INTENTS, INTENT_ORDER, DEFAULT_PRIORITY,
  loadLog, loadWeights, saveWeights, runREC, resetRouter, routerStats,
} from "./router.js";

const mono = `'SF Mono','Menlo','Consolas',monospace`;
const C = {
  bg: "#0b0b0f", s1: "#131318", s2: "#1b1b22", s3: "#232330",
  border: "#282838", text: "#d4d4e4", dim: "#65657e", accent: "#6e56cf",
  green: "#30a46c", red: "#e5484d", orange: "#f76b15",
};

const Box = ({ title, sub, children }) => (
  <div style={{ background: C.s1, border: `1px solid ${C.border}`, borderRadius: 10, padding: "14px 18px", marginBottom: 10 }}>
    <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginBottom: sub ? 2 : 10 }}>{title}</div>
    {sub && <div style={{ fontSize: 11, color: C.dim, marginBottom: 10 }}>{sub}</div>}
    {children}
  </div>
);

const btn = (bg) => ({
  padding: "7px 14px", fontSize: 11, fontWeight: 600, borderRadius: 7, border: "none",
  cursor: "pointer", background: bg, color: bg === C.s2 ? C.dim : "#fff",
});

const moveBtn = (disabled) => ({
  fontSize: 10, fontFamily: mono, fontWeight: 600, padding: "4px 9px", borderRadius: 5,
  border: "none", cursor: disabled ? "default" : "pointer", background: C.s2, color: C.dim,
  opacity: disabled ? 0.4 : 1,
});

const formatTime = (ts) => {
  if (!ts) return "";
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(ts);
};

// Routing section of the Optimize tab. Self-contained: reads/writes the
// router's localStorage state directly.
export default function RoutingPanel({ installed }) {
  const [weights, setWeights] = useState(loadWeights);
  const [log, setLog] = useState(loadLog);
  const [showLog, setShowLog] = useState(false);

  const refresh = useCallback(() => {
    setWeights(loadWeights());
    setLog(loadLog());
  }, []);

  const effectiveWeights = useMemo(() => {
    const w = {};
    for (const it of INTENT_ORDER) w[it] = (weights && weights[it]) || DEFAULT_PRIORITY[it];
    return w;
  }, [weights]);

  const stats = useMemo(() => routerStats(log), [log]);
  const conf = stats.confidence;
  const confTotal = conf.high + conf.low + conf.zero;
  const recentLog = useMemo(() => [...log].slice(-20).reverse(), [log]);
  const maxIntent = Math.max(1, ...INTENT_ORDER.map(i => stats.byIntent[i]));

  // Manual reorder is the strongest REC signal — it writes weights directly.
  const reorderWeight = (intent, idx, dir) => {
    const j = idx + dir;
    const current = effectiveWeights[intent];
    if (j < 0 || j >= current.length) return;
    const next = [...current];
    [next[idx], next[j]] = [next[j], next[idx]];
    const updated = { ...(weights || {}) };
    for (const it of INTENT_ORDER) updated[it] = updated[it] || DEFAULT_PRIORITY[it];
    updated[intent] = next;
    updated.lastUpdated = new Date().toISOString();
    updated.totalEvaluations = (weights && weights.totalEvaluations) || 0;
    saveWeights(updated);
    refresh();
  };

  const triggerREC = () => { runREC(installed); refresh(); };

  const resetRouterState = () => {
    if (!window.confirm("Reset routing? This clears the routing log and learned weights, reverting to defaults.")) return;
    resetRouter();
    refresh();
  };

  return (
    <>
      <Box title="Auto-Routing" sub="With the chat model selector set to ⚡ Auto, each prompt is classified by intent and routed to the best installed model. Feedback tunes these rankings over time.">
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button onClick={triggerREC} style={btn(C.accent)}>↬ Recalculate now</button>
          <button onClick={resetRouterState} style={btn(C.red)}>Reset to defaults</button>
        </div>
        {weights?.lastUpdated && (
          <div style={{ fontSize: 10, fontFamily: mono, color: C.dim, marginTop: 8 }}>
            weights updated {formatTime(new Date(weights.lastUpdated).getTime())} · {weights.totalEvaluations || 0} evaluations applied
          </div>
        )}
      </Box>

      <Box title="Routing stats">
        <div style={{ fontFamily: mono, fontSize: 12, color: C.dim }}>
          <div style={{ marginBottom: 8 }}>
            <span style={{ color: C.text }}>{stats.total}</span> routed messages ·{" "}
            <span style={{ color: C.text }}>{stats.satisfaction == null ? "—" : `${Math.round(stats.satisfaction * 100)}%`}</span> satisfaction
            {stats.evaluated > 0 ? ` (${stats.evaluated} evaluated)` : ""}
          </div>
          {INTENT_ORDER.map(i => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "2px 0" }}>
              <span style={{ width: 96, color: INTENTS[i].color }}>{INTENTS[i].icon} {INTENTS[i].label}</span>
              <div style={{ flex: 1, height: 8, background: C.s3, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${stats.byIntent[i] / maxIntent * 100}%`, background: INTENTS[i].color }} />
              </div>
              <span style={{ width: 30, textAlign: "right", color: C.text }}>{stats.byIntent[i]}</span>
            </div>
          ))}
        </div>
      </Box>

      <Box title="Classifier confidence" sub="How decisive intent detection has been">
        {confTotal === 0 ? (
          <div style={{ fontSize: 12, color: C.dim }}>No routing decisions logged yet.</div>
        ) : (<>
          <div style={{ fontFamily: mono, fontSize: 12, color: C.dim, lineHeight: 1.9 }}>
            <div><span style={{ color: C.green }}>● high</span> (3+ matches): {conf.high} · {Math.round(conf.high / confTotal * 100)}%</div>
            <div><span style={{ color: C.orange }}>○ low</span> (1-2 matches): {conf.low} · {Math.round(conf.low / confTotal * 100)}%</div>
            <div><span style={{ color: C.dim }}>○ none</span> (fallback): {conf.zero} · {Math.round(conf.zero / confTotal * 100)}%</div>
          </div>
          {(conf.low + conf.zero) > conf.high && confTotal >= 5 && (
            <div style={{ marginTop: 8, fontSize: 11, color: C.orange }}>
              Low confidence is frequent — installing more specialized models (e.g. a dedicated coder or reasoning model) would let the router make sharper choices.
            </div>
          )}
        </>)}
      </Box>

      <Box title="Learned weights" sub="Routing order per intent — the first installed match wins. Reorder to override.">
        {INTENT_ORDER.map(intent => (
          <div key={intent} style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: INTENTS[intent].color, marginBottom: 4 }}>
              {INTENTS[intent].icon} {INTENTS[intent].label}
            </div>
            {effectiveWeights[intent].map((m, idx) => {
              const inst = installed.some(i => i.name.toLowerCase().includes(m.toLowerCase()));
              return (
                <div key={m + idx} style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 0" }}>
                  <span style={{ width: 18, fontSize: 10, fontFamily: mono, color: C.dim }}>{idx + 1}.</span>
                  <span style={{ flex: 1, fontSize: 12, fontFamily: mono, color: inst ? C.text : C.dim }}>
                    {m}{inst ? "" : " · not installed"}
                  </span>
                  <button onClick={() => reorderWeight(intent, idx, -1)} disabled={idx === 0} style={moveBtn(idx === 0)}>↑</button>
                  <button onClick={() => reorderWeight(intent, idx, 1)} disabled={idx === effectiveWeights[intent].length - 1} style={moveBtn(idx === effectiveWeights[intent].length - 1)}>↓</button>
                </div>
              );
            })}
          </div>
        ))}
      </Box>

      <Box title="Routing log">
        <button onClick={() => setShowLog(s => !s)} style={btn(C.s2)}>{showLog ? "Hide" : "Show"} recent decisions ({log.length})</button>
        {showLog && (
          <div style={{ marginTop: 10 }}>
            {recentLog.length === 0 ? (
              <div style={{ fontSize: 12, color: C.dim }}>No decisions logged yet.</div>
            ) : recentLog.map(e => {
              const net = (e.signals || []).reduce((a, s) => a + (s.value || 0), 0);
              return (
                <div key={e.id} style={{ fontFamily: mono, fontSize: 10, color: C.dim, padding: "4px 0", borderBottom: `1px solid ${C.s3}` }}>
                  <span style={{ color: INTENTS[e.intent]?.color || C.dim }}>{INTENTS[e.intent]?.icon} {e.intent}</span>
                  {" → "}<span style={{ color: C.text }}>{e.modelChosen}</span>
                  {" · conf "}{e.confidence}
                  {" · "}{e.evaluated
                    ? <span style={{ color: net >= 0 ? C.green : C.red }}>{net >= 0 ? "+" : ""}{net.toFixed(1)}</span>
                    : "pending"}
                  {e.alternateModel ? <span style={{ color: C.orange }}> · picked {e.alternateModel}</span> : ""}
                  {e.failures?.length ? <span style={{ color: C.red }}> · failed: {e.failures.join(", ")}</span> : ""}
                </div>
              );
            })}
          </div>
        )}
      </Box>
    </>
  );
}
