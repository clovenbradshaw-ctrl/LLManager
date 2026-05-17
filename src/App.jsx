import { useState, useEffect, useCallback, useRef } from "react";
import Chat from "./Chat.jsx";
import RoutingPanel from "./RoutingPanel.jsx";
import MatrixGate from "./MatrixGate.jsx";
import MatrixChat from "./MatrixChat.jsx";
import { loadLog, runREC } from "./router.js";
import { loadSession } from "./matrix.js";

const GATE_SKIP_KEY = "llm-manager-gate-skipped";

const MODEL_CATALOG = [
  { id: "gemma2:2b", params: "2B", vram: 1.5, speed: "fast", use: "Light tasks, quick responses" },
  { id: "llama3.2:3b", params: "3B", vram: 2.2, speed: "fast", use: "General chat, summarization" },
  { id: "phi3:mini", params: "3.8B", vram: 2.5, speed: "fast", use: "Reasoning, code, math" },
  { id: "mistral", params: "7B", vram: 4.4, speed: "medium", use: "General purpose, tool use" },
  { id: "llama3.1:8b", params: "8B", vram: 5.0, speed: "medium", use: "Strong all-rounder" },
  { id: "gemma2:9b", params: "9B", vram: 5.5, speed: "medium", use: "Quality reasoning" },
  { id: "qwen2.5:7b", params: "7B", vram: 4.7, speed: "medium", use: "Code, math, multilingual" },
  { id: "deepseek-r1:8b", params: "8B", vram: 5.0, speed: "medium", use: "Chain-of-thought reasoning" },
  { id: "qwen2.5:14b", params: "14B", vram: 9.0, speed: "slow", use: "High quality code & writing" },
  { id: "phi3:medium", params: "14B", vram: 8.5, speed: "slow", use: "Strong reasoning & analysis" },
  { id: "codellama:13b", params: "13B", vram: 7.9, speed: "slow", use: "Code generation" },
  { id: "mixtral:8x7b", params: "47B MoE", vram: 15, speed: "slow", use: "MoE — fast for its quality" },
  { id: "nomic-embed-text", params: "137M", vram: 0.3, speed: "instant", use: "Embeddings for RAG" },
  { id: "llava", params: "7B", vram: 5.0, speed: "medium", use: "Image + text understanding" },
  { id: "starcoder2:7b", params: "7B", vram: 4.5, speed: "medium", use: "Code completion" },
];

const QUANT_LEVELS = [
  { q: "Q2_K", bpw: 2.6, note: "Heavy quality loss — only worth it to squeeze a 70B in", tone: "bad" },
  { q: "Q3_K_M", bpw: 3.4, note: "Noticeable loss — last resort on tight memory", tone: "warn" },
  { q: "Q4_0", bpw: 4.5, note: "Legacy 4-bit — fine, slightly behind K-quants", tone: "ok" },
  { q: "Q4_K_M", bpw: 4.8, note: "Sweet spot — best speed/quality balance", tone: "best" },
  { q: "Q5_K_M", bpw: 5.7, note: "Higher quality, a little slower & larger", tone: "best" },
  { q: "Q6_K", bpw: 6.6, note: "Near-lossless — diminishing returns", tone: "ok" },
  { q: "Q8_0", bpw: 8.5, note: "Minimal loss but ~2x the memory bandwidth cost", tone: "warn" },
  { q: "F16", bpw: 16, note: "Full precision — rarely worth it for local inference", tone: "warn" },
];

const SPEED_TIERS = [
  { tier: "gemma2:2b", tps: "100+ tok/s", note: "Fastest — light tasks, quick replies" },
  { tier: "llama3.2:3b · phi3:mini", tps: "60–80 tok/s", note: "Noticeably smarter, still snappy" },
  { tier: "mistral · llama3.1:8b", tps: "30–40 tok/s", note: "Strong quality, comfortable" },
  { tier: "qwen2.5:14b", tps: "15–20 tok/s", note: "Best quality you can run comfortably" },
];

const mono = `'SF Mono','Menlo','Consolas',monospace`;
const sans = `-apple-system,system-ui,sans-serif`;
const C = {
  bg: "#0b0b0f", s1: "#131318", s2: "#1b1b22", s3: "#232330",
  border: "#282838", text: "#d4d4e4", dim: "#65657e", accent: "#6e56cf",
  green: "#30a46c", red: "#e5484d", orange: "#f76b15",
};

const Pill = ({ color, children }) => (
  <span style={{ fontSize: 10, fontFamily: mono, padding: "2px 8px", borderRadius: 99, background: color + "22", color, fontWeight: 600 }}>{children}</span>
);

const CopyBlock = ({ text, id, label, copy, copied }) => (
  <div style={{ marginBottom: 8 }}>
    {label && <div style={{ fontSize: 10, color: C.dim, marginBottom: 3 }}>{label}</div>}
    <div style={{ display: "flex", gap: 6 }}>
      <code style={{ flex: 1, fontSize: 11, fontFamily: mono, background: C.bg, padding: "8px 12px", borderRadius: 6, color: C.green, border: `1px solid ${C.border}`, wordBreak: "break-all", whiteSpace: "pre-wrap" }}>{text}</code>
      <button onClick={() => copy(text, id)} style={{ padding: "6px 12px", fontSize: 10, fontFamily: mono, fontWeight: 600, borderRadius: 6, border: "none", cursor: "pointer", background: copied === id ? C.green : C.accent, color: copied === id ? "#000" : "#fff", whiteSpace: "nowrap" }}>{copied === id ? "✓" : "copy"}</button>
    </div>
  </div>
);

const ActBtn = ({ onClick, disabled, color, children }) => (
  <button onClick={onClick} disabled={disabled} style={{
    fontSize: 10, fontFamily: mono, fontWeight: 600, padding: "4px 10px", borderRadius: 5,
    border: "none", cursor: disabled ? "default" : "pointer", whiteSpace: "nowrap",
    background: color || C.s2, color: color ? "#fff" : C.dim, opacity: disabled ? 0.5 : 1,
  }}>{children}</button>
);

const Box = ({ title, sub, children }) => (
  <div style={{ background: C.s1, border: `1px solid ${C.border}`, borderRadius: 10, padding: "14px 18px", marginBottom: 10 }}>
    <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginBottom: sub ? 2 : 10 }}>{title}</div>
    {sub && <div style={{ fontSize: 11, color: C.dim, marginBottom: 10 }}>{sub}</div>}
    {children}
  </div>
);

export default function App() {
  const [tab, setTab] = useState("chat");
  const [settingsSection, setSettingsSection] = useState("connection");
  const [matrixSession, setMatrixSession] = useState(loadSession);
  const [gateDone, setGateDone] = useState(
    () => !!loadSession() || localStorage.getItem(GATE_SKIP_KEY) === "1",
  );
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");
  const [ollamaUp, setOllamaUp] = useState(null);
  const [ollamaVer, setOllamaVer] = useState("");
  const [installed, setInstalled] = useState([]);
  const [running, setRunning] = useState([]);
  const [hw, setHw] = useState(null);

  const [model, setModel] = useState("");
  const [keepAlivePref, setKeepAlivePref] = useState("10m");
  const [copied, setCopied] = useState(null);
  const [pulling, setPulling] = useState({}); // { [name]: { status, completed, total, error } }
  const [busy, setBusy] = useState({});       // { [name]: "load" | "unload" | "delete" }

  // ── Hardware ──
  useEffect(() => {
    const d = {
      cores: navigator.hardwareConcurrency || "?",
      ram: navigator.deviceMemory ? `${navigator.deviceMemory} GB (browser estimate)` : "not exposed",
      rawRam: navigator.deviceMemory || null,
    };
    const ua = navigator.userAgent;
    d.os = ua.includes("Mac") ? "macOS" : ua.includes("Linux") ? "Linux" : ua.includes("Win") ? "Windows" : navigator.platform;
    d.arch = ua.includes("Mac") ? (ua.includes("Intel") ? "Intel x86_64" : "Apple Silicon") : "unknown";
    d.gpu = "—";
    if (navigator.gpu) {
      navigator.gpu.requestAdapter().then(a => {
        if (a) a.requestAdapterInfo().then(i => {
          d.gpu = [i.vendor, i.architecture, i.description].filter(Boolean).join(" ") || "WebGPU available";
          setHw({ ...d });
        }).catch(() => setHw({ ...d }));
        else setHw({ ...d });
      }).catch(() => setHw({ ...d }));
    }
    setHw(d);
  }, []);

  // ── Probe ──
  const probe = useCallback(async () => {
    setOllamaUp(null);
    try {
      const vR = await fetch(`${ollamaUrl}/api/version`, { signal: AbortSignal.timeout(3000) });
      if (!vR.ok) { setOllamaUp(false); return; }
      setOllamaVer((await vR.json()).version || "?");
      setOllamaUp(true);
      const tR = await fetch(`${ollamaUrl}/api/tags`, { signal: AbortSignal.timeout(5000) });
      if (tR.ok) {
        const t = await tR.json();
        setInstalled(t.models || []);
        if (!model && t.models?.length) setModel(t.models[0].name);
      }
      const pR = await fetch(`${ollamaUrl}/api/ps`, { signal: AbortSignal.timeout(3000) });
      if (pR.ok) setRunning((await pR.json()).models || []);
    } catch {
      // A normal fetch fails both when the server is down AND when it's up
      // but rejecting this page's origin (CORS). A no-cors request still
      // resolves (as an opaque response) if something is actually listening,
      // so it tells the two cases apart.
      try {
        await fetch(`${ollamaUrl}/api/version`, { mode: "no-cors", signal: AbortSignal.timeout(3000) });
        setOllamaUp("cors");
      } catch {
        setOllamaUp(false);
      }
    }
  }, [ollamaUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { probe(); }, [probe]);

  // REC on app load: refresh routing weights once installed models are known
  // (prunes uninstalled models and log entries older than 30 days).
  const recRan = useRef(false);
  useEffect(() => {
    if (recRan.current || !installed.length) return;
    recRan.current = true;
    if (loadLog().length) runREC(installed);
  }, [installed]);

  // ── Model management ──
  const pullModel = async (name) => {
    if (pulling[name] && !pulling[name].error) return;
    setPulling(p => ({ ...p, [name]: { status: "starting…" } }));
    try {
      const res = await fetch(`${ollamaUrl}/api/pull`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
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
          setPulling(p => ({ ...p, [name]: { status: j.status, completed: j.completed, total: j.total } }));
        }
      }
      setPulling(p => { const n = { ...p }; delete n[name]; return n; });
      probe();
    } catch (e) {
      setPulling(p => ({ ...p, [name]: { status: `error: ${e.message}`, error: true } }));
      setTimeout(() => setPulling(p => { const n = { ...p }; delete n[name]; return n; }), 5000);
    }
  };

  const setKeepAlive = async (name, keep_alive, action) => {
    setBusy(b => ({ ...b, [name]: action }));
    try {
      await fetch(`${ollamaUrl}/api/generate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: name, keep_alive }),
      });
    } catch { /* probe below reflects real state */ }
    setBusy(b => { const n = { ...b }; delete n[name]; return n; });
    probe();
  };
  const loadModel = (name) => setKeepAlive(name, keepAlivePref === "-1" ? -1 : keepAlivePref, "load");
  const unloadModel = (name) => setKeepAlive(name, 0, "unload");

  const deleteModel = async (name) => {
    if (!window.confirm(`Delete ${name}? This removes the model files from disk.`)) return;
    setBusy(b => ({ ...b, [name]: "delete" }));
    try {
      await fetch(`${ollamaUrl}/api/delete`, {
        method: "DELETE", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
    } catch { /* probe below reflects real state */ }
    setBusy(b => { const n = { ...b }; delete n[name]; return n; });
    probe();
  };

  // ── Helpers ──
  const copy = (t, id) => { navigator.clipboard.writeText(t); setCopied(id); setTimeout(() => setCopied(null), 1500); };
  const fmtB = b => !b ? "—" : (b / (1024 ** 3)) >= 1 ? `${(b / (1024 ** 3)).toFixed(1)} GB` : `${(b / (1024 ** 2)).toFixed(0)} MB`;
  const ramGB = navigator.deviceMemory || null;
  const getRec = m => {
    if (!ramGB) return null;
    const a = ramGB * 0.75;
    if (m.vram <= a * 0.5) return { t: "easy", c: C.green };
    if (m.vram <= a) return { t: "fits", c: C.accent };
    if (m.vram <= ramGB) return { t: "tight", c: C.orange };
    return { t: "too big", c: C.red };
  };

  // ── Snippet generator for other apps ──
  const snippet = (m) => `// Drop this into any browser app to call ${m}
async function askLLM(prompt, options = {}) {
  const res = await fetch("${ollamaUrl}/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "${m}",
      messages: [
        ...(options.system ? [{ role: "system", content: options.system }] : []),
        { role: "user", content: prompt }
      ],
      temperature: options.temperature ?? 0.7,
      max_tokens: options.max_tokens ?? 2048,
      stream: false,
    }),
  });
  const data = await res.json();
  return data.choices[0].message.content;
}

// Usage:
// const answer = await askLLM("summarize this text...");
// const code = await askLLM("write a sort function", { system: "You are a senior engineer.", temperature: 0.2 });`;

  const streamSnippet = (m) => `// Streaming version — prints tokens as they arrive
async function askLLMStream(prompt, onToken) {
  const res = await fetch("${ollamaUrl}/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "${m}", prompt, stream: true }),
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const lines = decoder.decode(value).split("\\n").filter(Boolean);
    for (const line of lines) {
      try {
        const j = JSON.parse(line);
        if (j.response) { full += j.response; onToken(j.response, full); }
      } catch {}
    }
  }
  return full;
}

// Usage:
// await askLLMStream("explain monads", (token, soFar) => {
//   document.getElementById("output").textContent = soFar;
// });`;

  const pageOrigin = typeof window !== "undefined" ? window.location.origin : "";

  // OLLAMA_ORIGINS setup for the official desktop app, which ignores your
  // shell environment — the variable has to be set at the OS level.
  const originFix = {
    macOS: `launchctl setenv OLLAMA_ORIGINS "*"
# then fully quit Ollama (menu-bar icon → Quit) and reopen it`,
    Windows: `setx OLLAMA_ORIGINS "*"
:: then quit Ollama from the system tray and reopen it`,
    Linux: `systemctl edit ollama.service
# add under [Service]:   Environment="OLLAMA_ORIGINS=*"
sudo systemctl daemon-reload && sudo systemctl restart ollama`,
  };
  // Detected OS first, then the rest.
  const originFixOrder = ["macOS", "Windows", "Linux"]
    .sort((a, b) => (a === hw?.os ? -1 : b === hw?.os ? 1 : 0));

  const corsNote = `# Terminal users: start Ollama with the origins allowed
OLLAMA_ORIGINS="*" ollama serve

# Or restrict to specific origins:
OLLAMA_ORIGINS="${pageOrigin || "https://myapp.com"},http://localhost:3000" ollama serve`;

  if (!gateDone) {
    return (
      <MatrixGate
        onLogin={s => { setMatrixSession(s); setGateDone(true); }}
        onSkip={() => { localStorage.setItem(GATE_SKIP_KEY, "1"); setGateDone(true); }}
      />
    );
  }

  return (
    <div style={{ fontFamily: sans, background: C.bg, color: C.text, height: "100vh", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "16px 20px 12px", borderBottom: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10, flexShrink: 0 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700 }}><span style={{ color: C.accent }}>◆</span> LLM Manager</div>
          <div style={{ fontSize: 11, fontFamily: mono, color: C.dim, marginTop: 2 }}>
            {ollamaUp === true ? "🟢" : ollamaUp === "cors" ? "🟠" : ollamaUp === false ? "🔴" : "⏳"}
            {" Ollama "}
            {ollamaUp === true ? `v${ollamaVer}` : ollamaUp === "cors" ? "blocked" : "offline"} · {installed.length} models
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {["chat", "matrix", "settings"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "7px 16px", fontSize: 12, fontWeight: 600, borderRadius: 8, cursor: "pointer", textTransform: "capitalize",
              border: `1px solid ${tab === t ? C.accent : C.border}`, background: tab === t ? C.accent : "transparent", color: tab === t ? "#fff" : C.dim,
            }}>{t}</button>
          ))}
        </div>
      </div>

      {tab === "chat" ? (
        <div style={{ flex: 1, minHeight: 0 }}>
          <Chat ollamaUrl={ollamaUrl} installed={installed} ollamaUp={ollamaUp} />
        </div>
      ) : tab === "matrix" ? (
        <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
          <MatrixChat
            session={matrixSession}
            onLogin={setMatrixSession}
            onLogout={() => setMatrixSession(null)}
            ollamaUrl={ollamaUrl}
            ollamaUp={ollamaUp}
            model={model}
            models={installed}
          />
        </div>
      ) : (
      <div style={{ flex: 1, overflowY: "auto" }}>
      <div style={{ padding: "16px 20px", maxWidth: 820, margin: "0 auto" }}>

        {/* ═══ SETTINGS — sub-nav ═══ */}
        <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
          {[["connection", "Connection"], ["models", "Models"], ["optimize", "Optimize"], ["connect", "Connect"]].map(([id, label]) => (
            <button key={id} onClick={() => setSettingsSection(id)} style={{
              padding: "6px 14px", fontSize: 11.5, fontWeight: 600, borderRadius: 7, cursor: "pointer",
              border: `1px solid ${settingsSection === id ? C.accent : C.border}`,
              background: settingsSection === id ? "rgba(110,86,207,.18)" : "transparent",
              color: settingsSection === id ? C.text : C.dim,
            }}>{label}</button>
          ))}
        </div>

        {/* ═══ CONNECTION ═══ */}
        {settingsSection === "connection" && (<>
          <Box title="Hardware" sub="Browser-reported — approximate for some values">
            {hw && (
              <div style={{ fontFamily: mono, fontSize: 12, lineHeight: 2, color: C.dim }}>
                {[["OS", hw.os], ["Arch", hw.arch], ["Cores", hw.cores], ["RAM", hw.ram], ["GPU", hw.gpu]].map(([k, v]) => (
                  <div key={k}><span style={{ color: C.text, display: "inline-block", width: 70 }}>{k}</span>{v}</div>
                ))}
              </div>
            )}
            <div style={{ marginTop: 10 }}>
              <CopyBlock copy={copy} copied={copied} id="hw" text='system_profiler SPHardwareDataType | grep -E "Chip|Memory|Cores|Model"' label="Get exact specs in terminal" />
            </div>
          </Box>

          <Box title="Ollama">
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
              <input value={ollamaUrl} onChange={e => setOllamaUrl(e.target.value)} style={{ flex: 1, padding: "8px 12px", fontSize: 12, fontFamily: mono, background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 6 }} />
              <button onClick={probe} style={{ padding: "7px 14px", fontSize: 11, fontWeight: 600, background: C.s2, color: C.dim, border: `1px solid ${C.border}`, borderRadius: 6, cursor: "pointer" }}>Probe</button>
            </div>
            {ollamaUp === false && (
              <div style={{ background: C.red + "12", border: `1px solid ${C.red}30`, borderRadius: 8, padding: "12px 16px" }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: C.red, marginBottom: 8 }}>Not reachable</div>
                <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 10 }}>
                  Install the <strong style={{ color: C.text }}>official Ollama app</strong> from{" "}
                  <a href="https://ollama.com/download" target="_blank" rel="noopener noreferrer" style={{ color: C.accent }}>ollama.com/download</a>.
                  It runs the server as a background service automatically — a menu-bar icon on
                  macOS, a system-tray icon on Windows — so you don't need to keep a terminal open.
                </div>
                <CopyBlock copy={copy} copied={copied} text="ollama pull gemma2:2b" id="fp" label="Then pull your first model" />
                <div style={{ fontSize: 11, color: C.dim, marginTop: 8, lineHeight: 1.6 }}>
                  Prefer the terminal? Run <code style={{ fontFamily: mono, color: C.accent }}>OLLAMA_ORIGINS="*" ollama serve</code> —
                  the origins flag lets this page reach it.
                </div>
              </div>
            )}
            {ollamaUp === "cors" && (
              <div style={{ background: C.orange + "12", border: `1px solid ${C.orange}40`, borderRadius: 8, padding: "12px 16px" }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: C.orange, marginBottom: 8 }}>Running — but blocking this page</div>
                <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 12 }}>
                  Ollama is up at <code style={{ fontFamily: mono, color: C.text }}>{ollamaUrl}</code>, but it's
                  refusing requests from this page's origin
                  (<code style={{ fontFamily: mono, color: C.text }}>{pageOrigin}</code>).
                  The official desktop app ignores your shell environment, so set{" "}
                  <code style={{ fontFamily: mono, color: C.accent }}>OLLAMA_ORIGINS</code> at the OS level,
                  then fully quit and reopen Ollama.
                </div>
                {originFixOrder.map(os => (
                  <CopyBlock key={os} copy={copy} copied={copied} id={`fix-${os}`} text={originFix[os]}
                    label={os === hw?.os ? `${os} (your system)` : os} />
                ))}
                <div style={{ fontSize: 11, color: C.dim, marginTop: 8, lineHeight: 1.6 }}>
                  Running both the Ollama app and <code style={{ fontFamily: mono, color: C.accent }}>ollama serve</code> at
                  once? Only one process can own port 11434 — the app takes it, and the terminal
                  server (with your origins flag) never sees the request. Either configure the app
                  above, or quit the app and use the terminal server.
                </div>
                <div style={{ marginTop: 10 }}>
                  <button onClick={probe} style={{ padding: "6px 14px", fontSize: 11, fontWeight: 600, background: C.s2, color: C.text, border: `1px solid ${C.border}`, borderRadius: 6, cursor: "pointer" }}>
                    Re-check connection
                  </button>
                </div>
              </div>
            )}
            {ollamaUp === true && (<>
              {running.length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: C.dim, marginBottom: 4 }}>Loaded in memory:</div>
                  {running.map(m => (
                    <div key={m.name} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", fontSize: 12, fontFamily: mono, borderBottom: `1px solid ${C.s3}` }}>
                      <span style={{ color: C.green }}>{m.name}</span>
                      <span style={{ color: C.dim }}>{fmtB(m.size)}</span>
                    </div>
                  ))}
                </div>
              )}
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: 11, color: C.dim, fontFamily: mono }}>Keep loaded for</span>
                <select value={keepAlivePref} onChange={e => setKeepAlivePref(e.target.value)} style={{ padding: "5px 10px", fontSize: 11, fontFamily: mono, background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 6 }}>
                  <option value="5m">5m — Ollama default</option>
                  <option value="10m">10m</option>
                  <option value="1h">1h</option>
                  <option value="24h">24h — stays warm all day</option>
                  <option value="-1">forever (until unload)</option>
                </select>
                <span style={{ fontSize: 10, color: C.dim }}>longer = no cold-load delay</span>
              </div>
              <div style={{ fontSize: 11, color: C.dim, marginBottom: 4 }}>Installed ({installed.length}):</div>
              {installed.map(m => {
                const loaded = running.some(r => r.name === m.name);
                const act = busy[m.name];
                return (
                  <div key={m.name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, padding: "6px 0", fontSize: 12, borderBottom: `1px solid ${C.s3}` }}>
                    <span style={{ fontFamily: mono, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.name} <span style={{ color: C.dim, fontSize: 10 }}>· {fmtB(m.size)} · {m.details?.quantization_level || ""}</span></span>
                    <div style={{ display: "flex", gap: 6, alignItems: "center", flexShrink: 0 }}>
                      {loaded && <Pill color={C.green}>LOADED</Pill>}
                      {loaded
                        ? <ActBtn onClick={() => unloadModel(m.name)} disabled={!!act}>{act === "unload" ? "unloading…" : "unload"}</ActBtn>
                        : <ActBtn onClick={() => loadModel(m.name)} disabled={!!act}>{act === "load" ? "loading…" : "load"}</ActBtn>}
                      <ActBtn onClick={() => deleteModel(m.name)} disabled={!!act} color={C.red}>{act === "delete" ? "deleting…" : "delete"}</ActBtn>
                    </div>
                  </div>
                );
              })}
            </>)}
          </Box>
        </>)}

        {/* ═══ MODELS ═══ */}
        {settingsSection === "models" && (<>
          <Box title="Model Catalog" sub={ramGB ? `~${ramGB} GB detected → ~${(ramGB * .75).toFixed(0)} GB usable for models` : "RAM not exposed by browser — check terminal"}>
            {!ramGB && <div style={{ marginBottom: 12 }}><CopyBlock copy={copy} copied={copied} id="ram" text='sysctl -n hw.memsize | awk "{print $1/1073741824\" GB\"}"' label="Check actual RAM" /></div>}
            {MODEL_CATALOG.map(m => {
              const rec = getRec(m);
              const inst = installed.some(i => i.name.startsWith(m.id.split(":")[0]));
              return (
                <div key={m.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 0", borderBottom: `1px solid ${C.s3}` }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                      <span style={{ fontSize: 13, fontFamily: mono, fontWeight: 600 }}>{m.id}</span>
                      {inst && <Pill color={C.green}>INSTALLED</Pill>}
                      {rec && <Pill color={rec.c}>{rec.t}</Pill>}
                    </div>
                    <div style={{ fontSize: 11, color: C.dim, marginTop: 2 }}>{m.params} · ~{m.vram} GB · {m.speed} · {m.use}</div>
                  </div>
                  {(() => {
                    const pp = pulling[m.id];
                    if (pp) {
                      const pct = pp.total ? Math.round((pp.completed || 0) / pp.total * 100) : null;
                      return (
                        <div style={{ width: 170, flexShrink: 0 }}>
                          <div style={{ fontSize: 9, fontFamily: mono, color: pp.error ? C.red : C.dim, marginBottom: 3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                            {pp.status}{pct != null ? ` · ${pct}%` : ""}
                          </div>
                          <div style={{ height: 5, background: C.s3, borderRadius: 3, overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${pct || 0}%`, background: pp.error ? C.red : C.accent, transition: "width .2s" }} />
                          </div>
                        </div>
                      );
                    }
                    return (
                      <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                        <ActBtn onClick={() => copy(`ollama pull ${m.id}`, `p-${m.id}`)}>{copied === `p-${m.id}` ? "✓" : "copy"}</ActBtn>
                        <ActBtn onClick={() => pullModel(m.id)} color={C.accent}>{inst ? "re-pull" : "pull"}</ActBtn>
                      </div>
                    );
                  })()}
                </div>
              );
            })}
          </Box>
        </>)}

        {/* ═══ OPTIMIZE ═══ */}
        {settingsSection === "optimize" && (() => {
          const tone = { best: C.green, ok: C.accent, warn: C.orange, bad: C.red };
          const serverEnv = `# Keep models warm across requests (default 5m)
launchctl setenv OLLAMA_KEEP_ALIVE 24h

# Flash attention — faster, smaller KV cache
launchctl setenv OLLAMA_FLASH_ATTENTION 1

# Quantize the KV cache to halve its memory footprint
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0

# Then restart the server for the env vars to take effect
ollama serve`;
          return (<>
            <RoutingPanel installed={installed} />

            <Box title="Quantization" sub="The biggest speed lever. Fewer bits per weight = less memory to move per token = faster.">
              <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 12 }}>
                A 7B model is ~14&nbsp;GB at F16 but only ~4.4&nbsp;GB at Q4_K_M — and runs 3–4x faster.
                Quality loss is minimal down to Q4. Below that it degrades fast.
              </div>
              {QUANT_LEVELS.map(l => (
                <div key={l.q} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 0", borderBottom: `1px solid ${C.s3}` }}>
                  <span style={{ fontFamily: mono, fontSize: 12, fontWeight: 600, width: 64, flexShrink: 0, color: tone[l.tone] }}>{l.q}</span>
                  <span style={{ fontFamily: mono, fontSize: 10, color: C.dim, width: 64, flexShrink: 0 }}>{l.bpw} bpw</span>
                  <span style={{ fontSize: 11, color: C.dim }}>{l.note}</span>
                </div>
              ))}
            </Box>

            <Box title="Speed on Apple Silicon" sub="Approximate warm throughput — first token is slower if the model cold-loads.">
              {SPEED_TIERS.map(s => (
                <div key={s.tier} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 0", borderBottom: `1px solid ${C.s3}` }}>
                  <span style={{ fontFamily: mono, fontSize: 11, fontWeight: 600, width: 180, flexShrink: 0 }}>{s.tier}</span>
                  <Pill color={C.accent}>{s.tps}</Pill>
                  <span style={{ fontSize: 11, color: C.dim }}>{s.note}</span>
                </div>
              ))}
              <div style={{ fontSize: 11, color: C.dim, lineHeight: 1.6, marginTop: 12 }}>
                The hard ceiling is <strong style={{ color: C.text }}>memory bandwidth</strong> (~100&nbsp;GB/s on an M3).
                Every token reads the whole model from memory, so token speed scales with model size.
                Quantization helps by shrinking how much data moves per token — everything else is marginal.
              </div>
            </Box>

            <Box title="Keep models warm" sub="Cold-loading a model into GPU memory can take 30s+. Keep it resident between requests.">
              <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 10 }}>
                Use the keep-alive selector on the <strong style={{ color: C.text }}>Connection</strong> section to load a model
                and pin it in memory. To make Ollama keep every model warm by default, set the env var below.
              </div>
              <CopyBlock copy={copy} copied={copied} id="opt-env" text={serverEnv} label="Server tuning — paste into terminal" />
            </Box>

            <Box title="Context window" sub="Ollama defaults num_ctx to 4096. Smaller windows shrink the KV cache and speed up the first token.">
              <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6 }}>
                Every request can pass <code style={{ fontFamily: mono, color: C.accent }}>options.num_ctx</code> to size
                the context window; set a server-wide default with the
                <code style={{ fontFamily: mono, color: C.accent }}> OLLAMA_CONTEXT_LENGTH</code> env var.
                Keep it at 2048–4096 for short prompts; raise it only when you actually feed in long documents,
                since a larger window means a larger KV cache and a slower first token.
              </div>
            </Box>

            <Box title="Free up memory & MoE tradeoffs">
              <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 10 }}>
                Every GB you free is a GB Ollama can use. Electron apps (Slack, Discord, VS Code) and browser tabs
                are the usual culprits — check what's eating memory:
              </div>
              <CopyBlock copy={copy} copied={copied} id="opt-top" text="top -o mem" label="See what's using memory" />
              <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginTop: 10 }}>
                <strong style={{ color: C.text }}>MoE models</strong> (e.g. Mixtral 8x7B) activate only a fraction of their
                weights per token, so they run faster than a dense model of the same total size — but they still need
                <em> all</em> the weights in memory. They only win when you have memory to spare; otherwise a good dense
                7–8B model is faster.
              </div>
            </Box>
          </>);
        })()}

        {/* ═══ CONNECT ═══ */}
        {settingsSection === "connect" && (<>
          <Box title="Connect Other Browser Apps" sub="Any web app running locally can call your Ollama models directly via fetch. Copy these snippets into your app code.">
            <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 14 }}>
              Ollama exposes an <strong style={{ color: C.text }}>OpenAI-compatible API</strong> at <code style={{ fontFamily: mono, color: C.accent }}>{ollamaUrl}/v1/chat/completions</code>.
              Any browser app, React component, or HTML page can call it. No API keys, no auth — it's your machine.
            </div>

            {installed.length > 0 ? (
              <>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Pick a model for the snippet:</div>
                {installed.map(m => (
                  <button key={m.name} onClick={() => setModel(m.name)} style={{
                    padding: "5px 12px", fontSize: 11, fontFamily: mono, borderRadius: 6, border: "none", cursor: "pointer", marginRight: 6, marginBottom: 6,
                    background: model === m.name ? C.accent : C.s2, color: model === m.name ? "#fff" : C.dim,
                  }}>{m.name}</button>
                ))}
              </>
            ) : (
              <div style={{ fontSize: 12, color: C.orange }}>No models installed. Pull one from Settings → Models first.</div>
            )}

            {model && (<>
              <div style={{ marginTop: 16 }}>
                <CopyBlock copy={copy} copied={copied} id="snip-simple" label="Simple — one-shot request, returns the text response" text={snippet(model)} />
              </div>
              <div style={{ marginTop: 8 }}>
                <CopyBlock copy={copy} copied={copied} id="snip-stream" label="Streaming — tokens arrive as they're generated" text={streamSnippet(model)} />
              </div>
            </>)}
          </Box>

          <Box title="CORS Setup" sub="Why a hosted page can't reach a freshly-installed Ollama">
            <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 10 }}>
              Ollama only answers browser requests whose origin is listed in{" "}
              <code style={{ fontFamily: mono, color: C.accent }}>OLLAMA_ORIGINS</code> — by default just
              localhost. A page served from GitHub Pages (or any other host) is blocked until you add it.
              For the terminal server, pass the variable when you start it:
            </div>
            <CopyBlock copy={copy} copied={copied} id="cors" text={corsNote} />
            <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginTop: 12 }}>
              The <strong style={{ color: C.text }}>official desktop app</strong> ignores shell variables —
              set them at the OS level instead, then quit and reopen Ollama:
            </div>
            {originFixOrder.map(os => (
              <div key={os} style={{ marginTop: 8 }}>
                <CopyBlock copy={copy} copied={copied} id={`conn-fix-${os}`} text={originFix[os]}
                  label={os === hw?.os ? `${os} (your system)` : os} />
              </div>
            ))}
          </Box>

          <Box title="API Reference">
            <div style={{ fontFamily: mono, fontSize: 11, color: C.dim, lineHeight: 2.2 }}>
              <div><span style={{ color: C.text, display: "inline-block", width: 200 }}>OpenAI chat completions</span> <span style={{ color: C.green }}>POST</span> {ollamaUrl}/v1/chat/completions</div>
              <div><span style={{ color: C.text, display: "inline-block", width: 200 }}>Generate (Ollama native)</span> <span style={{ color: C.green }}>POST</span> {ollamaUrl}/api/generate</div>
              <div><span style={{ color: C.text, display: "inline-block", width: 200 }}>Embeddings</span> <span style={{ color: C.green }}>POST</span> {ollamaUrl}/api/embeddings</div>
              <div><span style={{ color: C.text, display: "inline-block", width: 200 }}>List models</span> <span style={{ color: C.green }}>GET</span>&nbsp; {ollamaUrl}/api/tags</div>
              <div><span style={{ color: C.text, display: "inline-block", width: 200 }}>Running models</span> <span style={{ color: C.green }}>GET</span>&nbsp; {ollamaUrl}/api/ps</div>
              <div><span style={{ color: C.text, display: "inline-block", width: 200 }}>Model info</span> <span style={{ color: C.green }}>POST</span> {ollamaUrl}/api/show</div>
            </div>
          </Box>
        </>)}
      </div>
      </div>
      )}
    </div>
  );
}
