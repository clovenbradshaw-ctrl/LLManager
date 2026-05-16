import { useState, useEffect, useCallback } from "react";
import Chat from "./Chat.jsx";

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
  const [tab, setTab] = useState("status");
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");
  const [ollamaUp, setOllamaUp] = useState(null);
  const [ollamaVer, setOllamaVer] = useState("");
  const [installed, setInstalled] = useState([]);
  const [running, setRunning] = useState([]);
  const [hw, setHw] = useState(null);

  const [model, setModel] = useState("");
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
  const loadModel = (name) => setKeepAlive(name, "10m", "load");
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
          {["status", "chat", "models", "connect"].map(t => (
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
      ) : (
      <div style={{ flex: 1, overflowY: "auto" }}>
      <div style={{ padding: "16px 20px", maxWidth: 820, margin: "0 auto" }}>

        {/* ═══ STATUS ═══ */}
        {tab === "status" && (<>
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
        {tab === "models" && (<>
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

        {/* ═══ CONNECT ═══ */}
        {tab === "connect" && (<>
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
              <div style={{ fontSize: 12, color: C.orange }}>No models installed. Pull one from the Models tab first.</div>
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
