import { useState, useEffect, useCallback, useRef } from "react";

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

export default function App() {
  const [tab, setTab] = useState("status");
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");
  const [ollamaUp, setOllamaUp] = useState(null);
  const [ollamaVer, setOllamaVer] = useState("");
  const [installed, setInstalled] = useState([]);
  const [running, setRunning] = useState([]);
  const [hw, setHw] = useState(null);

  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [generating, setGenerating] = useState(false);
  const [messages, setMessages] = useState([]);
  const [chatMode, setChatMode] = useState("thread");
  const [copied, setCopied] = useState(null);
  const chatEndRef = useRef(null);

  // Keep the latest message in view as the conversation grows.
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, generating]);

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
    } catch { setOllamaUp(false); }
  }, [ollamaUrl]);

  useEffect(() => { probe(); }, [probe]);

  // ── Send a chat message ──
  const send = async () => {
    const text = prompt.trim();
    if (!text || !model || generating) return;

    // In thread mode the prior turns are sent as context; in one-shot mode
    // each prompt is answered on its own with no history.
    const context = chatMode === "thread" ? messages : [];
    const userMsg = { id: `${Date.now()}-u`, role: "user", content: text };

    setMessages(prev => [...prev, userMsg]);
    setPrompt("");
    setGenerating(true);
    const t0 = Date.now();
    try {
      const apiMessages = [...context, userMsg]
        .filter(m => !m.error)
        .map(m => ({ role: m.role, content: m.content }));
      const r = await fetch(`${ollamaUrl}/v1/chat/completions`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, messages: apiMessages, stream: false }),
      });
      const data = await r.json();
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      setMessages(prev => [...prev, {
        id: `${Date.now()}-a`, role: "assistant", model, elapsed,
        content: data.choices?.[0]?.message?.content || JSON.stringify(data),
        usage: data.usage || {},
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        id: `${Date.now()}-a`, role: "assistant", content: `Error: ${e.message}`, error: true,
      }]);
    }
    setGenerating(false);
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

  const Pill = ({ color, children }) => (
    <span style={{ fontSize: 10, fontFamily: mono, padding: "2px 8px", borderRadius: 99, background: color + "22", color, fontWeight: 600 }}>{children}</span>
  );

  const CopyBlock = ({ text, id, label }) => (
    <div style={{ marginBottom: 8 }}>
      {label && <div style={{ fontSize: 10, color: C.dim, marginBottom: 3 }}>{label}</div>}
      <div style={{ display: "flex", gap: 6 }}>
        <code style={{ flex: 1, fontSize: 11, fontFamily: mono, background: C.bg, padding: "8px 12px", borderRadius: 6, color: C.green, border: `1px solid ${C.border}`, wordBreak: "break-all", whiteSpace: "pre-wrap" }}>{text}</code>
        <button onClick={() => copy(text, id)} style={{ padding: "6px 12px", fontSize: 10, fontFamily: mono, fontWeight: 600, borderRadius: 6, border: "none", cursor: "pointer", background: copied === id ? C.green : C.accent, color: copied === id ? "#000" : "#fff", whiteSpace: "nowrap" }}>{copied === id ? "✓" : "copy"}</button>
      </div>
    </div>
  );

  const Box = ({ title, sub, children }) => (
    <div style={{ background: C.s1, border: `1px solid ${C.border}`, borderRadius: 10, padding: "14px 18px", marginBottom: 10 }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginBottom: sub ? 2 : 10 }}>{title}</div>
      {sub && <div style={{ fontSize: 11, color: C.dim, marginBottom: 10 }}>{sub}</div>}
      {children}
    </div>
  );

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

  const corsNote = `# If other browser apps get CORS errors, restart Ollama with:
OLLAMA_ORIGINS="*" ollama serve

# Or restrict to specific origins:
OLLAMA_ORIGINS="http://localhost:3000,https://myapp.com" ollama serve`;

  return (
    <div style={{ fontFamily: sans, background: C.bg, color: C.text, minHeight: "100vh" }}>
      <div style={{ padding: "16px 20px 12px", borderBottom: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700 }}><span style={{ color: C.accent }}>◆</span> LLM Manager</div>
          <div style={{ fontSize: 11, fontFamily: mono, color: C.dim, marginTop: 2 }}>
            {ollamaUp === true ? "🟢" : ollamaUp === false ? "🔴" : "⏳"} Ollama {ollamaUp ? `v${ollamaVer}` : "offline"} · {installed.length} models
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {["status", "run", "models", "connect"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "7px 16px", fontSize: 12, fontWeight: 600, borderRadius: 8, cursor: "pointer", textTransform: "capitalize",
              border: `1px solid ${tab === t ? C.accent : C.border}`, background: tab === t ? C.accent : "transparent", color: tab === t ? "#fff" : C.dim,
            }}>{t}</button>
          ))}
        </div>
      </div>

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
              <CopyBlock id="hw" text='system_profiler SPHardwareDataType | grep -E "Chip|Memory|Cores|Model"' label="Get exact specs in terminal" />
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
                <CopyBlock text="brew install ollama" id="inst" label="1. Install" />
                <CopyBlock text="ollama serve" id="srv" label="2. Start server" />
                <CopyBlock text="ollama pull gemma2:2b" id="fp" label="3. Pull a model" />
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
              {installed.map(m => (
                <div key={m.name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "5px 0", fontSize: 12, borderBottom: `1px solid ${C.s3}` }}>
                  <span style={{ fontFamily: mono }}>{m.name} <span style={{ color: C.dim, fontSize: 10 }}>· {fmtB(m.size)} · {m.details?.quantization_level || ""}</span></span>
                  {running.some(r => r.name === m.name) && <Pill color={C.green}>LOADED</Pill>}
                </div>
              ))}
            </>)}
          </Box>
        </>)}

        {/* ═══ RUN ═══ */}
        {tab === "run" && (
          ollamaUp === false ? (
            <Box title="Chat">
              <div style={{ fontSize: 12, color: C.red }}>Ollama not running — check the Status tab for setup.</div>
            </Box>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 130px)" }}>
              {/* ── Controls ── */}
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 10 }}>
                <select value={model} onChange={e => setModel(e.target.value)} style={{ flex: 1, minWidth: 160, padding: "8px 12px", fontSize: 12, fontFamily: mono, background: C.s1, color: C.text, border: `1px solid ${C.border}`, borderRadius: 8 }}>
                  {installed.length === 0 && <option value="">no models installed</option>}
                  {installed.map(m => <option key={m.name} value={m.name}>{m.name} ({fmtB(m.size)})</option>)}
                </select>
                <div style={{ display: "flex", background: C.s1, border: `1px solid ${C.border}`, borderRadius: 8, padding: 2 }}>
                  {[["thread", "Thread"], ["oneshot", "One-shot"]].map(([v, label]) => (
                    <button key={v} onClick={() => setChatMode(v)} title={v === "thread" ? "Each reply sees the whole conversation" : "Each prompt is answered on its own, no memory"} style={{
                      padding: "6px 14px", fontSize: 11, fontWeight: 600, borderRadius: 6, border: "none", cursor: "pointer",
                      background: chatMode === v ? C.accent : "transparent", color: chatMode === v ? "#fff" : C.dim,
                    }}>{label}</button>
                  ))}
                </div>
                {messages.length > 0 && (
                  <button onClick={() => setMessages([])} style={{ padding: "7px 14px", fontSize: 11, fontWeight: 600, background: C.s2, color: C.dim, border: `1px solid ${C.border}`, borderRadius: 8, cursor: "pointer" }}>New chat</button>
                )}
              </div>

              {/* ── Messages ── */}
              <div style={{ flex: 1, overflowY: "auto", background: C.s1, border: `1px solid ${C.border}`, borderRadius: 10, padding: "16px 18px" }}>
                {messages.length === 0 && !generating && (
                  <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: C.dim, fontSize: 13, textAlign: "center" }}>
                    <div style={{ fontSize: 28, marginBottom: 8 }}>◆</div>
                    <div>Start a conversation.</div>
                    <div style={{ fontSize: 11, marginTop: 4 }}>
                      {chatMode === "thread" ? "Thread mode — replies remember the whole conversation." : "One-shot mode — each prompt is answered independently."}
                    </div>
                  </div>
                )}
                {messages.map(m => (
                  <div key={m.id} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start", marginBottom: 14 }}>
                    <div style={{ maxWidth: "85%" }}>
                      <div style={{
                        fontSize: 13, lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word",
                        padding: "10px 14px", borderRadius: 12,
                        background: m.role === "user" ? C.accent : C.s2,
                        color: m.error ? C.red : m.role === "user" ? "#fff" : C.text,
                        border: m.role === "user" ? "none" : `1px solid ${C.border}`,
                      }}>{m.content}</div>
                      {m.role === "assistant" && !m.error && (
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 4, fontSize: 10, fontFamily: mono, color: C.dim }}>
                          <span>{m.model} · {m.elapsed}s · {m.usage?.total_tokens || "?"} tokens</span>
                          <button onClick={() => copy(m.content, m.id)} style={{
                            padding: "2px 8px", fontSize: 10, fontFamily: mono, borderRadius: 5, border: "none", cursor: "pointer",
                            background: copied === m.id ? C.green : "transparent", color: copied === m.id ? "#000" : C.dim,
                          }}>{copied === m.id ? "✓" : "copy"}</button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {generating && (
                  <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 14 }}>
                    <div style={{ padding: "10px 14px", borderRadius: 12, background: C.s2, border: `1px solid ${C.border}`, fontSize: 13, color: C.dim }}>
                      Generating…
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* ── Input ── */}
              <div style={{ display: "flex", gap: 8, alignItems: "flex-end", marginTop: 10 }}>
                <textarea
                  value={prompt} onChange={e => setPrompt(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                  placeholder="Type a message… Enter to send, Shift+Enter for newline"
                  rows={2}
                  style={{ flex: 1, padding: "10px 14px", fontSize: 13, background: C.s1, color: C.text, border: `1px solid ${C.border}`, borderRadius: 8, resize: "none", boxSizing: "border-box", lineHeight: 1.5, fontFamily: sans }}
                />
                <button onClick={send} disabled={generating || !model || !prompt.trim()} style={{
                  padding: "11px 24px", fontSize: 13, fontWeight: 700, borderRadius: 8, border: "none",
                  cursor: generating ? "wait" : "pointer", background: generating ? C.s2 : C.accent,
                  color: generating ? C.dim : "#fff", opacity: (!model || !prompt.trim()) ? 0.4 : 1,
                }}>{generating ? "…" : "Send"}</button>
              </div>
            </div>
          )
        )}

        {/* ═══ MODELS ═══ */}
        {tab === "models" && (<>
          <Box title="Model Catalog" sub={ramGB ? `~${ramGB} GB detected → ~${(ramGB * .75).toFixed(0)} GB usable for models` : "RAM not exposed by browser — check terminal"}>
            {!ramGB && <div style={{ marginBottom: 12 }}><CopyBlock id="ram" text='sysctl -n hw.memsize | awk "{print $1/1073741824\" GB\"}"' label="Check actual RAM" /></div>}
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
                  <button onClick={() => copy(`ollama pull ${m.id}`, `p-${m.id}`)} style={{
                    fontSize: 10, fontFamily: mono, padding: "5px 10px", borderRadius: 5, cursor: "pointer", border: "none", whiteSpace: "nowrap",
                    background: copied === `p-${m.id}` ? C.green : C.s2, color: copied === `p-${m.id}` ? "#000" : C.dim,
                  }}>{copied === `p-${m.id}` ? "✓" : "copy pull"}</button>
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
                <CopyBlock id="snip-simple" label="Simple — one-shot request, returns the text response" text={snippet(model)} />
              </div>
              <div style={{ marginTop: 8 }}>
                <CopyBlock id="snip-stream" label="Streaming — tokens arrive as they're generated" text={streamSnippet(model)} />
              </div>
            </>)}
          </Box>

          <Box title="CORS Setup" sub="If your browser app is on a different origin and gets blocked">
            <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 10 }}>
              By default Ollama allows requests from any origin. If you run into CORS errors, restart Ollama with the <code style={{ fontFamily: mono, color: C.accent }}>OLLAMA_ORIGINS</code> env var:
            </div>
            <CopyBlock id="cors" text={corsNote} />
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
  );
}
