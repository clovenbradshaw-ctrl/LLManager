import { useState } from "react";
import { C, mono, sans } from "./theme.js";
import { discoverHomeserver, login } from "./matrix.js";

const lbl = { fontSize: 11, color: C.dim, display: "block", marginBottom: 4 };

// Matrix sign-in form. Rendered full-screen as the app gate, or `embedded`
// inside the Matrix tab once the gate has been skipped.
export default function MatrixGate({ onLogin, onSkip, embedded }) {
  const [homeserver, setHomeserver] = useState("matrix.org");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    if (busy) return;
    setError("");
    setBusy(true);
    try {
      const hs = await discoverHomeserver(username.includes(":") ? username : homeserver);
      onLogin(await login(hs, username, password));
    } catch (err) {
      setError(err.message || "Login failed");
    }
    setBusy(false);
  };

  const field = {
    width: "100%", padding: "10px 12px", fontSize: 13, fontFamily: mono,
    background: C.bg, color: C.text, border: `1px solid ${C.border}`,
    borderRadius: 8, boxSizing: "border-box",
  };

  const card = (
    <form onSubmit={submit} style={{
      width: "100%", maxWidth: 380, background: C.s1,
      border: `1px solid ${C.border}`, borderRadius: 14, padding: 24, boxSizing: "border-box",
    }}>
      <div style={{ fontSize: 17, fontWeight: 700, color: C.text }}>
        <span style={{ color: C.accent }}>◆</span> Matrix sign-in
      </div>
      <div style={{ fontSize: 12, color: C.dim, marginTop: 6, lineHeight: 1.6 }}>
        Sign in to a Matrix account to chat across devices. The device running
        Ollama can answer those messages with your local LLM.
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 18 }}>
        <div>
          <label style={lbl}>Username</label>
          <input value={username} onChange={e => setUsername(e.target.value)}
            placeholder="@you:matrix.org" autoComplete="username" style={field} />
        </div>
        <div>
          <label style={lbl}>Password</label>
          <input value={password} onChange={e => setPassword(e.target.value)}
            type="password" autoComplete="current-password" style={field} />
        </div>
        <div>
          <label style={lbl}>Homeserver</label>
          <input value={homeserver} onChange={e => setHomeserver(e.target.value)}
            placeholder="matrix.org" style={field} />
          <div style={{ fontSize: 10, color: C.dim, marginTop: 4, lineHeight: 1.5 }}>
            A domain or a full base URL. If sign-in can't connect, the API may
            live elsewhere — open <code style={{ fontFamily: mono }}>https://&lt;domain&gt;/.well-known/matrix/client</code> and
            paste its <code style={{ fontFamily: mono }}>base_url</code> here.
            Ignored when the username already includes a server (@you:server).
          </div>
        </div>
      </div>
      {error && (
        <div style={{
          marginTop: 12, fontSize: 12, color: C.red, background: C.red + "14",
          border: `1px solid ${C.red}33`, borderRadius: 8, padding: "8px 12px",
        }}>{error}</div>
      )}
      <button type="submit" disabled={busy || !username || !password} style={{
        width: "100%", marginTop: 16, padding: "11px 0", fontSize: 13, fontWeight: 700,
        borderRadius: 9, border: "none", cursor: busy ? "wait" : "pointer",
        background: C.accent, color: "#fff", opacity: (!username || !password) ? 0.4 : 1,
      }}>{busy ? "Signing in…" : "Sign in"}</button>
      {onSkip && (
        <button type="button" onClick={onSkip} style={{
          width: "100%", marginTop: 10, padding: "8px 0", fontSize: 12,
          background: "transparent", color: C.dim, border: "none", cursor: "pointer",
        }}>Continue without login (local Ollama only) →</button>
      )}
    </form>
  );

  if (embedded) return card;

  return (
    <div style={{
      fontFamily: sans, background: C.bg, color: C.text, minHeight: "100vh",
      display: "flex", alignItems: "center", justifyContent: "center", padding: 20,
    }}>{card}</div>
  );
}
