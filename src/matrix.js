// Minimal Matrix client-server API client — browser fetch only, no SDK.
// Public homeservers (matrix.org and most others) enable CORS, so a static
// page can talk to them directly.

const API = "/_matrix/client/v3";
const SESSION_KEY = "llm-manager-matrix-session";

// Keep sync payloads small: short timelines, no presence/typing noise.
const SYNC_FILTER = JSON.stringify({
  presence: { types: [] },
  account_data: { types: [] },
  room: {
    timeline: { limit: 40 },
    ephemeral: { types: [] },
    account_data: { types: [] },
  },
});

export function loadSession() {
  try {
    const s = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
    return s && s.accessToken && s.homeserver && s.userId ? s : null;
  } catch {
    return null;
  }
}

export function saveSession(session) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

export function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

const authHeaders = (session) => ({ Authorization: `Bearer ${session.accessToken}` });

// Resolves "matrix.org", "https://matrix.org", or a "@user:server" handle to
// the homeserver's real API base URL via .well-known discovery.
export async function discoverHomeserver(input) {
  let domain = (input || "").trim();
  if (domain.startsWith("@")) domain = domain.slice(domain.indexOf(":") + 1);
  domain = domain.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
  if (!domain) throw new Error("Enter a homeserver");
  const base = `https://${domain}`;
  try {
    const r = await fetch(`${base}/.well-known/matrix/client`);
    if (r.ok) {
      const url = (await r.json())?.["m.homeserver"]?.base_url;
      if (url) return url.replace(/\/+$/, "");
    }
  } catch {
    // No well-known record — fall back to the bare domain.
  }
  return base;
}

export async function login(homeserver, user, password) {
  const r = await fetch(`${homeserver}${API}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      type: "m.login.password",
      identifier: { type: "m.id.user", user },
      password,
      initial_device_display_name: "LLM Manager",
    }),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `Login failed (HTTP ${r.status})`);
  const session = {
    homeserver,
    accessToken: j.access_token,
    userId: j.user_id,
    deviceId: j.device_id,
  };
  saveSession(session);
  return session;
}

export async function logout(session) {
  try {
    await fetch(`${session.homeserver}${API}/logout`, {
      method: "POST",
      headers: authHeaders(session),
    });
  } catch {
    // The token is dropped locally regardless of the server response.
  }
  clearSession();
}

// Long-polls the homeserver. `since` is the cursor from the previous call.
export async function sync(session, since, signal) {
  const params = new URLSearchParams({ timeout: "30000", filter: SYNC_FILTER });
  if (since) params.set("since", since);
  const r = await fetch(`${session.homeserver}${API}/sync?${params}`, {
    headers: authHeaders(session),
    signal,
  });
  if (r.status === 401) throw new Error("Matrix session expired — sign in again.");
  if (!r.ok) throw new Error(`Sync failed (HTTP ${r.status})`);
  return r.json();
}

export async function sendMessage(session, roomId, body, msgtype = "m.text") {
  const txnId = `llm-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const r = await fetch(
    `${session.homeserver}${API}/rooms/${encodeURIComponent(roomId)}/send/m.room.message/${txnId}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...authHeaders(session) },
      body: JSON.stringify({ msgtype, body }),
    },
  );
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `Send failed (HTTP ${r.status})`);
  return j.event_id;
}
