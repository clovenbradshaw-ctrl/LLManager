import { useState, useEffect, useRef, useMemo } from "react";
import { C, mono, sans } from "./theme.js";
import * as mx from "./matrix.js";
import MatrixGate from "./MatrixGate.jsx";

const MSG_CAP = 250;   // messages retained per room
const CTX_MSGS = 12;   // room messages sent to the LLM as conversation context

const localpart = (userId) => (userId || "").replace(/^@/, "").split(":")[0];

function roomDisplayName(events, fallback) {
  let name = "", alias = "";
  for (const ev of events) {
    if (ev.type === "m.room.name" && ev.content?.name) name = ev.content.name;
    if (ev.type === "m.room.canonical_alias" && ev.content?.alias) alias = ev.content.alias;
  }
  return name || alias || fallback;
}

export default function MatrixChat(props) {
  const { session, onLogin } = props;
  if (!session) {
    return (
      <div style={{ padding: "32px 20px", display: "flex", justifyContent: "center" }}>
        <div style={{ maxWidth: 380, width: "100%" }}>
          <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6, marginBottom: 14 }}>
            Matrix sign-in was skipped. Sign in here to chat across devices and
            route messages to your local LLM.
          </div>
          <MatrixGate embedded onLogin={onLogin} />
        </div>
      </div>
    );
  }
  // Remount cleanly whenever the account changes.
  return <MatrixChatInner key={session.userId} {...props} />;
}

function MatrixChatInner({ session, onLogout, ollamaUrl, ollamaUp, model, models }) {
  const [rooms, setRooms] = useState({});
  const [selectedRoomId, setSelectedRoomId] = useState(null);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [connected, setConnected] = useState(false);
  const [syncError, setSyncError] = useState("");
  const [bridgeRoomId, setBridgeRoomId] = useState(null);
  const [bridgeModel, setBridgeModel] = useState(model || "");
  const [answering, setAnswering] = useState(0);
  const chatEndRef = useRef(null);

  // Refs let the long-lived sync loop read current values without restarting.
  const sinceRef = useRef(null);
  const roomsRef = useRef(rooms);
  const bridgeRoomRef = useRef(bridgeRoomId);
  const bridgeModelRef = useRef(bridgeModel);
  const ollamaUrlRef = useRef(ollamaUrl);
  const answeredRef = useRef(new Set()); // event ids the bridge has handled
  useEffect(() => { roomsRef.current = rooms; }, [rooms]);
  useEffect(() => { bridgeRoomRef.current = bridgeRoomId; }, [bridgeRoomId]);
  useEffect(() => { bridgeModelRef.current = bridgeModel; }, [bridgeModel]);
  useEffect(() => { ollamaUrlRef.current = ollamaUrl; }, [ollamaUrl]);

  // Adopt the dashboard's model once one becomes available.
  useEffect(() => {
    if (!bridgeModel && model) setBridgeModel(model);
  }, [model, bridgeModel]);

  const roomList = useMemo(() =>
    Object.values(rooms).sort((a, b) => {
      const at = a.messages[a.messages.length - 1]?.ts || 0;
      const bt = b.messages[b.messages.length - 1]?.ts || 0;
      return bt - at;
    }), [rooms]);

  useEffect(() => {
    if (!selectedRoomId && roomList.length) setSelectedRoomId(roomList[0].id);
  }, [selectedRoomId, roomList]);

  const selectedRoom = selectedRoomId ? rooms[selectedRoomId] : null;

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [selectedRoom?.messages, answering]);

  // ── Merge a sync response's joined rooms into state ──
  const applyJoin = (join) => {
    setRooms(prev => {
      const next = { ...prev };
      for (const [roomId, data] of Object.entries(join)) {
        const stateEvents = data.state?.events || [];
        const timeline = data.timeline?.events || [];
        const existing = next[roomId] || { id: roomId, name: roomId, messages: [] };
        const name = roomDisplayName([...stateEvents, ...timeline], existing.name);
        const seen = new Set(existing.messages.map(m => m.id));
        const added = [];
        for (const ev of timeline) {
          if (ev.type !== "m.room.message" || seen.has(ev.event_id)) continue;
          const ct = ev.content || {};
          if (!["m.text", "m.notice", "m.emote"].includes(ct.msgtype)) continue;
          added.push({
            id: ev.event_id, sender: ev.sender, body: ct.body || "",
            msgtype: ct.msgtype, ts: ev.origin_server_ts || Date.now(),
          });
        }
        next[roomId] = {
          id: roomId, name,
          messages: [...existing.messages, ...added].slice(-MSG_CAP),
        };
      }
      return next;
    });
  };

  // ── Bridge: route an incoming message through Ollama, reply as a notice ──
  const answerWithOllama = async (roomId, triggerEvent) => {
    const llmModel = bridgeModelRef.current;
    if (!llmModel) {
      mx.sendMessage(session, roomId,
        "⚠️ LLM bridge: no Ollama model selected on the host device.", "m.notice")
        .catch(() => {});
      return;
    }
    setAnswering(n => n + 1);
    try {
      const msgs = (roomsRef.current[roomId]?.messages || []).slice(-CTX_MSGS);
      const ctx = msgs.some(m => m.id === triggerEvent.event_id)
        ? msgs
        : [...msgs, {
            id: triggerEvent.event_id, sender: triggerEvent.sender,
            body: triggerEvent.content.body, msgtype: "m.text",
          }];
      // A notice from us is a prior LLM answer; everything else is user input.
      const history = ctx.map(m => ({
        role: m.sender === session.userId && m.msgtype === "m.notice" ? "assistant" : "user",
        content: m.body,
      }));
      const r = await fetch(`${ollamaUrlRef.current}/api/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: llmModel, messages: history, stream: false }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
      await mx.sendMessage(session, roomId,
        j.message?.content?.trim() || "(empty response)", "m.notice");
    } catch (e) {
      await mx.sendMessage(session, roomId, `⚠️ LLM bridge error: ${e.message}`, "m.notice")
        .catch(() => {});
    }
    setAnswering(n => Math.max(0, n - 1));
  };

  // Bridge replies go out as m.notice and are never re-answered (only m.text
  // triggers the bridge), so the host can never answer its own output.
  const handleBridge = (join) => {
    const roomId = bridgeRoomRef.current;
    if (!roomId) return;
    for (const ev of join[roomId]?.timeline?.events || []) {
      if (ev.type !== "m.room.message") continue;
      if (ev.content?.msgtype !== "m.text") continue;
      if (answeredRef.current.has(ev.event_id)) continue;
      answeredRef.current.add(ev.event_id);
      answerWithOllama(roomId, ev);
    }
  };

  // ── Sync loop ──
  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    let firstSync = true;
    setSyncError("");

    (async () => {
      while (!cancelled) {
        try {
          const data = await mx.sync(session, sinceRef.current, controller.signal);
          if (cancelled) return;
          sinceRef.current = data.next_batch;
          const join = data.rooms?.join || {};
          if (Object.keys(join).length) applyJoin(join);
          setConnected(true);
          setSyncError("");
          // Skip the first batch so the bridge never answers history.
          if (!firstSync) handleBridge(join);
          firstSync = false;
        } catch (e) {
          if (cancelled || e.name === "AbortError") return;
          setConnected(false);
          setSyncError(e.message || "Sync error");
          await new Promise(res => setTimeout(res, 4000));
        }
      }
    })();

    return () => { cancelled = true; controller.abort(); };
  }, [session]);

  const send = async () => {
    const text = draft.trim();
    if (!text || !selectedRoomId || sending) return;
    setSending(true);
    setDraft("");
    try {
      await mx.sendMessage(session, selectedRoomId, text);
    } catch (e) {
      setSyncError(`Send failed: ${e.message}`);
      setDraft(text);
    }
    setSending(false);
  };

  const handleLogout = async () => {
    await mx.logout(session);
    onLogout();
  };

  const bridgeHere = bridgeRoomId && bridgeRoomId === selectedRoomId;
  const bridgeElsewhere = bridgeRoomId && bridgeRoomId !== selectedRoomId;

  return (
    <div style={{
      display: "grid", gridTemplateColumns: "280px minmax(0, 1fr)",
      height: "100%", minHeight: 520,
    }}>
      {/* ── Rooms sidebar ── */}
      <aside style={{
        borderRight: `1px solid ${C.border}`, background: C.s1,
        display: "flex", flexDirection: "column", minHeight: 0,
      }}>
        <div style={{ padding: 14, borderBottom: `1px solid ${C.border}` }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
            <span style={{ fontSize: 12, fontFamily: mono, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {session.userId}
            </span>
            <button onClick={handleLogout} style={{
              fontSize: 10, fontFamily: mono, fontWeight: 600, padding: "4px 9px",
              borderRadius: 5, border: `1px solid ${C.border}`, background: C.s2,
              color: C.dim, cursor: "pointer", flexShrink: 0,
            }}>logout</button>
          </div>
          <div style={{ fontSize: 10, fontFamily: mono, color: connected ? C.green : C.orange, marginTop: 6 }}>
            {connected ? "🟢 synced" : "⏳ connecting…"}
          </div>
        </div>
        <div style={{ padding: "10px 10px 6px", fontSize: 10, fontFamily: mono, color: C.dim, textTransform: "uppercase", letterSpacing: 0.7 }}>
          Rooms ({roomList.length})
        </div>
        <div style={{ overflowY: "auto", padding: "0 8px 12px", flex: 1 }}>
          {roomList.length === 0 ? (
            <div style={{ color: C.dim, fontSize: 12, lineHeight: 1.5, padding: "8px 10px" }}>
              No rooms yet. Join or create a room from any Matrix client and it
              will appear here.
            </div>
          ) : roomList.map(r => {
            const active = r.id === selectedRoomId;
            const last = r.messages[r.messages.length - 1];
            return (
              <button key={r.id} onClick={() => setSelectedRoomId(r.id)} style={{
                display: "block", width: "100%", textAlign: "left", padding: "10px 11px",
                marginBottom: 6, borderRadius: 9, cursor: "pointer", color: C.text,
                border: `1px solid ${active ? C.accent : "transparent"}`,
                background: active ? C.accent + "20" : "transparent",
              }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                  <span style={{ fontSize: 12, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.name}</span>
                  {r.id === bridgeRoomId && <span style={{ fontSize: 9, fontFamily: mono, color: C.accent, flexShrink: 0 }}>◆ bridge</span>}
                </div>
                <div style={{ fontSize: 10, color: C.dim, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginTop: 3 }}>
                  {last ? `${localpart(last.sender)}: ${last.body}` : "No messages yet"}
                </div>
              </button>
            );
          })}
        </div>
      </aside>

      {/* ── Conversation ── */}
      <main style={{ display: "flex", flexDirection: "column", minHeight: 0, background: C.bg }}>
        {/* Bridge control bar */}
        <div style={{ padding: "10px 18px", borderBottom: `1px solid ${C.border}`, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: C.text, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: "1 1 140px" }}>
            {selectedRoom ? selectedRoom.name : "Select a room"}
          </div>
          <select value={bridgeModel} onChange={e => setBridgeModel(e.target.value)} style={{
            padding: "6px 10px", fontSize: 11, fontFamily: mono, background: C.s1,
            color: C.text, border: `1px solid ${C.border}`, borderRadius: 7, maxWidth: 200,
          }}>
            {models.length === 0 && <option value="">no local models</option>}
            {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
          </select>
          <button
            onClick={() => setBridgeRoomId(prev => (prev ? null : selectedRoomId))}
            disabled={!selectedRoomId}
            title="When on, messages in this room are answered by your local Ollama model"
            style={{
              fontSize: 11, fontFamily: mono, fontWeight: 700, padding: "6px 12px",
              borderRadius: 7, border: "none", cursor: selectedRoomId ? "pointer" : "default",
              background: bridgeRoomId ? C.red : C.accent, color: "#fff",
              opacity: selectedRoomId ? 1 : 0.4,
            }}>
            {bridgeRoomId ? "Stop LLM bridge" : "Start LLM bridge"}
          </button>
        </div>

        {/* Bridge status line */}
        {(bridgeRoomId || ollamaUp === false) && (
          <div style={{
            padding: "6px 18px", fontSize: 11, fontFamily: mono,
            borderBottom: `1px solid ${C.border}`,
            background: bridgeHere ? C.accent + "14" : C.s1,
            color: bridgeElsewhere ? C.orange : bridgeRoomId ? C.accent : C.dim,
          }}>
            {bridgeHere && `◆ Bridge active — incoming messages here are answered by ${bridgeModel || "(no model)"}${answering ? ` · answering ${answering}…` : ""}`}
            {bridgeElsewhere && `◆ Bridge is running in "${rooms[bridgeRoomId]?.name || bridgeRoomId}", not this room`}
            {!bridgeRoomId && ollamaUp === false && "Ollama is offline on this device — start it (Status tab) before using the LLM bridge here."}
          </div>
        )}

        {/* Messages */}
        <div style={{ flex: 1, overflowY: "auto", padding: "18px 24px", minHeight: 0 }}>
          <div style={{ maxWidth: 820, margin: "0 auto" }}>
            {!selectedRoom || selectedRoom.messages.length === 0 ? (
              <div style={{ minHeight: "40vh", display: "flex", alignItems: "center", justifyContent: "center", textAlign: "center", color: C.dim }}>
                <div>
                  <div style={{ fontSize: 30, marginBottom: 10 }}>◆</div>
                  <div style={{ fontSize: 15, color: C.text, fontWeight: 700, marginBottom: 6 }}>
                    {selectedRoom ? "No messages yet" : "Pick a room"}
                  </div>
                  <div style={{ fontSize: 12, lineHeight: 1.6 }}>
                    Send a message here, or from any Matrix client on another
                    device. With the LLM bridge on, this device answers them.
                  </div>
                </div>
              </div>
            ) : selectedRoom.messages.map(m => {
              const mine = m.sender === session.userId;
              const isNotice = m.msgtype === "m.notice";
              return (
                <div key={m.id} style={{ display: "flex", justifyContent: mine ? "flex-end" : "flex-start", padding: "5px 0" }}>
                  <div style={{ maxWidth: "82%", minWidth: 0 }}>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", justifyContent: mine ? "flex-end" : "flex-start", marginBottom: 3 }}>
                      <span style={{ fontSize: 10, fontFamily: mono, color: C.dim }}>{localpart(m.sender)}</span>
                      {isNotice && <span style={{ fontSize: 9, fontFamily: mono, color: C.accent, fontWeight: 700 }}>LLM</span>}
                      <span style={{ fontSize: 9, fontFamily: mono, color: C.dim }}>
                        {new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(m.ts)}
                      </span>
                    </div>
                    <div style={{
                      background: mine && !isNotice ? C.accent : "transparent",
                      border: mine && !isNotice ? "none" : `1px solid ${isNotice ? C.accent + "55" : C.border}`,
                      color: mine && !isNotice ? "#fff" : C.text,
                      borderRadius: 12, padding: "9px 13px", fontSize: 13, lineHeight: 1.5,
                      whiteSpace: "pre-wrap", wordBreak: "break-word",
                      fontStyle: m.msgtype === "m.emote" ? "italic" : "normal",
                    }}>
                      {m.msgtype === "m.emote" ? `* ${localpart(m.sender)} ${m.body}` : m.body}
                    </div>
                  </div>
                </div>
              );
            })}
            <div ref={chatEndRef} />
          </div>
        </div>

        {syncError && (
          <div style={{ padding: "6px 24px", fontSize: 11, fontFamily: mono, color: C.red, background: C.red + "10" }}>
            {syncError}
          </div>
        )}

        {/* Composer */}
        <div style={{ borderTop: `1px solid ${C.border}`, padding: "14px 24px 18px", background: C.bg }}>
          <div style={{ maxWidth: 820, margin: "0 auto" }}>
            <div style={{ background: C.s1, border: `1px solid ${C.border}`, borderRadius: 14, padding: 10 }}>
              <textarea
                value={draft} onChange={e => setDraft(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder={selectedRoom ? "Message this room… Enter to send" : "Select a room first"}
                disabled={!selectedRoom}
                rows={2}
                style={{
                  width: "100%", padding: "8px 10px", fontSize: 14, background: "transparent",
                  color: C.text, border: "none", outline: "none", resize: "none",
                  boxSizing: "border-box", lineHeight: 1.5, fontFamily: sans,
                }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 10, fontFamily: mono, color: C.dim }}>
                  {bridgeHere ? "the LLM bridge will answer this" : "sent as a normal Matrix message"}
                </span>
                <button onClick={send} disabled={sending || !draft.trim() || !selectedRoom} style={{
                  padding: "8px 18px", fontSize: 12, fontWeight: 700, borderRadius: 9, border: "none",
                  cursor: sending ? "wait" : "pointer", background: sending ? C.s2 : C.accent,
                  color: sending ? C.dim : "#fff", opacity: (!draft.trim() || !selectedRoom) ? 0.4 : 1,
                }}>{sending ? "Sending…" : "Send"}</button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
