/* Per-chat memory — a lightweight "Experience Engine".

   In Memory mode a chat does not send its growing conversation history to
   the model. Instead, every turn is distilled into a small per-chat knowledge
   graph (entities, connections, definitions) kept in localStorage alongside
   the conversation. Each turn rebuilds a fixed-size prompt: system + projected
   dossier + one-turn position marker. The prompt stays the same size at turn 1
   and turn 10,000.

   This is a dependency-free adaptation of the architecture in experience.md:
   NER is mechanical (capitalised-run + date/number matching) rather than
   compromise.js, and the persistence layer is localStorage rather than
   Matrix, since the regular chat is a local-only, single-device app. */

export const MEMORY_SYSTEM = `You are a helpful assistant with long-term memory.
The [CONTEXT] block holds facts recalled from this conversation.
The [LIBRARY] block contains the full text of documents read into this chat —
treat them as source material you have already read and can quote directly.
The [POSITION] block notes what was discussed on the previous turn.
Use them to answer accurately and stay consistent. If they do not cover
something, answer normally from your own knowledge — never claim you have no
memory or that you cannot see a document. Keep responses concise unless asked
for detail.`;

export const EXTRACT_SYSTEM = `Extract new, durable facts from this exchange as JSON.
Return a JSON array of events. Each event is one of:
  { "op": "INS", "entity": "<name>", "kind": "person|place|org|thing|event|topic" }
  { "op": "CON", "from": "<entity>", "to": "<entity>", "type": "<relation>" }
  { "op": "DEF", "entity": "<name>", "field": "<attribute>", "value": "<value>" }
Only include facts worth remembering long-term: names, relationships, attributes,
preferences, and decisions. Ignore small talk and transient details.
Return [] if there is nothing worth remembering.
Return ONLY the JSON array — no markdown, no commentary.`;

/* Ingest: read a standalone block of text (a pasted note, an article, an
   uploaded document) into a knowledge graph. Same event vocabulary as the
   chat Extract step, but framed for a document rather than a conversation. */
export const INGEST_SYSTEM = `Read the text below and extract its durable facts as JSON.
Return a JSON array of events. Each event is one of:
  { "op": "INS", "entity": "<name>", "kind": "person|place|org|thing|event|topic" }
  { "op": "CON", "from": "<entity>", "to": "<entity>", "type": "<relation>" }
  { "op": "DEF", "entity": "<name>", "field": "<attribute>", "value": "<value>" }
Capture the named things, their relationships, their attributes, and any
definitions or decisions stated in the text. Ignore filler and rhetoric.
Return [] if there is nothing worth remembering.
Return ONLY the JSON array — no markdown, no commentary.`;

/* Split text into sentences for reading. Each sentence is read on its own so
   extraction stays focused — a long chunk makes the model skim and miss facts.
   Overlong runs (no punctuation) are hard-split; tiny fragments are merged
   back into the preceding sentence. */
export function splitSentences(text, maxLen = 1200) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return [];
  const raw = clean.match(/[^.!?]+[.!?]+(?:["'”’)\]]+)?|[^.!?]+$/g) || [clean];
  const pieces = [];
  for (let s of raw.map(x => x.trim()).filter(Boolean)) {
    while (s.length > maxLen) {
      let cut = s.lastIndexOf(" ", maxLen);
      if (cut < maxLen * 0.5) cut = maxLen;
      pieces.push(s.slice(0, cut).trim());
      s = s.slice(cut).trim();
    }
    if (s) pieces.push(s);
  }
  const out = [];
  for (const p of pieces) {
    if (out.length && p.length < 35) out[out.length - 1] += " " + p;
    else out.push(p);
  }
  return out;
}

/* Merge several knowledge graphs into one (entities/edges/defs only). Used to
   project a chat's own memory together with any opted-in library documents
   into a single graph before building the dossier. */
export function mergeMemory(...memories) {
  const out = { entities: {}, edges: {}, defs: {} };
  for (const m of memories) {
    if (!m) continue;
    for (const [id, e] of Object.entries(m.entities || {})) {
      if (!out.entities[id]) out.entities[id] = { ...e };
      else out.entities[id].mentions = (out.entities[id].mentions || 0) + (e.mentions || 0);
    }
    Object.assign(out.edges, m.edges || {});
    Object.assign(out.defs, m.defs || {});
  }
  return out;
}

/* ── Empty / clone helpers ── */
export const emptyMemory = () => ({ entities: {}, edges: {}, defs: {}, lastTurn: null });

export const cloneMemory = (m) => {
  if (!m) return emptyMemory();
  return {
    entities: { ...m.entities },
    edges: { ...m.edges },
    defs: { ...m.defs },
    lastTurn: m.lastTurn ? { ...m.lastTurn } : null,
  };
};

export const memoryStats = (m) => ({
  entities: m ? Object.keys(m.entities).length : 0,
  edges: m ? Object.keys(m.edges).length : 0,
  defs: m ? Object.keys(m.defs).length : 0,
});

const slug = (s) => String(s).trim().toLowerCase().replace(/\s+/g, " ");

/* ── The Signal: mechanical NER + keywords ── */

/* Capitalised words that are usually just sentence starters, not entities. */
const COMMON_CAPS = new Set([
  "the", "a", "an", "i", "i'm", "i've", "it", "this", "that", "these", "those",
  "what", "how", "why", "when", "where", "who", "which", "is", "are", "was",
  "do", "does", "did", "can", "could", "would", "should", "will", "yes", "no",
  "hi", "hello", "hey", "ok", "okay", "please", "thanks", "thank", "also",
  "but", "and", "or", "so", "if", "my", "your", "you", "we", "they", "he", "she",
]);

const STOPS = new Set(["what", "how", "why", "when", "where", "who", "is", "was",
  "are", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "but",
  "did", "do", "does", "has", "have", "had", "been", "be", "this", "that", "it",
  "with", "from", "about", "into", "not", "no", "yes", "can", "could", "would",
  "should", "will", "just", "also", "very", "much", "more", "some", "any", "all"]);

const DATE_RE = /\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*day\b|\b(?:today|tomorrow|yesterday|tonight)\b|\b(?:19|20)\d{2}\b/gi;

function extractNER(message) {
  const names = new Set();
  const re = /\b[A-Z][a-zA-Z]+(?:\s+(?:of|the|and|de|van|von)\s+)?(?:\s*[A-Z][a-zA-Z]+)*\b/g;
  let m;
  while ((m = re.exec(message))) {
    const name = m[0].trim().replace(/\s+/g, " ");
    if (name.length < 2) continue;
    if (COMMON_CAPS.has(name.toLowerCase())) continue;
    names.add(name);
  }
  return {
    names: [...names],
    dates: message.match(DATE_RE) || [],
    numbers: message.match(/\b\d+(?:[.,]\d+)?\b/g) || [],
  };
}

export function extractKeywords(message) {
  return message.toLowerCase().replace(/[?!.,;:'"]/g, "")
    .split(/\s+/)
    .filter(w => w.length > 2 && !STOPS.has(w));
}

export function signal(message) {
  return { ner: extractNER(message), keywords: extractKeywords(message) };
}

/* ── The Reach: search the graph using Signal output ── */
export function reach(sig, memory) {
  const entities = memory?.entities || {};
  const edges = memory?.edges || {};
  const cand = new Map(); // entityId -> score

  const add = (id, score) => cand.set(id, (cand.get(id) || 0) + score);

  for (const name of sig.ner.names) {
    const q = slug(name);
    for (const e of Object.values(entities)) {
      if (e.id.includes(q) || q.includes(e.id)) add(e.id, 3);
    }
  }
  for (const kw of sig.keywords) {
    for (const e of Object.values(entities)) {
      if (e.id.includes(kw)) add(e.id, 1);
    }
  }

  // 2-hop expansion along edges
  for (const id of [...cand.keys()]) {
    for (const edge of Object.values(edges)) {
      if (edge.from === id && !cand.has(edge.to)) add(edge.to, 1);
      if (edge.to === id && !cand.has(edge.from)) add(edge.from, 1);
    }
  }

  return [...cand.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([id]) => entities[id])
    .filter(Boolean);
}

/* ── The Dossier: format top entities into a compact context block ── */
export function buildDossier(entities, memory) {
  if (!entities.length) {
    return "[CONTEXT]\n(nothing remembered yet)\n[/CONTEXT]";
  }
  const edges = Object.values(memory?.edges || {});
  const defs = Object.values(memory?.defs || {});
  const block = entities.map(e => {
    const lines = [`${e.canonical} (${e.kind || "thing"})`];
    for (const d of defs.filter(d => d.entity === e.id)) {
      lines.push(`  ${d.field}: ${d.value}`);
    }
    for (const edge of edges.filter(g => g.from === e.id)) {
      const to = memory.entities[edge.to];
      lines.push(`  ${edge.type} → ${to ? to.canonical : edge.to}`);
    }
    return lines.join("\n");
  }).join("\n");
  return `[CONTEXT]\n${block}\n[/CONTEXT]`;
}

/* ── The Position Marker: one-turn memory, overwrites every turn ── */
export function buildPosition(lastTurn) {
  if (!lastTurn) return "";
  return `[POSITION]
Last entities: ${(lastTurn.entities || []).join(", ") || "none"}
Topic: ${lastTurn.topic || "none"}
Last user message: "${lastTurn.userMessage || ""}"
[/POSITION]`;
}

/* ── The Library card: the actual text of every document read into a chat,
   so the model is prompted with the source material itself — not just a list
   of entity names. Capped per document; this block is fixed-size relative to
   the conversation (it never grows as the chat goes on). */
export function buildLibrary(docs, perDocCap = 8000) {
  if (!docs || !docs.length) return "";
  const out = [
    "[LIBRARY]",
    "Full text of the documents read into this chat. Treat them as source",
    "material you have already read and may quote or summarise directly.",
  ];
  for (const d of docs) {
    const text = String(d.text || "").trim();
    const body = text.length > perDocCap
      ? text.slice(0, perDocCap) + " …[truncated]"
      : text;
    out.push("");
    out.push(`Document: "${d.title}"`);
    out.push(body || "(no readable text was extracted from this document)");
  }
  out.push("[/LIBRARY]");
  return out.join("\n");
}

/* ── The Extract: parse the model's JSON event list ── */
export function parseEvents(text) {
  if (!text) return [];
  let raw = text.replace(/```json/gi, "").replace(/```/g, "").trim();
  const start = raw.indexOf("[");
  const end = raw.lastIndexOf("]");
  if (start !== -1 && end !== -1) raw = raw.slice(start, end + 1);
  let parsed;
  try { parsed = JSON.parse(raw); } catch { return []; }
  if (!Array.isArray(parsed)) return [];
  return parsed.filter(e => {
    if (!e || typeof e !== "object") return false;
    if (e.op === "INS") return !!e.entity;
    if (e.op === "CON") return !!e.from && !!e.to;
    if (e.op === "DEF") return !!e.entity && !!e.field && e.value != null && String(e.value).trim() !== "";
    return false;
  });
}

/* Apply parsed events to a memory graph (mutates `memory`). Returns the
   count of events applied. */
export function applyEvents(memory, events) {
  let applied = 0;
  const ensure = (name, kind) => {
    const id = slug(name);
    if (!id) return null;
    if (!memory.entities[id]) {
      memory.entities[id] = { id, canonical: name, kind: kind || "thing", mentions: 0, created: Date.now() };
    }
    memory.entities[id].mentions = (memory.entities[id].mentions || 0) + 1;
    if (kind && memory.entities[id].kind === "thing") memory.entities[id].kind = kind;
    return id;
  };
  for (const e of events) {
    if (e.op === "INS") {
      if (ensure(e.entity, e.kind)) applied++;
    } else if (e.op === "CON") {
      const from = ensure(e.from);
      const to = ensure(e.to);
      const type = (e.type || "related to").trim();
      if (from && to) {
        memory.edges[`${from}::${type}::${to}`] = { from, to, type };
        applied++;
      }
    } else if (e.op === "DEF") {
      const id = ensure(e.entity);
      if (id) {
        memory.defs[`${id}::${e.field}`] = { entity: id, field: e.field, value: String(e.value ?? "") };
        applied++;
      }
    }
  }
  return applied;
}
