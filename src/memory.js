/* Per-chat memory — a lightweight "Experience Engine".

   In Memory mode a chat does not send its growing conversation history to
   the model. Instead, knowledge-bearing turns are distilled into a small
   per-chat knowledge graph (entities, connections, definitions) kept in
   localStorage alongside the conversation. Each turn rebuilds a fixed-size
   prompt: system + projected dossier + one-turn position marker. The prompt
   stays the same size at turn 1 and turn 10,000.

   This is a dependency-free adaptation of the architecture in experience.md:
   NER is mechanical (capitalised-run + date/number matching) rather than
   compromise.js, and the persistence layer is localStorage rather than
   Matrix, since the regular chat is a local-only, single-device app. */

export const MEMORY_SYSTEM = `You are a helpful assistant with long-term memory.
The [CONTEXT] block holds facts remembered from earlier in this conversation.
The [POSITION] block notes what was discussed on the previous turn.
Use them to stay consistent and personalised across the chat. If the context
is empty or does not cover something, answer normally from your own knowledge —
do not claim you have no memory. Keep responses concise unless asked for detail.`;

export const CASUAL_SYSTEM = `You are a helpful assistant. Reply naturally and concisely.`;

export const EXTRACT_SYSTEM = `Extract new, durable facts from this exchange as JSON.
Return a JSON array of events. Each event is one of:
  { "op": "INS", "entity": "<name>", "kind": "person|place|org|thing|event|topic" }
  { "op": "CON", "from": "<entity>", "to": "<entity>", "type": "<relation>" }
  { "op": "DEF", "entity": "<name>", "field": "<attribute>", "value": "<value>" }
Only include facts worth remembering long-term: names, relationships, attributes,
preferences, and decisions. Ignore small talk and transient details.
Return [] if there is nothing worth remembering.
Return ONLY the JSON array — no markdown, no commentary.`;

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

/* ── The Gate: is this turn knowledge-bearing? ── */
export function isKnowledgeBearing(message) {
  const ner = extractNER(message);
  const isQuestion = message.includes("?");
  const words = message.trim().split(/\s+/).filter(Boolean);
  const isSubstantive = words.length > 4;
  return ner.names.length > 0
    || ner.dates.length > 0
    || isQuestion
    || (isSubstantive && ner.numbers.length > 0);
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
    if (e.op === "DEF") return !!e.entity && !!e.field;
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
