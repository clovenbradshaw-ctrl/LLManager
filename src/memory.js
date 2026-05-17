/* Per-chat memory — a lightweight "Experience Engine".

   In Memory mode a chat does not send its growing conversation history to
   the model. Instead, text is read into a knowledge graph of sites (entities)
   and links (relations), kept in localStorage. Each turn rebuilds a fixed-size
   prompt: system + projected dossier + library + one-turn position marker.

   Reading is a stateful "walk": text is split into sentences, grouped into
   passages, and each passage is read by the model with the roster of sites
   already found in hand — so it resolves and enriches existing sites instead
   of duplicating them. A site carries a running `hypothesis` (a short prose
   description that gets revised), not scattered field/value pairs. The intent
   is to render the whole text as a machine-readable graph, clause by clause. */

export const MEMORY_SYSTEM = `You are a helpful assistant with long-term memory.
The [CONTEXT] block holds sites (people, organisations, places, etc.) recalled
from this chat, each with a short description.
The [LIBRARY] block contains the full text of documents read into this chat —
treat them as source material you have already read and can quote directly.
The [POSITION] block notes what was discussed on the previous turn.
Use them to answer accurately and stay consistent. If they do not cover
something, answer normally from your own knowledge — never claim you have no
memory or that you cannot see a document. Keep responses concise unless asked
for detail.`;

/* The Walk — read one passage into the graph, with the current roster of
   sites in hand. Used both for documents and for chat exchanges. */
export const WALK_SYSTEM = `You convert text into a knowledge graph, passage by passage.
You are given the KNOWN SITES already in the graph and one PASSAGE.
Render the PASSAGE completely: parse every clause, not just the notable ones.
The operations you return, taken together, must reconstruct what it says.

A "site" is one entity — a person, organisation, place, role, policy, event,
or concept. Never make a site out of a whole sentence or clause. Canonical
names are short.

Resolve against KNOWN SITES: if the passage refers to an existing site — by
full name, partial name, title, pronoun, or description — reuse that site's id.
Only create a site for something genuinely new.

Return ONLY a JSON array of operations:
  {"op":"SIG","id":"<slug>","canonical":"<short name>","kind":"person|org|place|role|policy|event|concept|document|thing","hypothesis":"<1-2 sentences on what this site is>"}
  {"op":"DEF","id":"<existing slug>","hypothesis":"<the site's full description, revised to fold in what this passage adds>"}
  {"op":"CON","from":"<slug>","to":"<slug>","relation":"<the predicate, a few words>","evidence":"<the clause this came from>"}
  {"op":"REC","id":"<existing slug>","canonical":"<corrected short name>","alias":"<the clumsy or wrong name to fold in>"}

How to render a clause: its subject and object become sites (SIG if new), the
predicate becomes a CON between them with the clause as evidence. Descriptive
or attributive clauses become DEF revisions of a site's hypothesis. Resolve
pronouns to their site.

ids are lowercase slugs of the canonical name (e.g. "michael perry").
Use REC when an earlier site was created under a clumsy name or a clause.
Ignore only navigation chrome, byline boilerplate, and ads.
Return [] only if the passage is entirely boilerplate.`;

/* JSON schema for Ollama's `format` parameter — constrained decoding for the
   walk. With this in hand the model cannot emit malformed JSON, so even a
   small model is structurally reliable; `parseWalk` still validates each
   operation's required fields by `op`. Property names mirror the WALK_SYSTEM
   operation shapes; only `op` is required since the rest vary per operation. */
export const WALK_FORMAT = {
  type: "array",
  items: {
    type: "object",
    properties: {
      op: { type: "string", enum: ["SIG", "DEF", "CON", "REC"] },
      id: { type: "string" },
      canonical: { type: "string" },
      kind: {
        type: "string",
        enum: ["person", "org", "place", "role", "policy", "event", "concept", "document", "thing"],
      },
      hypothesis: { type: "string" },
      from: { type: "string" },
      to: { type: "string" },
      relation: { type: "string" },
      evidence: { type: "string" },
      alias: { type: "string" },
    },
    required: ["op"],
  },
};

/* ── Splitting text for the walk ── */

/* Split text into sentences. Overlong runs (no punctuation) are hard-split;
   tiny fragments are merged back into the preceding sentence. */
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

/* Group sentences into passages of a few sentences each. The walk reads one
   passage per model call, carrying the roster of sites between passages. */
export function batchSentences(sentences, maxChars = 1400, maxCount = 5) {
  const batches = [];
  let buf = [], len = 0;
  for (const s of sentences) {
    if (buf.length && (len + s.length > maxChars || buf.length >= maxCount)) {
      batches.push(buf.join(" ")); buf = []; len = 0;
    }
    buf.push(s); len += s.length + 1;
  }
  if (buf.length) batches.push(buf.join(" "));
  return batches;
}

/* ── Empty / clone / merge helpers ── */
export const emptyMemory = () => ({ entities: {}, edges: {}, defs: {}, lastTurn: null });

export const cloneMemory = (m) => {
  if (!m) return emptyMemory();
  const entities = {};
  for (const [k, e] of Object.entries(m.entities || {})) {
    entities[k] = { ...e, aliases: [...(e.aliases || [])] };
  }
  const edges = {};
  for (const [k, g] of Object.entries(m.edges || {})) edges[k] = { ...g };
  return { entities, edges, defs: { ...(m.defs || {}) }, lastTurn: m.lastTurn ? { ...m.lastTurn } : null };
};

export const memoryStats = (m) => ({
  entities: m ? Object.keys(m.entities || {}).length : 0,
  edges: m ? Object.keys(m.edges || {}).length : 0,
  defs: m ? Object.keys(m.defs || {}).length : 0,
});

/* Merge several graphs into one — a chat's own memory plus any opted-in
   library documents — before projecting the dossier. */
export function mergeMemory(...memories) {
  const out = { entities: {}, edges: {}, defs: {} };
  for (const m of memories) {
    if (!m) continue;
    for (const [id, e] of Object.entries(m.entities || {})) {
      if (!out.entities[id]) {
        out.entities[id] = { ...e, aliases: [...(e.aliases || [])] };
      } else {
        const o = out.entities[id];
        o.mentions = (o.mentions || 0) + (e.mentions || 0);
        if ((e.hypothesis || "").length > (o.hypothesis || "").length) o.hypothesis = e.hypothesis;
      }
    }
    Object.assign(out.edges, m.edges || {});
    Object.assign(out.defs, m.defs || {});
  }
  return out;
}

const slug = (s) => String(s || "").trim().toLowerCase().replace(/\s+/g, " ");

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

/* ── The Roster: the sites already in the graph, handed to the walk so it
   resolves against them instead of duplicating. ── */
export function buildRoster(memory, cap = 80) {
  const sites = Object.values(memory?.entities || {});
  if (!sites.length) return "KNOWN SITES: (none yet — this is the first passage)";
  const top = [...sites].sort((a, b) => (b.mentions || 0) - (a.mentions || 0)).slice(0, cap);
  const lines = top.map(s => {
    const h = (s.hypothesis || "").replace(/\s+/g, " ").trim();
    const snip = h.length > 140 ? h.slice(0, 140) + "…" : h;
    return `- ${s.id} (${s.kind || "thing"})${snip ? ": " + snip : ""}`;
  });
  return "KNOWN SITES:\n" + lines.join("\n");
}

/* ── The Reach: search the graph using Signal output ── */
export function reach(sig, memory) {
  const entities = memory?.entities || {};
  const edges = memory?.edges || {};
  const cand = new Map();
  const add = (id, score) => cand.set(id, (cand.get(id) || 0) + score);

  for (const name of sig.ner.names) {
    const q = slug(name);
    for (const e of Object.values(entities)) {
      if (e.id.includes(q) || q.includes(e.id)) add(e.id, 3);
    }
  }
  for (const kw of sig.keywords) {
    for (const e of Object.values(entities)) {
      if (e.id.includes(kw) || (e.hypothesis || "").toLowerCase().includes(kw)) add(e.id, 1);
    }
  }

  // 2-hop expansion along links
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

/* ── The Dossier: format the reached sites into a compact context block ── */
export function buildDossier(sites, memory) {
  if (!sites.length) {
    return "[CONTEXT]\n(nothing recalled yet)\n[/CONTEXT]";
  }
  const edges = Object.values(memory?.edges || {});
  const defs = Object.values(memory?.defs || {});
  const block = sites.map(s => {
    const lines = [`${s.canonical} (${s.kind || "thing"})`];
    if (s.hypothesis) lines.push("  " + s.hypothesis.replace(/\s+/g, " ").trim());
    for (const d of defs.filter(d => d.entity === s.id && d.value)) {
      lines.push(`  ${d.field}: ${d.value}`);
    }
    for (const edge of edges.filter(g => g.from === s.id)) {
      const to = memory.entities[edge.to];
      lines.push(`  ${edge.relation || edge.type} → ${to ? to.canonical : edge.to}`);
    }
    return lines.join("\n");
  }).join("\n");
  return `[CONTEXT]\n${block}\n[/CONTEXT]`;
}

/* ── The Position Marker: one-turn memory, overwrites every turn ── */
export function buildPosition(lastTurn) {
  if (!lastTurn) return "";
  return `[POSITION]
Last sites: ${(lastTurn.entities || []).join(", ") || "none"}
Topic: ${lastTurn.topic || "none"}
Last user message: "${lastTurn.userMessage || ""}"
[/POSITION]`;
}

/* ── The Library card: the actual text of every document read into a chat,
   so the model is prompted with the source material itself. Capped per
   document; fixed-size relative to the conversation. ── */
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

/* ── The Walk: parse and apply the model's operation list ── */
export function parseWalk(text) {
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
    if (e.op === "SIG") return !!(e.canonical || e.id);
    if (e.op === "DEF") return !!(e.id || e.canonical) && !!e.hypothesis;
    if (e.op === "CON") return !!e.from && !!e.to;
    if (e.op === "REC") return !!e.id && !!e.canonical;
    return false;
  });
}

/* Fold the site `fromId` into `intoId`: move its links, keep the richer
   hypothesis, then drop it. */
function mergeSite(memory, fromId, intoId) {
  const src = memory.entities[fromId], dst = memory.entities[intoId];
  if (!src || !dst || fromId === intoId) return;
  dst.mentions = (dst.mentions || 0) + (src.mentions || 0);
  if ((src.hypothesis || "").length > (dst.hypothesis || "").length) dst.hypothesis = src.hypothesis;
  dst.aliases = dst.aliases || [];
  if (src.canonical && !dst.aliases.includes(src.canonical)) dst.aliases.push(src.canonical);
  for (const a of src.aliases || []) if (!dst.aliases.includes(a)) dst.aliases.push(a);
  for (const [k, edge] of Object.entries(memory.edges)) {
    const from = edge.from === fromId ? intoId : edge.from;
    const to = edge.to === fromId ? intoId : edge.to;
    if (from !== edge.from || to !== edge.to) {
      delete memory.edges[k];
      if (from !== to) {
        memory.edges[`${from}::${edge.relation || edge.type}::${to}`] = { ...edge, from, to };
      }
    }
  }
  delete memory.entities[fromId];
}

/* Apply parsed walk operations to a graph (mutates `memory`). Returns the
   count of operations applied. */
export function applyWalk(memory, ops) {
  let applied = 0;
  const sites = memory.entities;

  /* Resolve a reference to a site id, creating the site if it is new.
     Checks aliases so a reference to a folded-in name still resolves. */
  const ensure = (rawId, canonical, kind) => {
    let id = slug(rawId || canonical);
    if (!id) return null;
    if (!sites[id]) {
      for (const s of Object.values(sites)) {
        if ((s.aliases || []).some(a => slug(a) === id)) { id = s.id; break; }
      }
    }
    if (!sites[id]) {
      sites[id] = {
        id, canonical: (canonical || rawId || id).trim(), kind: kind || "thing",
        hypothesis: "", aliases: [], mentions: 0, created: Date.now(),
      };
    }
    const site = sites[id];
    site.mentions = (site.mentions || 0) + 1;
    if (kind && kind !== "thing" && (!site.kind || site.kind === "thing")) site.kind = kind;
    return id;
  };

  for (const e of ops) {
    if (e.op === "SIG") {
      const id = ensure(e.id, e.canonical, e.kind);
      if (id) {
        const h = String(e.hypothesis || "").trim();
        if (h && h.length > (sites[id].hypothesis || "").length) sites[id].hypothesis = h;
        applied++;
      }
    } else if (e.op === "DEF") {
      const id = ensure(e.id, e.canonical);
      if (id) {
        sites[id].hypothesis = String(e.hypothesis).trim();
        applied++;
      }
    } else if (e.op === "CON") {
      const from = ensure(e.from);
      const to = ensure(e.to);
      const relation = String(e.relation || e.type || "related to").trim();
      if (from && to && from !== to) {
        memory.edges[`${from}::${relation}::${to}`] = {
          from, to, relation, type: relation,
          evidence: e.evidence ? String(e.evidence).trim() : undefined,
        };
        applied++;
      }
    } else if (e.op === "REC") {
      const id = slug(e.id);
      if (sites[id]) {
        const oldCanon = sites[id].canonical;
        sites[id].canonical = String(e.canonical).trim();
        sites[id].aliases = sites[id].aliases || [];
        const aliasName = e.alias || oldCanon;
        if (aliasName && !sites[id].aliases.includes(aliasName)) sites[id].aliases.push(aliasName);
        const aliasId = slug(aliasName);
        if (aliasId && aliasId !== id && sites[aliasId]) mergeSite(memory, aliasId, id);
        applied++;
      }
    }
  }
  return applied;
}
