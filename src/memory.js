/* Per-chat memory — a graph-backed "Experience Engine".

   In Memory mode a chat does not send its growing conversation history to
   the model. Everything is in the graph instead:

   - The Given-Log is an append-only record of every message. User messages
     and model responses each get a content hash (g_xxxxxxxx) and an `agent`
     field ("user", "model:<name>", "system:ingest"). A model response also
     records the dossier projection that produced it (dossierHash) and the
     evidence spans that projection was built from — so you can always ask
     "what did the model see, and where did it come from?".

   - Entities are content-addressed: an id is e_<hash> of the canonical name,
     and a short state version (e_3a7f21b4@8f2c) tracks how the entity has
     evolved. DEFs and edges carry a `source` pointing at the Given-Log entry
     that produced them.

   - Three model calls run per turn: READ (user-facing), EXTRACT (background
     event extraction) and MUTATE (background, triggered only on ambiguity).
     MUTATE produces exactly one action — FORK, MERGE, CORRECT, RECLASSIFY or
     NONE — and its triggers are mechanical (zero tokens). */

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

/* The Ingest walk — read one passage of a document into the graph, with the
   current roster of sites in hand. May emit AMBIG when unsure of a match. */
export const INGEST_SYSTEM = `You convert text into a knowledge graph, passage by passage.
You are given the KNOWN SITES already in the graph and one PASSAGE.
Render the PASSAGE completely: parse every clause, not just the notable ones.
The operations you return, taken together, must reconstruct what it says.

A "site" is one entity — a person, organisation, place, role, policy, event,
or concept. Never make a site out of a whole sentence or clause. Canonical
names are short.

Resolve against KNOWN SITES: each is listed as e_<hash>@<state> with its name.
If the passage clearly refers to an existing site, reuse its e_<hash> id.
If you are UNSURE whether a name matches an existing site, do NOT guess —
emit an AMBIG op naming the closest candidate. Only create a site for
something genuinely new.

Return ONLY a JSON array of operations:
  {"op":"SIG","name":"<short name>","kind":"person|org|place|role|policy|event|concept|document|thing","hypothesis":"<1-2 sentences on what this site is>"}
  {"op":"DEF","id":"<e_hash or new name>","hypothesis":"<the site's full description, revised to fold in what this passage adds>"}
  {"op":"CON","from":"<e_hash or name>","to":"<e_hash or name>","relation":"<the predicate, a few words>","evidence":"<the clause this came from>"}
  {"op":"REC","id":"<e_hash>","canonical":"<corrected short name>","alias":"<the clumsy or wrong name to fold in>"}
  {"op":"AMBIG","name":"<the name in the passage>","candidate":"<e_hash of the closest existing site>","span":"<the clause it appeared in>"}

How to render a clause: its subject and object become sites (SIG if new), the
predicate becomes a CON between them with the clause as evidence. Descriptive
or attributive clauses become DEF revisions of a site's hypothesis. Resolve
pronouns to their site.

Use e_<hash> ids for known sites; use a short canonical name for new ones
(the system assigns the hash). Ignore navigation chrome, bylines and ads.
Return [] only if the passage is entirely boilerplate.`;

/* The Extract — read a completed conversation turn into the graph. The
   exchange is labelled with its Given-Log ids; every DEF must carry the
   `source` Given-Log entry that produced it. */
export const EXTRACT_SYSTEM = `You extract new knowledge from one conversation exchange into a knowledge graph.
You are given the KNOWN SITES already in the graph and one EXCHANGE. The
exchange is labelled with Given-Log ids: User [g_xxxxxxxx] and Model [g_xxxxxxxx].

Return only what the exchange newly establishes — do not restate the roster.

Return ONLY a JSON array of operations:
  {"op":"SIG","name":"<short name>","kind":"person|org|place|role|policy|event|concept|document|thing","hypothesis":"<1-2 sentences on what this site is>"}
  {"op":"DEF","id":"<e_hash or name>","field":"<attribute>","value":"<the fact>","source":"<g_id this came from>"}
  {"op":"CON","from":"<e_hash or name>","to":"<e_hash or name>","relation":"<the predicate>","evidence":"<the clause>","source":"<g_id this came from>"}

For KNOWN SITES (listed as e_<hash>@<state>) reference them by their e_<hash>
id. For genuinely new sites, give a short canonical name — the system assigns
the hash on creation. Every DEF must carry a "source" field set to the
Given-Log id (g_xxxxxxxx) of the message it came from.
Return [] if the exchange establishes nothing new.`;

/* The Mutate — invoked only when a mechanical trigger flags an ambiguity.
   Produces exactly one action with a reason. */
export const MUTATE_SYSTEM = `You resolve one ambiguity in a knowledge graph. You are given the detail of
one or two sites and the reason this was flagged. Decide the single best action.

Return ONLY one JSON object, exactly one of:
  {"action":"MERGE","target":"<e_hash to keep>","other":"<e_hash to fold in>","reason":"<why>"}
  {"action":"FORK","name":"<the ambiguous name>","from":"<e_hash it was confused with>","kind":"<kind>","reason":"<why>"}
  {"action":"CORRECT","target":"<e_hash>","canonical":"<corrected name>","reason":"<why>"}
  {"action":"RECLASSIFY","target":"<e_hash>","kind":"<corrected kind>","reason":"<why>"}
  {"action":"NONE","reason":"<why no change is needed>"}

MERGE when two sites are the same entity. FORK when one name has been conflated
with a different entity and needs its own site. CORRECT when a site's canonical
name is wrong. RECLASSIFY when a site's kind is wrong. NONE when the graph is
already correct. Keep the reason to one sentence.`;

/* ── Content hashing — sync, for content-addressed ids ── */

function fnv(str) {
  let h = 0x811c9dc5;
  const s = String(str || "");
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h >>> 0;
}
/* A second, differently-seeded pass so 8 hex digits carry real entropy. */
function hash8(str) {
  const a = fnv(str).toString(16).padStart(8, "0");
  const b = fnv("" + str).toString(16).padStart(8, "0");
  return (a + b).slice(0, 8);
}
function hash4(str) {
  return fnv(str).toString(16).padStart(8, "0").slice(0, 4);
}

const slug = (s) => String(s || "").trim().toLowerCase().replace(/\s+/g, " ");

/* Content-addressed id for an entity, from its canonical name. */
export const entityHash = (canonical) => "e_" + hash8(slug(canonical));

/* Short state version for an entity — recomputed on every meaningful change
   so the dossier and register can show e_3a7f21b4@8f2c. */
export const stateOf = (e) =>
  hash4([e.canonical, e.kind, e.hypothesis, (e.aliases || []).join("|")].join(""));

const restate = (e) => { e.state = stateOf(e); return e; };

/* ── The Given-Log — append-only record of every message ── */

/* Build a Given-Log entry. `agent` is "user", "model:<name>" or
   "system:ingest". Model entries may carry the dossierHash + spans they were
   produced from. */
export function makeGiven({ agent, text, turn = null, dossierHash = null, spans = null }) {
  const ts = Date.now();
  const id = "g_" + hash8([agent, text, ts, Math.random()].join(""));
  return {
    id, agent, text: String(text || ""), ts, turn,
    dossierHash: dossierHash || null,
    spans: spans || null,
  };
}

export function appendGiven(memory, entry) {
  memory.givenLog = memory.givenLog || [];
  memory.givenLog.push(entry);
  return entry;
}

/* ── Splitting text for the walk ── */

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
export const emptyMemory = () => ({
  schema: "graph-2",
  givenLog: [],
  entities: {},
  edges: {},
  defs: {},
  mutations: [],
  lastTurn: null,
});

export const cloneMemory = (m) => {
  if (!m) return emptyMemory();
  const entities = {};
  for (const [k, e] of Object.entries(m.entities || {})) {
    entities[k] = { ...e, aliases: [...(e.aliases || [])] };
  }
  const edges = {};
  for (const [k, g] of Object.entries(m.edges || {})) edges[k] = { ...g };
  const defs = {};
  for (const [k, d] of Object.entries(m.defs || {})) defs[k] = { ...d };
  return {
    schema: "graph-2",
    givenLog: (m.givenLog || []).map(g => ({ ...g })),
    entities, edges, defs,
    mutations: (m.mutations || []).map(x => ({ ...x })),
    lastTurn: m.lastTurn ? { ...m.lastTurn } : null,
  };
};

export const memoryStats = (m) => ({
  entities: m ? Object.keys(m.entities || {}).length : 0,
  edges: m ? Object.keys(m.edges || {}).length : 0,
  defs: m ? Object.keys(m.defs || {}).length : 0,
  given: m ? (m.givenLog || []).length : 0,
  pending: m ? (m.mutations || []).filter(x => x.status === "pending").length : 0,
});

/* Merge several graphs into one — a chat's own memory plus any opted-in
   library documents — before projecting the dossier. */
export function mergeMemory(...memories) {
  const out = { entities: {}, edges: {}, defs: {}, givenLog: [] };
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
    out.givenLog.push(...(m.givenLog || []));
  }
  return out;
}

/* ── The Signal: mechanical NER + keywords ── */

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

const TITLE_RE = /\b(?:mr|mrs|ms|dr|prof|professor|president|senator|governor|mayor|sir|lord|lady|king|queen|gen|general|capt|captain|rev)\.?$/i;
const PLACE_PREP = new Set(["in", "at", "to", "from", "near", "into", "across", "through", "toward", "towards"]);
const ORG_SUFFIX = /\b(?:inc|corp|ltd|llc|co|company|agency|department|ministry|bureau|institute|university|college|bank|group|association|committee|commission|council)\b\.?$/i;

/* Guess an entity kind for an NER name from its surrounding words. Returns
   "person" | "place" | "org" | null. Mechanical, zero tokens. */
function guessKind(message, name, index) {
  const before = message.slice(Math.max(0, index - 24), index).trim();
  const prevWord = (before.match(/(\S+)\s*$/) || [, ""])[1];
  if (TITLE_RE.test(prevWord)) return "person";
  if (PLACE_PREP.has(prevWord.toLowerCase())) return "place";
  if (ORG_SUFFIX.test(name)) return "org";
  return null;
}

function extractNER(message) {
  const seen = new Map(); // name → guessed kind
  const re = /\b[A-Z][a-zA-Z]+(?:\s+(?:of|the|and|de|van|von)\s+)?(?:\s*[A-Z][a-zA-Z]+)*\b/g;
  let m;
  while ((m = re.exec(message))) {
    const name = m[0].trim().replace(/\s+/g, " ");
    if (name.length < 2) continue;
    if (COMMON_CAPS.has(name.toLowerCase())) continue;
    if (!seen.has(name)) seen.set(name, guessKind(message, name, m.index));
  }
  return {
    names: [...seen.keys()],
    typed: Object.fromEntries(seen),
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

/* ── Resolving a textual reference to an entity ── */

/* Find an existing entity by e_<hash> id, canonical name, or alias. */
function findEntity(memory, ref) {
  const sites = memory.entities || {};
  if (!ref) return null;
  if (/^e_[0-9a-f]{8}$/i.test(ref) && sites[ref]) return sites[ref];
  const want = slug(ref);
  for (const e of Object.values(sites)) {
    if (slug(e.canonical) === want) return e;
    if ((e.aliases || []).some(a => slug(a) === want)) return e;
  }
  // a hash without state suffix that does not exist, or stripped @state
  const bare = String(ref).split("@")[0];
  if (sites[bare]) return sites[bare];
  return null;
}

/* ── The Roster: the sites already in the graph, handed to the walk ── */
export function buildRoster(memory, cap = 80) {
  const sites = Object.values(memory?.entities || {});
  if (!sites.length) return "KNOWN SITES: (none yet — this is the first passage)";
  const top = [...sites].sort((a, b) => (b.mentions || 0) - (a.mentions || 0)).slice(0, cap);
  const lines = top.map(s => {
    const h = (s.hypothesis || "").replace(/\s+/g, " ").trim();
    const snip = h.length > 140 ? h.slice(0, 140) + "…" : h;
    return `- ${s.id}@${s.state || stateOf(s)} "${s.canonical}" (${s.kind || "thing"})${snip ? ": " + snip : ""}`;
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
      const c = slug(e.canonical);
      if (c.includes(q) || q.includes(c)) add(e.id, 3);
      else if ((e.aliases || []).some(a => slug(a).includes(q) || q.includes(slug(a)))) add(e.id, 2);
    }
  }
  for (const kw of sig.keywords) {
    for (const e of Object.values(entities)) {
      if (slug(e.canonical).includes(kw) || (e.hypothesis || "").toLowerCase().includes(kw)) add(e.id, 1);
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
    const lines = [`${s.canonical} [${s.id}@${s.state || stateOf(s)}] (${s.kind || "thing"})`];
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

/* The Spans: the underlying evidence the dossier was projected from, gathered
   mechanically. Each span is a clause (or fact) plus the Given-Log id it came
   from — so a model response can be traced back past the hypotheses to source. */
export function collectSpans(sites, memory) {
  const edges = Object.values(memory?.edges || {});
  const defs = Object.values(memory?.defs || {});
  const ids = new Set(sites.map(s => s.id));
  const spans = [];
  for (const d of defs) {
    if (ids.has(d.entity) && d.value) {
      spans.push({ entity: d.entity, kind: "def", text: `${d.field}: ${d.value}`, source: d.source || null });
    }
  }
  for (const e of edges) {
    if ((ids.has(e.from) || ids.has(e.to)) && e.evidence) {
      spans.push({ entity: e.from, kind: "edge", text: e.evidence, source: e.source || null });
    }
  }
  return spans;
}

/* The dossier hash — a content hash of the projected context block, recorded
   on the model's Given-Log entry so the projection that produced a response
   is identifiable later. */
export const dossierHashOf = (dossier) => "d_" + hash8(dossier || "");

/* ── The Position Marker: one-turn memory, overwrites every turn ── */
export function buildPosition(lastTurn) {
  if (!lastTurn) return "";
  return `[POSITION]
Last sites: ${(lastTurn.entities || []).join(", ") || "none"}
Topic: ${lastTurn.topic || "none"}
Last user message: "${lastTurn.userMessage || ""}"
[/POSITION]`;
}

/* ── The Library card: the actual text of every document read into a chat ── */
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
    if (e.op === "SIG") return !!(e.name || e.canonical || e.id);
    if (e.op === "DEF") return !!(e.id || e.canonical || e.name) && !!(e.hypothesis || e.value);
    if (e.op === "CON") return !!e.from && !!e.to;
    if (e.op === "REC") return !!e.id && !!e.canonical;
    if (e.op === "AMBIG") return !!e.name;
    return false;
  });
}

/* Fold the site `fromId` into `intoId`: move its links and defs, keep the
   richer hypothesis, then drop it. */
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
  for (const [k, d] of Object.entries(memory.defs || {})) {
    if (d.entity === fromId) {
      delete memory.defs[k];
      memory.defs[`${intoId}::${d.field}`] = { ...d, entity: intoId };
    }
  }
  delete memory.entities[fromId];
  restate(dst);
}

/* Apply parsed walk operations to a graph (mutates `memory`). `opts.source`
   is the default Given-Log id to attribute new DEFs/edges to. Returns the
   count of operations applied and any AMBIG ops the model emitted. */
export function applyWalk(memory, ops, opts = {}) {
  let applied = 0;
  const ambigs = [];
  const sites = memory.entities;
  memory.defs = memory.defs || {};
  const defaultSource = opts.source || null;

  /* Resolve a reference to an entity id, creating it if new. */
  const ensure = (ref, canonical, kind) => {
    const found = findEntity(memory, ref) || (canonical ? findEntity(memory, canonical) : null);
    let site = found;
    if (!site) {
      const name = (canonical || (/^e_[0-9a-f]{8}$/i.test(ref) ? "" : ref) || "").trim();
      if (!name) return null;
      const id = entityHash(name);
      if (sites[id]) {
        site = sites[id];
      } else {
        site = sites[id] = {
          id, state: "", canonical: name, kind: kind || "thing",
          hypothesis: "", aliases: [], mentions: 0,
          created: Date.now(), source: defaultSource,
        };
      }
    }
    site.mentions = (site.mentions || 0) + 1;
    if (kind && kind !== "thing" && (!site.kind || site.kind === "thing")) site.kind = kind;
    restate(site);
    return site.id;
  };

  for (const e of ops) {
    if (e.op === "AMBIG") {
      ambigs.push({
        name: String(e.name || "").trim(),
        candidate: String(e.candidate || "").trim(),
        span: String(e.span || "").trim(),
      });
    } else if (e.op === "SIG") {
      const id = ensure(e.id || e.name || e.canonical, e.name || e.canonical, e.kind);
      if (id) {
        const h = String(e.hypothesis || "").trim();
        if (h && h.length > (sites[id].hypothesis || "").length) sites[id].hypothesis = h;
        restate(sites[id]);
        applied++;
      }
    } else if (e.op === "DEF") {
      const id = ensure(e.id || e.name || e.canonical, e.name || e.canonical);
      if (id) {
        if (e.hypothesis) {
          sites[id].hypothesis = String(e.hypothesis).trim();
          restate(sites[id]);
        }
        if (e.field && e.value) {
          memory.defs[`${id}::${slug(e.field)}`] = {
            entity: id, field: String(e.field).trim(), value: String(e.value).trim(),
            source: e.source || defaultSource, created: Date.now(),
          };
        }
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
          source: e.source || defaultSource,
        };
        applied++;
      }
    } else if (e.op === "REC") {
      const site = findEntity(memory, e.id);
      if (site) {
        const oldCanon = site.canonical;
        site.canonical = String(e.canonical).trim();
        site.aliases = site.aliases || [];
        const aliasName = e.alias || oldCanon;
        if (aliasName && !site.aliases.includes(aliasName)) site.aliases.push(aliasName);
        const dup = aliasName ? findEntity(memory, aliasName) : null;
        if (dup && dup.id !== site.id) mergeSite(memory, dup.id, site.id);
        restate(site);
        applied++;
      }
    }
  }
  return { applied, ambigs };
}

/* ── MUTATE: triggers, prompt, parse, apply ── */

const REF_UNCERTAIN_RE = [
  /\b(?:may|might|could|possibly|perhaps|not necessarily|unsure)\b[^.?!]*?\b(e_[0-9a-f]{8})\b/i,
  /\b(e_[0-9a-f]{8})\b[^.?!]*?\b(?:may not|might not|could be different|not the same|not necessarily)\b/i,
];
const CORRECTION_RE = /\b(?:that(?:'s| is) not the same|different (?:person|place|thing|one|entity)|you(?:'re| are) (?:confusing|mixing)|not who i meant|i meant (?:a )?different|wrong (?:person|entity|one)|mixed (?:them|it) up)\b/i;

/* Detect ambiguities mechanically — zero tokens. Returns a list of triggers,
   each of which fires one MUTATE call. */
export function detectMutationTriggers({ memory, userMessage = "", modelResponse = "", ambigs = [] }) {
  const triggers = [];

  // Path 1: the model's response references an entity hash with uncertainty.
  for (const re of REF_UNCERTAIN_RE) {
    const m = re.exec(modelResponse || "");
    if (m && memory.entities[m[1]]) {
      triggers.push({ kind: "ref-uncertainty", target: m[1], detail: m[0].trim() });
    }
  }

  // Path 2: the ingest walk emitted an AMBIG op.
  for (const a of ambigs || []) {
    triggers.push({ kind: "ambig", target: a.candidate, name: a.name, detail: a.span || a.name });
  }

  // Path 3: NER tagged a name with a kind that conflicts with the matched
  // entity's stored kind.
  const sig = signal(userMessage || "");
  for (const name of sig.ner.names) {
    const guessed = sig.ner.typed[name];
    if (!guessed) continue;
    const ent = findEntity(memory, name);
    if (ent && ent.kind && ent.kind !== "thing" && ent.kind !== guessed) {
      triggers.push({
        kind: "type-mismatch", target: ent.id, name,
        detail: `"${name}" reads as ${guessed} but ${ent.id} is recorded as ${ent.kind}`,
      });
    }
  }

  // Path 4: the user signalled a correction.
  if (CORRECTION_RE.test(userMessage || "")) {
    triggers.push({ kind: "correction", target: null, detail: userMessage.slice(0, 200) });
  }

  return triggers;
}

/* Full detail of an entity, for the MUTATE prompt. */
function entityDetail(memory, id) {
  const e = memory.entities[id];
  if (!e) return null;
  const lines = [`${e.canonical} [${e.id}@${e.state || stateOf(e)}] (${e.kind || "thing"})`];
  if (e.aliases?.length) lines.push(`  aliases: ${e.aliases.join(", ")}`);
  if (e.hypothesis) lines.push(`  hypothesis: ${e.hypothesis}`);
  for (const d of Object.values(memory.defs || {})) {
    if (d.entity === id && d.value) lines.push(`  ${d.field}: ${d.value}`);
  }
  for (const g of Object.values(memory.edges || {})) {
    if (g.from === id) {
      const to = memory.entities[g.to];
      lines.push(`  ${g.relation || g.type} → ${to ? to.canonical : g.to}`);
    }
  }
  return lines.join("\n");
}

/* Build the MUTATE user message for one trigger. */
export function buildMutateUser(memory, trigger) {
  const parts = [`FLAGGED: ${trigger.detail}`, `REASON: ${trigger.kind}`];
  const seen = new Set();
  const addSite = (id) => {
    if (!id || seen.has(id)) return;
    const d = entityDetail(memory, id);
    if (d) { parts.push("", "SITE:", d); seen.add(id); }
  };
  addSite(trigger.target);
  if (trigger.name) {
    const other = findEntity(memory, trigger.name);
    if (other) addSite(other.id);
    else parts.push("", `The name "${trigger.name}" is not yet its own site.`);
  }
  if (trigger.kind === "type-mismatch" && trigger.name) {
    parts.push("", `The user's message reads "${trigger.name}" as a different kind.`);
  }
  return parts.join("\n");
}

/* Parse a MUTATE response into a single action. */
export function parseMutate(text) {
  if (!text) return null;
  let raw = text.replace(/```json/gi, "").replace(/```/g, "").trim();
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start !== -1 && end !== -1) raw = raw.slice(start, end + 1);
  let obj;
  try { obj = JSON.parse(raw); } catch { return null; }
  if (!obj || typeof obj !== "object") return null;
  const action = String(obj.action || "").toUpperCase();
  if (!["FORK", "MERGE", "CORRECT", "RECLASSIFY", "NONE"].includes(action)) return null;
  return {
    action,
    target: obj.target ? String(obj.target).split("@")[0] : null,
    other: obj.other ? String(obj.other).split("@")[0] : null,
    name: obj.name ? String(obj.name).trim() : null,
    from: obj.from ? String(obj.from).split("@")[0] : null,
    canonical: obj.canonical ? String(obj.canonical).trim() : null,
    kind: obj.kind ? String(obj.kind).trim() : null,
    reason: String(obj.reason || "").trim(),
  };
}

/* Build a mutation record from a parsed action. The auto-commit tier for
   MUTATE is PROMPT — every real action lands as `pending` and needs consent.
   NONE is logged but never surfaced. */
export function makeMutation(parsed, { trigger, msgId = null } = {}) {
  return {
    id: "m_" + hash8([parsed.action, parsed.reason, Date.now(), Math.random()].join("")),
    action: parsed.action,
    target: parsed.target,
    other: parsed.other,
    name: parsed.name,
    from: parsed.from,
    canonical: parsed.canonical,
    kind: parsed.kind,
    reason: parsed.reason,
    trigger: trigger ? trigger.kind : null,
    triggerDetail: trigger ? trigger.detail : null,
    msgId,
    ts: Date.now(),
    status: parsed.action === "NONE" ? "noop" : "pending",
  };
}

/* Apply an accepted mutation to the graph (mutates `memory`). */
export function applyMutation(memory, mut) {
  if (!mut) return false;
  if (mut.action === "MERGE" && mut.target && mut.other) {
    if (memory.entities[mut.target] && memory.entities[mut.other]) {
      mergeSite(memory, mut.other, mut.target);
      return true;
    }
  } else if (mut.action === "RECLASSIFY" && mut.target && mut.kind) {
    const e = memory.entities[mut.target];
    if (e) { e.kind = mut.kind; restate(e); return true; }
  } else if (mut.action === "CORRECT" && mut.target && mut.canonical) {
    const e = memory.entities[mut.target];
    if (e) {
      if (e.canonical && !e.aliases.includes(e.canonical)) e.aliases.push(e.canonical);
      e.canonical = mut.canonical;
      restate(e);
      return true;
    }
  } else if (mut.action === "FORK" && mut.name) {
    // Give the conflated name its own distinct site.
    let id = entityHash(mut.name);
    if (memory.entities[id]) id = entityHash(mut.name + "·" + mut.id);
    memory.entities[id] = {
      id, state: "", canonical: mut.name, kind: mut.kind || "thing",
      hypothesis: "", aliases: [], mentions: 1,
      created: Date.now(), source: null, forkedFrom: mut.from || null,
    };
    restate(memory.entities[id]);
    return true;
  }
  return false;
}
