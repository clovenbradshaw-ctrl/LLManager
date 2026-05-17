/* v3 prompts, builders and graph operations.

   Memory mode runs on the situated graph (see graph.js). A chat does not
   send its growing history to the model; every turn is projected into a
   fixed-size prompt and read back into the graph.

   Three model calls per knowledge-bearing turn:
     READ    — user-facing, answers from the projected [CTX]
     EXTRACT — background, reads the exchange into the graph as INS/CON/DEF/EVA
     MUTATE  — background, fires only when a mechanical trigger flags an
               ambiguity; produces one FORK/MERGE/CORRECT/RECLASSIFY/NONE

   SIG is mechanical only — the NER/keyword scan that produces retrieval
   candidates before the graph is touched. It is never stored. INS creates
   permanent identity; kind is carried as a revisable DEF; EVA records
   judgments; REC is reserved for MUTATE. */

import { Graph, mintGivenId } from "./graph.js";

export const INTERVALS = {
  ENTITY_HYPOTHESIS: 1,
  GROUP_HYPOTHESIS: 4,
  SECTION_HYPOTHESIS: 12,
  DOCUMENT_HYPOTHESIS: Infinity,
  SESSION_HYPOTHESIS: Infinity,
  CORPUS_HYPOTHESIS: 5,
};

/* ═══ Prompts: READ (user-facing) ═══ */

export const READ_SYSTEM = `You are an interpreter reading a situated knowledge graph.
Entity IDs are content hashes (e_xxxxxxxx). State versions (@xxxx) track changes.
Your responses and the user's messages are both recorded in the graph.

Answer using ONLY the [CTX] block. Say "I don't have that" if insufficient.
Do NOT use outside knowledge.

If an entity reference is ambiguous — a name that could refer to more than one
thing in the graph — say so plainly. Example: "This 'Hardy' may not be the same
as e_3a7f21b4 (Tom Hardy the actor)." The system will handle the resolution.

Reading [CTX]:
  E: hash@state | terrain | edges
  ~: canonical name, aka aliases
  h: current hypothesis
  →←: connection (type) target_hash
  =: field = value
  @: "verbatim source span"
  ⚠: unresolved conflict

Reading [POS]:
  prev: entity hashes from last turn
  topic: what we were discussing
  last: user's previous message`;

export const READ_CASUAL = `You are a helpful assistant. Be concise and natural.
Your messages are recorded in a knowledge graph.`;

/* ═══ Prompt: EXTRACT (background) ═══ */

export const EXTRACT_SYSTEM = `Extract new knowledge from this exchange as a JSON array.
Both the user's message and the model's response are Given-Log entries (IDs provided).
Return [] if nothing new. ONLY valid JSON, no markdown.

Event types:
{"op":"INS","entity":"<canonical>","terrain":"<T>"}
{"op":"CON","from":"<hash>","to":"<hash>","type":"<verb>"}
{"op":"DEF","entity":"<hash>","field":"<attr>","value":"<val>","source":"<given_id>"}
{"op":"EVA","entity":"<hash>","claim":"<field=value>","status":"holds|fails|contested","source":"<given_id>"}

Rules:
- Reference existing entities by their hash (from the register).
- For NEW entities, use "entity" with the canonical name. The system assigns the hash.
- INS creates identity + terrain only. Attributes go in DEF.
- The entity's kind (person, organization, place...) is a DEF: field "kind".
- "source" on DEF/EVA = the Given-Log ID of the message that produced this knowledge.
- A DEF is a fact. An EVA is a judgment (does a claim hold, fail, or is contested?).
- Skip greetings, filler, restatements of existing context.
- Entity names: lowercase-hyphenated for new canonical names.

Terrains: Entity, Network, Paradigm, Void, Kind, Field, Link, Atmosphere, Lens`;

/* ═══ Prompt: MUTATE (background, on ambiguity) ═══ */

export const MUTATE_SYSTEM = `You are resolving a graph ambiguity. Examine the evidence and
produce exactly ONE action as JSON. ONLY valid JSON, no markdown.

Actions:

FORK — one entity is actually two:
{"action":"FORK","source":"<hash>","new_canonical":"<name>","reason":"<why>",
 "reassign":[{"def_id":"<id>","reason":"<why this DEF belongs to the new entity>"}]}

MERGE — two entities are actually one:
{"action":"MERGE","keep":"<hash>","absorb":"<hash>","reason":"<why>",
 "new_aliases":["<alias1>"]}

CORRECT — a DEF is wrong:
{"action":"CORRECT","entity":"<hash>","field":"<field>",
 "old_value":"<wrong>","new_value":"<right>","reason":"<why>","source":"<given_id>"}

RECLASSIFY — terrain assignment is wrong:
{"action":"RECLASSIFY","entity":"<hash>","old_terrain":"<T>","new_terrain":"<T>","reason":"<why>"}

NONE — no action needed:
{"action":"NONE","reason":"<why the ambiguity is not real>"}

Always include "reason". The action is logged and must be auditable.`;

/* ═══ Prompt: INGEST (document walk, per passage) ═══ */

export const INGEST_SYSTEM = `Extract entities and claims from this passage as a JSON array.
Return [] if nothing extractable. ONLY valid JSON, no markdown.

Entity hashes from the register are e_xxxxxxxx. Reference them for known entities.
For NEW entities use canonical name; the system assigns the hash.

Event types:
{"op":"INS","entity":"<canonical>","terrain":"<T>"}
{"op":"CON","from":"<hash_or_name>","to":"<hash_or_name>","type":"<verb>"}
{"op":"DEF","entity":"<hash_or_name>","field":"<attr>","value":"<val>","span":"<exact words>"}
{"op":"EVA","entity":"<hash_or_name>","claim":"<claim>","status":"holds|fails|contested","span":"<exact words>"}

Example — "Smith, who left Boeing in 2019, now advises the Pentagon on drone policy."
Register has: e_a1b2c3d4 (Boeing) | Network
[
{"op":"INS","entity":"smith","terrain":"Entity"},
{"op":"DEF","entity":"smith","field":"kind","value":"person","span":"Smith"},
{"op":"DEF","entity":"smith","field":"former_employer","value":"Boeing, left 2019","span":"left Boeing in 2019"},
{"op":"CON","from":"smith","to":"e_a1b2c3d4","type":"former_employee"},
{"op":"DEF","entity":"smith","field":"advisory_role","value":"drone policy","span":"advises the Pentagon on drone policy"}
]

Rules:
- Do NOT re-INS entities from the register. Reference them by hash.
- "span" = EXACT words from the passage.
- INS only mints identity + terrain. All attributes go in DEF, including kind.
- Extract ALL entities, connections, claims. Multiple events per passage is normal.
- Rhetoric IS data. Author judgments are EVA events.
- If a name might refer to an existing entity but you are uncertain, flag it:
  {"op":"AMBIG","name":"<name>","candidate":"<hash>","span":"<exact words>"}
  The system will trigger a MUTATE call to resolve it.

Terrains: Entity, Network, Paradigm, Void, Kind, Field, Link, Atmosphere, Lens`;

/* ═══ Prompts: Hypothesis (one per level, strict nesting) ═══ */

export const HYPOTHESIS_ENTITY = `Write a one-sentence hypothesis for what this entity is about.
Under 150 characters. Specific enough that future evidence could revise it.
Your prior hypotheses and their triggers are shown below.
Build on the trajectory of understanding. Do not restart from scratch.
Write ONLY the sentence.`;

export const HYPOTHESIS_GROUP = `Write a one-sentence hypothesis for what this passage group is about.
You see ONLY entity hypotheses — not their underlying facts.
Under 150 characters. Capture the thread connecting them.
Prior group hypotheses show how the document is developing.
Write ONLY the sentence.`;

export const HYPOTHESIS_SECTION = `Write a one-sentence hypothesis for what this section is about.
You see ONLY group hypotheses — not entity detail.
Under 150 characters. Capture the argument or narrative arc.
Prior section hypotheses show the document's shape so far.
Write ONLY the sentence.`;

export const HYPOTHESIS_DOCUMENT = `Write a one-sentence hypothesis for what this document is about.
You see ONLY section hypotheses — not group or entity detail.
Under 200 characters. Capture the central finding or argument.
Prior document hypotheses in this corpus situate this document.
Write ONLY the sentence.`;

export const HYPOTHESIS_SESSION = `Write a one-sentence hypothesis for what this conversation was about.
Focus on the investigative thread: what was asked, discovered, shifted.
Under 200 characters.
Write ONLY the sentence.`;

export const HYPOTHESIS_CORPUS = `Write a one-sentence hypothesis for what this body of work is about.
You see ONLY document and session hypotheses.
Under 200 characters. Capture the overarching inquiry.
Write ONLY the sentence.`;

/* ═══ Prompts: Write mode ═══ */

export const WRITE_OUTLINE = `Create a document outline from the material below.
Return a JSON array:
[{"section":1,"topic":"...","entities":["<hash>"],"move":"INS|CON|DEF|EVA"}]

"move" = rhetorical operation:
  INS = introduce something new to the reader
  CON = reveal a connection between known things
  DEF = establish and support a claim
  EVA = assess whether a claim holds

Order: introduce before connect, connect before claim, claim before evaluate.
Return ONLY the JSON array.`;

export const WRITE_SECTION = `Write this section using ONLY the [CTX] block.
Ground every claim in context. Do not editorialize beyond evidence.
2-4 paragraphs.
[READER] shows entities already introduced. Do not re-introduce them.`;

/* ═══ Given-Log builders ═══ */

export function logUserMessage(text, signalOut, sessionId, turnNumber) {
  return {
    id: mintGivenId("user", text, Date.now()),
    type: "eo.given", agent: "user", mode: "conversation",
    text, ner: signalOut?.ner || null,
    session: sessionId, turn: turnNumber, timestamp: Date.now(),
  };
}

export function logModelResponse(text, model, dossierHash, sessionId, turnNumber) {
  return {
    id: mintGivenId("model", text, Date.now()),
    type: "eo.given", agent: `model:${model}`, mode: "response",
    text, dossierHash: dossierHash || null,
    session: sessionId, turn: turnNumber, timestamp: Date.now(),
  };
}

export function logPassage(text, documentId, passageIndex, source) {
  return {
    id: mintGivenId("document", text, Date.now()),
    type: "eo.given", agent: "system:walker", mode: "document",
    text, source: source || null,
    documentId, passageIndex, timestamp: Date.now(),
  };
}

/* ═══ Splitting text for the walk ═══ */

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

/* ═══ The Signal — mechanical NER + keywords (SIG: candidates only) ═══ */

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

function guessKind(message, name, index) {
  const before = message.slice(Math.max(0, index - 24), index).trim();
  const prevWord = (before.match(/(\S+)\s*$/) || [, ""])[1];
  if (TITLE_RE.test(prevWord)) return "person";
  if (PLACE_PREP.has(prevWord.toLowerCase())) return "place";
  if (ORG_SUFFIX.test(name)) return "organization";
  return null;
}

function extractNER(message) {
  const seen = new Map();
  const re = /\b[A-Z][a-zA-Z]+(?:\s+(?:of|the|and|de|van|von)\s+)?(?:\s*[A-Z][a-zA-Z]+)*\b/g;
  let m;
  while ((m = re.exec(message))) {
    const name = m[0].trim().replace(/\s+/g, " ");
    if (name.length < 2 || COMMON_CAPS.has(name.toLowerCase())) continue;
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
    .split(/\s+/).filter(w => w.length > 2 && !STOPS.has(w));
}

export function signal(message) {
  return { ner: extractNER(message), keywords: extractKeywords(message) };
}

const slug = (s) => String(s || "").trim().toLowerCase().replace(/\s+/g, " ");

/* ═══ The Reach — rank graph entities against the signal ═══ */

export function reach(sig, graph) {
  const cand = new Map();
  const add = (id, score) => cand.set(id, (cand.get(id) || 0) + score);
  for (const name of sig.ner.names) {
    const q = slug(name);
    for (const e of graph.allEntities()) {
      const c = slug(e.canonical);
      if (c.includes(q) || q.includes(c)) add(e.id, 3);
      else if ((e.aliases || []).some(a => slug(a).includes(q) || q.includes(slug(a)))) add(e.id, 2);
    }
  }
  for (const kw of sig.keywords) {
    for (const e of graph.allEntities()) {
      if (slug(e.canonical).includes(kw) || (e.hypothesis || "").toLowerCase().includes(kw)) add(e.id, 1);
    }
  }
  for (const id of [...cand.keys()]) {
    for (const edge of graph.getEdges(id)) {
      const other = edge.from === id ? edge.to : edge.from;
      if (!cand.has(other)) add(other, 1);
    }
  }
  return [...cand.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([id, score]) => ({ entity: graph.getEntity(id), tier: score >= 3 ? 1 : score >= 1 ? 2 : 3 }))
    .filter(r => r.entity);
}

/* ═══ The Dossier — projected [CTX] block ═══ */

export function buildDossier(rankedEntities, graph, maxTokens = 350) {
  let budget = maxTokens;
  const blocks = [];
  for (const { entity, tier } of rankedEntities) {
    if (!entity) continue;
    const stateHash = graph.stateHash(entity.id);
    const edges = graph.getEdges(entity.id);
    const defs = graph.getDefs(entity.id);
    const conflicts = graph.getConflicts(entity.id);
    const aliases = entity.aliases?.length ? entity.aliases.join(", ") : null;

    if (tier === 1 && budget >= 65) {
      let b = `E: ${entity.id}@${stateHash} | ${entity.terrain} | ${edges.length}`;
      b += `\n  ~ ${entity.canonical}${aliases ? ", aka " + aliases : ""}`;
      b += `\n  h: ${(entity.hypothesis || "?").slice(0, 130)}`;
      for (const edge of edges.slice(0, 3)) {
        const dir = edge.from === entity.id ? "→" : "←";
        const target = edge.from === entity.id ? edge.to : edge.from;
        b += `\n  ${dir} ${target} (${edge.type})`;
      }
      let spanDone = false;
      for (const def of defs.slice(0, 3)) {
        b += `\n  = ${def.field}: "${def.value}"`;
        if (def.span && !spanDone) { b += ` @"${def.span.slice(0, 50)}"`; spanDone = true; }
      }
      for (const c of conflicts.slice(0, 1)) {
        b += `\n  ⚠ ${c.field}: "${c.existing}" vs "${c.incoming}"`;
      }
      blocks.push(b); budget -= 65;
    } else if (tier === 2 && budget >= 35) {
      let b = `E: ${entity.id}@${stateHash} | ${entity.terrain}`;
      b += `\n  ~ ${entity.canonical}`;
      b += `\n  h: ${(entity.hypothesis || "?").slice(0, 130)}`;
      for (const def of defs.slice(0, 2)) b += `\n  = ${def.field}: "${def.value}"`;
      blocks.push(b); budget -= 35;
    } else if (budget >= 15) {
      blocks.push(`E: ${entity.id}@${stateHash} | ${entity.terrain}\n  ~ ${entity.canonical}\n  h: ${(entity.hypothesis || "?").slice(0, 100)}`);
      budget -= 15;
    } else break;
  }
  return blocks.length
    ? `[CTX]\n${blocks.join("\n\n")}\n[/CTX]`
    : "[CTX]\n(no matching entities)\n[/CTX]";
}

/* The dossier hash — identifies the projection that produced a response. */
export function dossierHashOf(dossier) {
  let h = 0x811c9dc5;
  const s = String(dossier || "");
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = (h * 0x01000193) >>> 0; }
  return "d_" + h.toString(16).padStart(8, "0");
}

/* Source spans behind the projected dossier — DEF spans and edge sources. */
export function collectSpans(rankedEntities, graph) {
  const spans = [];
  for (const { entity } of rankedEntities) {
    if (!entity) continue;
    for (const d of graph.getDefs(entity.id)) {
      spans.push({ entity: entity.id, kind: "def", text: `${d.field}: ${d.value}`, source: d.source || d.span || null });
    }
    for (const v of graph.getEvals(entity.id)) {
      spans.push({ entity: entity.id, kind: "eva", text: `${v.claim} [${v.status}]`, source: v.source || null });
    }
  }
  return spans;
}

/* ═══ The Position Marker ═══ */

export function buildPosition(lastTurn) {
  if (!lastTurn) return "";
  return `[POS]
prev: ${(lastTurn.entities || []).slice(0, 5).join(", ")}
topic: ${lastTurn.topic || "none"}
last: "${(lastTurn.userMessage || "").slice(0, 80)}"
[/POS]`;
}

/* ═══ The Register — known entities, so EXTRACT does not re-INS ═══ */

export function buildRegister(graph, cap = 20) {
  const entities = graph.allEntities()
    .sort((a, b) => (b.mentions || 0) - (a.mentions || 0)).slice(0, cap);
  if (!entities.length) return "Entity register: (empty — first pass)";
  return "Entity register:\n" + entities.map(e =>
    `${e.id}@${graph.stateHash(e.id)} | ${e.canonical}${e.aliases?.length ? " | aka: " + e.aliases.join(", ") : ""} | ${e.terrain}`
  ).join("\n");
}

export function buildReaderCursor(introduced) {
  if (introduced.length <= 10) return `[READER]\nIntroduced: ${introduced.join(", ")}\n[/READER]`;
  return `[READER]\n${introduced.length} entities introduced. Recent: ${introduced.slice(-5).join(", ")}\n[/READER]`;
}

/* ═══ EXTRACT / MUTATE prompt builders ═══ */

export function buildExtractPrompt(userMessage, modelResponse, userGivenId, modelGivenId, graph) {
  return `${buildRegister(graph)}

EXCHANGE:
User [${userGivenId}]: "${String(userMessage).slice(0, 4000)}"
Model [${modelGivenId}]: "${String(modelResponse).slice(0, 4000)}"`;
}

export function buildMutatePrompt(trigger, graph) {
  const { name, candidateHash, span, context } = trigger;
  let prompt = `Ambiguous reference: "${name || "(unnamed)"}"`;
  if (span) prompt += `\nSource span: "${span}"`;
  if (context) prompt += `\nContext: "${context}"`;
  const candidate = candidateHash ? graph.getEntity(candidateHash) : null;
  if (candidate) {
    prompt += `\n\nExisting entity that might match:`;
    prompt += `\n  ${candidate.id}@${graph.stateHash(candidate.id)}`;
    prompt += `\n  canonical: ${candidate.canonical}`;
    prompt += `\n  terrain: ${candidate.terrain}`;
    prompt += `\n  h: ${candidate.hypothesis || "?"}`;
    for (const d of graph.getDefs(candidate.id).slice(0, 6)) {
      prompt += `\n  [${d.id}] = ${d.field}: "${d.value}"`;
    }
    for (const c of graph.getConflicts(candidate.id)) {
      prompt += `\n  ⚠ ${c.field}: "${c.existing}" vs "${c.incoming}"`;
    }
  }
  prompt += `\n\nIs "${name || "this"}" the same entity as ${candidateHash || "an existing one"}, or different? Decide one action.`;
  return prompt;
}

/* ═══ Hypothesis prompt builders (strict nesting) ═══ */

export function getHypothesisSystemPrompt(level) {
  return {
    entity: HYPOTHESIS_ENTITY, group: HYPOTHESIS_GROUP, section: HYPOTHESIS_SECTION,
    document: HYPOTHESIS_DOCUMENT, session: HYPOTHESIS_SESSION, corpus: HYPOTHESIS_CORPUS,
  }[level] || HYPOTHESIS_ENTITY;
}

function getChildInputs(level, id, graph) {
  switch (level) {
    case "entity": {
      const e = graph.getEntity(id);
      if (!e) return [];
      const defs = graph.getDefs(id), edges = graph.getEdges(id);
      const lines = [`${id}@${graph.stateHash(id)} | ${e.canonical} | ${e.terrain}`];
      if (defs.length) lines.push(`Facts: ${defs.map(d => `${d.field}="${d.value}"`).join(", ")}`);
      if (edges.length) lines.push(`Connections: ${edges.map(x => `→${x.to} (${x.type})`).join(", ")}`);
      return lines;
    }
    case "group":
      return graph.getEntitiesInRange(id.start, id.end)
        .map(e => `${e.id}@${graph.stateHash(e.id)} ${e.canonical}: "${e.hypothesis || "?"}"`);
    case "section":
      return graph.getPassageGroupDEFs(id.start, id.end)
        .map((g, i) => `group ${i + 1} (p${g.start + 1}-${g.end + 1}): "${g.hypothesis}"`);
    case "document":
      return graph.getSectionDEFs(id.documentId)
        .map((s, i) => `section ${i + 1}: "${s.hypothesis}"`);
    case "session":
      return [
        ...graph.getSessionDocumentDEFs().map(d => `doc "${d.title}": "${d.hypothesis}"`),
        ...graph.getSessionTopics().map(t => `topic: "${t}"`),
      ];
    case "corpus":
      return [
        ...graph.getRecentDocumentDEFs(10).map(d => `"${d.title}": "${d.hypothesis}"`),
        ...graph.getRecentSessionDEFs(5).map(s => `session: "${s.hypothesis}"`),
      ];
    default: return [];
  }
}

export function buildHypothesisPrompt(level, id, graph) {
  const children = getChildInputs(level, id, graph);
  const history = graph.getHypothesisHistory(level, id);
  let prompt = children.join("\n");
  if (history.length) {
    prompt += "\n\nPrior hypotheses (oldest first):";
    history.forEach((h, i) => {
      prompt += `\n  rev ${i + 1} (${h.after}, ${h.inputCount} inputs): "${h.text}"`;
    });
  }
  return prompt;
}

/* ═══ Apply EXTRACT / INGEST events to the graph ═══ */

export function parseEvents(out) {
  if (Array.isArray(out)) return out.filter(e => e && typeof e === "object" && e.op);
  if (typeof out !== "string") return [];
  let raw = out.replace(/```json/gi, "").replace(/```/g, "").trim();
  const a = raw.indexOf("["), b = raw.lastIndexOf("]");
  if (a !== -1 && b !== -1) raw = raw.slice(a, b + 1);
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(e => e && typeof e === "object" && e.op) : [];
  } catch { return []; }
}

/* Apply parsed events. Conflicting DEFs are written without retiring the
   prior value, so the dossier surfaces ⚠ and the next turn can flag it.
   Returns { applied, ambigs, conflicts }. */
export function applyEvents(graph, events, opts = {}) {
  const source = opts.source || null;
  let applied = 0;
  const ambigs = [], conflicts = [];
  for (const e of events) {
    if (e.op === "AMBIG") {
      ambigs.push({ name: String(e.name || "").trim(), candidateHash: String(e.candidate || "").trim(), span: String(e.span || "").trim() });
    } else if (e.op === "INS") {
      const ent = graph.ensureEntity(e.entity, { canonical: e.entity, terrain: e.terrain });
      if (ent && opts.passageIndex != null && !ent.passages.includes(opts.passageIndex)) {
        ent.passages.push(opts.passageIndex);
      }
      if (ent) applied++;
    } else if (e.op === "DEF") {
      const ent = graph.ensureEntity(e.entity, { canonical: e.entity });
      if (ent && e.field && e.value != null) {
        const prior = graph.getDef(ent.id, e.field);
        const isConflict = prior && prior.value !== String(e.value) && e.field !== "kind";
        if (isConflict) conflicts.push({ entity: ent.id, field: e.field, existing: prior.value, incoming: String(e.value) });
        graph.writeDef(ent.id, e.field, e.value, { source: e.source || source, span: e.span || null, supersede: !isConflict });
        applied++;
      }
    } else if (e.op === "CON") {
      const from = graph.ensureEntity(e.from, { canonical: e.from });
      const to = graph.ensureEntity(e.to, { canonical: e.to });
      if (from && to && from.id !== to.id) {
        graph.addEdge(from.id, to.id, e.type || "related to", { source: e.source || source });
        applied++;
      }
    } else if (e.op === "EVA") {
      const ent = graph.ensureEntity(e.entity, { canonical: e.entity });
      if (ent && e.claim) {
        graph.writeEval(ent.id, e.claim, e.status || "holds", { source: e.source || source, span: e.span || null });
        applied++;
      }
    }
  }
  return { applied, ambigs, conflicts };
}

/* ═══ MUTATE: triggers, parse, apply ═══ */

const REF_UNCERTAIN = [
  /\bmay not be (?:the same|identical)(?: as)? (e_[0-9a-f]{8})/i,
  /\bmight be (?:a )?different (?:from |than )?(e_[0-9a-f]{8})/i,
  /\bnot (?:the same|identical) (?:as |to )(e_[0-9a-f]{8})/i,
  /\b(e_[0-9a-f]{8})\b[^.?!]*?\b(?:may not|might not|not the same)\b/i,
];
const CORRECTION_RE = /\b(?:that(?:'s| is) not the same|those are the same|different (?:person|place|thing|one|entity)|you(?:'re| are) (?:confusing|mixing)|not who i meant|i meant (?:a )?different|wrong (?:person|entity|one)|mixed (?:them|it) up)\b/i;

/* Detect ambiguities mechanically — zero tokens. */
export function detectMutationTriggers(modelResponse, extractEvents, sig, graph) {
  const triggers = [];
  for (const re of REF_UNCERTAIN) {
    const m = re.exec(modelResponse || "");
    if (m && graph.getEntity(m[1])) {
      triggers.push({ type: "model_flagged", candidateHash: m[1], span: m[0].trim() });
    }
  }
  for (const evt of extractEvents || []) {
    if (evt.op === "AMBIG") {
      triggers.push({ type: "ingest_ambig", name: evt.name, candidateHash: evt.candidate, span: evt.span });
    }
  }
  for (const name of sig?.ner?.names || []) {
    const guessed = sig.ner.typed[name];
    if (!guessed) continue;
    const matches = graph.searchEntities(name);
    if (matches.length === 1) {
      const kindDef = graph.getDef(matches[0].id, "kind");
      if (kindDef && kindDef.value !== guessed) {
        triggers.push({
          type: "type_mismatch", name, candidateHash: matches[0].id,
          context: `reads as ${guessed} but recorded kind is ${kindDef.value}`,
        });
      }
    }
  }
  return triggers;
}

export function userCorrectionTrigger(userMessage, graph, lastEntities = []) {
  if (!CORRECTION_RE.test(userMessage || "")) return null;
  const candidateHash = lastEntities.find(id => graph.getEntity(id)) || null;
  return { type: "user_correction", candidateHash, context: String(userMessage).slice(0, 200) };
}

export function parseMutate(out) {
  let obj = out;
  if (typeof out === "string") {
    let raw = out.replace(/```json/gi, "").replace(/```/g, "").trim();
    const a = raw.indexOf("{"), b = raw.lastIndexOf("}");
    if (a !== -1 && b !== -1) raw = raw.slice(a, b + 1);
    try { obj = JSON.parse(raw); } catch { return null; }
  }
  if (!obj || typeof obj !== "object") return null;
  const action = String(obj.action || "").toUpperCase();
  if (!["FORK", "MERGE", "CORRECT", "RECLASSIFY", "NONE"].includes(action)) return null;
  return { ...obj, action, reason: String(obj.reason || "").trim() };
}

/* Apply a MUTATE action to the graph. Logs the mutation as an event. */
export function applyMutation(action, graph, triggerId) {
  const event = { timestamp: Date.now(), triggeredBy: triggerId || null, action: action.action, reason: action.reason };
  switch (action.action) {
    case "FORK": {
      const src = graph.getEntity(action.source);
      if (src && action.new_canonical) {
        const ne = graph.ensureEntity(action.new_canonical, { canonical: action.new_canonical, terrain: src.terrain });
        ne.forkedFrom = action.source;
        for (const r of action.reassign || []) graph.reassignDef(r.def_id, action.source, ne.id);
        event.op = "SEG"; event.sourceEntity = action.source; event.newEntity = ne.id;
      }
      break;
    }
    case "MERGE":
      if (graph.getEntity(action.keep) && graph.getEntity(action.absorb)) {
        graph.mergeEntities(action.keep, action.absorb);
        for (const a of action.new_aliases || []) graph.addAlias(action.keep, a);
        event.op = "CON"; event.type = "same_as"; event.keep = action.keep; event.absorb = action.absorb;
      }
      break;
    case "CORRECT": {
      const ent = graph.getEntity(action.entity);
      if (ent && action.field) {
        // Retire every live DEF on the field, then write the corrected one.
        for (const d of graph.getDefs(action.entity)) if (d.field === action.field) d.retired = true;
        graph.writeDef(action.entity, action.field, action.new_value, { source: action.source });
        event.op = "DEF"; event.entity = action.entity; event.field = action.field;
        event.oldValue = action.old_value; event.newValue = action.new_value;
      }
      break;
    }
    case "RECLASSIFY":
      if (graph.getEntity(action.entity)) {
        graph.updateTerrain(action.entity, action.new_terrain);
        event.op = "SEG"; event.entity = action.entity;
        event.oldTerrain = action.old_terrain; event.newTerrain = action.new_terrain;
      }
      break;
    case "NONE":
      event.op = "NUL";
      break;
  }
  graph.appendEvent(event);
  return event;
}

/* ═══ Auto-commit tiers ═══

   AUTO commits silently. PROMPT surfaces an inline consent pill. REQUIRE
   blocks until the user acts. Forks, merges, reclassifications and
   conflicting facts always need consent. */
export function commitTier(event, graph) {
  switch (event.op || event.action) {
    case "CON": return event.type === "same_as" ? "PROMPT" : "AUTO";
    case "SEG": return "PROMPT";
    case "FORK": case "MERGE": case "RECLASSIFY": return "PROMPT";
    case "DEF": case "CORRECT": {
      if (event.entity && event.field) {
        const existing = graph.getDef(event.entity, event.field);
        if (existing && event.newValue != null && existing.value !== event.newValue) return "PROMPT";
      }
      return "AUTO";
    }
    case "INS": return "AUTO";
    case "EVA": return event.status === "contested" ? "PROMPT" : "AUTO";
    case "REC": return "REQUIRE";
    default: return "AUTO";
  }
}

/* ═══ Consolidation gate ═══ */
export function shouldConsolidate(state) {
  const elapsed = Date.now() - (state.lastConsolidation || 0);
  if (elapsed < 24 * 60 * 60 * 1000) return false;
  if ((state.sessionsSinceConsolidation || 0) < INTERVALS.CORPUS_HYPOTHESIS) return false;
  if (state.consolidationLock) return false;
  return true;
}

/* Re-export Graph helpers so Chat.jsx has one import surface. */
export { Graph } from "./graph.js";
export { mergeGraphs } from "./graph.js";
