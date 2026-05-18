/* v3 prompts, builders and graph operations — over the SQLite local store.

   Memory mode runs on the situated graph in local-store.js. A chat does not
   send its growing history to the model; every turn is projected into a
   fixed-size prompt and read back into the graph.

   Three model calls per knowledge-bearing turn:
     READ    — user-facing, answers from the projected [CTX]
     EXTRACT — background, reads the exchange into the graph as INS/CON/DEF/EVA
     MUTATE  — background, fires only on a mechanical ambiguity trigger;
               produces one FORK/MERGE/CORRECT/RECLASSIFY/NONE

   SIG is mechanical only — the NER/keyword scan plus embedding recall that
   produces retrieval candidates before the graph is touched. INS creates
   permanent identity; kind is a revisable DEF; EVA records judgments; REC is
   reserved for MUTATE. All graph reads here run against the current scope —
   the caller sets it with store.setScope(chatId). */

import * as store from "./local-store.js";

export const INTERVALS = {
  ENTITY_HYPOTHESIS: 1,
  GROUP_HYPOTHESIS: 4,
  SECTION_HYPOTHESIS: 12,
  DOCUMENT_HYPOTHESIS: Infinity,
  SESSION_HYPOTHESIS: Infinity,
  CORPUS_HYPOTHESIS: 5,
};

/* ═══ Prompts: READ (user-facing) ═══ */

export const READ_SYSTEM = `Answer from the material below. Be direct and concise.

If the material covers the question, answer from it.
If it only partly covers it, say what you can see and what is still being read.
If it doesn't cover it at all, just answer normally.

The material is graded by how well it has been read — state your confidence
to match it:
  - Grounded entities (E:) and their facts have been read closely — answer
    with confidence.
  - PARTIAL entities and [partial] passages come from a document still being
    read — answer provisionally and say the reading is in progress.
  - UNREAD passages and [skim] passages are a quick embedding match from a
    document not yet read closely — flag the answer as a first impression.
When you are not working from a confident, grounded answer, say so plainly.

[READING] is your running understanding of the documents being read. When
the user asks what a document is about or what you have learned, answer from
[READING]: give its understanding and the key threads, and state how much
has been read. Do not just list raw passages back.
A "skim" line is a first-pass guess from a glance — say so, and treat it as
provisional. A "document" line is a synthesis from a full read — answer it
with confidence.

When the user asks what the material says or what you have, give a summary
of the key points — not just "yes" or a one-liner.

Never quote tags, hashes, or format markers back to the user.
Never narrate your process. Just answer the question.`;

export const READ_CASUAL = `You are a helpful assistant. Be concise and natural.
Your messages are recorded in a knowledge graph.`;

/* ═══ Prompt: GROUNDED (ask-with-documents) ═══
   The model interprets; the code supplies and presents the evidence. The
   numbered sources are the ONLY material it may draw facts from. */

export const GROUNDED_SYSTEM = `Answer the question using ONLY the numbered sources in [DOCS].

Return a single JSON object: { "runs": [ ... ] }.
Each run is an object: { "text": "...", "cites": [n], "ungrounded": true }.

Rules:
- Concatenated, every run's "text" forms the full answer. Split the answer so
  each factual claim is its own run, immediately followed by the source
  number(s) that support it in "cites".
- Every number in "cites" MUST be a source number shown in [DOCS]. Never
  invent a number and never cite a source that is not listed.
- Do NOT add facts that are not in the numbered sources. If the sources do
  not answer the question, say so plainly.
- If a sentence is your own inference and not supported by any source, put it
  in its own run with "ungrounded": true and no "cites". Use this sparingly.
- Paraphrase — do not quote the sources at length; the user sees them already.
- Keep the whole answer concise. Never restate these instructions or tags.

Return ONLY the JSON object. No prose outside it, no code fences.`;

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

EVA status meanings:
  "holds"     = the text SUPPORTS this claim (it happened, it's true)
  "fails"     = the text CONTRADICTS this claim (it didn't happen, it's false)
  "contested" = the text presents conflicting evidence

EVA is about whether the CLAIM holds, not whether the EVENT is good or bad.
A death is tragic but the claim that it occurred HOLDS.

Terrains: Entity, Network, Paradigm, Void, Kind, Field, Link, Atmosphere, Lens`;

/* ═══ Prompt: MUTATE (background, on ambiguity) ═══ */

export const MUTATE_SYSTEM = `You are resolving a graph ambiguity. Examine the evidence and
produce exactly ONE action as JSON. ONLY valid JSON, no markdown.

Actions:

FORK — the evidence clearly names two separate referents that were recorded
as one (not merely conflicting facts about a single entity):
{"action":"FORK","source":"<hash>","new_canonical":"<name>","reason":"<why>"}

MERGE — two entities are actually one:
{"action":"MERGE","keep":"<hash>","absorb":"<hash>","reason":"<why>",
 "new_aliases":["<alias1>"]}

CORRECT — a DEF is wrong:
{"action":"CORRECT","entity":"<hash>","field":"<field>",
 "old_value":"<wrong>","new_value":"<right>","reason":"<why>"}

RECLASSIFY — terrain assignment is wrong:
{"action":"RECLASSIFY","entity":"<hash>","old_terrain":"<T>","new_terrain":"<T>","reason":"<why>"}

NONE — no action needed:
{"action":"NONE","reason":"<why the ambiguity is not real>"}

Default to NONE. Conflicting facts about one entity are NOT grounds for a
FORK — contradictory DEFs are allowed to coexist as a recorded conflict, and
"we do not know this yet" is a valid resting state. Only FORK, MERGE or
CORRECT when the evidence clearly supports it; missing or incomplete
information is never enough. When in doubt, choose NONE.

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

Coreference rules:
- Pronouns (he, she, they, his, her, their, it) are NOT entities. They
  refer to entities already in the register. Use the register hash.
- Possessive phrases ("his mother", "his oldest brother") refer to a
  RELATIONSHIP, not a new entity. Create a CON edge to the known entity,
  or a DEF on the known entity.
  Example: "his mother died of cancer" with register showing
  e_c002f85f (Martin Short) →
  {"op":"DEF","entity":"e_c002f85f","field":"mother_death","value":"cancer","span":"his mother died of cancer"}
  NOT: {"op":"INS","entity":"his mother","terrain":"Entity"}
- Temporal phrases ("five years later", "three years after that", "at 20")
  are NOT entities. They are context for DEFs on existing entities. Include
  the temporal info in the DEF value.
  Example: "Five years later, his mother died of cancer" →
  {"op":"DEF","entity":"e_c002f85f","field":"mother_death","value":"cancer, ~5 years after brother's death","span":"Five years later, his mother died of cancer"}
- Surname-only references ("Short") resolve to the most recently mentioned
  entity with that surname in the register. Do NOT create a new entity.
- "he says" / "she says" — the speaker is the subject of the passage or the
  most recent named person. Reference by hash.

EVA status meanings:
  "holds"     = the text SUPPORTS this claim (it happened, it's true)
  "fails"     = the text CONTRADICTS this claim (it didn't happen, it's false)
  "contested" = the text presents conflicting evidence
EVA is about whether the CLAIM holds, not whether the EVENT is good or bad.
A death is tragic but the claim that it occurred HOLDS.

AMBIG rules:
- If a name might refer to an existing entity but you are uncertain, flag it:
  {"op":"AMBIG","name":"<name>","candidate":"<hash>","span":"<exact words>"}
  The system will trigger a MUTATE call to resolve it.
- "candidate" must be a SINGLE entity hash from the register, not a list.
  Pick the most likely match.
- If no entity in the register is a plausible match, do NOT emit AMBIG.
  Just INS a new entity.
- AMBIG means: "this name MIGHT be the same as this specific entity but I'm
  not sure." It does NOT mean "I don't know which entity this could be out
  of all of them."
  Good:  {"op":"AMBIG","name":"his mother","candidate":"e_c002f85f","span":"his mother died"}
  Bad:   {"op":"AMBIG","name":"his mother","candidate":"e_a|e_b|e_c|e_d"}

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

export const HYPOTHESIS_SKIM = `You are skimming a document for the first time — only its opening lines and the names that jump out, not a close reading.
Write a one-sentence hypothesis for what this document is probably about.
Under 200 characters. A first impression: specific but provisional, the kind a reader forms in a few seconds.
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
    id: store.mintGivenId("user", text),
    type: "eo.given", agent: "user", mode: "conversation",
    text, ner: signalOut?.ner || null,
    session: sessionId, turn: turnNumber, timestamp: Date.now(),
  };
}

export function logModelResponse(text, model, dossierHash, sessionId, turnNumber) {
  return {
    id: store.mintGivenId("model", text),
    type: "eo.given", agent: `model:${model}`, mode: "response",
    text, dossierHash: dossierHash || null,
    session: sessionId, turn: turnNumber, timestamp: Date.now(),
  };
}

export function logPassage(text, documentId, passageIndex, source) {
  return {
    id: store.mintGivenId("document", text),
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

/* Break a long sentence into clauses at strong internal boundaries —
   semicolons, colons, spaced dashes, and commas before a coordinating or
   subordinating conjunction. Short sentences pass through whole; tiny
   fragments are merged back so a clause is never just a connective. */
const CLAUSE_CONJ = /,\s+(?=(?:and|but|or|so|yet|nor|because|although|though|while|whereas|which|who|that|when|where|if|since)\b)/i;

export function splitClauses(sentence, minLen = 160) {
  const s = String(sentence || "").trim();
  if (s.length <= minLen) return s ? [s] : [];
  const parts = s
    .split(/\s*[;:]\s+|\s+[—–]\s+/)
    .flatMap(p => p.split(CLAUSE_CONJ))
    .map(p => p.trim())
    .filter(Boolean);
  if (parts.length <= 1) return [s];
  const out = [];
  for (const p of parts) {
    if (out.length && p.length < 40) out[out.length - 1] += " " + p;
    else out.push(p);
  }
  return out;
}

/* The unit of the reading walk: one sentence per passage, with long
   sentences broken down to one clause each. Encoding at this grain gives
   the model far cleaner extractions than multi-sentence chunks. */
export function splitPassages(text) {
  return splitSentences(text).flatMap(s => splitClauses(s));
}

/* ═══ Chrome detection — rhetorical scaffolding with no extractable content ═══

   Chrome is transitions, attributions, narrative framing and filler. It
   carries no entities or claims of its own, so it is stored (for narrative
   context) but skipped by the LLM walk. Mechanical — no model call. */
const CHROME_PATTERNS = [
  /^in the (film|book|article|interview|documentary|piece|story)/i,
  /^(he|she|they|it) (said|says|added|noted|explained|responded|replied|continued|recalled)/i,
  /^(according to|as .+ (said|noted|put it))/i,
  /^(meanwhile|however|nonetheless|in contrast|on the other hand)/i,
  /^(this|that|it) (is|was|would be|has been) (a|an|the)\b/i,
];

export function classifyPassage(text, index, totalPassages) {
  const t = String(text || "").trim();
  const words = t ? t.split(/\s+/).length : 0;
  const hasQuotedSpeech = /["“”‘’]/.test(t);
  const hasProperNoun = /[A-Z][a-z]{2,}/.test(t);
  const hasNumber = /\d/.test(t);

  const isChromePattern = CHROME_PATTERNS.some(p => p.test(t));
  const isShort = words < 15;
  const lacksContent = !hasProperNoun && !hasNumber;

  if (isChromePattern && isShort && lacksContent) {
    return { type: "chrome", confidence: "high" };
  }
  if (isChromePattern && !hasQuotedSpeech) {
    return { type: "chrome", confidence: "medium" };
  }
  if (isShort && lacksContent && !hasQuotedSpeech) {
    return { type: "chrome", confidence: "low" };
  }
  return { type: "content", confidence: "high" };
}

/* Chrome passages are skipped by the walk, but their temporal/narrative
   information should attach to the nearest content passage (next preferred,
   previous as fallback). Returns an array of arrays: ctx[i] = chrome texts
   to prepend when walking content passage i. */
export function attachChromeContext(passages, classes) {
  const ctx = passages.map(() => []);
  for (let i = 0; i < passages.length; i++) {
    if (classes[i]?.type !== "chrome") continue;
    let target = -1;
    for (let j = i + 1; j < passages.length; j++) {
      if (classes[j]?.type === "content") { target = j; break; }
    }
    if (target === -1) {
      for (let j = i - 1; j >= 0; j--) {
        if (classes[j]?.type === "content") { target = j; break; }
      }
    }
    if (target !== -1) ctx[target].push(passages[i]);
  }
  return ctx;
}

/* ═══ The Gate — is a chat message knowledge-bearing? ═══

   Meta-conversation about the app ("i put instructions in there", "try now")
   and casual chatter carry no knowledge. They must skip EXTRACT, so the walk
   never mints entities for filler. Mechanical — no model call. */
const META_PATTERNS = [
  /^(i|we) (put|added|uploaded|pasted|imported|gave|sent)/i,
  /^(try|check|look|see) (now|again|this|that|it)/i,
  /^(do|did|can|could|will) you (have|see|find|remember|get|know)/i,
  /^what('s| is| are) in (the|my|our|that|this)/i,
  /^(yes|no|ok|okay|sure|thanks|thank you|hello|hi|hey|yep|nope|cool|nice|great)\b/i,
];

export function gateClassify(message) {
  const text = String(message || "").trim();
  const lower = text.toLowerCase();

  if (META_PATTERNS.some(p => p.test(lower))) {
    return { knowledgeBearing: false, intent: "meta" };
  }

  const words = text ? text.split(/\s+/).length : 0;
  const hasProperNoun = /[A-Z][a-z]{2,}/.test(text);
  if (words < 6 && !hasProperNoun) {
    return { knowledgeBearing: false, intent: "casual" };
  }

  return { knowledgeBearing: true, intent: "query" };
}

/* ═══ Follow-up reference resolution ═══

   "what's it say?" has zero keyword overlap with "Martin Short". When the
   message is a follow-up reference, fall back to the entities that were
   active last turn so the Reach still finds context. */
const FOLLOWUP_REFS = [
  "it", "that", "this", "the article", "the doc", "the document",
  "the stuff", "the text", "the file", "the piece", "the story",
];

export function resolveFollowUpReferences(message, position) {
  const lower = String(message || "").toLowerCase();
  const isFollowUp = FOLLOWUP_REFS.some(p =>
    new RegExp(`\\b${p.replace(/\s+/g, "\\s+")}\\b`).test(lower));
  if (isFollowUp && position?.entities?.length > 0) return position.entities;
  return null;
}

/* Zoom — how far out is this question pitched? The hypothesis hierarchy
   (entity → group → section → document) is a stack of zoom levels; the
   question selects which one to project into [READING]:

     entity   — names an entity the graph knows; wants its facts (from [CTX])
     group    — asks how things connect/relate; wants the threads
     document — asks what something is overall about; wants the synthesis
     section  — the default middle zoom

   This is the availability function over the hypothesis hierarchy. */
const ZOOM_THEME_RE = /\b(how|why|connects?|connection|relates?|related|relationship|themes?|patterns?|links?|tie|ties)\b/i;
const ZOOM_OVERVIEW_RE = /\b(summar(?:y|ise|ize)|overview|gist|overall|tell me about|what have you learned|what(?:'s| is| are| is this|'s this)|main (?:point|idea))\b/i;

export function selectZoom(message) {
  const q = String(message || "");
  const names = signal(q).ner.names;
  if (names.some(n => store.entities.search(n).length > 0)) return "entity";
  if (ZOOM_THEME_RE.test(q)) return "group";
  if (ZOOM_OVERVIEW_RE.test(q) || /\bwhat\b[^?]*\b(about|says?|covers?|discuss|learned)\b/i.test(q))
    return "document";
  return "section";
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

/* ═══ The Reach — SIG-level RAG ═══

   Standard RAG has one signal: embedding similarity. The Reach has five —
   embedding similarity, NER entity overlap, keyword overlap, co-occurrence
   density and passage position — and provenance on every result. It mixes
   grounded entities (structured, cheap per token) with unwalked passages
   (raw text), then deduplicates so a 350-token dossier is never wasted on a
   passage that an entity already covers. */

const STATUS_WEIGHT = { walked: 1.0, walking: 0.85, sig: 0.6 };

export async function retrieve(query, { topK = 6, position = null } = {}) {
  const myScope = store.getScope();
  const sig = signal(query);
  const queryEntities = sig.ner.names.map(n => n.toLowerCase());
  const queryKeywords = sig.keywords;
  let queryVec = null;
  try { queryVec = await store.embed(query); } catch { /* embeddings optional */ }
  store.setScope(myScope); // re-assert after the embed await

  const results = [];

  // ── Tier A: entities (all grades carry a centroid from the first pass) ──
  const hypMap = {};
  for (const h of store.vectors.dumpHypotheses()) hypMap[h.entityId] = h.vec;
  for (const row of store.vectors.dumpCentroids()) {
    const entity = store.entities.get(row.entityId);
    if (!entity) continue;
    const centroidSim = queryVec ? store.cosineSim(queryVec, row.vec) : 0;
    const hypSim = (queryVec && hypMap[row.entityId]) ? store.cosineSim(queryVec, hypMap[row.entityId]) : 0;
    const name = entity.canonical.toLowerCase();
    const nameMatch = queryEntities.some(qe => name.includes(qe) || qe.includes(name)
      || (entity.aliases || []).some(a => a.toLowerCase().includes(qe)));
    const kwMatch = queryKeywords.some(k => name.includes(k) || (entity.hypothesis || "").toLowerCase().includes(k));
    let score = (centroidSim * 0.3) + (hypSim * 0.3) + (nameMatch ? 0.4 : 0) + (kwMatch ? 0.1 : 0);
    score *= (STATUS_WEIGHT[entity.status] ?? 0.6);
    if (score > 0.12) {
      results.push({
        type: "entity", id: entity.id, status: entity.status,
        canonical: entity.canonical, ner: [entity.canonical], score,
      });
    }
  }

  // ── Tier B: unwalked passages, scored by five signals ──
  for (const row of store.vectors.dumpUnwalked()) {
    const pSig = signal(row.text);
    const passageEntities = pSig.ner.names.map(n => n.toLowerCase());
    const embSim = queryVec ? store.cosineSim(queryVec, row.vec) : 0;
    const nerOverlap = queryEntities.filter(qe =>
      passageEntities.some(pe => pe.includes(qe) || qe.includes(pe))).length;
    const nerScore = queryEntities.length ? nerOverlap / queryEntities.length : 0;
    const kwOverlap = queryKeywords.filter(k => pSig.keywords.includes(k)).length;
    const kwScore = queryKeywords.length ? kwOverlap / queryKeywords.length : 0;
    const cooccur = Math.min(nerOverlap / 3, 1);
    const total = store.documents.get(row.documentId)?.total || 0;
    const positionScore = (row.passageIdx === 0 || (total && row.passageIdx === total - 1)) ? 0.1 : 0;
    const score = (embSim * 0.35) + (nerScore * 0.25) + (kwScore * 0.15) + (cooccur * 0.15) + positionScore;
    if (score > 0.10) {
      results.push({
        type: "passage", id: row.id, status: "sig", text: row.text,
        passageIndex: row.passageIdx, documentId: row.documentId,
        ner: pSig.ner.names, score,
      });
    }
  }

  // ── Follow-up references — "what's it say?" carries no keyword signal.
  // Pull in the entities that were active last turn so CTX is never empty.
  const followUp = resolveFollowUpReferences(query, position);
  if (followUp) {
    for (const eid of followUp) {
      if (results.some(r => r.type === "entity" && r.id === eid)) continue;
      const entity = store.entities.get(eid);
      if (!entity) continue;
      results.push({
        type: "entity", id: entity.id, status: entity.status,
        canonical: entity.canonical, ner: [entity.canonical], score: 0.5,
      });
    }
  }

  results.sort((a, b) => b.score - a.score);

  // Deduplicate — a structured entity beats a raw passage that only repeats
  // names the entity already carries.
  const seen = new Set();
  const deduped = [];
  for (const r of results) {
    if (r.type === "entity") { seen.add(r.canonical.toLowerCase()); deduped.push(r); }
    else {
      const nl = (r.ner || []).map(n => n.toLowerCase());
      const covered = nl.length > 0 && nl.every(n => [...seen].some(s => s.includes(n) || n.includes(s)));
      if (!covered) deduped.push(r);
    }
  }
  return deduped.slice(0, topK);
}

/* Keyword fallback — entity-name and keyword overlap, no embeddings. Used
   only when a document has no stored passage vectors to score against
   (opted in from another chat's scope, or never embedded). */
function scorePassagesByKeyword(query, passages) {
  const qSig = signal(query || "");
  const qKw = qSig.keywords;
  const qNames = qSig.ner.names.map(n => n.toLowerCase());
  return passages.map(p => {
    const pSig = signal(p.text);
    const pl = p.text.toLowerCase();
    const kwHits = qKw.filter(k => pSig.keywords.includes(k) || pl.includes(k)).length;
    const nameHits = qNames.filter(n => pl.includes(n)).length;
    const kwScore = qKw.length ? kwHits / qKw.length : 0;
    const nameScore = qNames.length ? nameHits / qNames.length : 0;
    return { ...p, score: kwScore * 0.6 + nameScore * 0.4 };
  });
}

/* Document lookup — embedding-based RAG over opted-in library documents,
   backing the composer's "ask with documents" trigger.

   Standard embedding similarity: the query is embedded once, then scored by
   cosine against each document's precomputed passage vectors (written into
   this chat's scope by the first pass — see ensureDocEmbedded) — no
   per-query re-embedding, so the pull is fast. A document is only pulled
   here until it has been fully read; once its walk completes the graph
   projection ([CTX] entities) carries it instead, so a read document is
   skipped. Each passage is tagged with the document's reading confidence —
   [skim] (not yet read) or [partial] (walk in progress) — so the model can
   state how provisional its answer is. Documents with no stored vectors
   fall back to keyword overlap. Identical passage text is deduplicated.

   Returns { text, spans }: the [DOCS] block for the prompt, and one source
   span per pulled passage so the chunk it came from stays auditable. */
export async function lookupDocuments(query, docs, { topK = 6 } = {}) {
  const myScope = store.getScope();
  let queryVec = null;
  try { queryVec = await store.embed(query || ""); } catch { /* embeddings optional */ }
  store.setScope(myScope);
  const allUnwalked = store.vectors.dumpUnwalked();

  const scored = [];
  for (const doc of docs || []) {
    const text = (doc?.text || "").trim();
    if (!text) continue;
    const docPassages = splitPassages(text);

    const reg = store.documents.get(doc.id);
    const walked = reg?.walked || 0;
    const total = reg?.total || docPassages.length;
    // Fully read — the [CTX] graph projection carries it with confidence.
    if (total > 0 && walked >= total) continue;
    const confidence = walked > 0 ? "partial" : "skim";

    const vecRows = allUnwalked.filter(r => r.documentId === doc.id);
    if (queryVec && vecRows.length) {
      for (const r of vecRows) {
        scored.push({
          docId: doc.id, docTitle: doc.title, text: r.text,
          index: r.passageIdx ?? 0, confidence,
          score: store.cosineSim(queryVec, r.vec),
        });
      }
    } else {
      // No stored vectors — keyword fallback over the raw passages.
      for (const s of scorePassagesByKeyword(query,
        docPassages.map((p, i) => ({ text: p, index: i })))) {
        scored.push({ docId: doc.id, docTitle: doc.title, ...s, confidence });
      }
    }
  }
  if (!scored.length) return { text: "", spans: [] };

  const seen = new Set();
  const dedup = (rows) => rows.filter(r => {
    const k = r.text.trim().toLowerCase();
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  let top = dedup(scored.filter(r => r.score > 0).sort((a, b) => b.score - a.score))
    .slice(0, topK);
  if (!top.length) top = dedup(scored).slice(0, topK); // general question — openings

  const lines = top.map((r, i) => `${i + 1}: [${r.confidence}] "${r.text.trim()}"`);
  // Spans are first-class evidence — numbered, code-built, displayed beside the
  // answer. The model cites these numbers; it never mints span text.
  const spans = top.map((r, i) => {
    const t = r.text.trim();
    return {
      index: i + 1,
      entity: r.docTitle || r.docId,   // back-compat for PromptView
      kind: "passage",
      text: t.length > 400 ? t.slice(0, 400) + "…" : t,
      source: r.docTitle || "document",
      passageIndex: r.index,
      status: r.confidence === "partial" ? "walking" : "sig",
      similarity: r.score || 0,
    };
  });
  return { text: lines.length ? `[DOCS]\n${lines.join("\n")}\n[/DOCS]` : "", spans };
}

/* The first pass — mechanical NER, no model. Mints SIG entities the instant
   text arrives: an impression with a centroid but no provenance, weighted
   low until a walk reaches it. Records a mention per (entity, passage) and,
   given a Given-Log id, keeps the passage as an unwalked vector. Run on
   document passages and chat turns so material is chattable immediately. */
export async function firstPass(text, opts = {}) {
  const myScope = store.getScope();
  const sig = signal(text);
  const names = sig.ner.names.filter(n => n.length > 1);
  let vec = null;
  try { vec = await store.embed(String(text).slice(0, 500)); } catch { /* optional */ }
  store.setScope(myScope); // re-assert before the sync writes
  if (vec && opts.givenId) store.vectors.writeUnwalked(opts.givenId, vec);
  let created = 0;
  for (const name of names) {
    let ent = store.entities.search(name)
      .find(m => m.canonical.toLowerCase() === name.toLowerCase());
    if (!ent) {
      const id = store.mintEntityId(name);
      store.entities.create(id, name, "Entity", { status: "sig" });
      const kind = sig.ner.typed[name];
      if (kind) {
        const defId = "d_" + store.mintHash(`${id}::kind::${Date.now()}::${Math.random()}`);
        store.defs.write(defId, id, "kind", kind, { source: opts.givenId || "ner:firstpass" });
      }
      ent = store.entities.get(id);
      created++;
    }
    // While the entity is still an impression, fold this context into its
    // centroid (the drift is the surprise) and record the mention.
    if (ent && ent.status === "sig") {
      const drift = vec ? store.vectors.foldClause(ent.id, vec) : null;
      store.mentions.record(ent.id, {
        documentId: opts.documentId || null, passageIdx: opts.passageIdx ?? null,
        givenId: opts.givenId || null, context: String(text).slice(0, 400), drift,
      });
    }
  }
  return { created };
}

/* A provisional dossier entry for a SIG entity — impressionistic but not
   blind: passages seen, NER type, co-occurring words, the nearest grounded
   entity, and the drift trail. Honest about its epistemic state. */
function buildProvisionalEntry(entity) {
  const ms = store.mentions.forEntity(entity.id);
  const kindDef = store.defs.get(entity.id, "kind");
  let block = `≈ ${entity.canonical} | ${entity.terrain} | READING`;
  const docId = ms.find(m => m.document_id)?.document_id;
  if (docId) {
    const total = store.documents.get(docId)?.total || 0;
    block += ` (${ms.length} of ${total || "?"} passages seen)`;
  }
  if (kindDef) {
    const ps = ms.filter(m => m.passage_idx != null).map(m => m.passage_idx + 1);
    block += `\n  NER: ${kindDef.value}${ps.length ? ` (passages ${ps.join(", ")})` : ""}`;
  }
  const co = extractKeywords(ms.map(m => m.context || "").join(" ")).slice(0, 6);
  if (co.length) block += `\n  co-occurs: ${co.map(w => `"${w}"`).join(", ")}`;
  const near = store.vectors.nearestWalked(entity.id);
  if (near) block += `\n  similar to: ${near.entityId} ${near.canonical} (${near.sim.toFixed(2)})`;
  const drifts = ms.map(m => m.drift).filter(d => d != null);
  if (drifts.length > 1) block += `\n  drift: ${drifts.map(d => d.toFixed(2)).join(" → ")}`;
  return block;
}

/* ═══ The [STATUS] block — where the system is in its own reading ═══ */

export function buildStatus() {
  const c = store.statusCounts();
  const lines = [`knowledge: ${c.walked} grounded, ${c.walking} forming, ${c.sig} impressions`];
  for (const w of store.documents.inProgress()) {
    const pct = w.total ? Math.round((w.walked / w.total) * 100) : 0;
    lines.push(`reading: ${w.title || w.id} — ${pct}% (${w.walked}/${w.total} passages)`);
  }
  const shifts = store.recentShifts();
  if (shifts.length) {
    lines.push(`recent shifts: ${shifts.map(s => `${s.canonical} drifted ${Number(s.drift).toFixed(2)} on "${s.field}"`).join(", ")}`);
  }
  return `[STATUS]\n${lines.join("\n")}\n[/STATUS]`;
}

/* Where an entity's knowledge came from — one line, so the model can weigh
   a source document differently from something said in conversation. */
function originLine(entityId) {
  const o = store.entityOrigin(entityId);
  const parts = [...o.documents];
  if (o.classes.includes("conversation")) parts.push("this conversation");
  if (!parts.length && o.classes.includes("scan")) parts.push("an unread NER scan");
  if (!parts.length && o.classes.includes("inference")) parts.push("a system inference");
  return parts.length ? parts.join(" + ") : null;
}

/* ═══ Adaptive scale — compression activates only when the window overflows ═══

   A hypothesis is lossy compression of facts, forced by a finite prompt
   window. If the evidence already fits, compressing it is pure overhead and
   loses fidelity for nothing. Every layer — the walk, entity hypotheses,
   group/section/document hypotheses — exists to fit an unbounded document
   into a bounded window, and each activates only when the layer below
   overflows its budget. Below the window the system is plain RAG with
   provenance; above it, it compresses progressively up the hierarchy. */
export const DOSSIER_BUDGET = 350;   // tokens for the whole [CTX] dossier
const TIER_1_BUDGET = 65;            // tokens for one walked entity's slot

export function estimateTokens(text) {
  return Math.ceil(String(text || "").length / 4);
}

/* Tokens to show an entity's evidence raw — every DEF and edge, uncompressed. */
function entityEvidenceTokens(entityId) {
  return store.defs.getFor(entityId).length * 12 + store.edges.getFor(entityId).length * 8;
}

/* Does this entity's evidence overflow its dossier slot? Below the slot the
   DEFs are shown directly — more faithful than any hypothesis of them. */
export function shouldHypothesizeEntity(entityId) {
  return entityEvidenceTokens(entityId) > TIER_1_BUDGET;
}

/* Do the entity hypotheses in a passage range overflow the digest budget?
   Below it, the entity hypotheses are shown directly — no group needed. */
export function shouldHypothesizeGroup(start, end) {
  return store.entities.getInRange(start, end).length * 20 > 100;
}

/* Do the accumulated group hypotheses overflow — time to roll them up? */
export function shouldRollUpSection() {
  return store.hypotheses.getByLevel("group").length > 3;
}

/* The ingest strategy for a document, decided by its size against the window:
     raw       — fits the dossier whole; do not walk, read it directly
     defs-only — walk to DEFs, which still fit; no hypotheses
     full      — walk and compress through the hypothesis hierarchy */
export function shouldWalk(text) {
  const tokens = estimateTokens(text);
  if (tokens <= DOSSIER_BUDGET) return { strategy: "raw", hypothesize: false };
  if (tokens <= DOSSIER_BUDGET * 3) return { strategy: "defs-only", hypothesize: false };
  return { strategy: "full", hypothesize: true };
}

/* A grounded (walked) entity block — structured and provenanced. */
function formatWalked(entity, full = true) {
  const stateHash = store.computeStateHash(entity.id);
  const edges = store.edges.getFor(entity.id);
  const defs = store.defs.getFor(entity.id);
  const from = originLine(entity.id);
  let b = `E: ${entity.id}@${stateHash} | ${entity.terrain} | ${edges.length}`;
  b += `\n  ~ ${entity.canonical}${entity.aliases?.length ? ", aka " + entity.aliases.join(", ") : ""}`;
  if (from) b += `\n  from: ${from}`;

  if (shouldHypothesizeEntity(entity.id)) {
    // Evidence overflows the slot — show the compressed hypothesis + top DEFs.
    b += `\n  h: ${(entity.hypothesis || "?").slice(0, 130)}`;
    let spanDone = false;
    for (const def of defs.slice(0, full ? 3 : 2)) {
      b += `\n  = ${def.field}: "${def.value}"`;
      if (def.span && !spanDone && full) { b += ` @"${def.span.slice(0, 50)}"`; spanDone = true; }
    }
    if (full) {
      for (const c of store.defs.getConflicts(entity.id).slice(0, 1)) b += `\n  ⚠ ${c.field}: ${c.vals}`;
    }
  } else {
    // Evidence fits the slot — show every DEF and edge raw. The facts
    // themselves are more faithful than any hypothesis compressing them.
    for (const edge of edges.slice(0, 3)) {
      const dir = edge.from_id === entity.id ? "→" : "←";
      const target = edge.from_id === entity.id ? edge.to_id : edge.from_id;
      b += `\n  ${dir} ${target} (${edge.type})`;
    }
    let spanDone = false;
    for (const def of defs) {
      b += `\n  = ${def.field}: "${def.value}"`;
      if (def.span && !spanDone) { b += ` @"${def.span.slice(0, 50)}"`; spanDone = true; }
    }
  }
  return b;
}

/* A forming (walking) entity block — partial structure, still being read. */
function formatWalking(entity) {
  const from = originLine(entity.id);
  let b = `E: ${entity.id}@${store.computeStateHash(entity.id)} | ${entity.terrain} | PARTIAL`;
  b += `\n  ~ ${entity.canonical}`;
  if (from) b += `\n  from: ${from}`;
  b += `\n  h: ${entity.hypothesis || "(forming)"}`;
  for (const def of store.defs.getFor(entity.id).slice(0, 2)) b += `\n  = ${def.field}: "${def.value}"`;
  return b;
}

/* ═══ Format the retrieved results into the [CTX] block ═══

   Mixed-grade: grounded entities (E:), forming entities (PARTIAL), SIG
   impressions and raw unwalked passages (≈). Each carries its provenance. */
export function formatRetrieved(results, maxTokens = 350) {
  let budget = maxTokens;
  const blocks = [];
  for (const r of results) {
    if (budget <= 0) break;
    if (r.type === "entity") {
      const entity = store.entities.get(r.id);
      if (!entity) continue;
      if (entity.status === "walked") {
        if (budget >= 65) { blocks.push(formatWalked(entity, true)); budget -= 65; }
        else if (budget >= 35) { blocks.push(formatWalked(entity, false)); budget -= 35; }
      } else if (entity.status === "walking") {
        if (budget >= 40) { blocks.push(formatWalking(entity)); budget -= 40; }
      } else {
        if (budget >= 25) { blocks.push(buildProvisionalEntry(entity)); budget -= 25; }
      }
    } else { // raw unwalked passage
      const doc = store.documents.get(r.documentId);
      const docLabel = doc?.title ? `document "${doc.title}"` : "an imported document";
      if (budget >= 40) {
        const t = r.text.length > 200 ? r.text.slice(0, 200) + "…" : r.text;
        let b = `≈ passage ${r.passageIndex + 1} | from ${docLabel} | UNREAD`;
        b += `\n  "${t}"`;
        if (r.ner?.length) b += `\n  mentions: ${r.ner.join(", ")}`;
        blocks.push(b); budget -= 40;
      } else if (budget >= 20) {
        blocks.push(`≈ passage ${r.passageIndex + 1} from ${docLabel} | "${r.text.slice(0, 120)}…"`);
        budget -= 20;
      }
    }
  }
  return blocks.length
    ? `[CTX]\n${blocks.join("\n\n")}\n[/CTX]`
    : "[CTX]\n(no matching context)\n[/CTX]";
}

export function dossierHashOf(dossier) {
  let h = 0x811c9dc5;
  const s = String(dossier || "");
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = (h * 0x01000193) >>> 0; }
  return "d_" + h.toString(16).padStart(8, "0");
}

/* Source spans behind the projected dossier — DEF spans, EVA judgments and
   the raw text of any unwalked passages, each tagged with its origin. */
export function collectSpans(results) {
  const spans = [];
  for (const r of results || []) {
    if (r.type === "entity") {
      for (const d of store.defs.getFor(r.id)) {
        spans.push({ entity: r.id, kind: "def", text: `${d.field}: ${d.value}`, source: d.source || d.span || null });
      }
      for (const v of store.evals.getFor(r.id)) {
        spans.push({ entity: r.id, kind: "eva", text: `${v.claim} [${v.status}]`, source: v.source || null });
      }
    } else if (r.type === "passage") {
      spans.push({ entity: r.documentId, kind: "passage", text: r.text.slice(0, 200), source: r.id });
    }
  }
  return spans;
}

/* ═══ The Reading Digest — projected running understanding ═══

   The walk produces a hierarchy of hypotheses as it reads: group (every few
   passages), section (every dozen), document (the whole). They are the
   system's own synthesis — exactly what a "what have you learned / what is
   this about" question wants — but the Reach only retrieves entities and raw
   passages, so they were never shown to the model.

   This projects them at the zoom the question calls for (see selectZoom).
   The document hypothesis is the headline, marked honestly: a SKIM is the
   first-pass guess from NER and embeddings before the walk; a walked
   DOCUMENT hypothesis is situated understanding synthesised up through the
   hierarchy. Every group/section hypothesis is written for a completed
   passage range, so the set already covers exactly what the reading cursor
   has reached. */
export function buildReadingDigest(query = "") {
  const sectionHyps = store.hypotheses.getByLevel("section");
  const groupHyps = store.hypotheses.getByLevel("group");
  const docs = store.documents.all();
  if (!docs.length && !sectionHyps.length && !groupHyps.length) return "";

  const zoom = selectZoom(query);
  const lines = [];

  // Per-document headline — title, reading cursor, and the document
  // hypothesis marked as a skim guess or a walked synthesis.
  for (const d of docs) {
    const total = d.total || 0, walked = d.walked || 0;
    const docHyp = store.hypotheses.getCurrent("document", d.id);
    const fully = total > 0 && walked >= total;
    const progress = !total ? ""
      : fully ? ` — fully read (${total} passages)`
      : ` — ${Math.round((walked / total) * 100)}% read (${walked}/${total} passages)`;
    if (docHyp) {
      const skim = docHyp.after_label === "skim" || !fully;
      lines.push(skim
        ? `skim of "${d.title || "document"}"${progress} — a first-pass guess, not yet fully read: "${docHyp.text}"`
        : `document "${d.title || "document"}"${progress}: "${docHyp.text}"`);
    }
  }

  // Detail layer at the selected zoom. An entity-zoom question is answered
  // from [CTX]; the others get the threads (group) or sections.
  if (zoom !== "entity") {
    const detail = zoom === "group"
      ? (groupHyps.length ? groupHyps : sectionHyps)
      : (sectionHyps.length ? sectionHyps : groupHyps);
    const label = detail === groupHyps ? "thread" : "section";
    detail.slice(0, 10).forEach((h, i) => lines.push(`  ${label} ${i + 1}: "${h.text}"`));
  }

  return lines.length ? `[READING]\n${lines.join("\n")}\n[/READING]` : "";
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

export function buildRegister(cap = 20) {
  const entities = store.entities.getAll()
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0)).slice(0, cap);
  if (!entities.length) return "Entity register: (empty — first pass)";
  return "Entity register:\n" + entities.map(e =>
    `${e.id}@${store.computeStateHash(e.id)} | ${e.canonical}${e.aliases?.length ? " | aka: " + e.aliases.join(", ") : ""} | ${e.terrain}`
  ).join("\n");
}

export function buildReaderCursor(introduced) {
  if (introduced.length <= 10) return `[READER]\nIntroduced: ${introduced.join(", ")}\n[/READER]`;
  return `[READER]\n${introduced.length} entities introduced. Recent: ${introduced.slice(-5).join(", ")}\n[/READER]`;
}

/* ═══ EXTRACT / MUTATE prompt builders ═══ */

export function buildExtractPrompt(userMessage, modelResponse, userGivenId, modelGivenId) {
  return `${buildRegister()}

EXCHANGE:
User [${userGivenId}]: "${String(userMessage).slice(0, 4000)}"
Model [${modelGivenId}]: "${String(modelResponse).slice(0, 4000)}"`;
}

export function buildMutatePrompt(trigger) {
  const { name, candidateHash, span, context } = trigger;
  let prompt = `Ambiguous reference: "${name || "(unnamed)"}"`;
  if (span) prompt += `\nSource span: "${span}"`;
  if (context) prompt += `\nContext: "${context}"`;
  const candidate = candidateHash ? store.entities.get(candidateHash) : null;
  if (candidate) {
    prompt += `\n\nExisting entity that might match:`;
    prompt += `\n  ${candidate.id}@${store.computeStateHash(candidate.id)}`;
    prompt += `\n  canonical: ${candidate.canonical}`;
    prompt += `\n  terrain: ${candidate.terrain}`;
    prompt += `\n  h: ${candidate.hypothesis || "?"}`;
    for (const d of store.defs.getFor(candidate.id).slice(0, 6)) {
      prompt += `\n  = ${d.field}: "${d.value}"`;
    }
    for (const c of store.defs.getConflicts(candidate.id)) {
      prompt += `\n  ⚠ ${c.field}: ${c.vals}`;
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

function getChildInputs(level, id) {
  switch (level) {
    case "entity": {
      const e = store.entities.get(id);
      if (!e) return [];
      const defs = store.defs.getFor(id), edges = store.edges.getFor(id);
      const lines = [`${id}@${store.computeStateHash(id)} | ${e.canonical} | ${e.terrain}`];
      if (defs.length) lines.push(`Facts: ${defs.map(d => `${d.field}="${d.value}"`).join(", ")}`);
      if (edges.length) lines.push(`Connections: ${edges.map(x => `→${x.to_id} (${x.type})`).join(", ")}`);
      return lines;
    }
    case "group":
      return store.entities.getInRange(id.start, id.end)
        .map(e => `${e.id}@${store.computeStateHash(e.id)} ${e.canonical}: "${e.hypothesis || "?"}"`);
    case "section":
      return store.hypotheses.getByLevel("group")
        .map((g, i) => `group ${i + 1}: "${g.text}"`);
    case "document":
      return store.hypotheses.getByLevel("section")
        .map((s, i) => `section ${i + 1}: "${s.text}"`);
    case "session":
      return store.hypotheses.getByLevel("document")
        .map(d => `document: "${d.text}"`);
    case "corpus":
      return [
        ...store.hypotheses.getByLevel("document").map(d => `document: "${d.text}"`),
        ...store.hypotheses.getByLevel("session").map(s => `session: "${s.text}"`),
      ];
    default: return [];
  }
}

/* A skim prompt — passage openings plus the names NER caught, the way a
   reader glances over a document before reading it closely. */
export function buildSkimPrompt(passages) {
  const skim = passages.slice(0, 12).map(p => p.slice(0, 200).trim()).join("\n---\n");
  const names = [...new Set(passages.flatMap(p => signal(p).ner.names))].slice(0, 20);
  let prompt = "";
  if (names.length) prompt += `Names that jump out: ${names.join(", ")}\n\n`;
  prompt += `Skim (passage openings):\n${skim}`;
  return prompt;
}

export function buildHypothesisPrompt(level, id) {
  const children = getChildInputs(level, id);
  const targetKey = typeof id === "string" ? id
    : id?.documentId || id?.sessionId || (id?.start != null ? `${id.start}-${id.end}` : "_");
  const history = store.hypotheses.getHistory(level, targetKey);
  let prompt = children.join("\n");
  if (history.length) {
    prompt += "\n\nPrior hypotheses (oldest first):";
    history.forEach((h, i) => {
      prompt += `\n  rev ${i + 1} (${h.after_label || ""}, ${h.input_count || 0} inputs): "${h.text}"`;
    });
  }
  return prompt;
}

/* ═══ Parsing ═══ */

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

/* Parse a grounded answer into runs. `spanCount` bounds valid citation
   numbers — any cite outside [1, spanCount] is dropped. On malformed output
   the whole reply collapses into one uncited run so the answer still renders. */
export function parseRuns(out, spanCount = 0) {
  const raw = typeof out === "string" ? out : "";
  let obj = out;
  if (typeof out === "string") {
    let s = out.replace(/```json/gi, "").replace(/```/g, "").trim();
    const a = s.indexOf("{"), b = s.lastIndexOf("}");
    if (a !== -1 && b !== -1) s = s.slice(a, b + 1);
    try { obj = JSON.parse(s); } catch { obj = null; }
  }
  const list = Array.isArray(obj) ? obj : (obj && Array.isArray(obj.runs) ? obj.runs : null);
  if (!list) return [{ text: raw.trim() || "(no answer)" }];

  const runs = [];
  for (const r of list) {
    if (!r || typeof r.text !== "string" || !r.text) continue;
    const run = { text: r.text };
    if (r.ungrounded === true) {
      run.ungrounded = true;
    } else if (Array.isArray(r.cites)) {
      const cites = r.cites
        .map(n => Number(n))
        .filter(n => Number.isInteger(n) && n >= 1 && n <= spanCount);
      if (cites.length) run.cites = cites;
    }
    runs.push(run);
  }
  return runs.length ? runs : [{ text: raw.trim() || "(no answer)" }];
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

/* ═══ MUTATE: triggers and apply ═══ */

const REF_UNCERTAIN = [
  /\bmay not be (?:the same|identical)(?: as)? (e_[0-9a-f]{8})/i,
  /\bmight be (?:a )?different (?:from |than )?(e_[0-9a-f]{8})/i,
  /\bnot (?:the same|identical) (?:as |to )(e_[0-9a-f]{8})/i,
  /\b(e_[0-9a-f]{8})\b[^.?!]*?\b(?:may not|might not|not the same)\b/i,
];
const CORRECTION_RE = /\b(?:that(?:'s| is) not the same|those are the same|different (?:person|place|thing|one|entity)|you(?:'re| are) (?:confusing|mixing)|not who i meant|i meant (?:a )?different|wrong (?:person|entity|one)|mixed (?:them|it) up)\b/i;

export function detectMutationTriggers(modelResponse, extractEvents, sig) {
  const triggers = [];
  for (const re of REF_UNCERTAIN) {
    const m = re.exec(modelResponse || "");
    if (m && store.entities.get(m[1])) {
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
    const matches = store.entities.search(name);
    if (matches.length === 1) {
      const kindDef = store.defs.get(matches[0].id, "kind");
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

export function userCorrectionTrigger(userMessage, lastEntities = []) {
  if (!CORRECTION_RE.test(userMessage || "")) return null;
  const candidateHash = lastEntities.find(id => store.entities.get(id)) || null;
  return { type: "user_correction", candidateHash, context: String(userMessage).slice(0, 200) };
}

/* Apply a MUTATE action to the graph. Logs the mutation. */
export function applyMutation(action, triggerId) {
  switch (action.action) {
    case "FORK": {
      const src = store.entities.get(action.source);
      if (src && action.new_canonical) {
        const id = store.mintEntityId(action.new_canonical);
        store.entities.create(id, action.new_canonical, src.terrain, { forkedFrom: action.source });
      }
      break;
    }
    case "MERGE":
      if (store.entities.get(action.keep) && store.entities.get(action.absorb)) {
        store.mergeEntities(action.keep, action.absorb);
        for (const a of action.new_aliases || []) store.entities.addAlias(action.keep, a);
      }
      break;
    case "CORRECT": {
      const ent = store.entities.get(action.entity);
      if (ent && action.field) {
        for (const d of store.defs.getFor(action.entity)) if (d.field === action.field) store.defs.retire(d.id);
        const defId = "d_" + store.mintHash(`${action.entity}::${action.field}::${Date.now()}::${Math.random()}`);
        store.defs.write(defId, action.entity, action.field, action.new_value, { source: "mutate:correct" });
      }
      break;
    }
    case "RECLASSIFY":
      if (store.entities.get(action.entity)) store.entities.updateTerrain(action.entity, action.new_terrain);
      break;
    case "NONE":
    default:
      break;
  }
  store.mutations.log(action.action, action, action.reason, triggerId || null, "applied");
}

/* ═══ Auto-commit tiers ═══

   AUTO commits silently. PROMPT surfaces an inline consent pill. REQUIRE
   blocks until the user acts. */
export function commitTier(action) {
  switch (action) {
    case "FORK": case "MERGE": case "RECLASSIFY": return "PROMPT";
    case "CORRECT": return "PROMPT";
    case "NONE": return "AUTO";
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
