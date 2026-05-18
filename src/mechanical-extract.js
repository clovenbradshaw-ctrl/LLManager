/* Mechanical extraction — Phase A of the two-phase walk.

   The expensive way to read a document is to send every passage to the LLM
   and ask it to generate structure from raw text. Generation is hard for
   small models. This module does the reading mechanically — NLP + regex,
   no LLM, a few milliseconds per clause — and produces a structured DRAFT
   of INS/DEF/CON events with spans and heuristic coreference.

   Classification is then mostly mechanical too, grounded in the EO
   reference: terrain follows from the extracted kind, DEF/EVA from a
   judgment lexicon. The LLM is called (Phase B) only for the cases the
   reference marks genuinely ambiguous — an organization (Entity vs
   Network), a quoted work (Kind vs Entity) — a tiny two-way choice. Most
   passages make zero LLM calls; the model is a scarce adjudicator, not a
   parser or generator.

   compromise.js does English NER. Everything else is domain regex. The
   pipeline is English-only and falls back to the full LLM walk on text it
   cannot handle (see shouldFallback). */

import nlp from "compromise";
import { cosineSim } from "./local-store.js";

/* Cosine margin below which the nearest two terrains count as a tie — only
   then does a referent's terrain go to the LLM. MiniLM cosine spreads are
   narrow, so this is deliberately small. */
const TIE_MARGIN = 0.04;

/* Split text into clauses — the minimal unit of extraction. Sentence-ending
   punctuation, semicolons, and colons before a capital or quote. Shorter
   clauses mean tighter span attribution. */
export function splitClauses(text) {
  const raw = String(text || "").split(/(?<=[.!?;])\s+|(?<=:)\s+(?=[“"A-Z])/);
  return raw.map(c => c.trim()).filter(c => c.length > 5);
}

/* Find quoted titles BEFORE NER runs. "Marty, Life Is Short" is one entity
   (a work), not three people named Marty, Life and Short. */
export function extractQuotedTitles(text) {
  const titles = [];
  const pattern = /[“"]\s*([A-Z][^"”]{3,60})\s*[”"]/g;
  let m;
  while ((m = pattern.exec(String(text || ""))) !== null) {
    titles.push({ title: m[1].trim(), fullMatch: m[0], index: m.index });
  }
  return titles;
}

/* Classify a clause as content (extractable) or chrome (scaffolding —
   attributions, transitions, filler). Chrome is stored but yields no events. */
export function classifyChrome(text) {
  const patterns = [
    /^in the (film|book|article|interview|documentary)/i,
    /^(he|she|they|it) (said|says|added|noted|explained|responded|replied)/i,
    /^(according to|as .+ (said|noted|put it))/i,
  ];
  const t = String(text || "").trim();
  const words = t.split(/\s+/).length;
  const hasProperNoun = /[A-Z][a-z]{2,}/.test(t.replace(/^[""“]/, ""));
  const isPattern = patterns.some(p => p.test(t));
  if (isPattern && words < 12 && !hasProperNoun) return { type: "chrome", confidence: "high" };
  if (words < 8 && !hasProperNoun) return { type: "chrome", confidence: "medium" };
  return { type: "content", confidence: "high" };
}

/* Strip leading/trailing punctuation and possessives from an NER result. */
function cleanName(name) {
  return String(name || "")
    .replace(/^[“”"',.\s]+/, "")
    .replace(/[“”"',.\s]+$/, "")
    .replace(/’s$/i, "")
    .replace(/'s$/i, "")
    .trim();
}

/* Reject the pronouns, articles and lowercase strings compromise sometimes
   returns as names. */
const NON_NAMES = new Set([
  "he", "she", "they", "it", "his", "her", "their", "its",
  "this", "that", "the", "when", "five", "in the", "there",
  "what", "who", "where", "how", "a", "an",
]);
function isRealName(name) {
  const cleaned = cleanName(name);
  if (cleaned.length < 2) return false;
  if (NON_NAMES.has(cleaned.toLowerCase())) return false;
  if (!/[A-Z]/.test(cleaned)) return false;
  return true;
}

/* compromise tags common-noun openers ("brother David Short") into a person
   span; keep the trailing proper-noun run. */
function trimLeadingCommonNoun(name) {
  const words = cleanName(name).split(/\s+/);
  let start = 0;
  while (start < words.length - 1 && !/^[A-Z]/.test(words[start])) start++;
  return words.slice(start).join(" ");
}

/* Common-noun referents — phrases like "drought", "the political climate",
   "dark comedy" that are NOT named entities but DO name something the graph
   should hold (an ambient condition, an interpretive frame, a relational
   field). compromise's doc.nouns() surfaces them; the filter below is the
   anti-spam gate. */
const REFERENT_STOPS = new Set([
  "the", "a", "an", "this", "that", "these", "those", "it", "its",
  "thing", "things", "way", "ways", "time", "times", "year", "years",
  "day", "days", "lot", "lots", "part", "parts", "case", "fact", "point",
  "number", "sort", "one", "ones", "something", "someone", "anyone",
  "everyone", "nothing", "everything", "anything", "he", "she", "they",
  "we", "you", "what", "who", "how", "why", "when", "where", "side",
]);
const LEADING_DET = /^(?:the|a|an|this|that|these|those|his|her|their|its|our|my|your|some|any|no)\s+/i;

export function extractReferents(doc, result) {
  const named = new Set();
  for (const list of [result.people, result.places, result.orgs, result.titles]) {
    for (const n of list) for (const tok of n.toLowerCase().split(/\s+/)) named.add(tok);
  }
  const seen = new Set();
  const out = [];
  for (const rawPhrase of doc.nouns().out("array")) {
    const trimmed = String(rawPhrase || "")
      .replace(/^[“”"',.;:\s]+/, "").replace(/[“”"',.;:\s]+$/, "").trim();
    if (!trimmed) continue;
    const rawTokens = trimmed.split(/\s+/);
    // A proper-noun-headed phrase is NER's job — skip it here.
    if (/^[A-Z]/.test(rawTokens[rawTokens.length - 1])) continue;
    const phrase = trimmed.replace(LEADING_DET, "").toLowerCase().trim();
    if (phrase.length < 4 || seen.has(phrase)) continue;
    const tokens = phrase.split(/\s+/);
    if (tokens.some(t => named.has(t))) continue;            // already a named entity
    if (tokens.every(t => REFERENT_STOPS.has(t) || NON_NAMES.has(t))) continue;
    if (/^\d/.test(phrase) || /^(year|day|month|week|hour)s?$/.test(phrase)) continue;
    seen.add(phrase);
    out.push({ phrase, headNoun: tokens[tokens.length - 1], words: tokens.length });
  }
  // Prefer multi-word phrases; cap per clause so a passage cannot spam the graph.
  out.sort((a, b) => b.words - a.words);
  result.referents = out.slice(0, 3).map(({ phrase, headNoun }) => ({ phrase, headNoun }));
  return result.referents;
}

/* Extract structured features from one clause. corefStack is shared across
   the document, most-recent-first, and is mutated here. */
export function extractClause(text, clauseIndex, allTitles, corefStack) {
  const doc = nlp(String(text || ""));
  const result = {
    people: [], places: [], orgs: [], titles: [],
    deaths: [], relationships: [], temporals: [],
    numbers: [], quotes: [], coreferences: [], referents: [],
  };
  const titleNames = new Set((allTitles || []).map(t => t.title.toLowerCase()));

  // ── NER, cleaned ──
  const people = doc.people().out("array").map(trimLeadingCommonNoun).map(cleanName).filter(isRealName);
  const places = doc.places().out("array").map(cleanName).filter(isRealName);
  const orgs = doc.organizations().out("array").map(cleanName).filter(isRealName);
  result.people = [...new Set(people.filter(p => !titleNames.has(p.toLowerCase())))];
  result.places = [...new Set(places)];
  result.orgs = [...new Set(orgs)];
  for (const t of allTitles || []) {
    if (text.includes(t.title) || text.includes(t.fullMatch)) result.titles.push(t.title);
  }

  // ── Coreference stack — push newly seen proper names to the front ──
  for (const p of result.people) {
    const idx = corefStack.indexOf(p);
    if (idx > -1) corefStack.splice(idx, 1);
    corefStack.unshift(p);
  }
  const referent = corefStack[0] || null;

  // possessives — "his mother" → referent's mother
  let pm;
  const possessiveP = /\b(his|her|their)\s+(\w+(?:\s\w+)?)/gi;
  while ((pm = possessiveP.exec(text)) !== null) {
    result.coreferences.push({ pronoun: pm[0], resolvedTo: referent, type: "possessive" });
  }
  // pronoun subjects — "he said" → referent said
  let ps;
  const pronounP = /\b(he|she)\s+(says?|said|responded|added|noted|emphasized)\b/gi;
  while ((ps = pronounP.exec(text)) !== null) {
    result.coreferences.push({ pronoun: ps[1], resolvedTo: referent, type: "subject" });
  }
  // surname-only — "Short" → "Martin Short" from the stack
  let sn;
  const surnameP = /\b([A-Z][a-z]+)\b/g;
  while ((sn = surnameP.exec(text)) !== null) {
    const surname = sn[1];
    const fullName = corefStack.find(n => n.endsWith(" " + surname) && n !== surname);
    if (fullName && !result.people.includes(surname)) {
      result.coreferences.push({ pronoun: surname, resolvedTo: fullName, type: "surname" });
    }
  }

  // ── Death patterns ──
  let dm;
  const deathP = /(?:(\w+(?:\s\w+)?)|his\s+(\w+(?:\s\w+)?))\s+died\s+(?:of|in|from)\s+(.+?)(?:\.|;|,|$)/gi;
  while ((dm = deathP.exec(text)) !== null) {
    const who = dm[1] || (referent ? `${referent}'s ${dm[2]}` : dm[2]);
    result.deaths.push({ who, relation: dm[2] || null, cause: dm[3].trim(), span: dm[0].trim() });
  }
  let sp;
  const strokeP = /(?:(\w+(?:\s\w+)?)|his\s+(\w+))\s+had\s+a\s+(\w+)\s+and\s+passed\s+away/gi;
  while ((sp = strokeP.exec(text)) !== null) {
    const who = sp[1] || (referent ? `${referent}'s ${sp[2]}` : sp[2]);
    result.deaths.push({ who, relation: sp[2] || null, cause: sp[3], span: sp[0].trim() });
  }

  // ── Relationship patterns — "Michael Short, Martin's brother" ──
  let rm;
  const relP = /([A-Z]\w+(?:\s[A-Z]\w+)?)\s*,\s*([A-Z]\w+)(?:'s|’s)\s+(brother|sister|mother|father|son|daughter|wife|husband|friend|colleague)/gi;
  while ((rm = relP.exec(text)) !== null) {
    result.relationships.push({
      person: cleanName(rm[1]), of: cleanName(rm[2]), relation: rm[3], span: rm[0].trim(),
    });
  }

  // ── Temporal expressions — context, never entities ──
  const tempPatterns = [
    /when\s+\w+(?:\s\w+)?\s+was\s+(\d+)/gi,
    /at\s+(\d+)/gi,
    /(\w+)\s+years?\s+(later|after|before|earlier)/gi,
    /leaving\s+\w+,?\s+at\s+(\d+)/gi,
  ];
  for (const tp of tempPatterns) {
    let tm;
    while ((tm = tp.exec(text)) !== null) result.temporals.push({ text: tm[0], value: tm[1] });
  }

  // ── Numerical facts, resolved to the current referent ──
  let nm;
  const numP = /(?:was|at|aged?)\s+(\d+)/gi;
  while ((nm = numP.exec(text)) !== null) {
    result.numbers.push({ value: nm[1], span: nm[0], entity: referent });
  }
  let om;
  const ordP = /the\s+(youngest|oldest|eldest)\s+of\s+(\w+)/gi;
  while ((om = ordP.exec(text)) !== null) {
    result.numbers.push({ value: `${om[1]} of ${om[2]}`, span: om[0], entity: referent });
  }

  // ── Quoted speech ──
  let qm;
  const quoteP = /[“"]([^"”]+)[”"]/g;
  while ((qm = quoteP.exec(text)) !== null) {
    const quoteText = qm[1].trim();
    if (titleNames.has(quoteText.toLowerCase())) continue;
    if (quoteText.length > 10) result.quotes.push({ text: quoteText, speaker: referent });
  }

  // ── Common-noun referents — runs after NER so it can skip named entities ──
  extractReferents(doc, result);

  return result;
}

/* ═══ EO classification — mechanical, per the canonical reference ═══

   The EO reference fixes most of the classification deterministically, so
   it never reaches the LLM:

   Terrain (Site = Domain × Object). A named person, place or event is a
   Figure of Existence — an Entity. Only two extracted kinds are genuinely
   ambiguous: an organization is an Entity or a Network, a quoted work is a
   Kind or an Entity. Those, and only those, become an LLM task.

   Operator (DEF vs EVA). DEF establishes what holds — a fact, a value, a
   status. EVA renders judgment — testing a particular against a standard.
   A clause carrying evaluative language is EVA; a plain factual claim is
   DEF. The split is a lexicon check, not an LLM call. */
const KIND_TERRAIN = {
  person: "Entity",
  place: "Entity",
  event: "Entity",
  organization: null,   // Entity or Network — the LLM decides
  work: null,           // Kind or Entity — the LLM decides
};

export function kindTerrain(kind) {
  const t = KIND_TERRAIN[kind];
  return { terrain: t || "?", ambiguous: !t };
}

/* Evaluative language — judgments, verdicts, tests against a standard. */
const EVA_LEXICON = /\b(rough|brilliant|great|terrible|poor|excellent|impressive|disappointing|successful|fail(s|ed|ing)?|wonderful|awful|remarkable|outstanding|mediocre|fell short|exceeded|lived up to|confirm(s|ed)?|refut(es|ed)|the point|worth it|beautiful|smart|stupid|best|worst|better|worse|should(n'?t)?|ought to|matters?)\b/i;

/* DEF establishes what holds; EVA renders a judgment. A clause with no
   evaluative language is a plain factual claim — a DEF. */
export function classifyDefEva(text) {
  return EVA_LEXICON.test(String(text || "")) ? "EVA" : "DEF";
}

/* Convert one clause's extraction into draft graph events plus the LLM
   classification tasks the review step needs. knownEntities dedups INS
   across the document. Most clauses produce zero LLM tasks — terrain comes
   from the kind, DEF/EVA from the lexicon; only an organization or a quoted
   work defers to the LLM. */
export function clauseToEvents(extraction, knownEntities) {
  const events = [];
  const llmTasks = [];

  const groups = [
    { list: extraction.people, kind: "person" },
    { list: extraction.places, kind: "place" },
    { list: extraction.orgs, kind: "organization" },
    { list: extraction.titles, kind: "work" },
  ];
  for (const { list, kind } of groups) {
    for (const name of list) {
      const key = name.toLowerCase();
      if (knownEntities.has(key)) continue;
      knownEntities.add(key);
      const { terrain, ambiguous } = kindTerrain(kind);
      events.push({ op: "INS", entity: name, terrain, span: name, mechanical: true });
      events.push({ op: "DEF", entity: name, field: "kind", value: kind, span: name, mechanical: true });
      if (ambiguous) llmTasks.push({ task: "CLASSIFY_TERRAIN", entity: name, kind });
    }
  }
  for (const d of extraction.deaths) {
    events.push({ op: "DEF", entity: d.who, field: "death", value: d.cause, span: d.span, mechanical: true });
  }
  for (const r of extraction.relationships) {
    events.push({ op: "CON", from: r.person, to: r.of, type: r.relation, span: r.span, mechanical: true });
  }
  for (const n of extraction.numbers) {
    if (n.entity) {
      events.push({ op: "DEF", entity: n.entity, field: "age_or_count", value: n.value, span: n.span, mechanical: true });
    }
  }
  for (const c of extraction.coreferences) {
    if (c.resolvedTo && c.type === "surname") {
      events.push({ op: "COREF", pronoun: c.pronoun, resolvedTo: c.resolvedTo, type: c.type, mechanical: true });
    }
  }
  for (const q of extraction.quotes) {
    const op = classifyDefEva(q.text);
    events.push(op === "EVA"
      ? { op: "EVA", entity: q.speaker || "?", claim: q.text.slice(0, 100), status: "holds", span: q.text, mechanical: true }
      : { op: "DEF", entity: q.speaker || "?", field: "quote", value: q.text.slice(0, 100), span: q.text, mechanical: true });
  }
  return { events, llmTasks };
}

/* Should this clause fall back to a full LLM INGEST call? Mechanical
   extraction handles ~80% of cases; a content clause it found nothing in
   needs the LLM. */
export function shouldFallback(clauseText, chrome, extraction) {
  if (chrome.type === "chrome") return false;
  const n = extraction.people.length + extraction.places.length + extraction.orgs.length
    + extraction.deaths.length + extraction.relationships.length
    + extraction.numbers.length + extraction.quotes.length;
  return n === 0 && String(clauseText || "").split(/\s+/).length > 8;
}

/* Main entry — extract a whole document mechanically. */
export function extractDocument(text) {
  const t0 = (typeof performance !== "undefined" ? performance.now() : Date.now());
  const titles = extractQuotedTitles(text);
  const clauses = splitClauses(text);

  const corefStack = [];
  const knownEntities = new Set();
  const allEvents = [];
  const allLLMTasks = [];
  const clauseResults = [];

  for (let i = 0; i < clauses.length; i++) {
    const clause = clauses[i];
    const chrome = classifyChrome(clause);
    const extraction = extractClause(clause, i, titles, corefStack);
    const { events, llmTasks } = clauseToEvents(extraction, knownEntities);
    for (const evt of events) { evt.clauseIndex = i; evt.clauseText = clause; }
    for (const task of llmTasks) task.clauseIndex = i;
    allEvents.push(...events);
    allLLMTasks.push(...llmTasks);
    clauseResults.push({
      clause, index: i, chrome, extraction, events, llmTasks,
      fallback: shouldFallback(clause, chrome, extraction),
    });
  }

  return {
    clauses: clauseResults,
    events: allEvents,
    llmTasks: allLLMTasks,
    corefStack, titles,
    stats: {
      elapsed: (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0,
      clauseCount: clauses.length,
      mechanicalEvents: allEvents.length,
      llmTaskCount: allLLMTasks.length,
    },
  };
}

/* ═══ Phase B — the LLM review ═══ */

const TERRAINS = "Entity, Network, Paradigm, Void, Kind, Field, Link, Atmosphere, Lens";

export const REVIEW_SYSTEM =
  "Classify each item exactly as asked. You are editing a structured draft, "
  + "not writing prose. Return ONLY a JSON array: [{\"index\":1,\"result\":\"...\"}].";

/* Build the review prompt for one clause's tasks — only the items the
   mechanical rules could not settle. Each is a two-way choice between the
   terrains the EO reference marks ambiguous, so the call stays tiny. */
export function buildReviewPrompt(llmTasks) {
  if (!llmTasks || !llmTasks.length) return null;
  let prompt = "Classify each item. Return a JSON array, one object per item.\n\n";
  llmTasks.forEach((task, i) => {
    if (task.task !== "CLASSIFY_TERRAIN") return;
    const opts = task.kind === "organization"
      ? "Entity (one specific organization) or Network (a system, platform, market or industry)"
      : task.kind === "work"
      ? "Entity (one specific work) or Kind (a genre or category)"
      : `one of: ${TERRAINS}`;
    prompt += `${i + 1}. "${task.entity}" — choose ${opts}\n`;
  });
  prompt += `\nReturn ONLY a JSON array like: [{"index":1,"result":"Entity"}]`;
  return prompt;
}

export const REVIEW_SCHEMA = {
  type: "array",
  items: {
    type: "object",
    properties: { index: { type: "integer" }, result: { type: "string" } },
    required: ["index", "result"],
  },
};

/* Parse the review call's JSON array, tolerating markdown fences or prose
   around it. */
export function parseReview(out) {
  const s = typeof out === "string" ? out : JSON.stringify(out ?? "");
  try {
    const direct = JSON.parse(s);
    if (Array.isArray(direct)) return direct.filter(r => r && Number.isFinite(r.index));
  } catch { /* fall through to bracket scan */ }
  const a = s.indexOf("["), b = s.lastIndexOf("]");
  if (a > -1 && b > a) {
    try {
      const arr = JSON.parse(s.slice(a, b + 1));
      if (Array.isArray(arr)) return arr.filter(r => r && Number.isFinite(r.index));
    } catch { /* give up — the draft applies without review */ }
  }
  return [];
}

const VALID_TERRAINS = new Set(TERRAINS.split(", "));

/* Merge the LLM's terrain classifications back into the draft INS events.
   COREF events are an audit trail only and are dropped here. */
export function mergeReviewResults(draftEvents, llmTasks, llmResults) {
  const resultMap = {};
  for (const r of llmResults || []) resultMap[r.index] = r.result;

  (llmTasks || []).forEach((task, i) => {
    const result = resultMap[i + 1];
    if (result && task.task === "CLASSIFY_TERRAIN") {
      const ins = draftEvents.find(e =>
        e.op === "INS" && e.entity.toLowerCase() === String(task.entity).toLowerCase());
      if (ins && VALID_TERRAINS.has(result)) ins.terrain = result;
    }
  });

  // Any ambiguous INS the review did not resolve defaults to Entity.
  for (const e of draftEvents) {
    if (e.op === "INS" && (!e.terrain || e.terrain === "?")) e.terrain = "Entity";
  }
  return draftEvents.filter(e => e.op !== "COREF");
}
