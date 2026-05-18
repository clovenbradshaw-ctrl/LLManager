/* EO Classifier — Chat Mode 2.0 pipeline.

   Chat mode 2.0 replaces the library-and-folders ingest flow: instead of
   loading documents into a corpus, every message pasted into the chat is
   run straight through this pipeline. Material is cleaned, split into
   clauses, classified against 27 EO reference centroids, and folded into
   an in-memory graph. Questions are answered from that graph.

   Ported from the standalone "EO Classifier" prototype. Embeddings reuse
   the shared all-MiniLM-L6-v2 pipeline from local-store.js, so the model
   is downloaded and cached once for the whole app. */

import nlp from "compromise";
import { embed, cosineSim } from "./local-store.js";

/* ── The 27 reference centroids — operator × terrain × stance ── */

export const OPERATORS = {
  NUL: "Observing without acting, recording that nothing happened, noting an absence, deliberate restraint, an unchanged condition persists",
  SIG: "Directing attention to something notable, pointing out a trend or pattern, flagging what stands out, highlighting a change worth noticing",
  INS: "A new thing comes into existence for the first time, founding, creating, building, launching, establishing something that did not exist before",
  SEG: "Drawing a boundary, partitioning into categories, filtering, separating inside from outside, distinguishing one group from another",
  CON: "Connecting two things across a boundary, establishing a relationship, linking, associating, one thing depends on or relates to another",
  SYN: "Multiple things combine into an emergent whole greater than its parts, merging, integrating, a new unity arises from combination",
  DEF: "Stating a fact, reporting what happened, recording a measurement, a price rose, a number increased, someone said something, an event occurred",
  EVA: "Rendering a judgment, assessing quality or fitness, an opinion about whether something is good or bad, testing against a standard, evaluating",
  REC: "A fundamental revolution in how everything is understood, not a change within a framework but replacing the entire framework, a conversion that makes all prior understanding obsolete, extremely rare",
};

export const TERRAINS = {
  Void: "An ambient condition or absence, background substrate, weather, mood, diffuse environment",
  Entity: "A specific nameable thing, this person, this object, this event, bounded and graspable",
  Kind: "A type or category, not a particular instance but the recurring class, a genre or species",
  Field: "An ambient relational environment, unwritten rules, implicit hierarchy, power dynamics",
  Link: "A specific connection between two things, this bond, this contract, this dependency",
  Network: "An architecture of connections, a system, web, ecosystem, platform viewed as structural whole",
  Atmosphere: "An ambient interpretive mood, cultural climate, shared assumptions, meaning-conditions",
  Lens: "A specific interpretive frame, one diagnosis, one take, a particular analytical viewpoint",
  Paradigm: "A worldview, ideology, system of interpretation through which everything gets filtered",
};

export const STANCES = {
  Clearing: "Dissolving ambient conditions, making space, removing what was there",
  Dissecting: "Taking apart a specific thing, analysis, investigation, cutting this-not-that",
  Unraveling: "Deconstructing a recurring pattern, showing how a system works or fails",
  Tending: "Maintaining conditions, the gardener stance, sustaining without forcing",
  Binding: "Connecting specific things, tying this to that, the most common stance",
  Tracing: "Mapping regularities across time, following threads through a system",
  Cultivating: "Producing conditions for emergence without producing the thing itself",
  Making: "Building, creating, producing a specific thing, the most celebrated stance",
  Composing: "Producing regularities, designing structures that recur, architecture in the broad sense",
};

export const OP_COLORS = {
  NUL: "#6b7280", SIG: "#8b5cf6", INS: "#22c55e", SEG: "#f97316",
  CON: "#a78bfa", SYN: "#06b6d4", DEF: "#3b82f6", EVA: "#eab308", REC: "#ef4444",
};

/* ── Reference embeddings — computed once, then cached for the session ── */

let refVecs = null;
let refPromise = null;

export async function ensureCentroids(onProgress) {
  if (refVecs) return refVecs;
  if (refPromise) return refPromise;
  refPromise = (async () => {
    const out = { operators: {}, terrains: {}, stances: {} };
    const groups = [
      ["operators", OPERATORS], ["terrains", TERRAINS], ["stances", STANCES],
    ];
    let done = 0;
    const total = Object.keys(OPERATORS).length + Object.keys(TERRAINS).length
      + Object.keys(STANCES).length;
    for (const [key, defs] of groups) {
      for (const [name, def] of Object.entries(defs)) {
        out[key][name] = await embed(def);
        done++;
        if (onProgress) onProgress(`Computing reference embeddings ${done}/${total}…`);
      }
    }
    refVecs = out;
    return out;
  })();
  return refPromise;
}

/* ── Vector helpers ── */

export function nearestCentroid(vec, refMap) {
  let best = { name: null, score: -1 };
  let second = { name: null, score: -1 };
  for (const [name, ref] of Object.entries(refMap)) {
    const sim = cosineSim(vec, ref);
    if (sim > best.score) {
      second = { ...best };
      best = { name, score: sim };
    } else if (sim > second.score) {
      second = { name, score: sim };
    }
  }
  return { name: best.name, score: best.score, runnerUp: second.name, margin: best.score - second.score };
}

function averageVec(vecs) {
  const dim = vecs[0].length;
  const avg = new Array(dim).fill(0);
  for (const v of vecs) for (let d = 0; d < dim; d++) avg[d] += v[d] / vecs.length;
  return avg;
}

/* ── Text pre-cleaning ── */

function stripWebChrome(text) {
  const patterns = [
    /Skip to \w+/gi,
    /Site Index.*$/gm,
    /Site Information Navigation.*$/gm,
    /©\s*\d{4}.*/g,
    /Share full article\s*\d*/gi,
    /Editors' Picks/gi,
    /Read \d+ comments/gi,
    /See more on:.*$/gm,
    /Credit\.{2,}[^.]*?(?=\s[A-Z][a-z]|\s*$)/g,
    /\d+\s*min read/gi,
    /Listen\s*·\s*\d+:\d+/gi,
    /Section Navigation Search/gi,
    /Skip to contentSkip to site index/gi,
    /Related Content.*$/gm,
  ];
  for (const p of patterns) text = text.replace(p, "");
  return text.trim();
}

function cleanArticleText(text) {
  text = text.replace(/([a-z])([A-Z][a-z])/g, "$1. $2");
  text = text.replace(/Credit\.{2,}[^\n]*/g, "");
  text = text.replace(/\bBy\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+)?\b/g, "");
  text = text.replace(/\s{2,}/g, " ");
  return text.trim();
}

function protectAbbreviations(text) {
  return text
    .replace(/\bMr\.\s*/g, "Mr ")
    .replace(/\bMrs\.\s*/g, "Mrs ")
    .replace(/\bDr\.\s*/g, "Dr ")
    .replace(/\bSt\.\s*/g, "St ")
    .replace(/\bJr\.\s*/g, "Jr ")
    .replace(/\bSr\.\s*/g, "Sr ")
    .replace(/\bU\.S\.\s*/g, "US ")
    .replace(/\ba\.m\.\s*/g, "am ")
    .replace(/\bp\.m\.\s*/g, "pm ")
    .replace(/\b([A-Z])\.\s*([A-Z])\./g, "$1$2")
    .replace(/\b([A-Z])\.\s*/g, "$1 ");
}

function restoreAbbreviations(text) {
  return text.replace(/ /g, ". ").replace(/\.\s+\./g, ".").trim();
}

function splitClauses(text) {
  text = protectAbbreviations(text);
  return text.split(/(?<=[.!?;])\s+|(?<=:)\s+(?=[“"A-Z])/)
    .map(c => restoreAbbreviations(c.trim()))
    .filter(c => c.length > 8);
}

/* ── Name cleanup ── */

function cleanName(name) {
  return name.replace(/^[“”"',.\s]+/, "")
    .replace(/[“”"',.\s]+$/, "")
    .replace(/[’']s$/i, "").trim();
}

function isRealName(name) {
  const c = cleanName(name);
  if (c.length < 3) return false;
  const skip = ["he", "she", "they", "it", "his", "her", "their", "this", "that",
    "the", "when", "five", "in the", "there", "what", "who", "a", "an",
    "mr", "mrs", "dr", "president", "image", "credit", "quiz", "wait",
    "for", "but", "and", "the times", "new york", "associated press",
    "section", "search", "share", "listen", "site", "index", "navigation"];
  if (skip.includes(c.toLowerCase())) return false;
  if (/^\d/.test(c)) return false;
  if (!/[A-Z]/.test(c)) return false;
  if (c.split(/\s+/).length === 1 &&
      ["President", "Mr", "Mrs", "Dr", "Image", "Credit", "Quiz"].includes(c)) return false;
  return true;
}

function isChrome(text, nextClause) {
  const words = text.split(/\s+/).length;
  const trimmed = text.trim();

  if (nextClause) {
    const nextTrimmed = nextClause.trim();
    const nextStartsWithQuote = /^[“"‘']/.test(nextTrimmed);
    const hasAttribution = /\b(said|says|told|asked|spoke|added|replied|responded|noted|observed|argued|declared|explained|recalled|continued|insisted|warned|suggested|acknowledged|admitted)\b/i.test(trimmed);
    const hasName = /[A-Z][a-z]+(?:\s[A-Z][a-z]+)?/.test(trimmed);
    if (nextStartsWithQuote && hasAttribution && hasName) return false;
    if (/:\s*$/.test(trimmed) && nextStartsWithQuote) return false;
  }

  const containsNameAndSpeech = /[A-Z][a-z]+/.test(trimmed) &&
    /\b(said|says|told|asked|spoke|added|replied|responded|attended.*spoke)\b/i.test(trimmed);
  if (containsNameAndSpeech && words < 12) return false;

  if (words < 4) return true;
  if (/^\w+\s+\d{1,2},\s+\d{4}/.test(trimmed) && words < 8) return true;
  if (/^Image\b/i.test(trimmed)) return true;
  if (/^(Politics|Related|Quiz|Wait,)/i.test(trimmed) && words < 10) return true;
  if (/^(he|she|they)\s+(said|added|replied)\.?$/i.test(trimmed)) return true;
  if (/^For\s+Mr\.?\s*$/.test(trimmed)) return true;
  return false;
}

/* ── Per-clause extraction (compromise NLP + domain regex) ── */

function extractClause(text, corefStack) {
  const doc = nlp(text);
  let people = doc.people().out("array").map(cleanName).filter(isRealName);
  let places = doc.places().out("array").map(cleanName).filter(isRealName);
  let orgs = doc.organizations().out("array").map(cleanName).filter(isRealName);

  const artifactPattern = /credit|associated press|press\b|\.com|copyright|©/i;
  people = people.filter(n => !artifactPattern.test(n));
  places = places.filter(n => !artifactPattern.test(n));
  orgs = orgs.filter(n => !artifactPattern.test(n));

  for (const p of people) {
    const idx = corefStack.indexOf(p);
    if (idx > -1) corefStack.splice(idx, 1);
    corefStack.unshift(p);
  }

  const claims = [];

  const deathP = /(?:(\w+(?:\s\w+)?)|(?:his|her)\s+(\w+(?:\s\w+)?))\s+died\s+(?:of|in|from)\s+(.+?)(?:\.|;|,|$)/gi;
  let dm;
  while ((dm = deathP.exec(text)) !== null) {
    const who = dm[1] || (corefStack[0] ? `${corefStack[0]}'s ${dm[2]}` : dm[2]);
    claims.push({ entity: who, value: dm[3].trim(), span: dm[0].trim(), rawType: "death" });
  }

  const strokeP = /(?:(\w+(?:\s\w+)?)|(?:his|her)\s+(\w+))\s+had\s+a\s+(\w+)\s+and\s+passed\s+away/gi;
  let sp;
  while ((sp = strokeP.exec(text)) !== null) {
    const who = sp[1] || (corefStack[0] ? `${corefStack[0]}'s ${sp[2]}` : sp[2]);
    claims.push({ entity: who, value: sp[3], span: sp[0].trim(), rawType: "death" });
  }

  const relP = /([A-Z]\w+(?:\s[A-Z]\w+)?)\s*,\s*([A-Z]\w+)(?:[’']s)\s+(brother|sister|mother|father|son|daughter|wife|husband|friend|colleague)/gi;
  let rm;
  while ((rm = relP.exec(text)) !== null) {
    claims.push({ entity: cleanName(rm[1]), value: cleanName(rm[2]), span: rm[0].trim(), rawType: "relationship", relType: rm[3] });
  }

  const ageP = /(?:was|at|aged?)\s+(\d+)/gi;
  let am;
  while ((am = ageP.exec(text)) !== null) {
    claims.push({ entity: corefStack[0] || "?", value: am[1], span: am[0], rawType: "age" });
  }

  for (const p of people) claims.push({ entity: p, value: "person", span: p, rawType: "kind" });
  for (const p of places) claims.push({ entity: p, value: "place", span: p, rawType: "kind" });
  for (const o of orgs) claims.push({ entity: o, value: "organization", span: o, rawType: "kind" });

  const quoteP = /[“"]([^"”]{10,})[”"]/g;
  let qm;
  while ((qm = quoteP.exec(text)) !== null) {
    claims.push({ entity: corefStack[0] || "?", value: qm[1].trim(), span: qm[0].trim(), rawType: "quote" });
  }

  const ordP = /the\s+(youngest|oldest|eldest)\s+of\s+(\w+)/gi;
  let om;
  while ((om = ordP.exec(text)) !== null) {
    claims.push({ entity: corefStack[0] || "?", value: om[0], span: om[0], rawType: "rank" });
  }

  return { people, places, orgs, claims, clause: text };
}

/* ── Hypothesis register — when to pause and read interpretively ── */

function shouldRead(register, clauseIndex, frameEvents, claimCount) {
  if (clauseIndex < 3) return null;

  if (claimCount !== undefined && claimCount >= 4) {
    return { reason: "density", claims: claimCount };
  }

  const recent = register.clauseVecs.slice(-4);
  if (recent.length >= 4) {
    const rc = averageVec(recent.map(c => c.vec));
    const dc = averageVec(register.clauseVecs.map(c => c.vec));
    const drift = 1 - cosineSim(rc, dc);
    if (drift > 0.12) return { reason: "drift", drift };
  }

  if (frameEvents.some(e => e.type === "surprise")) return { reason: "surprise" };

  for (let a = 0; a < register.frames.length; a++) {
    for (let b = a + 1; b < register.frames.length; b++) {
      const overlap = register.frames[a].confirmedBy
        .filter(c => c >= clauseIndex - 4 && register.frames[b].confirmedBy.includes(c));
      if (overlap.length >= 2)
        return { reason: "convergence", frames: [register.frames[a].id, register.frames[b].id] };
    }
  }

  if (register.frames.length > 0) {
    const lastConfirm = Math.max(...register.frames.map(f =>
      f.confirmedBy.length > 0 ? Math.max(...f.confirmedBy) : -1));
    if (clauseIndex - lastConfirm > 6) return { reason: "silence", gap: clauseIndex - lastConfirm };
  }

  return null;
}

/* ── Main pipeline — process one block of pasted material ──

   Returns { results, clauses, register }. Clause indices are local (0-based)
   to this block; appendToGraph offsets them into the cumulative graph. */

export async function processText(rawText, onProgress) {
  await ensureCentroids(onProgress);

  let text = stripWebChrome(String(rawText || ""));
  text = cleanArticleText(text);

  const clauses = splitClauses(text);
  const corefStack = [];
  const results = [];
  const register = { frames: [], history: [], clauseVecs: [], triggerPoints: [] };
  let frameCounter = 0;

  for (let i = 0; i < clauses.length; i++) {
    if (onProgress) onProgress(`Reading clause ${i + 1}/${clauses.length}…`);
    const clause = clauses[i];
    const nextClause = i < clauses.length - 1 ? clauses[i + 1] : null;

    if (isChrome(clause, nextClause)) {
      results.push({
        clauseIndex: i, clause, entity: "(chrome — skipped)", value: "", span: clause,
        rawType: "chrome",
        operator: { name: "—", score: 0, runnerUp: "", margin: 1 },
        terrain: { name: "—", score: 0, runnerUp: "", margin: 1 },
        stance: { name: "—", score: 0, runnerUp: "", margin: 1 },
        frameEvents: [],
      });
      continue;
    }

    const extraction = extractClause(clause, corefStack);

    for (const claim of extraction.claims) {
      const spanVec = await embed(claim.span);
      results.push({
        clauseIndex: i, clause: extraction.clause, entity: claim.entity,
        value: claim.value, span: claim.span, rawType: claim.rawType, relType: claim.relType,
        operator: nearestCentroid(spanVec, refVecs.operators),
        terrain: nearestCentroid(spanVec, refVecs.terrains),
        stance: nearestCentroid(spanVec, refVecs.stances),
        frameEvents: [],
      });
    }

    const clauseVec = await embed(clause);
    register.clauseVecs.push({ index: i, vec: clauseVec });
    const op = nearestCentroid(clauseVec, refVecs.operators);
    const te = nearestCentroid(clauseVec, refVecs.terrains);
    const st = nearestCentroid(clauseVec, refVecs.stances);

    // Score the clause against active frames.
    const frameEvents = [];
    for (const frame of register.frames) {
      const simContent = cosineSim(clauseVec, frame.vec);
      if (simContent > 0.50) {
        frame.confirmedBy.push(i);
        frame.strength = frame.confirmedBy.length / (i - frame.generatedAt + 1);
        frameEvents.push({ type: "confirm", frameId: frame.id, sim: simContent.toFixed(2), text: frame.text });
        register.history.push({ event: "confirmed", frame: frame.id, at: i, sim: simContent });
      } else if (simContent < 0.18 && frame.strength > 0.3) {
        if (!frame.surprisedBy) frame.surprisedBy = [];
        frame.surprisedBy.push(i);
        frameEvents.push({ type: "surprise", frameId: frame.id, sim: simContent.toFixed(2), text: frame.text });
        register.history.push({ event: "surprised", frame: frame.id, at: i, sim: simContent });
      }
    }

    const claimCount = results.filter(r => r.clauseIndex === i && r.rawType !== "clause" && r.rawType !== "chrome").length;
    const trigger = shouldRead(register, i, frameEvents, claimCount);

    if (trigger && register.clauseVecs.length >= 2) {
      const recentVecs = register.clauseVecs.slice(-4).map(c => c.vec);
      const rc = averageVec(recentVecs);
      const dc = averageVec(register.clauseVecs.map(c => c.vec));
      const drift = 1 - cosineSim(rc, dc);
      const secT = nearestCentroid(rc, refVecs.terrains);
      const secO = nearestCentroid(rc, refVecs.operators);
      const secEnt = [...new Set(results
        .filter(r => r.clauseIndex >= i - 3 && r.clauseIndex <= i && r.entity && !r.entity.startsWith("("))
        .map(r => r.entity))].slice(0, 4);
      const mechText = (secEnt.length > 0 ? secEnt.join(", ") + ": " : "")
        + secO.name + " at " + secT.name + (drift > 0.12 ? " [SHIFT]" : "");

      register.triggerPoints.push({ clauseIndex: i, trigger, drift, mechText });

      let matched = false;
      for (const ex of register.frames) {
        if (cosineSim(rc, ex.vec) > 0.55) {
          ex.text = mechText; ex.vec = rc; ex.revisedAt = i; ex.revision++; ex.drift = drift;
          register.history.push({ event: "refined", frame: ex.id, at: i, text: mechText });
          frameEvents.push({ type: "refine", frameId: ex.id, text: mechText });
          matched = true;
          break;
        }
      }
      if (!matched) {
        const nf = {
          id: `hyp_${frameCounter++}`, text: mechText, vec: rc, confirmedBy: [], surprisedBy: [],
          strength: 0.5, generatedAt: i, revisedAt: i, revision: 0, drift,
          source: "mechanical", trigger: trigger.reason,
        };
        register.frames.push(nf);
        register.history.push({ event: "created", frame: nf.id, at: i, text: mechText, trigger: trigger.reason });
        frameEvents.push({ type: "new_frame", frameId: nf.id, text: `[${trigger.reason}] ${mechText}` });
      }
      for (const f of register.frames) {
        const last = f.confirmedBy.length > 0 ? Math.max(...f.confirmedBy) : f.generatedAt;
        if (i - last > 8) f.strength *= 0.7;
      }
      register.frames = register.frames.filter(f => f.strength > 0.1 || i - f.generatedAt < 4);
    }

    results.push({
      clauseIndex: i, clause: extraction.clause, entity: "(whole clause)",
      value: clause.slice(0, 60) + (clause.length > 60 ? "…" : ""),
      span: clause, rawType: "clause", operator: op, terrain: te, stance: st,
      vec: clauseVec, frameEvents,
    });
  }

  return { results, clauses, register };
}

/* ── Cumulative graph — accumulates material across chat turns ── */

export function emptyGraph() {
  return { entities: {}, claims: [], clauses: [], vectors: {} };
}

/* Fold one processed block into the cumulative graph. clauseBase keeps
   clause indices globally unique across messages. Returns the count of
   new entities/claims/clauses added, for the turn summary. */
export async function appendToGraph(graph, results, clauseBase) {
  let newEntities = 0;
  const before = graph.claims.length;
  const seenClauses = new Set();

  for (const r of results) {
    const globalIndex = clauseBase + r.clauseIndex;

    if (!seenClauses.has(r.clauseIndex) &&
        (r.rawType === "clause" || r.rawType === "chrome")) {
      if (!graph.clauses.some(c => c.index === globalIndex)) {
        graph.clauses.push({ index: globalIndex, text: r.clause, vec: r.vec || null });
      }
      seenClauses.add(r.clauseIndex);
    }

    if (r.rawType === "chrome" || r.rawType === "clause") continue;

    const key = r.entity.toLowerCase();
    if (!key || key === "?" || key.startsWith("(")) continue;

    if (!graph.entities[key]) {
      graph.entities[key] = { name: r.entity, terrain: r.terrain.name, claims: [], edges: [] };
      newEntities++;
    }

    const claim = {
      entity: r.entity, value: r.value, span: r.span, rawType: r.rawType,
      operator: r.operator.name, site: r.terrain.name, resolution: r.stance.name,
      notation: `${r.operator.name}(${r.terrain.name}, ${r.stance.name})`,
      clauseIndex: globalIndex,
    };
    graph.entities[key].claims.push(claim);
    graph.claims.push(claim);

    if (r.rawType === "relationship" && r.value) {
      graph.entities[key].edges.push({ to: r.value, type: r.relType || "related", span: r.span });
    }
  }

  // Embed entities for retrieval.
  for (const [key, entity] of Object.entries(graph.entities)) {
    if (graph.vectors[key]) continue;
    const text = `${entity.name}: ${entity.claims.map(c => c.value).join(", ")}`;
    graph.vectors[key] = await embed(text.slice(0, 200));
  }

  return {
    newEntities,
    newClaims: graph.claims.length - before,
    newClauses: seenClauses.size,
  };
}

/* ── Retrieval — build a grounded dossier for a question ── */

export async function buildDossier(graph, query) {
  if (!graph || Object.keys(graph.entities).length === 0) {
    return { ctx: "[CTX]\n(graph empty)\n[/CTX]", docs: "", spans: [] };
  }

  const queryVec = await embed(query);
  const queryLower = query.toLowerCase();
  const spans = [];

  const stopWords = new Set(["what", "does", "that", "this", "with", "from", "about", "were",
    "have", "been", "their", "they", "them", "than", "into", "would", "could", "should",
    "which", "where", "when", "based", "tell", "explain", "write"]);
  const keyNouns = queryLower.split(/\s+/).filter(w => w.length > 3 && !stopWords.has(w));

  // Score entities.
  const entityScored = [];
  for (const [key, vec] of Object.entries(graph.vectors)) {
    const nameBoost = keyNouns.some(n => key.includes(n)) ? 0.15 : 0;
    entityScored.push({ key, sim: cosineSim(queryVec, vec) + nameBoost });
  }
  entityScored.sort((a, b) => b.sim - a.sim);

  const entityCount = Math.min(Math.max(3, Math.ceil(Object.keys(graph.entities).length / 4)), 8);
  let ctx = "[CTX]\n";
  for (const { key } of entityScored.slice(0, entityCount)) {
    const entity = graph.entities[key];
    if (!entity) continue;
    let block = `E: ${entity.name} | ${entity.terrain}`;
    for (const claim of entity.claims.slice(0, 4)) {
      block += `\n  ${claim.operator}: ${claim.value} @"${claim.span.slice(0, 60)}"`;
      spans.push({ text: claim.span, entity: entity.name, operator: claim.operator });
    }
    for (const edge of entity.edges.slice(0, 2)) block += `\n  → ${edge.to} (${edge.type})`;
    ctx += block + "\n\n";
  }
  ctx += "[/CTX]";

  // Score clauses (embedding + keyword overlap).
  const clauseScored = [];
  for (const clause of graph.clauses) {
    const cVec = clause.vec || await embed(clause.text.slice(0, 200));
    const textLower = clause.text.toLowerCase();
    const kwMatches = keyNouns.filter(n => textLower.includes(n)).length;
    const kwBoost = keyNouns.length > 0 ? (kwMatches / keyNouns.length) * 0.2 : 0;
    clauseScored.push({
      text: clause.text, index: clause.index,
      score: cosineSim(queryVec, cVec) + kwBoost, vec: cVec,
    });
  }
  clauseScored.sort((a, b) => b.score - a.score);

  const passageCount = Math.min(Math.max(5, Math.ceil(graph.clauses.length / 3)), 15);
  const passages = clauseScored.slice(0, passageCount).sort((a, b) => a.index - b.index);

  let docs = "[DOCS]\n";
  passages.forEach((c, i) => {
    docs += `${i + 1}: "${c.text.slice(0, 250)}"\n`;
    spans.push({ text: c.text, entity: "(passage)", operator: "source" });
  });
  docs += "[/DOCS]";

  return { ctx, docs, spans };
}

export const GROUNDED_SYSTEM =
  "Answer ONLY from the numbered [DOCS] sources below. "
  + "Every claim must cite at least one source number in parentheses. "
  + "If a claim is not directly supported by a specific source, do not make it. "
  + "Do not use outside knowledge — only what is in the sources. "
  + "If the sources do not answer the question fully, say what they cover and what is missing. "
  + "Be direct and specific. Quote or closely paraphrase the sources.";

/* ── Optional second pass — LLM interpretation at trigger points ──

   llm is an async (system, user) => string. Reads each trigger point and
   generates rhetorical-function hypotheses, scored retroactively against
   the rest of the document. */

export async function runSecondPass(register, llm, onProgress) {
  const triggers = register.triggerPoints || [];
  if (triggers.length === 0) return [];

  let frameCounter = register.frames.length + 100;
  const llmFrames = [];

  for (let t = 0; t < triggers.length; t++) {
    const tp = triggers[t];
    if (onProgress) onProgress(`Interpreting trigger ${t + 1}/${triggers.length} [${tp.trigger.reason}]…`);

    let context = "ACTIVE FRAMES:\n";
    for (const f of register.frames.filter(f => f.generatedAt <= tp.clauseIndex)) {
      context += `- ${f.id}: "${f.text}" (confirmed ${f.confirmedBy.length}x)\n`;
    }
    context += `\nTRIGGER: ${tp.trigger.reason}`;
    if (tp.drift) context += ` (drift=${tp.drift.toFixed(3)})`;
    context += `\nMECHANICAL READING: ${tp.mechText}\n`;

    const system = "You see a mechanical analysis of a text (ACTIVE FRAMES) and why "
      + "you were called (TRIGGER). Produce 1-2 hypotheses about what the text is "
      + "DOING — its rhetorical or narrative function, not what it says. "
      + 'Respond ONLY as JSON, no backticks: {"hypotheses":[{"text":"...","confidence":0.7}]}. '
      + "Keep each text under 30 words.";

    try {
      const raw = await llm(system, context);
      let parsed;
      try { parsed = JSON.parse(String(raw).replace(/```json|```/g, "").trim()); }
      catch { parsed = { hypotheses: [] }; }

      for (const hyp of (parsed.hypotheses || []).slice(0, 2)) {
        const textVec = await embed(hyp.text);
        const frame = {
          id: `read_${frameCounter++}`, text: hyp.text, vec: textVec,
          confirmedBy: [], surprisedBy: [], strength: hyp.confidence || 0.6,
          generatedAt: tp.clauseIndex, revisedAt: tp.clauseIndex, revision: 0,
          drift: tp.drift, source: "llm", trigger: tp.trigger.reason,
        };
        // Retroactive scoring against later clauses.
        for (const cv of register.clauseVecs) {
          if (cv.index <= tp.clauseIndex) continue;
          if (cosineSim(textVec, cv.vec) > 0.45) {
            frame.confirmedBy.push(cv.index);
            frame.strength = Math.min(1.0, frame.strength + 0.1);
          }
        }
        register.frames.push(frame);
        register.history.push({
          event: "created", frame: frame.id, at: tp.clauseIndex, text: frame.text,
          source: "llm", trigger: tp.trigger.reason, retroactive: frame.confirmedBy.length,
        });
        llmFrames.push(frame);
      }
    } catch (e) {
      register.history.push({ event: "llm_error", at: tp.clauseIndex, text: e.message });
    }
  }
  return llmFrames;
}

/* ── Heuristic: is this message a question for the graph, or new material? ──

   Material is what gets ingested; questions are answered from the graph.
   A short interrogative against a non-empty graph is a question; long
   text, or anything sent before the graph has content, is material. */
export function looksLikeQuestion(text, graphHasContent) {
  const t = String(text || "").trim();
  if (!graphHasContent) return false;
  const words = t.split(/\s+/).length;
  if (words > 60) return false;
  if (/\?\s*$/.test(t)) return true;
  if (words <= 40 && /^(who|what|when|where|why|how|which|did|do|does|is|are|was|were|can|could|should|would|tell me|explain|summarize|list|describe|write)\b/i.test(t)) {
    return true;
  }
  return false;
}
