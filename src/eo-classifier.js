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

/* The text's own volatility — the average clause-to-clause drift, measured
   over sliding windows. The trigger machinery calibrates against this so a
   naturally restless text is not flagged on every clause. */
function computeBaselineDrift(clauseVecs) {
  if (clauseVecs.length < 8) return 0.10;
  let totalDrift = 0, count = 0;
  for (let i = 4; i < clauseVecs.length; i++) {
    const recent = clauseVecs.slice(i - 4, i).map(c => c.vec);
    const prior = clauseVecs.slice(Math.max(0, i - 8), i - 4).map(c => c.vec);
    if (prior.length < 2) continue;
    totalDrift += 1 - cosineSim(averageVec(recent), averageVec(prior));
    count++;
  }
  return count > 0 ? totalDrift / count : 0.10;
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

  /* ── Argumentative structure ──
     The patterns above were built for news; these catch the propositions,
     causal claims, contrasts and definitions that carry essayistic and
     philosophical text. They are imperfect seeds — false positives are fine,
     they exist to feed salience and the fold, not to be final readings. */

  // Proposition: X is/are/was/were Y
  const propP = /(?:^|[.;]\s+)([A-Z][^.;]{0,50}?)\s+(?:is|are|was|were)\s+((?:not\s+)?[^.;]{5,80}?)(?:\.|;|$)/gm;
  let pm;
  while ((pm = propP.exec(text)) !== null) {
    const subj = pm[1].trim();
    const pred = pm[2].trim();
    if (pred.length > 4 && subj.length > 2) {
      claims.push({ entity: subj, value: pred, span: pm[0].trim(), rawType: "proposition" });
    }
  }

  // Causal: X produces/creates/generates/leads to Y
  const causeP = /([A-Za-z][^.;]{0,50}?)\s+(?:produces?|creates?|generates?|enables?|leads?\s+to|results?\s+in|gives?\s+rise\s+to)\s+([^.;]{5,80}?)(?:\.|;|$)/gi;
  let cm;
  while ((cm = causeP.exec(text)) !== null) {
    claims.push({ entity: cm[1].trim(), value: cm[2].trim(), span: cm[0].trim(), rawType: "causal" });
  }

  // Contrastive: not X but Y / X rather than Y
  const contrastP = /(?:not\s+(.{3,40}?)\s+but\s+(.{3,40}?)(?:\.|;|,|$))|(?:(.{3,40}?)\s+rather\s+than\s+(.{3,40}?)(?:\.|;|,|$))/gi;
  let crm;
  while ((crm = contrastP.exec(text)) !== null) {
    const neg = (crm[1] || crm[4] || "").trim();
    const pos = (crm[2] || crm[3] || "").trim();
    if (neg && pos) {
      claims.push({ entity: pos, value: `not ${neg}`, span: crm[0].trim(), rawType: "contrast" });
    }
  }

  // Definitional: X means Y / X refers to Y / by X I mean Y
  const defnP = /(?:by\s+)?["“”']?([A-Za-z][^.;"“”]{2,40}?)["“”']?\s+(?:means?|refers?\s+to|signifies?|denotes?)\s+([^.;]{5,60}?)(?:\.|;|$)/gi;
  let dfm;
  while ((dfm = defnP.exec(text)) !== null) {
    claims.push({ entity: dfm[1].trim(), value: dfm[2].trim(), span: dfm[0].trim(), rawType: "definition" });
  }

  // Drop a `kind` claim repeated verbatim inside this clause — the same
  // entity+kind generated more than once carries no extra information.
  const seenKinds = new Set();
  const deduped = claims.filter(c => {
    if (c.rawType === "kind") {
      const k = `${c.entity}::${c.value}`;
      if (seenKinds.has(k)) return false;
      seenKinds.add(k);
    }
    return true;
  });

  return { people, places, orgs, claims: deduped, clause: text };
}

/* ── Hypothesis register — when to pause and read interpretively ──

   The trigger replaces the old metronome: a clause is read interpretively
   only when something demands it — drift past the text's own baseline
   volatility, a clause that surprises an active frame, a long silence with
   no frame confirmed, a dense claim cluster, or two frames converging.
   Returns an array of reason objects, or null. */
function shouldRead(register, clauseIndex, clauseVec, extraction, results) {
  const reasons = [];

  if (register.clauseVecs.length >= 8) {
    const rc = averageVec(register.clauseVecs.slice(-4).map(c => c.vec));
    const dc = averageVec(register.clauseVecs.map(c => c.vec));
    const drift = 1 - cosineSim(rc, dc);
    const baseline = register.baselineDrift || 0.10;
    if (drift > baseline * 1.5) reasons.push({ type: "drift", drift, baseline });
  }

  for (const frame of register.frames) {
    if (frame.strength < 0.3) continue;
    const sim = cosineSim(clauseVec, frame.vec);
    if (sim < 0.18) reasons.push({ type: "surprise", frame: frame.id, sim });
  }

  if (register.frames.length > 0) {
    const lastConfirm = Math.max(
      ...register.frames.map(f =>
        f.confirmedBy.length > 0 ? Math.max(...f.confirmedBy) : f.generatedAt)
    );
    if (clauseIndex - lastConfirm > 6) reasons.push({ type: "silence", gap: clauseIndex - lastConfirm });
  }

  if (extraction && extraction.claims.length > 0) {
    const words = extraction.clause.split(/\s+/).length;
    const density = extraction.claims.length / words;
    if (density > 0.12) reasons.push({ type: "density", density, claims: extraction.claims.length, words });
  }

  for (let i = 0; i < register.frames.length; i++) {
    for (let j = i + 1; j < register.frames.length; j++) {
      const recentI = register.frames[i].confirmedBy.filter(c => c >= clauseIndex - 3);
      const recentJ = register.frames[j].confirmedBy.filter(c => c >= clauseIndex - 3);
      if (recentI.filter(c => recentJ.includes(c)).length >= 2) {
        reasons.push({ type: "convergence", frames: [register.frames[i].id, register.frames[j].id] });
      }
    }
  }

  // Operator mode shift — the dominant operator of the recent window differs
  // from the window before it. This catches argumentative turns (DEF→EVA→REC)
  // that leave the vocabulary, and so the embedding drift, unchanged.
  if (register.clauseVecs.length >= 12 && Array.isArray(results)) {
    const clauseResults = results.filter(r => r.rawType === "clause" && r.operator);
    const recent = clauseResults.slice(-6);
    const prior = clauseResults.slice(-12, -6);
    if (recent.length >= 3 && prior.length >= 3) {
      const mode = (arr) => {
        const counts = {};
        for (const r of arr) counts[r.operator.name] = (counts[r.operator.name] || 0) + 1;
        return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0];
      };
      const recentMode = mode(recent);
      const priorMode = mode(prior);
      if (recentMode && priorMode && recentMode !== priorMode) {
        reasons.push({ type: "mode_shift", from: priorMode, to: recentMode });
      }
    }
  }

  return reasons.length > 0 ? reasons : null;
}

/* ── Salience — how much a clause matters, scored without the LLM ──

   A continuous score over claim count, proximity to known entities, proximity
   to active frames, and lexical density, penalised for very short clauses.
   `entityVecs` is the set of entity/claim vectors known so far (this block
   plus any prior graph) — the clause is salient if it lands near them. */
function salience(clause, clauseVec, extraction, register, entityVecs) {
  let score = 0;

  // Propositional claims assert argumentative structure — they matter far
  // more than bare entity tags. `kind` claims still count, just much less.
  const propClaims = extraction.claims.filter(c =>
    c.rawType === "proposition" || c.rawType === "causal"
    || c.rawType === "contrast" || c.rawType === "definition"
    || c.rawType === "quote");
  const kindClaims = extraction.claims.filter(c => c.rawType === "kind");
  score += propClaims.length * 0.25;
  score += kindClaims.length * 0.05;

  let bestEntitySim = 0;
  for (const vec of entityVecs) {
    const sim = cosineSim(clauseVec, vec);
    if (sim > bestEntitySim) bestEntitySim = sim;
  }
  score += bestEntitySim * 0.15; // reduced — proximity is weaker evidence

  let bestFrameSim = 0;
  for (const frame of register.frames) {
    const sim = cosineSim(clauseVec, frame.vec);
    if (sim > bestFrameSim) bestFrameSim = sim;
  }
  score += bestFrameSim * 0.2;

  const words = clause.split(/\s+/);
  const stops = /^(the|this|that|with|from|have|been|will|would|could|should|they|their|them|these|those|into|onto|upon|also|just|than|then|when|what|which|where|who|whom|whose|more|most|some|such|each|every|both|were|does|done|here|there|about|after|before|under|over|between|through|during|without|within|along|among|against|toward|across|behind|beyond|above|below|a|an|and|but|for|nor|yet|so|or|is|are|was|it|its|not)$/i;
  const contentWords = words.filter(w => w.length > 3 && !stops.test(w));
  const lexDensity = words.length > 0 ? contentWords.length / words.length : 0;
  score += lexDensity * 0.15;

  // Copula density — "is/are/means" assertions are structurally load-bearing
  // in argumentative text even when the vocabulary is plain.
  const copulas = (clause.match(/\b(?:is|are|was|were|means?|becomes?)\b/gi) || []).length;
  score += Math.min(copulas * 0.08, 0.16);

  // Short clauses are discounted but no longer killed — a five-word thesis
  // ("Every mind is partial") must survive to be classified.
  if (words.length < 6) score *= 0.6;
  if (words.length < 3) score *= 0.4;

  return score;
}

/* ── Main pipeline — process one block of pasted material ──

   Returns { results, clauses, register }. Clause indices are local (0-based)
   to this block; appendToGraph offsets them into the cumulative graph.

   Every clause is embedded and scored for salience against the live entity
   set and the hypothesis register. A clause below the salience floor is
   logged inert — not classified — and revisited retroactively once the rest
   of the block is in. Interpretive reading is driven by the trigger, not a
   metronome. No LLM calls. `priorGraph`, when supplied, seeds the entity
   set so salience can see what earlier turns established. */

export async function processText(rawText, onProgress, priorGraph) {
  await ensureCentroids(onProgress);
  const t0 = Date.now();

  let text = stripWebChrome(String(rawText || ""));
  text = cleanArticleText(text);

  const clauses = splitClauses(text);
  const corefStack = [];
  const results = [];
  const register = {
    frames: [], history: [], clauseVecs: [], triggerPoints: [], flags: [],
    baselineDrift: 0.10,
  };
  let frameCounter = 0;
  // The entity vectors salience scores proximity against — seeded from the
  // cumulative graph, then grown with each claim span found in this block.
  const entityVecs = priorGraph && priorGraph.vectors
    ? Object.values(priorGraph.vectors).slice() : [];

  for (let i = 0; i < clauses.length; i++) {
    if (onProgress) onProgress(`Clause ${i + 1}/${clauses.length}…`);
    const clause = clauses[i];

    // Embed every clause first — salience needs the vector.
    const clauseVec = await embed(clause);
    register.clauseVecs.push({ index: i, vec: clauseVec });

    // Extract claims (cheap NLP).
    const extraction = extractClause(clause, corefStack);

    // Score salience against the live entity set + register.
    const sal = salience(clause, clauseVec, extraction, register, entityVecs);

    if (sal < 0.15) {
      results.push({
        clauseIndex: i, clause, entity: "(inert)", value: "",
        span: clause, rawType: "inert", salience: sal, vec: clauseVec,
        operator: { name: "—", score: 0, runnerUp: "", margin: 1 },
        terrain: { name: "—", score: 0, runnerUp: "", margin: 1 },
        stance: { name: "—", score: 0, runnerUp: "", margin: 1 },
        frameEvents: [],
      });
      continue;
    }

    // Classify each claim.
    for (const claim of extraction.claims) {
      const spanVec = await embed(claim.span);
      entityVecs.push(spanVec);
      results.push({
        clauseIndex: i, clause: extraction.clause, entity: claim.entity,
        value: claim.value, span: claim.span, rawType: claim.rawType,
        relType: claim.relType, salience: sal,
        operator: nearestCentroid(spanVec, refVecs.operators),
        terrain: nearestCentroid(spanVec, refVecs.terrains),
        stance: nearestCentroid(spanVec, refVecs.stances),
        frameEvents: [],
      });
    }

    // Classify whole clause.
    const op = nearestCentroid(clauseVec, refVecs.operators);
    const te = nearestCentroid(clauseVec, refVecs.terrains);
    const st = nearestCentroid(clauseVec, refVecs.stances);

    // ── Score clause against active frames ──
    const frameEvents = [];
    for (const frame of register.frames) {
      const sim = cosineSim(clauseVec, frame.vec);
      if (sim > 0.50) {
        frame.confirmedBy.push(i);
        frame.strength = frame.confirmedBy.length / (i - frame.generatedAt + 1);
        frameEvents.push({ type: "confirm", frameId: frame.id, sim: sim.toFixed(2), text: frame.text });
        register.history.push({ event: "confirmed", frame: frame.id, at: i, sim });
      } else if (sim < 0.20 && frame.strength > 0.3) {
        frameEvents.push({ type: "surprise", frameId: frame.id, sim: sim.toFixed(2), text: frame.text });
        register.history.push({ event: "surprised", frame: frame.id, at: i, sim });
      }
    }

    // ── Update baseline drift periodically ──
    if (i === 20 || (i > 20 && i % 30 === 0)) {
      register.baselineDrift = computeBaselineDrift(register.clauseVecs);
    }

    // ── Trigger function replaces metronome ──
    const triggers = shouldRead(register, i, clauseVec, extraction, results);

    if (triggers) {
      const recentVecs = register.clauseVecs.slice(-4).map(c => c.vec);
      const rc = averageVec(recentVecs);
      const allV = register.clauseVecs.map(c => c.vec);
      const dc = averageVec(allV);
      const drift = 1 - cosineSim(rc, dc);

      const secT = nearestCentroid(rc, refVecs.terrains);
      const secO = nearestCentroid(rc, refVecs.operators);
      const secEnt = [...new Set(
        results.filter(r =>
          r.clauseIndex >= i - 3 && r.clauseIndex <= i
          && r.entity && !r.entity.startsWith("("))
        .map(r => r.entity)
      )].slice(0, 4);

      const triggerLabel = triggers.map(t => t.type).join("+");
      const hypText = (secEnt.length > 0 ? secEnt.join(", ") + ": " : "")
        + secO.name + " at " + secT.name
        + " [" + triggerLabel + "]"
        + (drift > (register.baselineDrift || 0.10) * 1.5 ? " SHIFT" : "");

      // Keep a trigger point so the optional deep read can interpret it.
      register.triggerPoints.push({
        clauseIndex: i, trigger: { reason: triggerLabel, reasons: triggers },
        drift, mechText: hypText,
      });

      let matched = false;
      for (const ex of register.frames) {
        if (cosineSim(rc, ex.vec) > 0.55) {
          ex.text = hypText; ex.vec = rc; ex.revisedAt = i;
          ex.revision++; ex.drift = drift; ex.triggers = triggers;
          register.history.push({ event: "refined", frame: ex.id, at: i, text: hypText, triggers });
          frameEvents.push({ type: "refine", frameId: ex.id, text: hypText });
          matched = true; break;
        }
      }
      if (!matched) {
        const nf = {
          id: `hyp_${frameCounter++}`, text: hypText, vec: rc,
          confirmedBy: [], surprisedBy: [], strength: 0.5, generatedAt: i,
          revisedAt: i, revision: 0, drift, triggers, source: "mechanical",
        };
        register.frames.push(nf);
        register.history.push({ event: "created", frame: nf.id, at: i, text: hypText, triggers });
        frameEvents.push({ type: "new_frame", frameId: nf.id, text: hypText });
      }

      for (const f of register.frames) {
        const last = f.confirmedBy.length > 0 ? Math.max(...f.confirmedBy) : f.generatedAt;
        if (i - last > 8) f.strength *= 0.7;
      }
      register.frames = register.frames.filter(f => f.strength > 0.1 || i - f.generatedAt < 4);
    }

    // ── Flag where surface ≠ function ──
    const flagReasons = [];
    if (triggers) {
      const recentOps = results.slice(-8)
        .filter(r => r.rawType === "clause")
        .map(r => r.operator.name);
      const modeOp = recentOps.sort((a, b) =>
        recentOps.filter(v => v === b).length - recentOps.filter(v => v === a).length
      )[0];
      if (op.name === modeOp) flagReasons.push("drift+static");
      if (triggers.some(t => t.type === "surprise") && op.name === "DEF") flagReasons.push("surprise+DEF");
      if (triggers.some(t => t.type === "density")) flagReasons.push("density");
      if (triggers.some(t => t.type === "silence")) flagReasons.push("silence");
      if (triggers.some(t => t.type === "mode_shift")) flagReasons.push("mode_shift");
    }
    const needsReading = flagReasons.length > 0;
    if (needsReading) {
      register.flags.push({
        clauseIndex: i, mechanicalOp: op.name, reason: flagReasons.join("+"),
        text: clause, functionalOp: null, functionalReason: null,
      });
      register.history.push({ event: "flagged", at: i, text: `${op.name} · ${flagReasons.join("+")}` });
    }

    results.push({
      clauseIndex: i, clause: extraction.clause, entity: "(whole clause)",
      value: clause.slice(0, 60) + (clause.length > 60 ? "…" : ""),
      span: clause, rawType: "clause", salience: sal,
      operator: op, terrain: te, stance: st,
      vec: clauseVec, frameEvents, needsReading, flagReasons,
      mechanicalOp: op.name, triggers,
    });
  }

  // ── Retroactive salience revision ──
  // Recompute salience for inert clauses using the FINAL entity set + register.
  // A clause inert at position 5 may matter once entities appear later.
  if (onProgress) onProgress("Revising salience…");
  const inertIndices = [];
  for (let j = 0; j < results.length; j++) {
    if (results[j].rawType === "inert") inertIndices.push(j);
  }

  let revived = 0;
  for (const j of inertIndices) {
    const r = results[j];
    const cv = register.clauseVecs.find(c => c.index === r.clauseIndex);
    if (!cv) continue;

    // Re-extract (coref stack is stale but claims are structural).
    const reExtraction = extractClause(r.clause, []);
    const newSal = salience(r.clause, cv.vec, reExtraction, register, entityVecs);

    // Update the stored salience regardless — a clause that stays inert still
    // gets its final judgment, so the display fades it by what it became.
    r.salience = newSal;

    if (newSal >= 0.15) {
      // Revive: classify it now.
      r.rawType = "revived";
      r.entity = "(whole clause)";
      r.value = r.clause.slice(0, 60) + (r.clause.length > 60 ? "…" : "");

      r.operator = nearestCentroid(cv.vec, refVecs.operators);
      r.terrain = nearestCentroid(cv.vec, refVecs.terrains);
      r.stance = nearestCentroid(cv.vec, refVecs.stances);
      r.mechanicalOp = r.operator.name;
      r.vec = cv.vec;

      // Also insert any claims we missed.
      for (const claim of reExtraction.claims) {
        const spanVec = await embed(claim.span);
        entityVecs.push(spanVec);
        results.splice(j, 0, {
          clauseIndex: r.clauseIndex, clause: r.clause,
          entity: claim.entity, value: claim.value,
          span: claim.span, rawType: claim.rawType,
          relType: claim.relType, salience: newSal,
          operator: nearestCentroid(spanVec, refVecs.operators),
          terrain: nearestCentroid(spanVec, refVecs.terrains),
          stance: nearestCentroid(spanVec, refVecs.stances),
          frameEvents: [],
        });
      }
      revived++;
    }
  }

  if (onProgress) {
    const inertCount = results.filter(r => r.rawType === "inert").length;
    const flaggedCount = results.filter(r => r.needsReading).length;
    onProgress(`Done. ${results.length} classifications, ${register.frames.length} active frames, `
      + `${inertCount} inert, ${revived} revived, ${flaggedCount} flagged `
      + `in ${Date.now() - t0}ms.`);
  }

  return { results, clauses, register };
}

/* ── Cumulative graph — accumulates material across chat turns ── */

export function emptyGraph() {
  return { entities: {}, claims: [], clauses: [], vectors: {}, frames: [] };
}

/* Fold one processed block into the cumulative graph. clauseBase keeps
   clause indices globally unique across messages. When the block's register
   is supplied, its interpretive frames are folded in too (with globalised
   clause indices), so point-based folds can reconstruct the active reading at
   any clause. Returns the count of new entities/claims/clauses added. */
export async function appendToGraph(graph, results, clauseBase, register) {
  let newEntities = 0;
  const before = graph.claims.length;
  const seenClauses = new Set();

  for (const r of results) {
    const globalIndex = clauseBase + r.clauseIndex;

    if (!seenClauses.has(r.clauseIndex) &&
        (r.rawType === "clause" || r.rawType === "revived" || r.rawType === "inert")) {
      if (!graph.clauses.some(c => c.index === globalIndex)) {
        graph.clauses.push({
          index: globalIndex, text: r.clause, vec: r.vec || null,
          needsReading: !!r.needsReading,
          flagReason: (r.flagReasons && r.flagReasons.join("+")) || null,
          mechanicalOp: r.mechanicalOp || null, functionalOp: null,
          salience: r.salience != null ? r.salience : null,
        });
      }
      seenClauses.add(r.clauseIndex);
    }

    if (r.rawType === "clause" || r.rawType === "revived" || r.rawType === "inert") continue;

    const key = r.entity.toLowerCase();
    if (!key || key === "?" || key.startsWith("(")) continue;

    if (!graph.entities[key]) {
      graph.entities[key] = { name: r.entity, terrain: r.terrain.name, claims: [], edges: [], mentions: [] };
      newEntities++;
    }
    const ent = graph.entities[key];
    if (!ent.mentions) ent.mentions = [];

    // Every appearance is a mention — recorded even when the claim itself is
    // dropped as a duplicate, so a fold can see where an entity recurs.
    ent.mentions.push(globalIndex);

    // A `kind` claim ("X is a person") repeats at every mention and carries
    // almost no information — keep only the first per entity+value. The
    // mention list above still tracks where the entity appeared.
    if (r.rawType === "kind"
        && ent.claims.some(c => c.rawType === "kind" && c.value === r.value)) {
      continue;
    }

    const claim = {
      entity: r.entity, value: r.value, span: r.span, rawType: r.rawType,
      operator: r.operator.name, site: r.terrain.name, resolution: r.stance.name,
      notation: `${r.operator.name}(${r.terrain.name}, ${r.stance.name})`,
      clauseIndex: globalIndex,
    };
    ent.claims.push(claim);
    graph.claims.push(claim);

    if (r.rawType === "relationship" && r.value) {
      ent.edges.push({ to: r.value, type: r.relType || "related", span: r.span });
    }
  }

  // Mentions sorted and de-duplicated across the whole graph.
  for (const ent of Object.values(graph.entities)) {
    if (ent.mentions) ent.mentions = [...new Set(ent.mentions)].sort((a, b) => a - b);
  }

  // Embed entities for retrieval.
  for (const [key, entity] of Object.entries(graph.entities)) {
    if (graph.vectors[key]) continue;
    const text = `${entity.name}: ${entity.claims.map(c => c.value).join(", ")}`;
    graph.vectors[key] = await embed(text.slice(0, 200));
  }

  if (register) foldFrames(graph, register, clauseBase);

  return {
    newEntities,
    newClaims: graph.claims.length - before,
    newClauses: seenClauses.size,
  };
}

/* Fold a block's register frames into the cumulative graph. Frame clause
   indices are local to the block; they are offset by clauseBase so a fold can
   ask "which frames were active up to global clause N". Idempotent — calling
   it again after a deep read (which adds LLM frames) updates in place. */
export function foldFrames(graph, register, clauseBase) {
  if (!graph.frames) graph.frames = [];
  for (const f of register.frames || []) {
    const id = `${clauseBase}:${f.id}`;
    const record = {
      id,
      text: f.text,
      strength: f.strength || 0,
      generatedAt: clauseBase + f.generatedAt,
      source: f.source || "mechanical",
      trigger: f.trigger || null,
    };
    const existing = graph.frames.find(x => x.id === id);
    if (existing) Object.assign(existing, record);
    else graph.frames.push(record);
  }
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

export async function runSecondPass(register, llm, onProgress, gate) {
  const triggers = register.triggerPoints || [];
  if (triggers.length === 0) return [];

  let frameCounter = register.frames.length + 100;
  const llmFrames = [];

  for (let t = 0; t < triggers.length; t++) {
    const tp = triggers[t];
    if (onProgress) onProgress(`Interpreting trigger ${t + 1}/${triggers.length} [${tp.trigger.reason}]…`);

    // Yield the single LLM to anything with priority (a user question).
    if (gate) await gate();

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

/* ── Functional reclassification — the LLM names what the centroid missed ──

   The shadow found the flagged clauses; the LLM only looks at one in context
   and says which operator is actually functioning. The disagreement is
   written back into the graph and logged in the register as `reclassified`. */

export const RECLASSIFY_SYSTEM =
  `You are classifying the FUNCTION of a clause in context. The mechanical `
  + `classifier tagged it by surface form, but a surprise, drift or silence `
  + `signal suggests its function diverges from its surface. Given the `
  + `surrounding text and the active interpretive frames, what is the clause `
  + `actually DOING?\n\n`
  + `The operators:\n`
  + `NUL - noting absence or restraint\n`
  + `SIG - directing attention to something notable\n`
  + `INS - something new comes into existence\n`
  + `SEG - drawing a boundary, separating\n`
  + `CON - connecting across a boundary\n`
  + `SYN - combining into an emergent whole\n`
  + `DEF - stating a fact (may still be the correct call)\n`
  + `EVA - rendering judgment, evaluating\n`
  + `REC - replacing the entire frame of understanding\n\n`
  + `Respond ONLY as JSON, no backticks: {"operator":"...","reason":"...under 20 words"}`;

/* llm is an async (system, user) => string. clauses is the local clause
   array for this block; graph/clauseBase are optional — when supplied, the
   functional operator is written back onto the cumulative graph's clause. */
export async function reclassifyFlags(register, clauses, graph, clauseBase, llm, onProgress, gate) {
  const flags = register.flags || [];
  const validOps = new Set(Object.keys(OPERATORS));
  const changed = [];

  for (let f = 0; f < flags.length; f++) {
    const flag = flags[f];
    if (flag.functionalOp) continue; // already read
    if (onProgress) onProgress(`Reading flagged clause ${f + 1}/${flags.length} [${flag.reason}]…`);

    // Yield the single LLM to anything with priority (a user question).
    if (gate) await gate();

    const ci = flag.clauseIndex;
    const start = Math.max(0, ci - 2);
    const end = Math.min(clauses.length - 1, ci + 2);
    const context = [];
    for (let c = start; c <= end; c++) context.push(`[${c + 1}] ${clauses[c]}`);

    const frameContext = register.frames
      .filter(fr => fr.generatedAt <= ci)
      .map(fr => `- ${fr.text} (strength ${Math.round((fr.strength || 0) * 100)}%)`)
      .join("\n") || "(no active frames)";

    const user = `ACTIVE FRAMES:\n${frameContext}\n\nCONTEXT:\n${context.join("\n")}\n\n`
      + `The mechanical classifier tagged the clause below as ${flag.mechanicalOp} `
      + `(flagged: ${flag.reason}).\n\nCLASSIFY THIS CLAUSE:\n[${ci + 1}] "${flag.text}"`;

    try {
      const raw = await llm(RECLASSIFY_SYSTEM, user);
      let parsed;
      try { parsed = JSON.parse(String(raw).replace(/```json|```/g, "").trim()); }
      catch { parsed = null; }
      const operator = parsed && String(parsed.operator || "").trim().toUpperCase();
      if (!operator || !validOps.has(operator)) {
        register.history.push({ event: "reclassify_failed", at: ci, text: flag.mechanicalOp });
        continue;
      }
      flag.functionalOp = operator;
      flag.functionalReason = String(parsed.reason || "").trim();
      flag.confirmed = operator === flag.mechanicalOp;

      if (graph && Number.isFinite(clauseBase)) {
        const gc = graph.clauses.find(c => c.index === clauseBase + ci);
        if (gc) gc.functionalOp = operator;
      }

      if (flag.confirmed) {
        register.history.push({
          event: "flag_confirmed", at: ci, text: `${operator} holds — ${flag.functionalReason}`,
        });
      } else {
        register.history.push({
          event: "reclassified", at: ci, mechanicalOp: flag.mechanicalOp,
          functionalOp: operator, text: flag.functionalReason,
        });
        changed.push(flag);
      }
    } catch (e) {
      register.history.push({ event: "reclassify_failed", at: ci, text: e.message });
    }
  }
  return changed;
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

/* ── Point-based entity folds — situated, windowed compression ──

   A fold is never global. foldEntityAt compresses an entity's claims UP TO a
   point in the text, from one site's perspective, into a single committed
   reading — what the entity SEEMS TO BE at that moment, not what it is overall.

   The same entity folded at the same site at successive points is a reading
   diary: the sequence is the arc. The same entity folded at different sites at
   one point produces readings that disagree, and that disagreement is where
   the text is doing its most interesting work.

   llm is an async (system, user) => string. */

export async function foldEntityAt(graph, entityKey, site, upToClause, llm) {
  const entity = graph.entities[entityKey];
  if (!entity) return null;

  // Only claims from clauses up to this point, at this site.
  const windowedClaims = entity.claims
    .filter(c => c.clauseIndex <= upToClause && c.site === site);
  if (windowedClaims.length < 2) return null;

  // The interpretive frames active at this point.
  const activeFrames = (graph.frames || [])
    .filter(f => f.generatedAt <= upToClause && f.strength > 0.2);

  // The ambient operator distribution up to this point.
  const opDist = {};
  for (const c of windowedClaims) {
    opDist[c.operator] = (opDist[c.operator] || 0) + 1;
  }

  const claimText = windowedClaims
    .map(c => `[c${c.clauseIndex + 1}] ${c.operator}: ${c.value} @"${c.span.slice(0, 60)}"`)
    .join("\n");

  const frameText = activeFrames.length > 0
    ? "\nACTIVE READING:\n" + activeFrames.map(f => `- ${f.text}`).join("\n")
    : "";

  const siteDefinition = TERRAINS[site] || site;

  const system =
    "Compress these claims into understanding FROM THIS SITE'S PERSPECTIVE "
    + "at this point in the text.\n\n"
    + `SITE: ${site} — ${siteDefinition}\n\n`
    + "This is what the text has revealed SO FAR. Not what is true overall. "
    + "What does this entity SEEM TO BE at this moment, read from this site?\n\n"
    + "One sentence. Under 30 words. Present tense. No hedging, no \"seems to\" "
    + "— commit to the reading the text has earned at this point.";

  const response = await llm(system, `Entity: ${entity.name}\n\n${claimText}${frameText}`);
  const text = String(response || "").trim();

  return {
    entity: entity.name,
    site,
    atClause: upToClause,
    text,
    vec: text ? await embed(text) : null,
    claimCount: windowedClaims.length,
    sources: windowedClaims.map(c => c.clauseIndex),
    opDistribution: opDist,
  };
}

/* Answer a question about one entity from its situated folds at a point in the
   text. Folds the entity at every site it occupies up to upToClause, then
   answers from those folds plus the raw windowed evidence — never from later
   text or outside knowledge. Returns { entity, atClause, answer, folds } or
   null when the entity is unknown or has no claims in the window. */
export async function askAboutEntity(graph, entityName, question, upToClause, llm, onProgress) {
  const entityKey = String(entityName || "").toLowerCase();
  const entity = graph.entities[entityKey];
  if (!entity) return null;

  const windowed = entity.claims.filter(c => c.clauseIndex <= upToClause);
  if (windowed.length === 0) return null;

  // Fold at every site occupied up to this point.
  const occupiedSites = [...new Set(windowed.map(c => c.site))];
  if (onProgress) {
    onProgress(`Folding ${entity.name} at ${occupiedSites.length} site${occupiedSites.length !== 1 ? "s" : ""}…`);
  }
  const folds = (await Promise.all(
    occupiedSites.map(site => foldEntityAt(graph, entityKey, site, upToClause, llm))
  )).filter(Boolean);

  const entityHypotheses = (graph.frames || [])
    .filter(f => f.generatedAt <= upToClause
      && f.text.toLowerCase().includes(entityKey)
      && f.strength > 0.2);

  const foldText = folds.length > 0
    ? folds.map(f => `@${f.site}: ${f.text}`).join("\n")
    : "(too few claims to fold — answer from the evidence below)";

  const hypText = entityHypotheses.length > 0
    ? "\nHYPOTHESES:\n" + entityHypotheses.map(h => `- ${h.text}`).join("\n")
    : "";

  const rawClaims = windowed
    .map(c => `[c${c.clauseIndex + 1}] "${c.span.slice(0, 80)}"`)
    .join("\n");

  if (onProgress) onProgress("Answering from the situated folds…");

  const system =
    "Answer from the situated understandings below. They represent what the "
    + `text has established AT THIS POINT (up to clause ${upToClause + 1}). Do `
    + "not use knowledge from later in the text, or outside knowledge about the "
    + "subject.\n\n"
    + "Each @Site line is a reading from a different perspective. Use whichever "
    + "perspectives best address the question. Note tensions between them when "
    + "relevant.\n\nGround your answer in the raw evidence. Be direct. 2-4 sentences.";

  const prompt =
    `ENTITY: ${entity.name}\n\n`
    + `SITUATED UNDERSTANDING:\n${foldText}${hypText}\n\n`
    + `EVIDENCE:\n${rawClaims}\n\n`
    + `QUESTION: ${question}`;

  const answer = await llm(system, prompt);

  return {
    entity: entity.name,
    atClause: upToClause,
    answer: String(answer || "").trim(),
    folds,
  };
}
