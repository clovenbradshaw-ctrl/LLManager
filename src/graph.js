/* The situated knowledge graph — v3.

   A Graph wraps a plain, serialisable `store` object (so it round-trips
   localStorage) and exposes the stateful API the v3 prompts and builders
   expect: content-addressed entities, DEFs/edges/EVAs with provenance, an
   append-only Given-Log and event log, and a per-level hypothesis history.

   Identity is content-addressed:
     - entity ids are e_<hash> minted from canonical name + creation time
     - a state version (@xxxx) is derived from the entity's current content
     - Given-Log ids are g_<hash> of agent + text + time

   Terrains classify entities: Entity, Network, Paradigm, Void, Kind, Field,
   Link, Atmosphere, Lens. */

export const TERRAINS = [
  "Entity", "Network", "Paradigm", "Void", "Kind", "Field", "Link", "Atmosphere", "Lens",
];

/* ── Sync content hashing ──

   A synchronous hash producing 64 hex chars, mixed from several FNV-1a
   passes with different seeds. Not cryptographic, but stable and
   collision-resistant enough for local content addressing. */
function fnv(str, seed) {
  let h = seed >>> 0;
  const s = String(str || "");
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h >>> 0;
}
export function sha256(str) {
  const seeds = [0x811c9dc5, 0x01000193, 0x9e3779b9, 0x85ebca6b,
                 0xc2b2ae35, 0x27d4eb2f, 0x165667b1, 0xff51afd7];
  let out = "";
  for (let i = 0; i < seeds.length; i++) {
    out += fnv(str + "#" + i, seeds[i]).toString(16).padStart(8, "0");
  }
  return out;
}

/* Mint a permanent entity id from INS content. */
export function mintEntityId(canonical, timestamp = Date.now()) {
  return "e_" + sha256(`${String(canonical).toLowerCase().trim()}::${timestamp}`).slice(0, 8);
}

/* Mint a Given-Log id from a message. */
export function mintGivenId(agent, text, timestamp = Date.now()) {
  return "g_" + sha256(`${agent}::${String(text).slice(0, 100)}::${timestamp}`).slice(0, 8);
}

const slug = (s) => String(s || "").trim().toLowerCase().replace(/\s+/g, " ");
let seq = 0;
const localId = (p) => p + Date.now().toString(36) + (seq++).toString(36) + Math.random().toString(36).slice(2, 5);

/* ── An empty store ── */
export function emptyStore() {
  return {
    schema: "graph-3",
    entities: {},      // id → { id, canonical, terrain, aliases, hypothesis, created, forkedFrom, passages }
    defs: {},          // defId → { id, entity, field, value, source, span, supersedes, retired, created }
    edges: {},         // edgeId → { id, from, to, type, source, created }
    evals: {},         // evalId → { id, entity, claim, status, source, span, created }
    given: [],         // Given-Log entries (append-only)
    events: [],        // mutation / commit event log (append-only)
    hypotheses: {},     // "level::key" → [ { text, after, inputCount, ts } ]
    documents: {},     // docId → { id, title, passageCount, groups, sections, hypothesis }
    pending: [],       // commits awaiting user consent (PROMPT/REQUIRE tier)
    lastTurn: null,
  };
}

/* ── The Graph ── */
export class Graph {
  constructor(store) {
    this.store = store && store.schema === "graph-3" ? store : emptyStore();
  }

  /* The plain object for persistence. */
  toJSON() { return this.store; }
  static from(store) { return new Graph(store); }

  /* ── Entities ── */
  getEntity(id) { return this.store.entities[id] || null; }
  allEntities() { return Object.values(this.store.entities); }

  createEntity(id, props = {}) {
    const e = {
      id,
      canonical: props.canonical || id,
      terrain: props.terrain || "Entity",
      aliases: props.aliases ? [...props.aliases] : [],
      hypothesis: props.hypothesis || "",
      forkedFrom: props.forkedFrom || null,
      passages: props.passages ? [...props.passages] : [],
      mentions: 0,
      created: Date.now(),
    };
    this.store.entities[id] = e;
    return e;
  }

  /* Resolve a textual reference (id, canonical name or alias) to an entity. */
  resolve(ref) {
    if (!ref) return null;
    const bare = String(ref).split("@")[0];
    if (this.store.entities[bare]) return this.store.entities[bare];
    const want = slug(ref);
    for (const e of this.allEntities()) {
      if (slug(e.canonical) === want) return e;
      if ((e.aliases || []).some(a => slug(a) === want)) return e;
    }
    return null;
  }

  /* Resolve or mint. New entities are content-addressed from the name. */
  ensureEntity(ref, { canonical, terrain } = {}) {
    const found = this.resolve(ref) || (canonical ? this.resolve(canonical) : null);
    if (found) {
      found.mentions = (found.mentions || 0) + 1;
      if (terrain && (!found.terrain || found.terrain === "Entity") && terrain !== "Entity") {
        found.terrain = terrain;
      }
      return found;
    }
    const name = (canonical || (/^e_[0-9a-f]{8}$/i.test(ref) ? "" : ref) || "").trim();
    if (!name) return null;
    const e = this.createEntity(mintEntityId(name), { canonical: name, terrain });
    e.mentions = 1;
    return e;
  }

  updateTerrain(id, terrain) {
    const e = this.getEntity(id);
    if (e) e.terrain = terrain;
  }

  addAlias(id, alias) {
    const e = this.getEntity(id);
    if (e && alias && !e.aliases.includes(alias)) e.aliases.push(alias);
  }

  searchEntities(query) {
    const q = slug(query);
    if (!q) return [];
    return this.allEntities().filter(e => {
      const c = slug(e.canonical);
      return c.includes(q) || q.includes(c)
        || (e.aliases || []).some(a => slug(a).includes(q) || q.includes(slug(a)));
    });
  }

  /* ── DEFs ── */
  getDefs(entityId) {
    return Object.values(this.store.defs).filter(d => d.entity === entityId && !d.retired);
  }
  getDef(entityId, field) {
    return Object.values(this.store.defs)
      .find(d => d.entity === entityId && d.field === field && !d.retired) || null;
  }
  writeDef(entityId, field, value, opts = {}) {
    // Supersede any prior live DEF on the same field.
    const prior = this.getDef(entityId, field);
    if (prior) prior.retired = true;
    const d = {
      id: localId("d_"), entity: entityId, field: String(field), value: String(value),
      source: opts.source || null, span: opts.span || null,
      supersedes: prior ? prior.id : (opts.supersedes || null),
      retired: false, created: Date.now(),
    };
    this.store.defs[d.id] = d;
    return d;
  }
  reassignDef(defId, fromEntity, toEntity) {
    const d = this.store.defs[defId];
    if (d && d.entity === fromEntity) d.entity = toEntity;
  }

  /* DEF conflicts — two live DEFs on the same field with different values. */
  getConflicts(entityId) {
    const byField = {};
    for (const d of this.getDefs(entityId)) (byField[d.field] ||= []).push(d);
    const conflicts = [];
    for (const [field, list] of Object.entries(byField)) {
      if (list.length > 1) {
        const vals = [...new Set(list.map(d => d.value))];
        if (vals.length > 1) conflicts.push({ field, existing: vals[0], incoming: vals[1] });
      }
    }
    return conflicts;
  }

  /* ── Edges ── */
  getEdges(entityId) {
    return Object.values(this.store.edges).filter(e => e.from === entityId || e.to === entityId);
  }
  addEdge(from, to, type, opts = {}) {
    const id = `${from}::${type}::${to}`;
    const e = { id, from, to, type: String(type), source: opts.source || null, created: Date.now() };
    this.store.edges[id] = e;
    return e;
  }

  /* ── EVAs ── */
  getEvals(entityId) {
    return Object.values(this.store.evals).filter(e => e.entity === entityId);
  }
  writeEval(entityId, claim, status, opts = {}) {
    const e = {
      id: localId("v_"), entity: entityId, claim: String(claim),
      status: status || "holds", source: opts.source || null, span: opts.span || null,
      created: Date.now(),
    };
    this.store.evals[e.id] = e;
    return e;
  }

  /* ── Merge: fold `absorb` into `keep` ── */
  mergeEntities(keep, absorb) {
    const src = this.getEntity(absorb), dst = this.getEntity(keep);
    if (!src || !dst || keep === absorb) return;
    dst.mentions = (dst.mentions || 0) + (src.mentions || 0);
    if ((src.hypothesis || "").length > (dst.hypothesis || "").length) dst.hypothesis = src.hypothesis;
    if (src.canonical && !dst.aliases.includes(src.canonical)) dst.aliases.push(src.canonical);
    for (const a of src.aliases || []) if (!dst.aliases.includes(a)) dst.aliases.push(a);
    for (const d of Object.values(this.store.defs)) if (d.entity === absorb) d.entity = keep;
    for (const v of Object.values(this.store.evals)) if (v.entity === absorb) v.entity = keep;
    for (const [k, e] of Object.entries(this.store.edges)) {
      const from = e.from === absorb ? keep : e.from;
      const to = e.to === absorb ? keep : e.to;
      if (from !== e.from || to !== e.to) {
        delete this.store.edges[k];
        if (from !== to) this.store.edges[`${from}::${e.type}::${to}`] = { ...e, id: `${from}::${e.type}::${to}`, from, to };
      }
    }
    delete this.store.entities[absorb];
  }

  /* ── State version ── */
  stateHash(entityId) {
    const e = this.getEntity(entityId);
    if (!e) return "0000";
    const defs = this.getDefs(entityId)
      .sort((a, b) => a.field.localeCompare(b.field))
      .map(d => `${d.field}=${d.value}`);
    const edges = this.getEdges(entityId)
      .sort((a, b) => a.to.localeCompare(b.to))
      .map(e2 => `${e2.type}→${e2.to}`);
    return sha256([entityId, e.terrain, e.hypothesis || "", ...defs, ...edges].join("|")).slice(0, 4);
  }

  /* ── Given-Log ── */
  appendGiven(entry) { this.store.given.push(entry); return entry; }
  given(id) { return this.store.given.find(g => g.id === id) || null; }

  /* ── Event log ── */
  appendEvent(event) {
    const e = { id: localId("ev_"), ...event };
    this.store.events.push(e);
    return e;
  }

  /* ── Pending commits (awaiting consent) ── */
  addPending(commit) { this.store.pending.push(commit); return commit; }
  resolvePending(id, status) {
    const c = this.store.pending.find(p => p.id === id);
    if (c) c.status = status;
    return c;
  }
  pendingCommits() { return this.store.pending.filter(p => !p.status || p.status === "pending"); }

  /* ── Hypothesis history (per level + key) ── */
  hypKey(level, id) {
    if (typeof id === "string") return `${level}::${id}`;
    if (id && id.documentId) return `${level}::${id.documentId}`;
    if (id && id.sessionId) return `${level}::${id.sessionId}`;
    if (id && id.start != null) return `${level}::${id.start}-${id.end}`;
    return `${level}::_`;
  }
  getHypothesisHistory(level, id) {
    return this.store.hypotheses[this.hypKey(level, id)] || [];
  }
  recordHypothesis(level, id, text, inputCount) {
    const key = this.hypKey(level, id);
    (this.store.hypotheses[key] ||= []).push({
      text, after: new Date().toISOString().slice(0, 16).replace("T", " "),
      inputCount: inputCount || 0, ts: Date.now(),
    });
  }

  /* ── Document structure (for the hypothesis hierarchy) ── */
  registerDocument(docId, title, passageCount) {
    this.store.documents[docId] = {
      id: docId, title: title || "Untitled", passageCount: passageCount || 0,
      groups: [], sections: [], hypothesis: "",
    };
    return this.store.documents[docId];
  }
  getDocument(docId) { return this.store.documents[docId] || null; }

  getEntitiesInRange(start, end) {
    return this.allEntities().filter(e =>
      (e.passages || []).some(p => p >= start && p <= end));
  }
  recordGroup(docId, start, end, hypothesis) {
    const doc = this.store.documents[docId];
    if (doc) doc.groups.push({ start, end, hypothesis });
  }
  recordSection(docId, start, end, hypothesis) {
    const doc = this.store.documents[docId];
    if (doc) doc.sections.push({ start, end, hypothesis });
  }
  getPassageGroupDEFs(start, end) {
    const groups = [];
    for (const doc of Object.values(this.store.documents)) {
      for (const g of doc.groups) {
        if (g.start >= start && g.end <= end) groups.push(g);
      }
    }
    return groups;
  }
  getSectionDEFs(docId) {
    const doc = this.store.documents[docId];
    return doc ? doc.sections : [];
  }
  getSessionDocumentDEFs() {
    return Object.values(this.store.documents).map(d => ({ title: d.title, hypothesis: d.hypothesis }));
  }
  getSessionTopics() {
    return this.store.lastTurn?.topic ? [this.store.lastTurn.topic] : [];
  }
  getRecentDocumentDEFs(n = 10) {
    return Object.values(this.store.documents).slice(-n).map(d => ({ title: d.title, hypothesis: d.hypothesis }));
  }
  getRecentSessionDEFs(n = 5) {
    return (this.store.hypotheses["session::_"] || []).slice(-n).map(h => ({ hypothesis: h.text }));
  }

  /* ── Stats ── */
  stats() {
    return {
      entities: Object.keys(this.store.entities).length,
      defs: Object.values(this.store.defs).filter(d => !d.retired).length,
      edges: Object.keys(this.store.edges).length,
      evals: Object.keys(this.store.evals).length,
      given: this.store.given.length,
      pending: this.pendingCommits().length,
    };
  }
}

/* Merge several graphs into a fresh read-only Graph — a chat's own graph
   plus any opted-in library documents — for projection. */
export function mergeGraphs(...graphs) {
  const out = emptyStore();
  for (const g of graphs) {
    if (!g) continue;
    const s = g.store || g;
    Object.assign(out.entities, s.entities || {});
    Object.assign(out.defs, s.defs || {});
    Object.assign(out.edges, s.edges || {});
    Object.assign(out.evals, s.evals || {});
    out.given.push(...(s.given || []));
    Object.assign(out.documents, s.documents || {});
  }
  return new Graph(out);
}
