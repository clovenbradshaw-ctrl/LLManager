/* Local Store — SQLite + Embeddings.

   One SQLite database, OPFS-persisted. Graph structure and embedding
   vectors in the same store. The Reach queries this directly every turn —
   entity lookup by name, terrain scoping, N-hop edge traversal, DEF lookup
   by field, conflict grouping, hypothesis history by level — all indexed.

   Per-chat isolation: every graph and vector row carries a `scope` (the
   chat id). One database holds many isolated per-chat graphs. Call
   setScope(chatId) before any read or write.

   What travels (Matrix sync, later): Given-Log, entities, DEFs, edges,
   hypotheses text. What stays local: embedding vectors — projections
   derived from text, rebuilt from the stored text if the database is wiped. */

import sqlite3InitModule from "@sqlite.org/sqlite-wasm";
import { pipeline, env } from "@xenova/transformers";

const DIMS = 384;
const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";

/* Allow the model to be fetched from the HF CDN; cache in the browser. */
env.allowLocalModels = false;

/* ── Init ── */

let db = null;
let embedder = null;
let scope = "default";
let persistent = false;

export function setScope(chatId) { scope = chatId || "default"; }
export function getScope() { return scope; }
export function ready() { return !!db; }
export function isPersistent() { return persistent; }

export async function init() {
  if (db) return { db, persistent };
  const sqlite3 = await sqlite3InitModule();
  try {
    // OPFS SyncAccessHandle pool — persists without cross-origin isolation,
    // so it works on a plain static host (GitHub Pages).
    const pool = await sqlite3.installOpfsSAHPoolVfs({ name: "eo-opfs" });
    db = new pool.OpfsSAHPoolDb("/eo-graph.db");
    persistent = true;
  } catch {
    // No OPFS — fall back to an in-memory database for this session.
    db = new sqlite3.oo1.DB(":memory:");
    persistent = false;
  }
  db.exec(SCHEMA);
  return { db, persistent };
}

async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline("feature-extraction", EMBED_MODEL, { quantized: true });
  }
  return embedder;
}

export async function embed(text) {
  const model = await getEmbedder();
  const out = await model(String(text || ""), { pooling: "mean", normalize: true });
  return new Float32Array(out.data);
}

/* ── Schema ── */

const SCHEMA = `
  CREATE TABLE IF NOT EXISTS given (
    scope       TEXT NOT NULL,
    id          TEXT NOT NULL,
    agent       TEXT NOT NULL,
    mode        TEXT NOT NULL,
    text        TEXT NOT NULL,
    document_id TEXT,
    passage_idx INTEGER,
    session_id  TEXT,
    turn        INTEGER,
    dossier_hash TEXT,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (scope, id)
  );

  CREATE TABLE IF NOT EXISTS entities (
    scope       TEXT NOT NULL,
    id          TEXT NOT NULL,
    canonical   TEXT NOT NULL,
    terrain     TEXT NOT NULL,
    hypothesis  TEXT,
    aliases     TEXT DEFAULT '[]',
    forked_from TEXT,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (scope, id)
  );

  CREATE TABLE IF NOT EXISTS edges (
    scope       TEXT NOT NULL,
    id          TEXT NOT NULL,
    from_id     TEXT NOT NULL,
    to_id       TEXT NOT NULL,
    type        TEXT NOT NULL,
    source      TEXT,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (scope, id)
  );

  CREATE TABLE IF NOT EXISTS defs (
    scope       TEXT NOT NULL,
    id          TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    field       TEXT NOT NULL,
    value       TEXT NOT NULL,
    span        TEXT,
    source      TEXT,
    supersedes  TEXT,
    retired     INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (scope, id)
  );

  CREATE TABLE IF NOT EXISTS evals (
    scope       TEXT NOT NULL,
    id          TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    claim       TEXT NOT NULL,
    status      TEXT NOT NULL,
    span        TEXT,
    source      TEXT,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (scope, id)
  );

  CREATE TABLE IF NOT EXISTS hypotheses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope       TEXT NOT NULL,
    level       TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    text        TEXT NOT NULL,
    revision    INTEGER NOT NULL DEFAULT 1,
    after_label TEXT,
    input_count INTEGER,
    grounded_in TEXT,
    created_at  INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS mutations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope       TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT NOT NULL,
    reason      TEXT NOT NULL,
    triggered_by TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS vec_clauses (
    scope       TEXT NOT NULL,
    id          TEXT NOT NULL,
    entity_id   TEXT,
    vector      BLOB NOT NULL,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (scope, id)
  );

  CREATE TABLE IF NOT EXISTS vec_centroids (
    scope       TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    vector      BLOB NOT NULL,
    clause_count INTEGER NOT NULL DEFAULT 0,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (scope, entity_id)
  );

  CREATE TABLE IF NOT EXISTS vec_hypotheses (
    scope       TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    vector      BLOB NOT NULL,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (scope, entity_id)
  );

  CREATE INDEX IF NOT EXISTS idx_defs_entity   ON defs(scope, entity_id);
  CREATE INDEX IF NOT EXISTS idx_defs_field    ON defs(scope, entity_id, field);
  CREATE INDEX IF NOT EXISTS idx_edges_from    ON edges(scope, from_id);
  CREATE INDEX IF NOT EXISTS idx_edges_to      ON edges(scope, to_id);
  CREATE INDEX IF NOT EXISTS idx_given_session ON given(scope, session_id);
  CREATE INDEX IF NOT EXISTS idx_given_doc     ON given(scope, document_id);
  CREATE INDEX IF NOT EXISTS idx_hyp_target    ON hypotheses(scope, level, target_id);
  CREATE INDEX IF NOT EXISTS idx_vec_clause_ent ON vec_clauses(scope, entity_id);
  CREATE INDEX IF NOT EXISTS idx_entities_terrain ON entities(scope, terrain);
`;

/* ── Vector ↔ BLOB ── */

function toBlob(vec) { return new Uint8Array(vec.buffer, vec.byteOffset, vec.byteLength); }
function toVec(blob) { return new Float32Array(new Uint8Array(blob).buffer); }

export function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d === 0 ? 0 : dot / d;
}

/* ── Given-Log ── */

export const given = {
  write(entry) {
    db.exec({
      sql: `INSERT OR IGNORE INTO given (scope,id,agent,mode,text,document_id,passage_idx,session_id,turn,dossier_hash,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
      bind: [scope, entry.id, entry.agent, entry.mode, entry.text, entry.documentId || null,
             entry.passageIndex ?? null, entry.sessionId || entry.session || null, entry.turn ?? null,
             entry.dossierHash || null, entry.timestamp || Date.now()],
    });
  },
  get(id) { return db.selectObject("SELECT * FROM given WHERE scope=? AND id=?", [scope, id]); },
  getBySession(sessionId) {
    return db.selectObjects("SELECT * FROM given WHERE scope=? AND session_id=? ORDER BY created_at", [scope, sessionId]);
  },
  getByDocument(documentId) {
    return db.selectObjects("SELECT * FROM given WHERE scope=? AND document_id=? ORDER BY passage_idx", [scope, documentId]);
  },
  getAll() { return db.selectObjects("SELECT * FROM given WHERE scope=? ORDER BY created_at", [scope]); },
};

/* ── Entities ── */

export const entities = {
  get(id) {
    const r = db.selectObject("SELECT * FROM entities WHERE scope=? AND id=?", [scope, id]);
    if (r) r.aliases = JSON.parse(r.aliases || "[]");
    return r;
  },
  getAll() {
    return db.selectObjects("SELECT * FROM entities WHERE scope=? ORDER BY created_at", [scope])
      .map(r => ({ ...r, aliases: JSON.parse(r.aliases || "[]") }));
  },
  create(id, canonical, terrain, { hypothesis = null, aliases = [], forkedFrom = null } = {}) {
    const now = Date.now();
    db.exec({
      sql: `INSERT OR REPLACE INTO entities (scope,id,canonical,terrain,hypothesis,aliases,forked_from,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)`,
      bind: [scope, id, canonical, terrain, hypothesis, JSON.stringify(aliases), forkedFrom, now, now],
    });
  },
  updateHypothesis(id, text) {
    db.exec({ sql: "UPDATE entities SET hypothesis=?, updated_at=? WHERE scope=? AND id=?", bind: [text, Date.now(), scope, id] });
  },
  updateTerrain(id, terrain) {
    db.exec({ sql: "UPDATE entities SET terrain=?, updated_at=? WHERE scope=? AND id=?", bind: [terrain, Date.now(), scope, id] });
  },
  updateCanonical(id, canonical) {
    db.exec({ sql: "UPDATE entities SET canonical=?, updated_at=? WHERE scope=? AND id=?", bind: [canonical, Date.now(), scope, id] });
  },
  addAlias(id, alias) {
    const e = entities.get(id);
    if (!e || e.aliases.includes(alias)) return;
    e.aliases.push(alias);
    db.exec({ sql: "UPDATE entities SET aliases=?, updated_at=? WHERE scope=? AND id=?", bind: [JSON.stringify(e.aliases), Date.now(), scope, id] });
  },
  search(query) {
    const q = `%${String(query).toLowerCase()}%`;
    return db.selectObjects(
      `SELECT * FROM entities WHERE scope=? AND (LOWER(canonical) LIKE ? OR LOWER(aliases) LIKE ? OR LOWER(hypothesis) LIKE ?)`,
      [scope, q, q, q]
    ).map(r => ({ ...r, aliases: JSON.parse(r.aliases || "[]") }));
  },
  byTerrain(terrain) {
    return db.selectObjects("SELECT * FROM entities WHERE scope=? AND terrain=?", [scope, terrain])
      .map(r => ({ ...r, aliases: JSON.parse(r.aliases || "[]") }));
  },
  getInRange(startPassage, endPassage) {
    return db.selectObjects(`
      SELECT DISTINCT e.* FROM entities e
      JOIN defs d ON d.entity_id = e.id AND d.scope = e.scope
      JOIN given g ON d.source = g.id AND g.scope = e.scope
      WHERE e.scope=? AND g.passage_idx BETWEEN ? AND ?
    `, [scope, startPassage, endPassage])
    .map(r => ({ ...r, aliases: JSON.parse(r.aliases || "[]") }));
  },
  remove(id) {
    db.exec({ sql: "DELETE FROM entities WHERE scope=? AND id=?", bind: [scope, id] });
  },
};

/* ── Edges ── */

export const edges = {
  create(fromId, toId, type, source = null) {
    const id = `${fromId}::${type}::${toId}`;
    db.exec({
      sql: `INSERT OR IGNORE INTO edges (scope,id,from_id,to_id,type,source,created_at) VALUES (?,?,?,?,?,?,?)`,
      bind: [scope, id, fromId, toId, type, source, Date.now()],
    });
  },
  getFor(entityId) {
    return db.selectObjects("SELECT * FROM edges WHERE scope=? AND (from_id=? OR to_id=?)", [scope, entityId, entityId]);
  },
  getNeighbors(entityId, hops = 2) {
    let frontier = new Set([entityId]);
    const visited = new Set([entityId]);
    for (let h = 0; h < hops; h++) {
      const next = new Set();
      for (const nid of frontier) {
        const rows = db.selectObjects("SELECT from_id, to_id FROM edges WHERE scope=? AND (from_id=? OR to_id=?)", [scope, nid, nid]);
        for (const r of rows) {
          if (!visited.has(r.from_id)) { next.add(r.from_id); visited.add(r.from_id); }
          if (!visited.has(r.to_id)) { next.add(r.to_id); visited.add(r.to_id); }
        }
      }
      frontier = next;
    }
    visited.delete(entityId);
    return [...visited];
  },
  reTarget(fromEntity, toEntity) {
    db.exec({ sql: "UPDATE edges SET from_id=? WHERE scope=? AND from_id=?", bind: [toEntity, scope, fromEntity] });
    db.exec({ sql: "UPDATE edges SET to_id=? WHERE scope=? AND to_id=?", bind: [toEntity, scope, fromEntity] });
  },
};

/* ── DEFs ── */

export const defs = {
  write(id, entityId, field, value, { span = null, source = null, supersedes = null, retired = 0 } = {}) {
    db.exec({
      sql: `INSERT OR REPLACE INTO defs (scope,id,entity_id,field,value,span,source,supersedes,retired,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)`,
      bind: [scope, id, entityId, field, value, span, source, supersedes, retired ? 1 : 0, Date.now()],
    });
  },
  getFor(entityId) {
    return db.selectObjects("SELECT * FROM defs WHERE scope=? AND entity_id=? AND retired=0 ORDER BY created_at", [scope, entityId]);
  },
  get(entityId, field) {
    return db.selectObject(
      "SELECT * FROM defs WHERE scope=? AND entity_id=? AND field=? AND retired=0 ORDER BY created_at DESC LIMIT 1",
      [scope, entityId, field]
    );
  },
  getAll() { return db.selectObjects("SELECT * FROM defs WHERE scope=?", [scope]); },
  retire(id) { db.exec({ sql: "UPDATE defs SET retired=1 WHERE scope=? AND id=?", bind: [scope, id] }); },
  reassign(id, toEntity) { db.exec({ sql: "UPDATE defs SET entity_id=? WHERE scope=? AND id=?", bind: [toEntity, scope, id] }); },
  getConflicts(entityId) {
    return db.selectObjects(`
      SELECT field, GROUP_CONCAT(DISTINCT value) AS vals, COUNT(DISTINCT value) AS cnt
      FROM defs WHERE scope=? AND entity_id=? AND retired=0 GROUP BY field HAVING cnt > 1
    `, [scope, entityId]);
  },
};

/* ── EVAls ── */

export const evals = {
  write(id, entityId, claim, status, { span = null, source = null } = {}) {
    db.exec({
      sql: `INSERT OR REPLACE INTO evals (scope,id,entity_id,claim,status,span,source,created_at)
            VALUES (?,?,?,?,?,?,?,?)`,
      bind: [scope, id, entityId, claim, status, span, source, Date.now()],
    });
  },
  getFor(entityId) {
    return db.selectObjects("SELECT * FROM evals WHERE scope=? AND entity_id=? ORDER BY created_at", [scope, entityId]);
  },
};

/* ── Hypotheses ── */

export const hypotheses = {
  write(level, targetId, text, { afterLabel = null, inputCount = null, groundedIn = null } = {}) {
    const existing = db.selectObject(
      "SELECT MAX(revision) AS rev FROM hypotheses WHERE scope=? AND level=? AND target_id=?",
      [scope, level, targetId]
    );
    const rev = (existing?.rev || 0) + 1;
    db.exec({
      sql: `INSERT INTO hypotheses (scope,level,target_id,text,revision,after_label,input_count,grounded_in,created_at)
            VALUES (?,?,?,?,?,?,?,?,?)`,
      bind: [scope, level, targetId, text, rev, afterLabel, inputCount,
             groundedIn ? JSON.stringify(groundedIn) : null, Date.now()],
    });
    if (level === "entity") entities.updateHypothesis(targetId, text);
  },
  getHistory(level, targetId) {
    return db.selectObjects(
      "SELECT * FROM hypotheses WHERE scope=? AND level=? AND target_id=? ORDER BY revision",
      [scope, level, targetId]
    ).map(r => ({ ...r, groundedIn: r.grounded_in ? JSON.parse(r.grounded_in) : [] }));
  },
  getCurrent(level, targetId) {
    return db.selectObject(
      "SELECT * FROM hypotheses WHERE scope=? AND level=? AND target_id=? ORDER BY revision DESC LIMIT 1",
      [scope, level, targetId]
    );
  },
  getByLevel(level) {
    return db.selectObjects(`
      SELECT h.* FROM hypotheses h
      INNER JOIN (SELECT target_id, MAX(revision) AS max_rev FROM hypotheses WHERE scope=? AND level=? GROUP BY target_id) m
      ON h.target_id = m.target_id AND h.revision = m.max_rev
      WHERE h.scope=? AND h.level=?
    `, [scope, level, scope, level]);
  },
};

/* ── Mutations ── */

export const mutations = {
  log(action, detail, reason, triggeredBy = null, status = "pending") {
    db.exec({
      sql: "INSERT INTO mutations (scope,action,detail,reason,triggered_by,status,created_at) VALUES (?,?,?,?,?,?,?)",
      bind: [scope, action, JSON.stringify(detail), reason, triggeredBy, status, Date.now()],
    });
  },
  setStatus(id, status) {
    db.exec({ sql: "UPDATE mutations SET status=? WHERE scope=? AND id=?", bind: [status, scope, id] });
  },
  getPending() {
    return db.selectObjects("SELECT * FROM mutations WHERE scope=? AND status='pending' ORDER BY created_at", [scope])
      .map(r => ({ ...r, detail: JSON.parse(r.detail) }));
  },
  getAll() {
    return db.selectObjects("SELECT * FROM mutations WHERE scope=? ORDER BY created_at", [scope])
      .map(r => ({ ...r, detail: JSON.parse(r.detail) }));
  },
};

/* ── Embeddings ── */

export const vectors = {
  /* A walked clause — embed it, fold it into the entity's centroid as a
     running mean, then discard the vector. SIG is ephemeral: once INS has
     created the anchor and DEF has attached the claims with spans, the
     clause embedding that found the entity is scaffolding the graph
     structure has replaced. The centroid is the compressed memory of every
     context the entity appeared in; it absorbs each clause in real time and
     never needs the individual vectors back. */
  foldClause(entityId, vec) {
    if (!entityId) return;
    const existing = db.selectObject(
      "SELECT vector, clause_count FROM vec_centroids WHERE scope=? AND entity_id=?", [scope, entityId]);
    if (existing) {
      const old = toVec(existing.vector), n = existing.clause_count;
      const updated = new Float32Array(DIMS);
      for (let i = 0; i < DIMS; i++) updated[i] = (old[i] * n + vec[i]) / (n + 1);
      db.exec({
        sql: "UPDATE vec_centroids SET vector=?, clause_count=?, updated_at=? WHERE scope=? AND entity_id=?",
        bind: [toBlob(updated), n + 1, Date.now(), scope, entityId],
      });
    } else {
      db.exec({
        sql: "INSERT INTO vec_centroids (scope,entity_id,vector,clause_count,updated_at) VALUES (?,?,?,?,?)",
        bind: [scope, entityId, toBlob(vec), 1, Date.now()],
      });
    }
  },
  async addClause(id, entityId, text) {
    const vec = await embed(text);
    vectors.foldClause(entityId, vec);
    return vec;
  },
  /* A Given-Log clause — chat history or an unwalked document. No graph
     structure yet, so SIG is all there is: keep the vector until the
     material is walked, then prune it. */
  async addGivenVector(id, text) {
    const vec = await embed(text);
    db.exec({
      sql: "INSERT OR REPLACE INTO vec_clauses (scope,id,entity_id,vector,created_at) VALUES (?,?,?,?,?)",
      bind: [scope, id, null, toBlob(vec), Date.now()],
    });
    return vec;
  },
  /* After a document walk completes, tear down any clause vectors for the
     entities it walked — the centroid has absorbed them. */
  pruneWalkedVectors(documentId) {
    const walked = db.selectObjects(`
      SELECT DISTINCT d.entity_id FROM defs d
      JOIN given g ON d.source = g.id AND g.scope = d.scope
      WHERE d.scope=? AND g.document_id=?`, [scope, documentId]);
    for (const { entity_id } of walked) {
      db.exec({ sql: "DELETE FROM vec_clauses WHERE scope=? AND entity_id=?", bind: [scope, entity_id] });
    }
  },
  async updateHypothesisVec(entityId, hypothesisText) {
    const vec = await embed(hypothesisText);
    db.exec({
      sql: "INSERT OR REPLACE INTO vec_hypotheses (scope,entity_id,vector,updated_at) VALUES (?,?,?,?)",
      bind: [scope, entityId, toBlob(vec), Date.now()],
    });
    return vec;
  },
  async findSimilarEntities(queryText, topK = 5) {
    const qv = await embed(queryText);
    return db.selectObjects("SELECT entity_id, vector FROM vec_centroids WHERE scope=?", [scope])
      .map(r => ({ entityId: r.entity_id, similarity: cosineSim(qv, toVec(r.vector)) }))
      .sort((a, b) => b.similarity - a.similarity).slice(0, topK);
  },
  async findSimilarByHypothesis(queryText, topK = 5) {
    const qv = await embed(queryText);
    return db.selectObjects("SELECT entity_id, vector FROM vec_hypotheses WHERE scope=?", [scope])
      .map(r => ({ entityId: r.entity_id, similarity: cosineSim(qv, toVec(r.vector)) }))
      .sort((a, b) => b.similarity - a.similarity).slice(0, topK);
  },
  async findSimilarGiven(queryText, topK = 5) {
    const qv = await embed(queryText);
    return db.selectObjects("SELECT id, vector FROM vec_clauses WHERE scope=? AND entity_id IS NULL", [scope])
      .map(r => ({ id: r.id, similarity: cosineSim(qv, toVec(r.vector)) }))
      .sort((a, b) => b.similarity - a.similarity).slice(0, topK);
  },
  async compareToEntity(mentionText, entityId) {
    const mv = await embed(mentionText);
    const cRow = db.selectObject("SELECT vector FROM vec_centroids WHERE scope=? AND entity_id=?", [scope, entityId]);
    const hRow = db.selectObject("SELECT vector FROM vec_hypotheses WHERE scope=? AND entity_id=?", [scope, entityId]);
    return {
      entityId,
      clausePoolSimilarity: cRow ? cosineSim(mv, toVec(cRow.vector)) : 0,
      hypothesisSimilarity: hRow ? cosineSim(mv, toVec(hRow.vector)) : 0,
    };
  },
  async resolveMatch(mentionText, candidateId) {
    const cmp = await vectors.compareToEntity(mentionText, candidateId);
    if (cmp.clausePoolSimilarity > 0.75 && cmp.hypothesisSimilarity > 0.65) {
      return { decision: "SAME", confidence: "high", ...cmp };
    }
    if (cmp.clausePoolSimilarity > 0.55 || cmp.hypothesisSimilarity > 0.55) {
      return { decision: "UNCERTAIN", confidence: "low", ...cmp };
    }
    return { decision: "DIFFERENT", confidence: "high", ...cmp };
  },
  stats() {
    const clauses = db.selectObject("SELECT COUNT(*) AS n FROM vec_clauses WHERE scope=?", [scope])?.n || 0;
    const centroids = db.selectObject("SELECT COUNT(*) AS n FROM vec_centroids WHERE scope=?", [scope])?.n || 0;
    const hyps = db.selectObject("SELECT COUNT(*) AS n FROM vec_hypotheses WHERE scope=?", [scope])?.n || 0;
    return { clauses, centroids, hypotheses: hyps, bytesEstimate: clauses * (DIMS * 4) };
  },
};

/* ── Hashing ── sync content hashing, so id minting and state versions do
   not introduce awaits into the apply path (which would let the active
   scope change mid-write). 64 hex chars mixed from seeded FNV-1a passes. */

function fnv(str, seed) {
  let h = seed >>> 0;
  const s = String(str || "");
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h >>> 0;
}
export function sha256hex(str) {
  const seeds = [0x811c9dc5, 0x01000193, 0x9e3779b9, 0x85ebca6b,
                 0xc2b2ae35, 0x27d4eb2f, 0x165667b1, 0xff51afd7];
  let out = "";
  for (let i = 0; i < seeds.length; i++) out += fnv(str + "#" + i, seeds[i]).toString(16).padStart(8, "0");
  return out;
}
export function mintHash(input) { return sha256hex(input).slice(0, 8); }
export function mintEntityId(canonical) {
  return "e_" + mintHash(`${String(canonical).toLowerCase().trim()}::${Date.now()}::${Math.random()}`);
}
export function mintGivenId(agent, text) {
  return "g_" + mintHash(`${agent}::${String(text).slice(0, 100)}::${Date.now()}::${Math.random()}`);
}
export function computeStateHash(entityId) {
  const e = entities.get(entityId);
  if (!e) return "0000";
  const d = defs.getFor(entityId).map(x => `${x.field}=${x.value}`).sort();
  const eg = edges.getFor(entityId).map(x => `${x.type}→${x.to_id}`).sort();
  return sha256hex([entityId, e.terrain, e.hypothesis || "", ...d, ...eg].join("|")).slice(0, 4);
}

/* ── Apply events (from EXTRACT or INGEST) ── */

function resolveId(ref, events) {
  if (!ref) return null;
  if (String(ref).startsWith("e_")) return entities.get(ref) ? ref : null;
  const ins = events.find(e => e.op === "INS" && e.entity === ref && e._resolvedId);
  if (ins) return ins._resolvedId;
  const matches = entities.search(ref);
  return matches.length ? matches[0].id : null;
}

export async function applyEvents(events, sourceGivenId = null) {
  // Phase 1 (async): embed every DEF span up front.
  const spanVecs = [];
  for (let i = 0; i < events.length; i++) {
    const e = events[i];
    if (e.op === "DEF" && e.span && e.span.length > 10) spanVecs[i] = await embed(e.span);
  }
  // Phase 2 (sync): apply every event — no await, so the active scope
  // cannot change between writes.
  const results = { applied: 0, skipped: 0, ambigs: [], newEntities: [] };
  events.forEach((evt, i) => {
    const now = Date.now();
    if (evt.op === "INS") {
      const existing = entities.search(evt.entity)
        .find(m => m.canonical.toLowerCase() === String(evt.entity).toLowerCase());
      if (existing) { evt._resolvedId = existing.id; results.applied++; return; }
      const id = mintEntityId(evt.entity);
      entities.create(id, evt.entity, evt.terrain || "Entity");
      evt._resolvedId = id;
      results.newEntities.push(id);
      results.applied++;
    } else if (evt.op === "CON") {
      const from = resolveId(evt.from, events), to = resolveId(evt.to, events);
      if (from && to && from !== to) { edges.create(from, to, evt.type || "related to", sourceGivenId); results.applied++; }
      else results.skipped++;
    } else if (evt.op === "DEF") {
      const eid = resolveId(evt.entity, events);
      if (eid && evt.field && evt.value != null) {
        const prior = defs.get(eid, evt.field);
        const conflict = prior && prior.value !== String(evt.value) && evt.field !== "kind";
        if (prior && !conflict) defs.retire(prior.id);
        const defId = "d_" + mintHash(`${eid}::${evt.field}::${now}::${Math.random()}`);
        defs.write(defId, eid, evt.field, evt.value, {
          span: evt.span || null, source: sourceGivenId || evt.source || null,
          supersedes: prior ? prior.id : null,
        });
        if (spanVecs[i]) vectors.foldClause(eid, spanVecs[i]);
        results.applied++;
      } else results.skipped++;
    } else if (evt.op === "EVA") {
      const eid = resolveId(evt.entity, events);
      if (eid && evt.claim) {
        const evalId = "v_" + mintHash(`${eid}::${evt.claim}::${now}::${Math.random()}`);
        evals.write(evalId, eid, evt.claim, evt.status || "holds", {
          span: evt.span || null, source: sourceGivenId || evt.source || null,
        });
        results.applied++;
      } else results.skipped++;
    } else if (evt.op === "AMBIG") {
      results.ambigs.push({ name: evt.name, candidateHash: evt.candidate, span: evt.span });
      results.skipped++;
    }
  });
  return results;
}

/* ── Full entity state, for dossier building ── */

export async function getEntityFull(entityId) {
  const e = entities.get(entityId);
  if (!e) return null;
  return {
    ...e,
    stateHash: await computeStateHash(entityId),
    defs: defs.getFor(entityId),
    edges: edges.getFor(entityId),
    evals: evals.getFor(entityId),
    conflicts: defs.getConflicts(entityId),
    hypothesisHistory: hypotheses.getHistory("entity", entityId),
  };
}

/* ── Rebuild vectors from text (after a wipe or model change) ── */

export async function rebuildVectors() {
  db.exec({ sql: "DELETE FROM vec_clauses WHERE scope=?", bind: [scope] });
  db.exec({ sql: "DELETE FROM vec_centroids WHERE scope=?", bind: [scope] });
  db.exec({ sql: "DELETE FROM vec_hypotheses WHERE scope=?", bind: [scope] });
  for (const d of defs.getAll()) {
    if (d.span && d.span.length > 10) await vectors.addClause(d.id, d.entity_id, d.span);
  }
  for (const g of given.getAll()) {
    if (g.text && g.text.length > 20) await vectors.addGivenVector(g.id, g.text.slice(0, 500));
  }
  for (const e of entities.getAll()) {
    if (e.hypothesis) await vectors.updateHypothesisVec(e.id, e.hypothesis);
  }
}

/* ── Drop a chat's whole graph ── */

export function dropScope(chatId) {
  const s = chatId || scope;
  for (const t of ["given", "entities", "edges", "defs", "evals", "hypotheses", "mutations", "vec_clauses", "vec_centroids", "vec_hypotheses"]) {
    db.exec({ sql: `DELETE FROM ${t} WHERE scope=?`, bind: [s] });
  }
}

export function stats() {
  return {
    entities: db.selectObject("SELECT COUNT(*) AS n FROM entities WHERE scope=?", [scope])?.n || 0,
    defs: db.selectObject("SELECT COUNT(*) AS n FROM defs WHERE scope=? AND retired=0", [scope])?.n || 0,
    edges: db.selectObject("SELECT COUNT(*) AS n FROM edges WHERE scope=?", [scope])?.n || 0,
    evals: db.selectObject("SELECT COUNT(*) AS n FROM evals WHERE scope=?", [scope])?.n || 0,
    given: db.selectObject("SELECT COUNT(*) AS n FROM given WHERE scope=?", [scope])?.n || 0,
    pending: db.selectObject("SELECT COUNT(*) AS n FROM mutations WHERE scope=? AND status='pending'", [scope])?.n || 0,
  };
}
