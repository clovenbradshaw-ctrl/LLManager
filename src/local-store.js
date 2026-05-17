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
  // Idempotent migrations for databases created before a column existed.
  for (const alter of [
    "ALTER TABLE entities ADD COLUMN status TEXT NOT NULL DEFAULT 'sig'",
    "ALTER TABLE defs ADD COLUMN drift REAL",
  ]) {
    try { db.exec(alter); } catch { /* column already exists */ }
  }
  return { db, persistent };
}

async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline("feature-extraction", EMBED_MODEL, { quantized: true });
  }
  return embedder;
}

/* Race a promise against a timeout so a stalled model load or inference
   surfaces as an error the caller can handle, never an unbounded hang. */
function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${label} timed out`)), ms)),
  ]);
}

/* Embedding cache — the model call is the expensive part of the first pass.
   The same passage embedded into a second chat (a document opted in from
   elsewhere) is a cache hit, so the embedding work happens once no matter
   how many chats reason over the document. Vectors are treated as
   immutable, so the cached Float32Array is shared by reference. */
const embedCache = new Map();
const EMBED_CACHE_MAX = 800;

export async function embed(text) {
  const key = String(text || "");
  const hit = embedCache.get(key);
  if (hit) { embedCache.delete(key); embedCache.set(key, hit); return hit; }
  const model = await withTimeout(getEmbedder(), 120000, "embedding model load");
  const out = await withTimeout(
    model(key, { pooling: "mean", normalize: true }), 30000, "embed");
  const vec = new Float32Array(out.data);
  embedCache.set(key, vec);
  if (embedCache.size > EMBED_CACHE_MAX) embedCache.delete(embedCache.keys().next().value);
  return vec;
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
    status      TEXT NOT NULL DEFAULT 'sig',  -- sig | walking | walked
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
    drift       REAL,
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

  -- Unwalked vectors: chat messages and unprocessed document passages.
  -- No graph structure yet — SIG-level only. Deleted once walked.
  CREATE TABLE IF NOT EXISTS vec_unwalked (
    scope       TEXT NOT NULL,
    id          TEXT NOT NULL,
    vector      BLOB NOT NULL,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (scope, id)
  );

  -- SIG-level mentions: one row per (entity, passage) the first-pass NER
  -- scan saw, with the passage context and the drift it caused. The raw
  -- material of a provisional dossier entry. Pruned once the entity walks.
  CREATE TABLE IF NOT EXISTS sig_mentions (
    scope       TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    document_id TEXT,
    passage_idx INTEGER,
    given_id    TEXT,
    context     TEXT,
    drift       REAL,
    created_at  INTEGER NOT NULL
  );

  -- Documents being walked, with reading progress for the [STATUS] block.
  CREATE TABLE IF NOT EXISTS documents (
    scope       TEXT NOT NULL,
    id          TEXT NOT NULL,
    title       TEXT,
    total       INTEGER NOT NULL DEFAULT 0,
    walked      INTEGER NOT NULL DEFAULT 0,
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
  CREATE INDEX IF NOT EXISTS idx_defs_drift    ON defs(scope, entity_id, drift);
  CREATE INDEX IF NOT EXISTS idx_edges_from    ON edges(scope, from_id);
  CREATE INDEX IF NOT EXISTS idx_edges_to      ON edges(scope, to_id);
  CREATE INDEX IF NOT EXISTS idx_given_session ON given(scope, session_id);
  CREATE INDEX IF NOT EXISTS idx_given_doc     ON given(scope, document_id);
  CREATE INDEX IF NOT EXISTS idx_hyp_target    ON hypotheses(scope, level, target_id);
  CREATE INDEX IF NOT EXISTS idx_vec_unwalked ON vec_unwalked(scope);
  CREATE INDEX IF NOT EXISTS idx_entities_terrain ON entities(scope, terrain);
  CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(scope, status);
  CREATE INDEX IF NOT EXISTS idx_sig_mentions ON sig_mentions(scope, entity_id);
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
  create(id, canonical, terrain, { hypothesis = null, aliases = [], forkedFrom = null, status = "sig" } = {}) {
    const now = Date.now();
    db.exec({
      sql: `INSERT OR REPLACE INTO entities (scope,id,canonical,terrain,hypothesis,aliases,forked_from,status,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)`,
      bind: [scope, id, canonical, terrain, hypothesis, JSON.stringify(aliases), forkedFrom, status, now, now],
    });
  },
  updateHypothesis(id, text) {
    db.exec({ sql: "UPDATE entities SET hypothesis=?, updated_at=? WHERE scope=? AND id=?", bind: [text, Date.now(), scope, id] });
  },
  /* Epistemic register: 'sig' (NER impression) → 'walking' (partial walk)
     → 'walked' (grounded, provenanced). Never downgrades. */
  setStatus(id, status) {
    const e = entities.get(id);
    if (!e) return;
    const rank = { sig: 0, walking: 1, walked: 2 };
    if ((rank[status] ?? 0) < (rank[e.status] ?? 0)) return;
    db.exec({ sql: "UPDATE entities SET status=?, updated_at=? WHERE scope=? AND id=?", bind: [status, Date.now(), scope, id] });
  },
  markWalked(ids) {
    for (const id of ids || []) {
      entities.setStatus(id, "walked");
      mentions.prune(id); // SIG scaffolding superseded by the grounded entity
    }
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
  write(id, entityId, field, value, { span = null, source = null, supersedes = null, retired = 0, drift = null } = {}) {
    db.exec({
      sql: `INSERT OR REPLACE INTO defs (scope,id,entity_id,field,value,span,source,supersedes,retired,drift,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
      bind: [scope, id, entityId, field, value, span, source, supersedes, retired ? 1 : 0, drift, Date.now()],
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

/* ── SIG mentions — the raw material of a provisional dossier entry ── */

export const mentions = {
  record(entityId, { documentId = null, passageIdx = null, givenId = null, context = null, drift = null } = {}) {
    db.exec({
      sql: `INSERT INTO sig_mentions (scope,entity_id,document_id,passage_idx,given_id,context,drift,created_at)
            VALUES (?,?,?,?,?,?,?,?)`,
      bind: [scope, entityId, documentId, passageIdx, givenId, context, drift, Date.now()],
    });
  },
  forEntity(entityId) {
    return db.selectObjects("SELECT * FROM sig_mentions WHERE scope=? AND entity_id=? ORDER BY created_at", [scope, entityId]);
  },
  prune(entityId) {
    db.exec({ sql: "DELETE FROM sig_mentions WHERE scope=? AND entity_id=?", bind: [scope, entityId] });
  },
};

/* ── Documents — reading progress, for the [STATUS] block ── */

export const documents = {
  register(id, title, total) {
    db.exec({
      sql: "INSERT OR REPLACE INTO documents (scope,id,title,total,walked,created_at) VALUES (?,?,?,?,?,?)",
      bind: [scope, id, title || "document", total || 0, 0, Date.now()],
    });
  },
  setWalked(id, n) {
    db.exec({ sql: "UPDATE documents SET walked=? WHERE scope=? AND id=?", bind: [n, scope, id] });
  },
  inProgress() {
    return db.selectObjects("SELECT * FROM documents WHERE scope=? AND walked < total ORDER BY created_at", [scope]);
  },
  get(id) {
    return db.selectObject("SELECT * FROM documents WHERE scope=? AND id=?", [scope, id]);
  },
  all() {
    return db.selectObjects("SELECT * FROM documents WHERE scope=? ORDER BY created_at", [scope]);
  },
};

/* ── Reading state, for the [STATUS] block ── */

export function statusCounts() {
  const rows = db.selectObjects("SELECT status, COUNT(*) AS n FROM entities WHERE scope=? GROUP BY status", [scope]);
  const get = (s) => rows.find(r => r.status === s)?.n || 0;
  return { sig: get("sig"), walking: get("walking"), walked: get("walked") };
}

export function recentShifts(sinceMs = 5 * 60 * 1000, limit = 3) {
  return db.selectObjects(`
    SELECT d.entity_id, e.canonical, d.drift, d.field FROM defs d
    JOIN entities e ON d.entity_id = e.id AND e.scope = d.scope
    WHERE d.scope=? AND d.drift > 0.15 AND d.created_at > ?
    ORDER BY d.created_at DESC LIMIT ?`,
    [scope, Date.now() - sinceMs, limit]);
}

/* ── Provenance — where a piece of knowledge came from ──

   Not every thought carries the same weight. A claim from an imported
   document is source material; a claim from the conversation is something
   said in chat; a hypothesis is the system's own inference. originOf
   classifies a source id; entityOrigin aggregates over an entity. */

export function originOf(sourceId) {
  if (!sourceId) return { class: "unknown", label: "unattributed" };
  const s = String(sourceId);
  if (s === "ner:firstpass") return { class: "scan", label: "an unread NER scan" };
  if (s.startsWith("mutate:")) return { class: "inference", label: "graph maintenance" };
  if (s.startsWith("inference")) return { class: "inference", label: "a system inference" };
  const g = given.get(s);
  if (!g) return { class: "unknown", label: "unattributed" };
  if (g.mode === "document" || g.agent === "system:walker" || g.agent === "system:firstpass") {
    const doc = g.document_id ? documents.get(g.document_id) : null;
    return { class: "document", label: doc?.title ? `document "${doc.title}"` : "an imported document" };
  }
  if (g.agent === "user") return { class: "conversation", label: "you said in this conversation" };
  if (g.agent && g.agent.startsWith("model:")) return { class: "conversation", label: "the assistant said in this conversation" };
  return { class: "unknown", label: "unattributed" };
}

export function entityOrigin(entityId) {
  const classes = new Set();
  const docs = new Set();
  const note = (src) => {
    const o = originOf(src);
    classes.add(o.class);
    if (o.class === "document") docs.add(o.label);
  };
  for (const d of defs.getFor(entityId)) note(d.source);
  for (const e of edges.getFor(entityId)) if (e.source) note(e.source);
  for (const m of mentions.forEntity(entityId)) note(m.given_id);
  return { classes: [...classes], documents: [...docs] };
}

/* ── Embeddings ── */

export const vectors = {
  /* Absorb a walked clause into the entity's centroid as a running mean,
     then discard the vector. Returns the drift — the cosine distance the
     centroid moved (0 = redundant, 0.3+ = rupture). SIG is ephemeral: once
     INS has the anchor and DEF the claims, the clause embedding is
     scaffolding the graph structure has replaced. The centroid is the
     compressed memory of every context the entity appeared in. */
  foldClause(entityId, vec) {
    if (!entityId) return 0;
    const existing = db.selectObject(
      "SELECT vector, clause_count FROM vec_centroids WHERE scope=? AND entity_id=?", [scope, entityId]);
    if (existing) {
      const old = toVec(existing.vector), n = existing.clause_count;
      const updated = new Float32Array(DIMS);
      for (let i = 0; i < DIMS; i++) updated[i] = (old[i] * n + vec[i]) / (n + 1);
      const drift = 1 - cosineSim(old, updated);
      db.exec({
        sql: "UPDATE vec_centroids SET vector=?, clause_count=?, updated_at=? WHERE scope=? AND entity_id=?",
        bind: [toBlob(updated), n + 1, Date.now(), scope, entityId],
      });
      return drift;
    }
    db.exec({
      sql: "INSERT INTO vec_centroids (scope,entity_id,vector,clause_count,updated_at) VALUES (?,?,?,?,?)",
      bind: [scope, entityId, toBlob(vec), 1, Date.now()],
    });
    return 1.0; // first clause — maximum novelty
  },
  async addClause(id, entityId, text) {
    return vectors.foldClause(entityId, await embed(text));
  },
  /* An unwalked vector — chat history or an unprocessed document passage.
     No graph structure yet; kept until the material is walked, then pruned. */
  writeUnwalked(id, vec) {
    db.exec({
      sql: "INSERT OR REPLACE INTO vec_unwalked (scope,id,vector,created_at) VALUES (?,?,?,?)",
      bind: [scope, id, toBlob(vec), Date.now()],
    });
  },
  async addUnwalked(id, text) {
    const vec = await embed(String(text).slice(0, 500));
    vectors.writeUnwalked(id, vec);
    return vec;
  },
  /* After a document walk completes, tear down its unwalked vectors — the
     centroids have absorbed the material. */
  pruneWalked(documentId) {
    db.exec({
      sql: "DELETE FROM vec_unwalked WHERE scope=? AND id IN (SELECT id FROM given WHERE scope=? AND document_id=?)",
      bind: [scope, scope, documentId],
    });
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
  /* Decoded vectors for the multi-signal retriever to score against. */
  dumpCentroids() {
    return db.selectObjects("SELECT entity_id, vector FROM vec_centroids WHERE scope=?", [scope])
      .map(r => ({ entityId: r.entity_id, vec: toVec(r.vector) }));
  },
  dumpHypotheses() {
    return db.selectObjects("SELECT entity_id, vector FROM vec_hypotheses WHERE scope=?", [scope])
      .map(r => ({ entityId: r.entity_id, vec: toVec(r.vector) }));
  },
  dumpUnwalked() {
    return db.selectObjects(`
      SELECT v.id, v.vector, g.text, g.passage_idx, g.document_id
      FROM vec_unwalked v JOIN given g ON v.id = g.id AND v.scope = g.scope
      WHERE v.scope=?`, [scope])
      .map(r => ({ id: r.id, vec: toVec(r.vector), text: r.text, passageIdx: r.passage_idx, documentId: r.document_id }));
  },
  /* The nearest walked (grounded) entity to a SIG entity's centroid —
     "this impression looks like something we already know." */
  nearestWalked(entityId, floor = 0.5) {
    const row = db.selectObject("SELECT vector FROM vec_centroids WHERE scope=? AND entity_id=?", [scope, entityId]);
    if (!row) return null;
    const v = toVec(row.vector);
    const walked = db.selectObjects(`
      SELECT c.entity_id, c.vector, e.canonical FROM vec_centroids c
      JOIN entities e ON c.entity_id = e.id AND e.scope = c.scope
      WHERE c.scope=? AND e.status='walked'`, [scope]);
    let best = null;
    for (const w of walked) {
      const s = cosineSim(v, toVec(w.vector));
      if (!best || s > best.sim) best = { entityId: w.entity_id, canonical: w.canonical, sim: s };
    }
    return best && best.sim >= floor ? best : null;
  },
  async findSimilarByHypothesis(queryText, topK = 5) {
    const qv = await embed(queryText);
    return db.selectObjects("SELECT entity_id, vector FROM vec_hypotheses WHERE scope=?", [scope])
      .map(r => ({ entityId: r.entity_id, similarity: cosineSim(qv, toVec(r.vector)) }))
      .sort((a, b) => b.similarity - a.similarity).slice(0, topK);
  },
  async findSimilarUnwalked(queryText, topK = 5) {
    const qv = await embed(queryText);
    return db.selectObjects("SELECT id, vector FROM vec_unwalked WHERE scope=?", [scope])
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
  /* ── Drift queries — drift accumulates on DEFs as the centroid moves ── */
  driftSince(entityId, since) {
    const row = db.selectObject(
      "SELECT SUM(drift) AS total, COUNT(*) AS n FROM defs WHERE scope=? AND entity_id=? AND drift IS NOT NULL AND created_at>?",
      [scope, entityId, since]);
    return { total: row?.total || 0, count: row?.n || 0 };
  },
  driftTrail(entityId) {
    return db.selectObjects(
      "SELECT field,value,drift,created_at FROM defs WHERE scope=? AND entity_id=? AND drift IS NOT NULL ORDER BY created_at",
      [scope, entityId]);
  },
  highDriftEntities(threshold = 0.15, since = 0) {
    return db.selectObjects(`
      SELECT entity_id, SUM(drift) AS total_drift, COUNT(*) AS clause_count
      FROM defs WHERE scope=? AND drift IS NOT NULL AND created_at>?
      GROUP BY entity_id HAVING total_drift>? ORDER BY total_drift DESC`,
      [scope, since, threshold]);
  },
  ruptures(threshold = 0.30) {
    return db.selectObjects(`
      SELECT d.*, e.canonical FROM defs d
      JOIN entities e ON d.entity_id=e.id AND e.scope=d.scope
      WHERE d.scope=? AND d.drift>? ORDER BY d.drift DESC`,
      [scope, threshold]);
  },
  stats() {
    const unwalked = db.selectObject("SELECT COUNT(*) AS n FROM vec_unwalked WHERE scope=?", [scope])?.n || 0;
    const centroids = db.selectObject("SELECT COUNT(*) AS n FROM vec_centroids WHERE scope=?", [scope])?.n || 0;
    const hyps = db.selectObject("SELECT COUNT(*) AS n FROM vec_hypotheses WHERE scope=?", [scope])?.n || 0;
    return { unwalked, centroids, hypotheses: hyps, bytesEstimate: (unwalked + centroids + hyps) * (DIMS * 4) };
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

/* Embed every DEF span in an event batch. Async — call before applyEvents
   so the apply itself has no awaits. */
export async function embedDefSpans(events) {
  const vecs = [];
  for (let i = 0; i < events.length; i++) {
    const e = events[i];
    if (e.op === "DEF" && e.span && e.span.length > 10) vecs[i] = await embed(e.span);
  }
  return vecs;
}

/* Apply events — fully synchronous. `spanVecs` (from embedDefSpans) carries
   the precomputed DEF-span vectors, so no await sits between setScope and
   the writes and the active scope cannot change mid-apply. */
export function applyEvents(events, sourceGivenId = null, spanVecs = []) {
  const results = { applied: 0, skipped: 0, ambigs: [], newEntities: [], ruptures: [], touched: [] };
  const touch = (id) => { if (id && !results.touched.includes(id)) results.touched.push(id); };
  events.forEach((evt, i) => {
    const now = Date.now();
    if (evt.op === "INS") {
      const existing = entities.search(evt.entity)
        .find(m => m.canonical.toLowerCase() === String(evt.entity).toLowerCase());
      if (existing) {
        // An impression the walk has now reached — upgrade sig → walking.
        if (existing.status === "sig") entities.setStatus(existing.id, "walking");
        evt._resolvedId = existing.id;
        touch(existing.id);
        results.applied++;
        return;
      }
      const id = mintEntityId(evt.entity);
      entities.create(id, evt.entity, evt.terrain || "Entity", { status: "walking" });
      evt._resolvedId = id;
      results.newEntities.push(id);
      touch(id);
      results.applied++;
    } else if (evt.op === "CON") {
      const from = resolveId(evt.from, events), to = resolveId(evt.to, events);
      if (from && to && from !== to) {
        edges.create(from, to, evt.type || "related to", sourceGivenId);
        touch(from); touch(to);
        results.applied++;
      } else results.skipped++;
    } else if (evt.op === "DEF") {
      const eid = resolveId(evt.entity, events);
      if (eid && evt.field && evt.value != null) {
        touch(eid);
        const prior = defs.get(eid, evt.field);
        const conflict = prior && prior.value !== String(evt.value) && evt.field !== "kind";
        if (prior && !conflict) defs.retire(prior.id);
        const defId = "d_" + mintHash(`${eid}::${evt.field}::${now}::${Math.random()}`);
        // Absorb the span into the centroid; the drift is the cosine
        // distance the centroid moved — stored on the DEF as a signal.
        const drift = spanVecs[i] ? vectors.foldClause(eid, spanVecs[i]) : null;
        defs.write(defId, eid, evt.field, evt.value, {
          span: evt.span || null, source: sourceGivenId || evt.source || null,
          supersedes: prior ? prior.id : null, drift,
        });
        if (drift !== null && drift > 0.30) {
          results.ruptures.push({ entityId: eid, defId, drift, field: evt.field });
        }
        results.applied++;
      } else results.skipped++;
    } else if (evt.op === "EVA") {
      const eid = resolveId(evt.entity, events);
      if (eid && evt.claim) {
        touch(eid);
        const evalId = "v_" + mintHash(`${eid}::${evt.claim}::${now}::${Math.random()}`);
        evals.write(evalId, eid, evt.claim, evt.status || "holds", {
          span: evt.span || null, source: sourceGivenId || evt.source || null,
        });
        results.applied++;
      } else results.skipped++;
    } else if (evt.op === "AMBIG") {
      // Reject multi-candidate AMBIGs — "candidate" must be a single hash,
      // not a pipe-joined dump of every entity in the register.
      if (evt.candidate && String(evt.candidate).includes("|")) {
        results.skipped++;
      } else {
        results.ambigs.push({ name: evt.name, candidateHash: evt.candidate, span: evt.span });
        results.skipped++;
      }
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
  db.exec({ sql: "DELETE FROM vec_unwalked WHERE scope=?", bind: [scope] });
  db.exec({ sql: "DELETE FROM vec_centroids WHERE scope=?", bind: [scope] });
  db.exec({ sql: "DELETE FROM vec_hypotheses WHERE scope=?", bind: [scope] });
  // Re-embed DEF spans entity by entity, folding into centroids; record the
  // drift back onto each DEF (approximate on rebuild — order is by created_at).
  for (const e of entities.getAll()) {
    for (const d of defs.getFor(e.id)) {
      if (d.span && d.span.length > 10) {
        const drift = await vectors.addClause(d.id, e.id, d.span);
        db.exec({ sql: "UPDATE defs SET drift=? WHERE scope=? AND id=?", bind: [drift, scope, d.id] });
      }
    }
    if (e.hypothesis) await vectors.updateHypothesisVec(e.id, e.hypothesis);
  }
  for (const g of given.getAll()) {
    if (!g.document_id && g.text && g.text.length > 20) await vectors.addUnwalked(g.id, g.text);
  }
}

/* ── Merge one entity into another ── */

export function mergeEntities(keep, absorb) {
  const k = entities.get(keep), a = entities.get(absorb);
  if (!k || !a || keep === absorb) return;
  db.exec({ sql: "UPDATE defs SET entity_id=? WHERE scope=? AND entity_id=?", bind: [keep, scope, absorb] });
  db.exec({ sql: "UPDATE evals SET entity_id=? WHERE scope=? AND entity_id=?", bind: [keep, scope, absorb] });
  edges.reTarget(absorb, keep);
  if (a.canonical) entities.addAlias(keep, a.canonical);
  for (const al of a.aliases) entities.addAlias(keep, al);
  entities.remove(absorb);
  db.exec({ sql: "DELETE FROM vec_centroids WHERE scope=? AND entity_id=?", bind: [scope, absorb] });
  db.exec({ sql: "DELETE FROM vec_hypotheses WHERE scope=? AND entity_id=?", bind: [scope, absorb] });
}

/* ── Copy one chat's whole graph into another scope (used by fork) ── */

const COPY_COLS = {
  given: "id,agent,mode,text,document_id,passage_idx,session_id,turn,dossier_hash,created_at",
  entities: "id,canonical,terrain,hypothesis,aliases,forked_from,created_at,updated_at",
  edges: "id,from_id,to_id,type,source,created_at",
  defs: "id,entity_id,field,value,span,source,supersedes,retired,drift,created_at",
  evals: "id,entity_id,claim,status,span,source,created_at",
  hypotheses: "level,target_id,text,revision,after_label,input_count,grounded_in,created_at",
  mutations: "action,detail,reason,triggered_by,status,created_at",
  vec_unwalked: "id,vector,created_at",
  vec_centroids: "entity_id,vector,clause_count,updated_at",
  vec_hypotheses: "entity_id,vector,updated_at",
};

export function copyScope(from, to) {
  for (const [t, cols] of Object.entries(COPY_COLS)) {
    db.exec({
      sql: `INSERT INTO ${t} (scope,${cols}) SELECT ?,${cols} FROM ${t} WHERE scope=?`,
      bind: [to, from],
    });
  }
}

/* ── Drop a chat's whole graph ── */

export function dropScope(chatId) {
  const s = chatId || scope;
  for (const t of ["given", "entities", "edges", "defs", "evals", "hypotheses", "mutations", "vec_unwalked", "vec_centroids", "vec_hypotheses"]) {
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
