/* The Library — a global store of documents that have been "read" into
   knowledge graphs.

   A document is ingested once: its text is chunked, each chunk is mechanically
   scanned for entities and then read by the model into a graph of entities,
   connections and definitions. The resulting graph is kept here, in
   localStorage, independent of any single chat.

   A chat opts in to a document (per-chat, see `convo.docs`); when it does, the
   document's graph is merged into that chat's memory projection so the model
   can draw on it. The same document can be opted into any number of chats. */

const LIB_KEY = "llmanager.library.v3";

export const loadLibrary = () => {
  try {
    const raw = localStorage.getItem(LIB_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

export const saveLibrary = (docs) => {
  try {
    localStorage.setItem(LIB_KEY, JSON.stringify(docs));
  } catch {
    /* quota — ignore, same policy as the chat store */
  }
};

/* The ingest job — a single in-progress document read, persisted so it
   survives a reload or crash. Reading walks one passage at a time; after each
   passage the job (its trace, partial graph and next index) is saved here. On
   startup an unfinished job is resumed from where it stopped. Only one job
   exists at a time — a new read may not start while one is running. */
const INGEST_JOB_KEY = "llmanager.ingestJob.v1";

export const loadIngestJob = () => {
  try {
    const raw = localStorage.getItem(INGEST_JOB_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === "object" && Array.isArray(parsed.passages)
      ? parsed
      : null;
  } catch {
    return null;
  }
};

export const saveIngestJob = (job) => {
  try {
    localStorage.setItem(INGEST_JOB_KEY, JSON.stringify(job));
  } catch {
    /* quota — ignore; progress simply won't survive a crash for huge docs */
  }
};

export const clearIngestJob = () => {
  try {
    localStorage.removeItem(INGEST_JOB_KEY);
  } catch {
    /* ignore */
  }
};

/* Quick count of what a document's graph holds, for UI badges. */
export const docStats = (doc) => ({
  entities: doc?.graph ? Object.keys(doc.graph.entities || {}).length : 0,
  edges: doc?.graph ? Object.keys(doc.graph.edges || {}).length : 0,
  defs: doc?.graph ? Object.values(doc.graph.defs || {}).filter(d => !d.retired).length : 0,
});
