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

/* Quick count of what a document's graph holds, for UI badges. */
export const docStats = (doc) => ({
  entities: doc?.graph ? Object.keys(doc.graph.entities || {}).length : 0,
  edges: doc?.graph ? Object.keys(doc.graph.edges || {}).length : 0,
  defs: doc?.graph ? Object.values(doc.graph.defs || {}).filter(d => !d.retired).length : 0,
});
