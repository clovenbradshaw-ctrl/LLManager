// graph.js — content graph: structured store + mechanical traversal.
// Prompt/response text is processed into nodes (entities) and edges
// (relations). Connection questions like "how is X linked to Y" are then
// answered by graph traversal instead of a fresh LLM prompt.

const STORE_KEY = "llm-content-graph";

export function emptyGraph() {
  return { nodes: {}, edges: [], sources: [] };
}

export function loadGraph() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return emptyGraph();
    const g = JSON.parse(raw);
    return { nodes: g.nodes || {}, edges: g.edges || [], sources: g.sources || [] };
  } catch {
    return emptyGraph();
  }
}

export function saveGraph(g) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(g)); } catch {}
}

export function normId(label) {
  return String(label).trim().toLowerCase().replace(/\s+/g, " ");
}

export function graphStats(g) {
  return {
    nodes: Object.keys(g.nodes).length,
    edges: g.edges.length,
    sources: g.sources.length,
  };
}

// Merge an extraction ({entities, relations}) into a new graph object.
export function mergeExtraction(g, extraction, source) {
  const next = {
    nodes: { ...g.nodes },
    edges: g.edges.map(e => ({ ...e, sources: [...e.sources] })),
    sources: [...g.sources],
  };
  const srcId = source?.id || `s${Date.now()}`;
  next.sources.push({ id: srcId, label: source?.label || "", ts: Date.now() });

  const ensureNode = (rawLabel, type) => {
    const label = String(rawLabel || "").trim();
    if (!label) return null;
    const id = normId(label);
    if (!next.nodes[id]) {
      next.nodes[id] = { id, label, type: type || "concept", mentions: [], firstSeen: Date.now() };
    }
    if (!next.nodes[id].mentions.includes(srcId)) next.nodes[id].mentions.push(srcId);
    return id;
  };

  let addedNodes = 0, addedEdges = 0;
  const before = Object.keys(next.nodes).length;

  for (const e of extraction.entities || []) {
    ensureNode(e.name || e.label, e.type);
  }
  addedNodes = Object.keys(next.nodes).length - before;

  for (const r of extraction.relations || []) {
    const from = ensureNode(r.from);
    const to = ensureNode(r.to);
    if (!from || !to || from === to) continue;
    const relation = String(r.relation || "related to").trim().slice(0, 60) || "related to";
    const existing = next.edges.find(x => x.from === from && x.to === to && x.relation === relation);
    if (existing) {
      existing.weight += 1;
      if (!existing.sources.includes(srcId)) existing.sources.push(srcId);
    } else {
      next.edges.push({ from, to, relation, weight: 1, sources: [srcId] });
      addedEdges++;
    }
  }
  return { graph: next, addedNodes, addedEdges };
}

// Undirected adjacency index: id -> [{ to, relation, dir, weight }].
export function buildAdjacency(g) {
  const adj = new Map();
  const add = (k, v) => { if (!adj.has(k)) adj.set(k, []); adj.get(k).push(v); };
  for (const e of g.edges) {
    add(e.from, { to: e.to, relation: e.relation, dir: "out", weight: e.weight });
    add(e.to, { to: e.from, relation: e.relation, dir: "in", weight: e.weight });
  }
  return adj;
}

export function neighbors(g, id) {
  return buildAdjacency(g).get(id) || [];
}

export function degree(g, id) {
  let d = 0;
  for (const e of g.edges) { if (e.from === id) d++; if (e.to === id) d++; }
  return d;
}

// BFS for all simple paths between two nodes, shortest first.
export function findPaths(g, fromId, toId, { maxDepth = 4, maxPaths = 12 } = {}) {
  if (!fromId || !toId || fromId === toId) return [];
  if (!g.nodes[fromId] || !g.nodes[toId]) return [];
  const adj = buildAdjacency(g);
  const results = [];
  const queue = [{ node: fromId, path: [{ node: fromId }], visited: new Set([fromId]) }];
  while (queue.length && results.length < maxPaths) {
    const { node, path, visited } = queue.shift();
    if (path.length - 1 >= maxDepth) continue;
    for (const edge of adj.get(node) || []) {
      if (visited.has(edge.to)) continue;
      const step = { node: edge.to, relation: edge.relation, dir: edge.dir };
      const newPath = [...path, step];
      if (edge.to === toId) {
        results.push(newPath);
        if (results.length >= maxPaths) break;
      } else {
        const nv = new Set(visited);
        nv.add(edge.to);
        queue.push({ node: edge.to, path: newPath, visited: nv });
      }
    }
  }
  return results.sort((a, b) => a.length - b.length);
}

export function pathToText(g, path) {
  let s = g.nodes[path[0].node]?.label || path[0].node;
  for (let i = 1; i < path.length; i++) {
    const step = path[i];
    const arrow = step.dir === "in" ? `<-[${step.relation}]-` : `-[${step.relation}]->`;
    s += ` ${arrow} ${g.nodes[step.node]?.label || step.node}`;
  }
  return s;
}

// Compact context block: the only text an LLM needs to answer a
// connection question, instead of the full conversation history.
export function pathsToContext(g, paths) {
  return paths.map((p, i) => `${i + 1}. ${pathToText(g, p)}`).join("\n");
}

export const EXTRACTION_PROMPT = `Extract a knowledge graph from the text below.
Return ONLY valid JSON, no prose or markdown, in this exact shape:
{"entities":[{"name":"...","type":"person|org|concept|place|thing"}],
 "relations":[{"from":"...","to":"...","relation":"short verb phrase"}]}
Use entity names exactly as they appear in the text. Every "from" and "to"
must also appear in entities. Keep each relation to 1-4 words.

Text:
"""
{TEXT}
"""`;

// Tolerant parse: small models often wrap JSON in prose or code fences.
export function parseExtraction(raw) {
  if (!raw) return null;
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) return null;
  try {
    const obj = JSON.parse(raw.slice(start, end + 1));
    return {
      entities: Array.isArray(obj.entities) ? obj.entities : [],
      relations: Array.isArray(obj.relations) ? obj.relations : [],
    };
  } catch {
    return null;
  }
}
