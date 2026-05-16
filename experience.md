# Adding Memory to a Chat App

## What you have

A browser chat app. Text input, model selector, send button. Each message goes to Ollama at `localhost:11434`, the response comes back, both render in the UI. No memory between sessions. Limited memory within a session (conversation history in the prompt, growing until it fills the context window).

## What you add

Seven things. Five are mechanical (zero token cost). Two involve the model. Matrix provides the persistence layer. Here is what goes where and how to wire it.

---

## The prompt, before and after

### Before (standard chat app):

```js
const messages = [
  { role: "system", content: systemPrompt },       // fixed
  ...conversationHistory,                           // GROWS every turn
  { role: "user", content: currentMessage }         // current
];
```

Problem: `conversationHistory` grows. At turn 20 it's 4,000 tokens. At turn 40 it's full.

### After (memory-backed):

```js
const messages = [
  { role: "system", content: systemPrompt },        // fixed, ~400 tokens
  { role: "system", content: dossier },              // projected, ~200-300 tokens
  { role: "system", content: positionMarker },       // cursor, ~50-80 tokens
  { role: "user", content: currentMessage }          // current
];
```

Fixed size. Same at turn 1 and turn 10,000. The `dossier` is mechanically assembled from the graph. The `positionMarker` tracks only the last turn's entities and topic. No conversation history in the prompt at all.

---

## The five mechanical steps (add these to your message handler)

### 1. The Gate

Before anything else, decide if this turn is knowledge-bearing.

```js
import nlp from 'compromise';

function isKnowledgeBearing(message) {
  const doc = nlp(message);
  const hasEntities = doc.people().length > 0
    || doc.places().length > 0
    || doc.organizations().length > 0
    || doc.topics().length > 0;
  const hasDates = doc.dates().length > 0;
  const hasNumbers = doc.numbers().length > 0;
  const isQuestion = message.includes('?');
  const isSubstantive = message.split(/\s+/).length > 4;

  return hasEntities || hasDates || isQuestion || (isSubstantive && hasNumbers);
}
```

If `false`, respond conversationally. Skip everything below. Write nothing to the graph.

### 2. The Signal

Run three branches in parallel. All mechanical.

```js
async function signal(message) {
  const [nerResult, embedding, keywords] = await Promise.all([
    // Branch A: NER (compromise.js, ~5ms)
    extractNER(message),
    // Branch B: Embed (transformers.js, ~50ms)
    embedMessage(message),
    // Branch C: Keywords (mechanical, ~1ms)
    extractKeywords(message),
  ]);
  return { ner: nerResult, embedding, keywords };
}

function extractNER(message) {
  const doc = nlp(message);
  return {
    people: doc.people().out('array'),
    places: doc.places().out('array'),
    orgs: doc.organizations().out('array'),
    dates: doc.dates().out('array'),
    topics: doc.topics().out('array'),
  };
}

function extractKeywords(message) {
  const stops = new Set(['what','how','why','when','where','who','is','was',
    'are','the','a','an','in','on','at','to','for','of','and','or','but',
    'did','do','does','has','have','had','been','be','this','that','it',
    'with','from','about','into','not','no','yes','can','could','would',
    'should','will','just','also','very','much','more','some','any','all']);
  return message.toLowerCase().replace(/[?!.,;:'"]/g, '')
    .split(/\s+/)
    .filter(w => w.length > 2 && !stops.has(w));
}
```

### 3. The Reach

Search the graph using Signal output. Returns ranked entity list.

```js
async function reach(signal, graph) {
  const candidates = new Map(); // entityId → score

  // Type-scoped NER match
  for (const name of [...signal.ner.people, ...signal.ner.places,
                       ...signal.ner.orgs, ...signal.ner.topics]) {
    const matches = graph.searchEntities(name.toLowerCase());
    for (const entity of matches) {
      candidates.set(entity.id, (candidates.get(entity.id) || 0) + 3);
    }
  }

  // Keyword substring match
  for (const kw of signal.keywords) {
    const matches = graph.searchByKeyword(kw);
    for (const entity of matches) {
      candidates.set(entity.id, (candidates.get(entity.id) || 0) + 1);
    }
  }

  // Embedding similarity (if clause-pool exists)
  if (signal.embedding) {
    const similar = graph.findSimilar(signal.embedding, 5);
    for (const { entity, score } of similar) {
      candidates.set(entity.id, (candidates.get(entity.id) || 0) + score * 2);
    }
  }

  // 2-hop expansion
  const direct = [...candidates.keys()];
  for (const id of direct) {
    const neighbors = graph.getNeighbors(id, 2);
    for (const neighbor of neighbors) {
      if (!candidates.has(neighbor.id)) {
        candidates.set(neighbor.id, 1);
      }
    }
  }

  // Rank and take top N
  return [...candidates.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([id]) => graph.getEntity(id));
}
```

### 4. The Dossier

Format the top entities into a compact context block.

```js
function buildDossier(entities, graph) {
  return '[CONTEXT]\n' + entities.map(e => {
    const edges = graph.getEdges(e.id);
    const defs = graph.getDefs(e.id);
    const edgeStr = edges.map(edge =>
      `  ${edge.op} → ${edge.target} ("${edge.type}")${edge.source ? ' src: ' + edge.source : ''}`
    ).join('\n');
    const defStr = defs.map(d =>
      `  DEF ${d.field}: ${d.value}${d.source ? ' src: ' + d.source : ''}`
    ).join('\n');
    return `${e.canonical} | ${e.terrain}/${e.subtype || ''} | ${edges.length} edges\n`
      + `  hyp: ${e.hypothesis || 'none'}\n`
      + edgeStr + '\n' + defStr;
  }).join('\n\n') + '\n[/CONTEXT]';
}
```

### 5. The Position Marker

One-turn memory. Overwrites every turn. Never grows.

```js
function buildPosition(lastTurn) {
  if (!lastTurn) return '';
  return `[POSITION]
Last entities: ${lastTurn.entities.join(', ')}
Topic: ${lastTurn.topic}
Last user message: "${lastTurn.userMessage}"
[/POSITION]`;
}

// Update after each turn:
lastTurn = {
  entities: dossierEntities.map(e => e.canonical),
  topic: extractTopic(currentMessage), // one phrase, mechanical
  userMessage: currentMessage.slice(0, 100),
};
```

---

## The two model steps

### 6. The Read

The only step the user waits for. Pure conversation. No structured output.

```js
const systemPrompt = `You answer questions using only the provided [CONTEXT].
If the context doesn't contain enough information, say so.
Do not invent information not present in the context.
Keep responses concise unless asked for detail.`;

async function read(systemPrompt, dossier, position, userMessage, model) {
  const res = await fetch('http://localhost:11434/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: systemPrompt + '\n\n' + dossier + '\n\n' + position },
        { role: 'user', content: userMessage },
      ],
      temperature: 0.7,
      stream: false,
    }),
  });
  const data = await res.json();
  return data.choices[0].message.content;
}
```

### 7. The Extract (background, after response is delivered)

Fires only on knowledge-bearing turns. Invisible to user.

```js
async function extract(userMessage, llmResponse, model) {
  const res = await fetch('http://localhost:11434/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: `Extract new knowledge from this exchange as JSON.
Return an array of events. Each event has:
  op: "INS" | "CON" | "DEF" | "EVA"
  entity: string (for INS, DEF, EVA)
  terrain: string (for INS only: Entity, Network, Paradigm, etc)
  subtype: string (for INS: person, organization, place, event, etc)
  from/to/type: strings (for CON)
  field/value: strings (for DEF)
  claim/status: strings (for EVA)
Return [] if nothing new. Return ONLY valid JSON, no markdown.` },
        { role: 'user', content: `User said: "${userMessage}"\n\nAssistant said: "${llmResponse}"` },
      ],
      temperature: 0,
      max_tokens: 500,
      stream: false,
    }),
  });
  const data = await res.json();
  const text = data.choices[0].message.content;
  try {
    return JSON.parse(text.replace(/```json|```/g, '').trim());
  } catch {
    return []; // fail silently, Given-Log entry still preserved
  }
}
```

---

## The Audit (mechanical, parallel with Extract)

```js
function audit(response, dossier, newEvents, graph) {
  const issues = [];

  // Provenance check: every new DEF must have a source
  for (const evt of newEvents) {
    if (evt.op === 'DEF' && !evt.source) {
      evt.source = 'user:conversation'; // default attribution
    }
  }

  // Conflict check: new DEFs vs existing DEFs
  for (const evt of newEvents) {
    if (evt.op === 'DEF') {
      const existing = graph.getDef(evt.entity, evt.field);
      if (existing && existing.value !== evt.value) {
        issues.push({
          type: 'conflict',
          entity: evt.entity,
          field: evt.field,
          existing: existing.value,
          existingSource: existing.source,
          incoming: evt.value,
          incomingSource: evt.source,
        });
      }
    }
  }

  // Hallucination check: proper nouns in response not in dossier
  const doc = nlp(response);
  const mentioned = [
    ...doc.people().out('array'),
    ...doc.places().out('array'),
    ...doc.organizations().out('array'),
  ];
  for (const name of mentioned) {
    if (!dossier.toLowerCase().includes(name.toLowerCase())) {
      issues.push({ type: 'hallucination', entity: name });
    }
  }

  return issues;
}
```

---

## Matrix storage

Matrix gives you three layers of persistence. Each maps to a component of the Experience Engine.

### Timeline events → Given-Log

Regular messages are already timeline events. They are append-only, timestamped, attributed to a sender. This IS a Given-Log. You do not need to build one. Matrix already has one.

Add custom event types for processed observations:

```js
// When the user sends a knowledge-bearing message
await matrixClient.sendEvent(roomId, 'eo.given', {
  agent: userId,
  mode: 'conversation',     // vs 'document', 'observation'
  text: userMessage,         // raw, verbatim
  ner: signal.ner,           // NER output, for future re-processing
  session: sessionId,
  turn: turnNumber,
});

// When ingesting a document clause
await matrixClient.sendEvent(roomId, 'eo.given', {
  agent: 'system:walker',
  mode: 'document',
  text: clauseText,
  source: { title: 'NYT article', url: '...', span: [charStart, charEnd] },
  session: ingestSessionId,
});
```

These are timeline events. Append-only. Permanent. Matrix never deletes timeline events (only redacts them, preserving the tombstone). This is Rule 3 enforced by the protocol.

### State events → Structure-Lattice (live index)

State events are key-value pairs per room. They have a `type` and a `state_key`. Setting the same type+state_key overwrites the previous value. This is perfect for the live entity index: the current state of each entity, its terrain, its connections.

```js
// Entity (state_key = entity id)
await matrixClient.sendStateEvent(roomId, 'eo.entity', entityId, {
  canonical: 'Boeing',
  terrain: 'Entity',
  subtype: 'corporation',
  hypothesis: 'Manufacturer of jets included in vague purchase agreement',
  created: timestamp,
  lastUpdated: timestamp,
});

// Edge (state_key = deterministic edge id)
const edgeId = `${fromId}::${type}::${toId}`;
await matrixClient.sendStateEvent(roomId, 'eo.edge', edgeId, {
  from: fromId,
  to: toId,
  type: 'subject_of',
  op: 'CON',
  source: 'eo.given:$eventId',  // provenance: links to Given-Log entry
  created: timestamp,
});

// DEF (state_key = entity::field)
const defKey = `${entityId}::${field}`;
await matrixClient.sendStateEvent(roomId, 'eo.def', defKey, {
  entity: entityId,
  field: field,
  value: value,
  source: 'eo.given:$eventId',  // provenance chain
  supersedes: previousDefEventId || null,  // σ function
  created: timestamp,
});
```

State events are queryable without paginating the entire timeline. `GET /state` returns all current state. This means the Reach can query the live entity index without reading the full Given-Log. The state is a materialized view of the log.

When a DEF is superseded, the state event is overwritten with the new value. But the old state event still exists in the timeline (state events are also timeline events). So you can reconstruct any previous state by replaying the timeline up to a point. This is the Horizon's `as-of` capability for free.

### State events → Meant-Graph (interpretations)

Interpretations are also state events, with a different type prefix to distinguish them from the structure index:

```js
// Interpretation (state_key = interpretation id)
await matrixClient.sendStateEvent(roomId, 'eo.meant', interpretationId, {
  entity: entityId,
  content: 'The vague agreement reflects diplomatic theater over substance',
  groundedIn: ['eo.given:$event1', 'eo.given:$event2'],  // π function
  supersedes: previousInterpretationId || null,            // σ function
  op: 'EVA',
  status: 'active',  // or 'superseded', 'retired'
  created: timestamp,
});
```

When an interpretation is superseded (σ), the state event is overwritten with `status: 'superseded'` and a reference to what replaced it. The old version persists in the timeline.

### Media store → Large data

Matrix rooms have a media upload endpoint. Files get an `mxc://` URI that is permanent and content-addressed.

```js
// Upload embedding vectors for a session's clause-pool
const poolData = JSON.stringify(clausePoolVectors); // could be megabytes
const blob = new Blob([poolData], { type: 'application/json' });
const mxcUri = await matrixClient.uploadContent(blob, {
  name: `clause-pool-${sessionId}.json`,
  type: 'application/json',
});

// Reference it from a state event
await matrixClient.sendStateEvent(roomId, 'eo.embeddings', sessionId, {
  mxcUri: mxcUri,
  model: 'all-MiniLM-L6-v2',
  dimensions: 384,
  clauseCount: vectors.length,
  created: timestamp,
});
```

Use the media store for:
- **Clause-pool embeddings** (384 floats × N clauses = potentially megabytes)
- **Full document text** (the raw source, before the walk)
- **Walk results** (the full event log from processing a document)
- **Snapshot exports** (periodic full graph dumps for backup/migration)

The state event references the `mxc://` URI. The app downloads and caches the media content in IndexedDB for fast local access. The media store is the cold tier; IndexedDB is the hot tier.

### Room topology

One room per knowledge domain, or one room per project:

```
!llm-memory-general:hyphae.social     — general conversation memory
!llm-memory-nashville:hyphae.social   — Nashville governance research
!llm-memory-eo:hyphae.social          — EO development
```

Each room contains:
- `m.room.message` events: the actual chat transcript
- `eo.given` events: processed observations (Given-Log)
- `eo.entity` state: live entity index (Structure-Lattice)
- `eo.edge` state: live connection index
- `eo.def` state: live claims with provenance
- `eo.meant` state: interpretations with groundedness chains
- `eo.embeddings` state: references to media-stored vectors

To query the graph, the app calls `GET /state` on the room. This returns all current `eo.entity`, `eo.edge`, `eo.def`, and `eo.meant` state events in one response. No pagination. No timeline scrolling. The live index is the state.

To reconstruct history, the app paginates the timeline. Every state change is also a timeline event. The Given-Log is the timeline filtered by `eo.given` type.

### Sync for multi-device

Matrix sync gives you this for free. Open the chat app on your phone. It syncs the room state. The graph is available on every device without custom sync logic. The state events ARE the graph index. Sync the state, you have the index.

E2EE works if the room is encrypted. The event content (entity names, claims, provenance) is encrypted at rest. The event types and state keys are not encrypted (Matrix limitation), so entity IDs are visible to the server but their content is not.

---

## The full message handler

```js
async function handleMessage(userMessage, {
  graph,           // local graph index (IndexedDB, hydrated from Matrix state)
  matrixClient,
  roomId,
  ollamaUrl,
  readModel,       // e.g. 'gemma2:2b' — fast, for the user
  extractModel,    // e.g. 'mistral' — better structured output, background
  lastTurn,        // position marker from previous turn
  sessionId,
}) {
  // 1. THE GATE
  if (!isKnowledgeBearing(userMessage)) {
    const response = await read(SYSTEM_PROMPT_CASUAL, '', '', userMessage, readModel);
    return { response, events: [], issues: [] };
  }

  // 2. THE SIGNAL (parallel)
  const sig = await signal(userMessage);

  // 3. THE REACH
  const entities = await reach(sig, graph);

  // 4. THE DOSSIER
  const dossier = buildDossier(entities, graph);

  // 5. THE POSITION
  const position = buildPosition(lastTurn);

  // 6. THE READ (user waits for this)
  const response = await read(SYSTEM_PROMPT, dossier, position, userMessage, readModel);

  // Show response to user immediately.
  // Everything below is background.

  // Write Given-Log entry to Matrix
  await matrixClient.sendEvent(roomId, 'eo.given', {
    agent: userId,
    mode: 'conversation',
    text: userMessage,
    ner: sig.ner,
    session: sessionId,
  });

  // 7a. THE EXTRACT (background)
  const events = await extract(userMessage, response, extractModel);

  // 7b. THE AUDIT (parallel with extract in practice, sequential here for clarity)
  const issues = audit(response, dossier, events, graph);

  // Write clean events to Matrix state + local graph
  const clean = events.filter(e =>
    !issues.some(i => i.type === 'conflict' && i.entity === e.entity && i.field === e.field)
  );

  for (const evt of clean) {
    if (evt.op === 'INS') {
      await matrixClient.sendStateEvent(roomId, 'eo.entity', evt.entity, {
        canonical: evt.entity,
        terrain: evt.terrain || 'Entity',
        subtype: evt.subtype || '',
        created: Date.now(),
      });
      graph.addEntity(evt);
    }
    if (evt.op === 'CON') {
      const edgeId = `${evt.from}::${evt.type}::${evt.to}`;
      await matrixClient.sendStateEvent(roomId, 'eo.edge', edgeId, {
        from: evt.from, to: evt.to, type: evt.type, op: 'CON',
        created: Date.now(),
      });
      graph.addEdge(evt);
    }
    if (evt.op === 'DEF') {
      const defKey = `${evt.entity}::${evt.field}`;
      await matrixClient.sendStateEvent(roomId, 'eo.def', defKey, {
        entity: evt.entity, field: evt.field, value: evt.value,
        source: evt.source || 'user:conversation',
        created: Date.now(),
      });
      graph.addDef(evt);
    }
  }

  // Update position marker for next turn
  const newLastTurn = {
    entities: entities.map(e => e.canonical),
    topic: sig.keywords.slice(0, 3).join(' '),
    userMessage: userMessage.slice(0, 100),
  };

  return { response, events: clean, issues, lastTurn: newLastTurn };
}
```

---

## What lives where

| Data | Hot tier (IndexedDB) | Cold tier (Matrix) | Size per entity |
|---|---|---|---|
| Entity index | Yes (full copy) | State events (`eo.entity`) | ~200 bytes |
| Edge index | Yes (full copy) | State events (`eo.edge`) | ~150 bytes |
| DEF claims | Yes (full copy) | State events (`eo.def`) | ~200 bytes |
| Interpretations | Yes (active only) | State events (`eo.meant`) | ~300 bytes |
| Given-Log | Recent session only | Timeline events (`eo.given`) | ~500 bytes |
| Chat transcript | Recent session only | Timeline events (`m.room.message`) | varies |
| Embeddings | Yes (full, from media) | Media store (mxc://) | ~1.5 KB per clause |
| Documents | No | Media store (mxc://) | varies |
| Walk results | No | Media store (mxc://) | varies |

On app startup: call `GET /state` on the room, hydrate IndexedDB with all `eo.entity`, `eo.edge`, `eo.def` state. Download embeddings from media store if not cached. The graph is ready. No timeline pagination needed for the live index.
