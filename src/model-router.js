/* Model Router — v3.

   Picks the best model per task across two runtimes:
     - Ollama (native, multi-model, faster) — a different model per task
     - WebGPU/WebLLM (zero setup) — one model holds for everything

   Tasks: ingest, extract, read, hypothesis, mutate, write. The router
   probes what is installed and assigns the first preferred model that is
   available, falling back to whatever exists. */

import { getBrowserEngine } from "./webllm.js";

/* ── Preferred models by task, in priority order ──

   qwen3:30b-a3b leads every task: a 30B MoE with only 3.3B active per token,
   so it is fast enough for chat and strong enough for extraction. One model
   for everything means no swaps and no GPU contention between tasks. */
const TASK_PREFERENCES = {
  ingest: ["qwen3:30b-a3b", "qwen3.5:9b", "qwen3:8b", "qwen2.5:7b", "mistral", "llama3.1:8b", "gemma2:9b"],
  extract: ["qwen3:30b-a3b", "qwen3.5:9b", "qwen3:8b", "qwen2.5:7b", "mistral", "llama3.1:8b"],
  read: ["qwen3:30b-a3b", "qwen3.5:4b", "qwen3:4b", "gemma2:2b", "qwen3:8b", "phi3:mini", "llama3.2:3b"],
  hypothesis: ["qwen3:30b-a3b", "qwen3.5:4b", "qwen3:4b", "gemma2:2b", "qwen3:8b"],
  mutate: ["qwen3:30b-a3b", "qwen3:8b", "qwen3.5:9b", "deepseek-r1:8b", "qwen2.5:7b", "llama3.1:8b"],
  write: ["qwen3:30b-a3b", "qwen3:8b", "qwen3.5:9b", "llama3.1:8b", "gemma2:9b", "mistral"],
};

/* ── WebGPU (MLC) equivalents — one model for all tasks ── */
const WEBGPU_MODELS = {
  universal: "Qwen3.5-9B-q4f16_1-MLC",
  fallbacks: [
    "Qwen3-8B-q4f16_1-MLC", "Qwen3.5-4B-q4f16_1-MLC", "Qwen3-4B-q4f16_1-MLC",
    "Llama-3.1-8B-Instruct-q4f16_1-MLC", "gemma-2-9b-it-q4f16_1-MLC",
    "Qwen2.5-7B-Instruct-q4f16_1-MLC", "gemma-2-2b-it-q4f16_1-MLC",
  ],
};

/* ── Runtime detection ── */

export async function probeOllama(baseUrl = "http://localhost:11434") {
  try {
    const vRes = await fetch(`${baseUrl}/api/version`, { signal: AbortSignal.timeout(3000) });
    if (!vRes.ok) return { available: false };
    const version = (await vRes.json()).version;
    const tRes = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(5000) });
    const models = tRes.ok ? (await tRes.json()).models || [] : [];
    const pRes = await fetch(`${baseUrl}/api/ps`, { signal: AbortSignal.timeout(3000) });
    const loaded = pRes.ok ? (await pRes.json()).models || [] : [];
    return {
      available: true, version, baseUrl,
      models: models.map(m => m.name),
      loaded: loaded.map(m => m.name),
    };
  } catch {
    return { available: false };
  }
}

export async function probeWebGPU() {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    return { available: false, reason: "WebGPU not supported" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { available: false, reason: "No GPU adapter" };
    return { available: true };
  } catch {
    return { available: false, reason: "WebGPU error" };
  }
}

/* ── Model selection ── */

export function selectModel(task, ollama, webgpu, webgpuLoaded = null) {
  if (ollama.available) {
    const prefs = TASK_PREFERENCES[task] || TASK_PREFERENCES.read;
    for (const preferred of prefs) {
      const match = ollama.models.find(m =>
        m === preferred || m.startsWith(preferred.split(":")[0] + ":"));
      if (match) return { runtime: "ollama", model: match, baseUrl: ollama.baseUrl };
    }
    if (ollama.models.length > 0) {
      return { runtime: "ollama", model: ollama.models[0], baseUrl: ollama.baseUrl };
    }
  }
  if (webgpu.available) {
    if (webgpuLoaded) return { runtime: "webgpu", model: webgpuLoaded };
    return { runtime: "webgpu", model: WEBGPU_MODELS.universal };
  }
  return null;
}

export function selectAllModels(ollama, webgpu, webgpuLoaded = null) {
  const assignments = {};
  if (ollama.available) {
    for (const task of Object.keys(TASK_PREFERENCES)) {
      assignments[task] = selectModel(task, ollama, webgpu, webgpuLoaded);
    }
  } else if (webgpu.available) {
    const model = webgpuLoaded || WEBGPU_MODELS.universal;
    for (const task of Object.keys(TASK_PREFERENCES)) {
      assignments[task] = { runtime: "webgpu", model };
    }
  }
  return assignments;
}

/* ── Unified call interface ── */

export async function callModel(assignment, systemPrompt, userPrompt, options = {}) {
  if (!assignment) throw new Error("No model available");
  return assignment.runtime === "ollama"
    ? callOllama(assignment, systemPrompt, userPrompt, options)
    : callWebGPU(assignment, systemPrompt, userPrompt, options);
}

const parseJson = (text) => {
  try { return JSON.parse(String(text).replace(/```json|```/g, "").trim()); }
  catch { return null; }
};

async function callOllama(assignment, systemPrompt, userPrompt, options) {
  const body = {
    model: assignment.model,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    stream: false,
    options: {
      temperature: options.temperature ?? 0.7,
      num_ctx: options.numCtx ?? 4096,
    },
  };
  if (options.jsonSchema) {
    body.format = options.jsonSchema;
    body.options.temperature = 0;
  }
  const res = await fetch(`${assignment.baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
  const data = await res.json();
  const text = data.message?.content || "";
  if (options.jsonSchema) return parseJson(text) ?? (options.jsonSchema.type === "array" ? [] : null);
  return text;
}

async function callWebGPU(assignment, systemPrompt, userPrompt, options) {
  const engine = await getBrowserEngine(assignment.model);
  const reply = await engine.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    temperature: options.jsonSchema ? 0 : (options.temperature ?? 0.7),
    max_tokens: options.maxTokens ?? 2048,
    ...(options.jsonSchema ? { response_format: { type: "json_object" } } : {}),
  });
  const text = reply.choices?.[0]?.message?.content || "";
  if (options.jsonSchema) return parseJson(text) ?? (options.jsonSchema.type === "array" ? [] : null);
  return text;
}

/* ── Structured-output schemas ── */

export const EVENT_SCHEMA = {
  type: "array",
  items: {
    type: "object",
    properties: {
      op: { type: "string", enum: ["INS", "CON", "DEF", "EVA", "AMBIG"] },
      entity: { type: "string" },
      terrain: { type: "string" },
      from: { type: "string" },
      to: { type: "string" },
      type: { type: "string" },
      field: { type: "string" },
      value: { type: "string" },
      span: { type: "string" },
      claim: { type: "string" },
      status: { type: "string", enum: ["holds", "fails", "contested"] },
      source: { type: "string" },
      name: { type: "string" },
      candidate: { type: "string" },
    },
    required: ["op"],
  },
};

export const MUTATE_SCHEMA = {
  type: "object",
  properties: {
    action: { type: "string", enum: ["FORK", "MERGE", "CORRECT", "RECLASSIFY", "NONE"] },
    source: { type: "string" },
    new_canonical: { type: "string" },
    keep: { type: "string" },
    absorb: { type: "string" },
    entity: { type: "string" },
    field: { type: "string" },
    old_value: { type: "string" },
    new_value: { type: "string" },
    old_terrain: { type: "string" },
    new_terrain: { type: "string" },
    reason: { type: "string" },
    new_aliases: { type: "array", items: { type: "string" } },
    reassign: { type: "array", items: { type: "object" } },
  },
  required: ["action", "reason"],
};

export const OUTLINE_SCHEMA = {
  type: "array",
  items: {
    type: "object",
    properties: {
      section: { type: "integer" },
      topic: { type: "string" },
      entities: { type: "array", items: { type: "string" } },
      move: { type: "string", enum: ["INS", "CON", "DEF", "EVA"] },
    },
    required: ["section", "topic", "entities", "move"],
  },
};

/* ── Task-specific callers ── */

export const callRead = (a, sys, user) =>
  callModel(a.read, sys, user, { temperature: 0.7 });

export const callExtract = (a, sys, user) =>
  callModel(a.extract, sys, user, { jsonSchema: EVENT_SCHEMA, temperature: 0 });

export const callIngest = (a, sys, user) =>
  callModel(a.ingest, sys, user, { jsonSchema: EVENT_SCHEMA, temperature: 0 });

export const callMutate = (a, sys, user) =>
  callModel(a.mutate, sys, user, { jsonSchema: MUTATE_SCHEMA, temperature: 0.2 });

export const callHypothesis = (a, sys, user) =>
  callModel(a.hypothesis, sys, user, { temperature: 0.4, maxTokens: 100 });

export const callWrite = (a, sys, user) =>
  callModel(a.write, sys, user, { temperature: 0.6, maxTokens: 1000 });

export const callOutline = (a, sys, user) =>
  callModel(a.write, sys, user, { jsonSchema: OUTLINE_SCHEMA, temperature: 0.3 });

/* ── Initialisation ── */

export async function initRouter(ollamaUrl = "http://localhost:11434") {
  const ollama = await probeOllama(ollamaUrl);
  const webgpu = await probeWebGPU();
  const assignments = selectAllModels(ollama, webgpu);
  return {
    ollama: ollama.available
      ? { up: true, version: ollama.version, models: ollama.models, loaded: ollama.loaded }
      : { up: false },
    webgpu: webgpu.available ? { up: true } : { up: false, reason: webgpu.reason },
    assignments,
    runtime: ollama.available ? "ollama" : webgpu.available ? "webgpu" : "none",
  };
}

export { TASK_PREFERENCES, WEBGPU_MODELS };
