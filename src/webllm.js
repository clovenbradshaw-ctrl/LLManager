// In-browser LLM runtime backed by WebLLM (WebGPU). The library is loaded
// lazily so the Ollama path keeps working even if it is never imported.

let modPromise = null;
const loadModule = () => {
  if (!modPromise) modPromise = import("@mlc-ai/web-llm");
  return modPromise;
};

export const webGPUAvailable = () =>
  typeof navigator !== "undefined" && !!navigator.gpu;

export async function listBrowserModels() {
  const webllm = await loadModule();
  const list = webllm.prebuiltAppConfig?.model_list || [];
  const embedding = webllm.ModelType?.embedding;
  return list
    // Embedding models can't serve chat completions — selecting one fails with
    // an LLMChatPipeline error, so keep them out of the chat model list.
    .filter(m => {
      if (embedding != null && m.model_type === embedding) return false;
      return !/(^|[-_/])embed/i.test(m.model_id || "");
    })
    .map(m => ({
      id: m.model_id,
      vramMB: m.vram_required_MB || null,
      lowResource: !!m.low_resource_required,
    }))
    .sort((a, b) => (a.vramMB || 1e9) - (b.vramMB || 1e9));
}

let engine = null;
let engineModelId = null;
let enginePromise = null;

export const loadedBrowserModel = () => (engine ? engineModelId : null);

export async function getBrowserEngine(modelId, onProgress) {
  if (engine && engineModelId === modelId) return engine;
  if (enginePromise && engineModelId === modelId) return enginePromise;

  if (engine) {
    try { await engine.unload(); } catch { /* ignore */ }
    engine = null;
  }
  engineModelId = modelId;
  const webllm = await loadModule();
  enginePromise = webllm
    .CreateMLCEngine(modelId, {
      initProgressCallback: report => onProgress && onProgress(report),
    })
    .then(e => { engine = e; enginePromise = null; return e; })
    .catch(err => { enginePromise = null; engineModelId = null; throw err; });
  return enginePromise;
}

export async function unloadBrowserEngine() {
  if (engine) {
    try { await engine.unload(); } catch { /* ignore */ }
  }
  engine = null;
  engineModelId = null;
  enginePromise = null;
}
