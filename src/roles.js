/* Role-based model selection for the memory pipeline.

   Distilling text into the knowledge graph is background work, and it does not
   need the same model as the conversation. The two graph-walk roles differ in
   what they need:

     extract — the per-turn walk. Fires every knowledge-bearing turn over a
               short exchange (a few sentences). High frequency, short input;
               favour a small fast model.
     ingest  — the document walk. Runs once per document over longer passages;
               favour a stronger model for thorough extraction.

   Structural validity is handled separately: the walk calls run with Ollama's
   `format` parameter (constrained decoding), so even a small model cannot
   produce malformed JSON. That shifts model choice from "can it produce valid
   JSON?" to "does it identify the right entities?" — which is why a fast 4B is
   enough for extract.

   Each role resolves to a concrete model by: an explicit override the user
   picked, else the first installed model matching the role's preference list,
   else a caller-supplied fallback (normally the conversation model). */

export const ROLES = {
  extract: {
    id: "extract",
    label: "Extract",
    desc: "Per-turn graph walk — fires every knowledge-bearing turn. Favours speed.",
    prefer: [
      "qwen3:4b", "qwen3:1.7b", "qwen3:0.6b", "qwen3",
      "qwen2.5:3b", "llama3.2:3b", "phi3:mini", "gemma2:2b", "qwen2.5",
    ],
  },
  ingest: {
    id: "ingest",
    label: "Ingest",
    desc: "Document walk — longer passages, run once per document. Favours quality.",
    prefer: [
      "qwen3:8b", "gemma2:9b", "llama3.1:8b", "qwen2.5:7b",
      "deepseek-r1:8b", "phi3:medium", "mistral", "qwen3",
    ],
  },
};

const ROLES_KEY = "llmanager.roles.v1";

/* The persisted override map: { extract: <model|null>, ingest: <model|null> }.
   A null/absent entry means "resolve automatically from the preference list". */
export const loadRoleConfig = () => {
  try {
    const raw = localStorage.getItem(ROLES_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
};

export const saveRoleConfig = (config) => {
  try {
    localStorage.setItem(ROLES_KEY, JSON.stringify(config || {}));
  } catch {
    /* quota — ignore, same policy as the rest of the app */
  }
};

/* Resolve a role to an installed model name. `installedNames` is the roster of
   model names to choose from; `fallback` is used when nothing else matches.
   `config` lets a caller pass the live config instead of re-reading storage. */
export const resolveRoleModel = (role, installedNames, fallback, config) => {
  const names = (installedNames || []).filter(Boolean);
  if (!names.length) return fallback || null;

  const cfg = config || loadRoleConfig();
  const override = cfg[role];
  if (override && names.includes(override)) return override;

  const def = ROLES[role];
  if (def) {
    for (const pref of def.prefer) {
      const match = names.find((n) => n.toLowerCase().includes(pref.toLowerCase()));
      if (match) return match;
    }
  }

  if (fallback && names.includes(fallback)) return fallback;
  return fallback || names[0] || null;
};
