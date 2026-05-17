# LLM Manager

A browser dashboard for running local LLMs — check hardware, browse a model
catalog, run prompts, and grab copy-paste API snippets.

## Providers

Pick a runtime with the **Ollama server / In-browser** toggle in the header:

- **Ollama server** — talks to a local [Ollama](https://ollama.com) instance
  (`http://localhost:11434`). Manage, pull, load, and chat with installed models.
- **In-browser** — runs models fully on-device via [WebLLM](https://github.com/mlc-ai/web-llm)
  and WebGPU. No server, no install: a model is downloaded once, cached by the
  browser, and executed locally on your GPU. Requires a WebGPU-capable browser
  (recent Chrome or Edge).

## Live site

Hosted on GitHub Pages at `https://<owner>.github.io/llmanager/`.

Nothing is sent to a remote server. The Ollama provider talks to Ollama on your
own machine; the in-browser provider runs entirely in the page. Chrome, Edge,
and Firefox allow an HTTPS page to reach `http://localhost`. Safari blocks it —
run the app locally there instead.

## Local development

```bash
npm install
npm run dev      # http://localhost:5173/
npm run build    # outputs the static site to docs/
npm run preview
```

## Deployment

The site is published with GitHub Pages' **Deploy from a branch** mode.

One-time setup: in the repo's **Settings → Pages**, set **Source** to
**Deploy from a branch**, then choose branch **`main`** and folder **`/docs`**.

To publish a change: run `npm run build` (which regenerates `docs/`), then
commit and push the updated `docs/` folder to `main`.

The Vite `base` is `"./"` so assets load with relative paths regardless of the
repository name or URL casing.
