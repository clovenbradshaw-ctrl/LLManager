# LLM Manager

A browser dashboard for a local [Ollama](https://ollama.com) instance — check
hardware, browse a model catalog, run prompts, and grab copy-paste API snippets.

## Live site

Hosted on GitHub Pages at `https://<owner>.github.io/llmanager/`.

The app talks to Ollama on your own machine (`http://localhost:11434`); nothing
is sent to a server. Chrome, Edge, and Firefox allow an HTTPS page to reach
`http://localhost`. Safari blocks it — run the app locally there instead.

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
