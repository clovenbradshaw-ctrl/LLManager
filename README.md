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
npm run dev      # http://localhost:5173/llmanager/
npm run build    # outputs to dist/
npm run preview
```

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the site
and publishes it to GitHub Pages. One-time setup: in the repo's
**Settings → Pages**, set **Source** to **GitHub Actions**.

The Vite `base` is `/llmanager/` to match the repository name. If you rename the
repo, update `base` in `vite.config.js`.
