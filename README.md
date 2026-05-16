# LLM Manager

A browser dashboard for a local [Ollama](https://ollama.com) instance — a
ChatGPT-style chat with saved conversations, hardware checks, a model catalog,
and copy-paste API snippets.

## Live site

Hosted on GitHub Pages at `https://<owner>.github.io/llmanager/`.

The app talks to Ollama on your own machine (`http://localhost:11434`); nothing
is sent to a server. Chrome, Edge, and Firefox allow an HTTPS page to reach
`http://localhost`. Safari blocks it — run the app locally there instead.

## Connecting to Ollama

Install the official Ollama app from [ollama.com/download](https://ollama.com/download).
It runs the server as a background service automatically (menu-bar icon on
macOS, system tray on Windows) — no terminal needed.

Ollama only answers browser requests whose origin is in `OLLAMA_ORIGINS`, which
defaults to localhost. A hosted page (GitHub Pages) is blocked until you add it.
The desktop app ignores shell variables, so set it at the OS level and then
quit/reopen Ollama:

```bash
# macOS
launchctl setenv OLLAMA_ORIGINS "*"

# Windows
setx OLLAMA_ORIGINS "*"

# Linux (systemd): systemctl edit ollama.service
#   Environment="OLLAMA_ORIGINS=*"
```

If you start Ollama from a terminal instead, pass it inline:
`OLLAMA_ORIGINS="*" ollama serve`. Note only one process can own port 11434 —
if the desktop app is running, the terminal server won't bind. The **Status**
tab detects this case and shows the exact fix.

## Matrix login & remote LLM bridge

The app opens to a **Matrix sign-in gate**. Sign in with any Matrix account
(homeserver discovery via `.well-known`, defaults to `matrix.org`), or skip it
to use the local Ollama dashboard only — the choice is remembered.

Once signed in, the **Matrix** tab is a lightweight chat client: it lists your
joined rooms and lets you read and send messages. The same app works on every
device — on your phone it is just a Matrix client.

On the device running Ollama, open a room and press **Start LLM bridge**. From
then on, every message that arrives in that room is routed to your local model
and the answer is posted back as a Matrix notice. So you can chat from your
phone (or any Matrix client) and have the device with the LLM answer for you.

Bridge replies are sent as `m.notice`, so the host never answers its own
output — no feedback loop.

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
