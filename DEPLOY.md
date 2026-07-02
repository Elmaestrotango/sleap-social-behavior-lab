# Deploying the course as one link

`serve.py` publishes the whole course — a landing page plus all six lessons in order — from a
**single URL**, using marimo's ASGI app server. It runs a **real Python kernel** (not
WebAssembly), so `numba` / `umap-learn` / `hdbscan` work — unlike a GitHub-Pages / WASM export,
which has no in-browser build for those and would break at lesson 03.

Each visitor gets their own **isolated kernel session**; they share the machine's CPU/RAM.

## Try it locally first

```bash
uv run python serve.py        # -> http://localhost:7860
```

Open the URL and click through lessons 1–6.

## What students see

Lessons are served in **run (app) mode**: markdown, plots, and every UI widget (sliders, the
labeling click-grid) work and re-run reactively, but the source code is read-only. That's ideal
for a class — nobody can corrupt a lesson, and sessions are isolated. If you want students to
*edit code*, have them run locally (`uv run marimo edit notebooks/01_load_sleap.py`) or use molab.

## Option A — Hugging Face Space (free, HF hosts it)

1. Create a new **Docker** Space at <https://huggingface.co/new-space> (SDK: Docker).
2. Push this repo to the Space's git remote (or point the Space at it). The included `Dockerfile`
   builds the environment and runs `serve.py` on port 7860 (HF's default).
3. Add this frontmatter to the top of the **Space's** `README.md` (only needed there):

   ```
   ---
   title: SLEAP Social Behavior Lab
   emoji: 🐭
   colorFrom: green
   colorTo: blue
   sdk: docker
   app_port: 7860
   ---
   ```

4. Wait for the build (it installs numba/umap/hdbscan — a few minutes the first time), then share
   the Space URL: `https://<user>-<space>.hf.space/`.

Notes: the free CPU tier is one shared 2-vCPU / 16 GB container (kernels are isolated but share
CPU) and sleeps after inactivity — open it once to wake it before class, or upgrade the hardware
for a large cohort.

## Option B — Self-host + tunnel (your box, one HTTPS link)

Run it on a lab machine / VM and expose a single URL:

```bash
uv run python serve.py                              # binds 0.0.0.0:7860
```

Then, in another shell, pick one tunnel:

```bash
cloudflared tunnel --url http://localhost:7860      # prints a https://…trycloudflare.com URL
# or, with Tailscale:
tailscale funnel 7860
```

Share the printed HTTPS URL. More CPU/RAM and no idle-sleep, but you keep the process running
(under `tmux` / `systemd`) for the duration of the course.

## Adding a lesson / changing order

Edit the `_LESSONS` list in `serve.py` (the `/NN` path sets the URL and order) and the table in
`home.py` (the landing page). Restart the server.
