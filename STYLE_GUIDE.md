# STYLE_GUIDE.md — Tone / Pedagogy Revision (all 14 notebooks)

This is the durable style layer that sits alongside `COURSE_DESIGN.md`. The 14 notebooks are
technically correct and the narrative structure and graphics are liked. The problem is **diction**:
the prose "reads a bit like a murder mystery novel." This course is a scientific **learning
exercise** for students who may be new to **both** behavioral neuroscience **and** Python.

You are **revising an existing, verified notebook**. Keep every working data-loading / computation /
self-check / no-live-UMAP cell **exactly as-is**. Rewrite the **prose, exercises, and figures** only.
Each notebook must still export cleanly headless.

---

## 1. Tone & diction (all 14)

- **Formal and plain.** Remove theatrical / mystery diction and dramatic beat-names. Banned framing
  includes: "the crisis", "the reversal", "the gut-punch", "the bet", "forbidden fruit", "the hunt",
  "graduation", "cash the check", "the threat", "the payoff", exclamatory hype, and ominous
  foreshadowing ("You are about to meet that problem in its rawest form."). State things directly.
- Keep a **light, professional framing**: the student is a behavioral neuroscientist studying social
  behavior. State that plainly, not cinematically.
- Drop email-style "FROM: Circuit Team → TO: Behavior Team" cold-open theatrics; open with the
  scientific purpose instead.

### The exact transformation wanted

> **TOO FLAMBOYANT:**
> "A neural experiment is coming: a laser that flips a hypothalamic switch, a probe in mPFC. But a
> manipulation is worthless without an objective readout of what each mouse actually does —
> hand-scoring won't survive review."

> **RIGHT REGISTER:**
> "Your job as a behavioral neuroscientist will be to study the role of the mPFC in social behavior.
> However, to understand behavior, we need a clearer understanding of what behavior IS. Today we will
> be using SLEAP, a software that ..."

---

## 2. Structure every section as: WHY → DEFINITIONS → METHOD

- **WHY** — open each notebook and each major section by explaining, in plain language, why it
  matters (the scientific purpose).
- **DEFINITIONS** — define the terms a newcomer needs *before* they can follow. Define jargon on
  first use, plainly, with a concrete example or analogy. Examples of terms to define: keypoint,
  behavior segmentation, what it means to "decode", principal component, clustering, Markov chain,
  allocentric, AUROC.
- **METHOD** — present the method. For **every function used**, state in plain words its **PURPOSE**,
  its **INPUTS**, and its **OUTPUTS**.
- Assume **no fluency** in neuroscience or coding.

---

## 3. Brain regions: ease off in Week 1 (NB01–07)

- In Week 1 dial **way** back on brain-region name-dropping and the "neural twin" comparisons. With
  no recordings yet, it reads as pretentious.
- **Remove** the neural-twin citation accordions and the running brain-twin device from Week-1 prose.
- A **single plain motivating sentence is the maximum**, e.g. "this is also how neuroscientists
  quantify behavior."
- **Keep genuine neural comparisons only where real neural data justifies them:** NB08 (the
  neural-demo decode check) and Week 2 (NB09–14, which use real imaging data). There, keep them but
  state them plainly.

---

## 4. Naming: remove "Hero" (all 14)

- **Delete the "Hero Event" branding everywhere.** Refer to the running example plainly as **"our
  example interaction"** or **"the example approach event."**
- Label the two interacting mice as **approacher** and **approachee** (or **subject** / **partner**).
- Keep the **same example event index as before** (`HERO = 909`, Cage 15, male aggression approach —
  approacher = Dom, approachee = Sub, bystander = Mid). Just stop calling it a Hero. You may keep the
  variable name in code if it is load-bearing, but never surface the word "Hero" in prose or titles.

---

## 5. Color scheme (HARD constraint, all 14)

Mice are colored **only by rank**, identically across all 14 notebooks. Use `cu.RANK_HEX` /
`cu.RANK_RGB`. **Never** introduce any other color mapping for mice.

| Rank | Name | Hex | RGB |
|------|------|-----|-----|
| 1 | **Dom** | `#d62728` (red) | (214, 39, 40) |
| 2 | **Int / Mid** | `#1f77b4` (blue) | (31, 119, 180) |
| 3 | **Sub** | `#2ca02c` (green) | (44, 160, 44) |
| 0 | unknown | `#969696` (gray) | (150, 150, 150) |

Neural notebooks with no rank may use neutral palettes, but any mouse-rank coloring uses this scheme.

---

## 6. Lean heavily on GIFs (all 14)

The subject is behavior; students learn by **seeing** it. Use the rendered skeleton GIF helpers
generously — prefer a GIF over a paragraph when illustrating a behavior:

- `cu.event_gif_bytes(kp_event, ranks, contact_rel, cell, fps)` → GIF bytes for one event.
- `cu.grid_gif_bytes(events, ncols, cell, fps)` → tiled grid GIF (max 5×5) of several events.
- `cu.gif_img_html(gif, width)` → wraps GIF bytes in an animating `<img>` data-URI for `mo.md` /
  `mo.Html` (a plain marimo image widget freezes the first frame).

Use them to show the example event, and to show what a method's output **means**: exemplars of a
cluster, of a high-frequency-wavelet event, of a correlated pair, of a predicted-aggression event.
Neural notebooks: lean on the real movies / rasters they already load.

---

## 7. Coding exercises: gentle, fill-in-the-blank (all 14)

- **Never** hand the student a daunting from-scratch code block. Give a **scaffold** with a few
  **blanked** lines/variables to fill:

  ```python
  # TODO: compute each mouse's speed.
  # Replace ____ with np.diff(cen, axis=0) — the per-frame change in centroid position.
  # The line below already has the centroids `cen` (shape (T, 2)) and takes the norm for you.
  speed = np.linalg.norm(____, axis=1)   # (T-1,) px/frame
  ```

- Be explicit about **exactly which line(s)** to edit, what each surrounding line already does, and
  what the resulting **plot** should look like: "you should see two curves; the red Dom mouse's speed
  should spike near contact."
- The output of every exercise is a **plot** the student compares against a described / expected
  picture.
- Keep the **revealable solution** (accordion) and the **tolerance-band self-check**, but keep the
  student-edited surface **small and clearly marked**.

---

## 8. Readout Board Gauge A bug — fix in every notebook that has a board

**Symptom.** Gauge A used a plotly Indicator with `mode="number+delta"` and
`delta={"reference": 11700}`. Because Gauge A's own value is smaller than 11,700 (e.g. 19 features in
NB02), the delta rendered as a confusing **negative** — `▼ -11,681` (the "-11k" that was reported) —
and the extra delta line crowded / overlapped the two-line title.

**Fix.**
- Use **`mode="number"`** (drop the delta). Show **only** the correct **positive** representation
  size for **this** notebook.
- Keep the **"was 11,700 raw"** context in the **title text**, not as a delta.
- Give the figure **enough height and top margin** that the two-line titles do not overlap
  (`height=230`, `margin.t≈95`).
- **No negative deltas anywhere.**

**Gauge A value per notebook:** NB01 = 11,700 · NB02 = 19 · NB03 = 19 · NB04 ≈ 6 · NB05 = 2 (map) /
1 (syllable) · later notebooks show their own representation size (NB08 = 1 decision).

The exact corrected `readout_fig` to paste is in `REVISION_CONTRACT` (Task 2) and in the contract
returned by the PREP agent.

Notebooks whose grep hit `number+delta`:

| Notebook | Line | Gauge | Verdict |
|----------|------|-------|---------|
| `02_body_eye_view.py` | 118 | A (size) | **BUG** — value 19, ref 11700 → **−11,681**. Fix to `mode="number"`. |
| `01_raw_signal.py` | 89 | A (size) | Pointless zero-delta (ref == value == 11700). Fix to `mode="number"` (baseline; no "was X raw"). |
| `06_reading_the_map.py` | 164 | B (rising AUROC) | delta ref == value → ≈0.000. Drop the delta; use `gauge+number`. |
| `08_decoder_graduates.py` | 156, 884 | B (rising AUC) | delta ref == value (0.86) → ≈0.000. Gauge A already `mode="number"`. Drop the delta on B. |

---

## 9. Preserve (do not break)

This is a prose / exercise / figure revision. Do **not** touch:

- All working code, data loading, and computed self-check **pin values**.
- The **no-live-UMAP** rule (UMAP is precomputed; live UMAP crashes the kernel).
- Valid **marimo** structure: one-assignment-per-name, `_`-prefixed locals, last-expression render,
  `hide_code` on prose cells, sliders placed adjacent to their output via `mo.vstack`.

**Re-verify each notebook exports headless after revising.**
