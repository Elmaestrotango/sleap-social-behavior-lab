"""Build data/train_derived.npz + data/heldout_derived.npz — precomputed metadata + features + PCA.

Everything expensive or metadata-y that the notebooks need is computed ONCE here and shipped, so no
student kernel does >2s of compute (molab-safety) and the sex / time-of-day exercises have clean
fields. Aligned row-for-row with train_events.npz / heldout_events.npz (same event order).

  cohort      (N,) str     cohort DATE ('20260222' | '12192025'), parsed from event_key
  cage        (N,) int     COHORT-UNIQUE physical-cage id = cohort_index*100 + cam (so cage 9 in
                           20260222 -> 9, cage 9 in 12192025 -> 109; they never merge). This is the
                           UNIT for the cage-level permutation test.
  sex         (N,) '<U1'   'M'/'F' looked up by (cohort, cam) — NOT by bare cam, which collides.
  tod_hour    (N,) float   approximate time-of-day (reverse cycle: lights ON 21-09)
  X           (N,19) f32   the 19 allocentric features (cu.features_batch)
  pca_scores  (N,10) f32   PCA scores (train PCA; heldout transformed by the SAME fit)
  feature_names, evr, pca_components, pca_mean, pca_std  (train file only)

Instructors:  uv run python tools/build_derived.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "course"))
import course_utils as cu  # noqa: E402

# cohort_meta.csv carries a `cohort` (date) column; sex is keyed by (cohort, cage) because cage
# numbers collide across cohorts (and even carry different sexes). cohort_index makes `cage`
# cohort-unique below.
_META_DF = pd.read_csv(os.path.join(ROOT, "data", "cohort_meta.csv"))
SEX = {(str(r["cohort"]), int(r["Cage"])): str(r["Sex"]) for _, r in _META_DF.iterrows()}
COHORT_INDEX = {"20260222": 0, "12192025": 1}
CAM_RE = re.compile(r"cam\.(\d+)")
HOUR_RE = re.compile(r"T(\d{2})")


def parse_meta(event_keys):
    cohort, cage, sex, tod = [], [], [], []
    for k in event_keys.astype(str):
        coh, stem, _, cf = k.split("|")
        date = coh.split("_")[0]                 # '20260222_dep' -> '20260222'
        cam = int(CAM_RE.search(stem).group(1))
        h = int(HOUR_RE.search(stem).group(1))
        ci = COHORT_INDEX.get(date, 9)
        cohort.append(date)
        cage.append(ci * 100 + cam)              # cohort-unique physical-cage id (int16-safe)
        sex.append(SEX.get((date, cam), "?"))    # sex by (cohort, cam), never bare cam
        tod.append((h + int(cf) / cu.FPS / 3600.0) % 24.0)
    return (np.array(cohort), np.array(cage, np.int16), np.array(sex), np.array(tod, np.float32))


def main():
    tr = cu.load_events(os.path.join(ROOT, "data", "train_events.npz"))
    he = cu.load_events(os.path.join(ROOT, "data", "heldout_events.npz"))

    Xtr = cu.features_batch(tr["kp"])                    # (N,19)
    Xhe = cu.features_batch(he["kp"])

    # PCA fit on train; transform both with the same fit (this IS how NB04/05 use it)
    Xz, mu, sd = cu.standardize(Xtr)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10, random_state=0).fit(Xz)
    scores_tr = pca.transform(Xz).astype(np.float32)
    scores_he = pca.transform(np.nan_to_num((Xhe - mu) / sd)).astype(np.float32)

    for tag, ev, X, scores in [("train", tr, Xtr, scores_tr), ("heldout", he, Xhe, scores_he)]:
        cohort, cage, sex, tod = parse_meta(ev["event_key"])
        out = dict(cohort=cohort, cage=cage, sex=sex, tod_hour=tod, X=X.astype(np.float32),
                   pca_scores=scores, feature_names=np.array(cu.FEATURE_NAMES))
        if tag == "train":
            out.update(evr=pca.explained_variance_ratio_.astype(np.float32),
                       pca_components=pca.components_.astype(np.float32),
                       pca_mean=mu.astype(np.float32), pca_std=sd.astype(np.float32))
        path = os.path.join(ROOT, "data", f"{tag}_derived.npz")
        np.savez_compressed(path, **out)
        print(f"  wrote {path} ({os.path.getsize(path)/1e6:.2f} MB)  "
              f"N={len(cage)}  cohort={dict(zip(*np.unique(cohort, return_counts=True)))}  "
              f"sex={dict(zip(*np.unique(sex, return_counts=True)))}  "
              f"cages={sorted(set(cage.tolist()))}  tod {tod.min():.1f}-{tod.max():.1f}h")
    print(f"  PCA evr (first 6): {np.round(pca.explained_variance_ratio_[:6], 3).tolist()} "
          f"(cum {pca.explained_variance_ratio_[:6].sum():.2f})")


if __name__ == "__main__":
    main()
