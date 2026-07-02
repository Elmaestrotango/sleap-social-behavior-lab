#!/usr/bin/env python
"""tools/trim_slp.py — trim a few real .slp files down to short clips for the 'load SLEAP' notebook.

Reads data/_scratch/slp_todo.json (written by build_dataset.py) and, for each entry, slices the
lab's cleaned prediction .slp to a small frame window around the event and saves it to
data/raw_slp/example_<tag>.slp. These tiny files let notebook 01 demonstrate reading real SLEAP
output with sleap-io; only the predicted keypoints are used (the video is never opened), so a
non-resolving video path on a student machine is harmless.

Run with an env that has sleap-io (the lab's sleap-nn env works):
    /nadata/snlkt/home/itang/miniconda3/envs/sleap-nn/bin/python tools/trim_slp.py
"""
import os, json, glob
import sleap_io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
OUT = os.path.join(DATA, "raw_slp")
SLP_BASE = "/snlkt/isaac/SLEAP_files/homecage/despotism/sleap_files"
PAD, WIN = 150, 250          # ~8 s clip around the event


def main():
    os.makedirs(OUT, exist_ok=True)
    todo = json.load(open(os.path.join(DATA, "_scratch", "slp_todo.json")))
    for item in todo:
        coh, stem, cs, tag = item["cohort"], item["stem"], item["contact_start"], item["tag"]
        hits = glob.glob(f"{SLP_BASE}/{coh}/cleaned/{stem}*_cleaned.slp")
        if not hits:
            print(f"  [skip] no .slp for {coh}/{stem}")
            continue
        labels = sio.load_slp(hits[0])
        lo, hi = cs - PAD, cs + WIN
        kept = [lf for lf in labels.labeled_frames if lo <= lf.frame_idx < hi]
        if not kept:
            print(f"  [skip] no labeled frames in [{lo},{hi}) for {stem}")
            continue
        trimmed = sio.Labels(labeled_frames=kept, videos=labels.videos,
                             skeletons=labels.skeletons, tracks=labels.tracks)
        outp = os.path.join(OUT, f"example_{tag}.slp")
        sio.save_slp(trimmed, outp)
        print(f"  wrote {outp}  ({len(kept)} frames around cs={cs})")


if __name__ == "__main__":
    main()
