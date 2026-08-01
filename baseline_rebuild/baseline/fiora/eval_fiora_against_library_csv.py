from pathlib import Path
import argparse
import json
import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd


def norm_key(x):
    s = str(x).strip()
    if s.endswith(".0"):
        try:
            return str(int(float(s)))
        except Exception:
            return s
    return s


def parse_mgf(fp):
    specs = {}
    cur_title = None
    mzs, ints = [], []

    def flush():
        nonlocal cur_title, mzs, ints
        if cur_title is not None:
            specs[norm_key(cur_title)] = (np.array(mzs, dtype=float), np.array(ints, dtype=float))
        cur_title = None
        mzs, ints = [], []

    for line in Path(fp).read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith("BEGIN IONS"):
            cur_title = None
            mzs, ints = [], []
        elif up.startswith("END IONS"):
            flush()
        elif up.startswith("TITLE="):
            cur_title = line.split("=", 1)[1].strip()
        elif "=" in line:
            continue
        else:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mzs.append(float(parts[0]))
                    ints.append(float(parts[1]))
                except Exception:
                    pass

    return specs


def peaks_from_json(s):
    obj = json.loads(s)
    return np.array(obj["mz"], dtype=float), np.array(obj["intensity"], dtype=float)


def filter_pred(mz, inten, top_k=0, rel_thresh=0.0):
    if len(mz) == 0:
        return mz, inten

    inten = np.asarray(inten, dtype=float)
    mz = np.asarray(mz, dtype=float)

    keep = np.ones(len(mz), dtype=bool)

    if rel_thresh and rel_thresh > 0:
        mx = float(np.max(inten)) if len(inten) else 0.0
        if mx > 0:
            keep &= inten >= mx * rel_thresh

    mz = mz[keep]
    inten = inten[keep]

    if top_k and top_k > 0 and len(mz) > top_k:
        idx = np.argsort(inten)[-top_k:]
        idx = idx[np.argsort(mz[idx])]
        mz = mz[idx]
        inten = inten[idx]

    return mz, inten


def binned_vec(mz, inten, bin_size=0.01):
    d = defaultdict(float)
    for m, v in zip(mz, inten):
        if not np.isfinite(m) or not np.isfinite(v):
            continue
        if v <= 0:
            continue
        b = int(round(float(m) / bin_size))
        d[b] += float(v)
    return d


def cosine(predicted_mz, predicted_intensity, library_mz, library_intensity, bin_size=0.01):
    prediction_bins = binned_vec(predicted_mz, predicted_intensity, bin_size)
    library_bins = binned_vec(library_mz, library_intensity, bin_size)

    if not prediction_bins or not library_bins:
        return 0.0

    dot = 0.0
    if len(prediction_bins) < len(library_bins):
        for key, value in prediction_bins.items():
            dot += value * library_bins.get(key, 0.0)
    else:
        for key, value in library_bins.items():
            dot += value * prediction_bins.get(key, 0.0)

    prediction_norm = math.sqrt(sum(value * value for value in prediction_bins.values()))
    library_norm = math.sqrt(sum(value * value for value in library_bins.values()))

    if prediction_norm <= 0 or library_norm <= 0:
        return 0.0
    return dot / (prediction_norm * library_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_mgf", required=True)
    ap.add_argument("--ref_csv", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test", "all"])
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--rel_thresh", type=float, default=0.0)
    ap.add_argument("--bin_size", type=float, default=0.01)
    args = ap.parse_args()

    pred = parse_mgf(args.pred_mgf)
    ref = pd.read_csv(args.ref_csv)

    if args.split != "all":
        ref = ref[ref["datasplit"] == args.split].copy()

    rows = []
    for _, r in ref.iterrows():
        key = norm_key(r["Name"])
        mz_ref, int_ref = peaks_from_json(r["peaks"])

        if key in pred:
            mz_pred, int_pred = pred[key]
            mz_pred, int_pred = filter_pred(mz_pred, int_pred, args.top_k, args.rel_thresh)
            c = cosine(mz_ref, int_ref, mz_pred, int_pred, args.bin_size)
            n_pred = len(mz_pred)
            missing = 0
        else:
            c = 0.0
            n_pred = 0
            missing = 1

        rows.append({
            "Name": key,
            "datasplit": r["datasplit"],
            "cos_0.01": c,
            "n_pred_peaks": n_pred,
            "missing_pred": missing,
        })

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("pred_mgf:", args.pred_mgf)
    print("ref_csv:", args.ref_csv)
    print("split:", args.split)
    print("top_k:", args.top_k)
    print("rel_thresh:", args.rel_thresh)
    print("n_ref:", len(out))
    print("n_pred_specs:", len(pred))
    print("missing:", int(out["missing_pred"].sum()))
    print("mean cos_0.01:", float(out["cos_0.01"].mean()))
    print("std cos_0.01:", float(out["cos_0.01"].std()))
    print("median cos_0.01:", float(out["cos_0.01"].median()))
    print("mean n_pred_peaks:", float(out["n_pred_peaks"].mean()))
    print("out_csv:", args.out_csv)


if __name__ == "__main__":
    main()
