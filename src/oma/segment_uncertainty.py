"""
Bat dinh tan so theo doan ban ghi (muc 5.2 bai bao).

Moi ban ghi duoc chia thanh ba doan bang nhau; moi doan duoc nhan dang lai voi
dung cac tham so cua pipeline.py. Do phan tan giua ba doan cua cac cuc trong
dai 1.0-1.4 Hz la mot uoc luong bat dinh thuc te hon do lech trong mot cum on
dinh (vi kich thich khong dung).

Chay:  python src/oma/segment_uncertainty.py
"""

import os
import sys
import csv
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from paths import DATA, RESULTS
from oma_lib import preprocess, ssi_cov, filter_poles, stable_poles, cluster_modes
from pipeline import FS_RAW, DEC, F_HIGH, I_BLOCK, ORDERS, XI_RANGE

SETUPS = [
    ("dot1/xl_thangdung.mat", "xl_thangdung", 1, "Vertical"),
    ("dot1/xl_ngang1.mat", "xl_ngang1", 1, "Lateral 1"),
    ("dot1/xl_ngang2.mat", "xl_ngang2", 1, "Lateral 2"),
    ("dot1/xl_ga.mat", "xl_ga", 1, "Three-channel"),
    ("dot2/xl_ngang.mat", "xl_ngang", 2, "Lateral"),
    ("dot2/xl_td1.mat", "xl_td1", 2, "Vertical d1"),
    ("dot2/xl_td2.mat", "xl_td2", 2, "Vertical d2ke"),
]
BAND = (1.0, 1.4)
N_SEG = 3
MIN_COUNT = 6          # doan ngan hon ban ghi day du nen ha nguong cum tu 8 xuong 6


def band_poles(y, fs):
    res = ssi_cov(y, 1.0 / fs, I_BLOCK, ORDERS)
    res = filter_poles(res, (0.5, 25.0), xi_range=XI_RANGE)
    modes = cluster_modes(res, stable_poles(res), min_count=MIN_COUNT)
    return sorted(m["f"] for m in modes if BAND[0] <= m["f"] <= BAND[1])


def main():
    rows = []
    for rel, var, camp, name in SETUPS:
        x = sio.loadmat(os.path.join(DATA, *rel.split("/")))[var].astype(float)
        y, fs = preprocess(x, FS_RAW, dec_factors=DEC, f_high=F_HIGH)
        segs = np.array_split(y, N_SEG, axis=0)
        per = [band_poles(s, fs) for s in segs]
        allp = [p for s in per for p in s]
        print(f"C{camp} {name:<14} " + " | ".join(
            " ".join(f"{p:.3f}" for p in s) if s else "-" for s in per))
        rows.append(dict(campaign=camp, setup=name,
                         segments=" | ".join(" ".join(f"{p:.4f}" for p in s) for s in per),
                         n_poles=len(allp),
                         f_min=round(min(allp), 4) if allp else "",
                         f_max=round(max(allp), 4) if allp else ""))
    out = os.path.join(RESULTS, "tables", "table_segment_uncertainty.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[luu]", out)


if __name__ == "__main__":
    main()
