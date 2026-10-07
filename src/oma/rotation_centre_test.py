"""
Phep kiem phan biet co che bang so lieu do (muc 6.1 cua bai bao).

Trong cac setup ngang, kenh 1, 2, 3, 5 do phuong ngang, kenh 4 do phuong dung.
Voi mot mode ngang-lac cua dai cung, tam xoay o cao do z_c:
    bien do dung tai kenh 4 / bien do ngang tai mat dai = d / |z_top - z_c|
voi d la khoang cach tu kenh 4 toi truc xoay (d <= 3.54 m neu khung dong tam).
    - xoay quanh tam cao (mode ngang thu nhat, z_c ~ 50-100 m): ty so ~ 0.03-0.08
    - lac quanh tam thap (z_c ~ 1-2 m, ngay duoi mat dai):     ty so ~ 0.5-1.5
Mode xoan cho kenh ngang o hai goc doi dien NGUOC PHA va kenh tam ~ 0.

Ty so do duoc la CAN TREN cua thanh phan xoay vi con co nhieu va do nhay ngang
cua cam bien do dung.

Chay:  python src/oma/rotation_centre_test.py
"""

import os
import sys
import csv
import numpy as np
import scipy.io as sio
from scipy import signal, linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from paths import DATA, RESULTS
from oma_lib import preprocess

FS_RAW = 51200 / 31            # Hz = 1651.61; wf_increment = 31/51200 s trong metadata TDMS
BAND = (1.00, 1.40)
CASES = [
    ("dot1/xl_ngang1.mat", "xl_ngang1", 1, "Lateral 1"),
    ("dot1/xl_ngang2.mat", "xl_ngang2", 1, "Lateral 2"),
    ("dot2/xl_ngang.mat", "xl_ngang", 2, "Lateral"),
]
HORIZ = [0, 1, 2, 4]      # kenh 1, 2, 3, 5
VERT = 3                  # kenh 4


def analyse(y, fs, nperseg=8192):
    f, P = signal.csd(y[:, :, None], y[:, None, :], fs=fs, nperseg=nperseg, axis=0)
    m = (f >= BAND[0]) & (f <= BAND[1])
    sv = np.array([linalg.svdvals(P[k])[:2] for k in range(len(f))])
    k = np.where(m)[0][np.argmax(sv[m, 0])]
    U, s, _ = linalg.svd(P[k])
    phi = U[:, 0] / U[np.argmax(np.abs(U[:, 0])), 0]
    h = np.abs(phi[HORIZ])
    # coherence kenh dung voi kenh ngang tam
    coh = np.abs(P[k, VERT, 4]) ** 2 / (np.abs(P[k, VERT, VERT]) * np.abs(P[k, 4, 4]))
    # ty so dung/ngang theo bien do PSD, khong qua vec-to ky di
    psd_ratio = np.sqrt(np.abs(P[k, VERT, VERT]) / np.mean(np.abs(P[k, HORIZ, HORIZ])))
    # pha cac kenh ngang so voi kenh tam
    ph = np.degrees(np.angle(phi[HORIZ] / phi[4]))
    return dict(f_peak=float(f[k]), vh_svd=float(abs(phi[VERT]) / h.mean()),
                vh_psd=float(psd_ratio), coherence=float(coh),
                sv2_over_sv1=float(s[1] / s[0]),
                centre_over_corner=float(h[3] / h[:3].mean()),
                max_horiz_phase=float(np.max(np.abs(ph))))


def main():
    rows = []
    print(f"{'setup':<20}{'f_pk':>7}{'v/h svd':>9}{'v/h psd':>9}{'coh 4-5':>9}"
          f"{'s2/s1':>7}{'tam/goc':>9}{'pha max':>9}")
    for rel, var, camp, name in CASES:
        x = sio.loadmat(os.path.join(DATA, *rel.split("/")))[var].astype(float)
        y, fs = preprocess(x, FS_RAW, dec_factors=(4, 8), f_high=0.3)
        r = analyse(y, fs)
        rows.append(dict(campaign=camp, setup=name, **{k: round(v, 4) for k, v in r.items()}))
        print(f"{name + ', C' + str(camp):<20}{r['f_peak']:7.3f}{r['vh_svd']:9.3f}{r['vh_psd']:9.3f}"
              f"{r['coherence']:9.3f}{r['sv2_over_sv1']:7.3f}{r['centre_over_corner']:9.3f}"
              f"{r['max_horiz_phase']:9.1f}")
    # dang mode SSI tung cuc cua setup ngang dot 2 (huong kenh co trong log MACEC)
    from pipeline import analyze
    r = analyze(os.path.join(DATA, "dot2", "xl_ngang.mat"), "xl_ngang", "Lateral, campaign 2")
    print("\n  Dang mode SSI, setup ngang dot 2 (kenh 1 2 3 [4=dung] 5 6=thap):")
    for mo in r["modes"]:
        if mo["f"] > 2.0:
            continue
        phi = mo["phi"] / mo["phi"][np.argmax(np.abs(mo["phi"][HORIZ]))] if False else mo["phi"]
        ref = phi[4]                                   # kenh tam
        rel = phi / ref
        h = np.abs(rel[HORIZ])
        ph = np.degrees(np.angle(rel[HORIZ]))
        vh = abs(rel[VERT]) / h.mean()
        pattern = "tinh tien" if np.max(np.abs(ph)) < 30 and h.min() > 0.5 * h.max() else "khac"
        print(f"   f = {mo['f']:.3f} Hz: |ngang 1,2,3,5| = {np.round(h, 2)}, pha = {np.round(ph, 0)},"
              f" dung/ngang = {vh:.3f}, MPC = {mo['mpc']:.2f} -> {pattern}")
        rows.append(dict(campaign=2, setup=f"Lateral SSI {mo['f']:.3f} Hz", f_peak=round(mo["f"], 4),
                         vh_svd=round(float(vh), 4), vh_psd="", coherence="", sv2_over_sv1="",
                         centre_over_corner=round(float(h[3] / h[:3].mean()), 4),
                         max_horiz_phase=round(float(np.max(np.abs(ph))), 1)))
    out = os.path.join(RESULTS, "tables", "table_rotation_centre.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[luu]", out)


if __name__ == "__main__":
    main()
