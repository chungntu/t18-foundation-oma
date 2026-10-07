"""
So sanh MUC DO HUY DONG KENH giua dai 1 Hz va dai 14-24 Hz.

Chi so: so kenh huy dong hieu dung (inverse participation ratio)
    N_eff = (sum |phi_i|^2)^2 / sum |phi_i|^4
  - mode toan cuc: moi kenh cung tham gia -> N_eff gan bang so kenh
  - mode cuc bo:   mot vai kenh chi phoi   -> N_eff gan 1

Dang mode lay tu vec-to ky di thu nhat cua ma tran CPSD tai dinh pho trong dai.

Chay:  python oma/participation.py
"""

import os, sys
import numpy as np
import scipy.io as sio
from scipy import signal, linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # src/
from paths import DATA, RESULTS, FIGURES
from oma_lib import preprocess

FS_RAW = 51200 / 31            # Hz = 1651.61; wf_increment = 31/51200 s trong metadata TDMS

CASES = [
    ("dot1/xl_thangdung.mat", "xl_thangdung", "Vertical, campaign 1"),
    ("dot1/xl_ngang1.mat", "xl_ngang1", "Lateral 1, campaign 1"),
    ("dot1/xl_ngang2.mat", "xl_ngang2", "Lateral 2, campaign 1"),
    ("dot1/xl_ga.mat", "xl_ga", "Three-channel, campaign 1"),
    ("dot2/xl_ngang.mat", "xl_ngang", "Lateral, campaign 2"),
    ("dot2/xl_td1.mat", "xl_td1", "Vertical d1, campaign 2"),
    ("dot2/xl_td2.mat", "xl_td2", "Vertical d2ke, campaign 2"),
]
BANDS = [("dai 1 Hz   ", (1.00, 1.40)), ("dai 14-24Hz", (13.0, 24.5))]


def shape_at_peak(y, fs, band, nperseg=8192):
    f, P = signal.csd(y[:, :, None], y[:, None, :], fs=fs, nperseg=nperseg, axis=0)
    sv = np.array([linalg.svdvals(P[k]) for k in range(len(f))])
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return None
    k = np.where(m)[0][np.argmax(sv[m, 0])]
    U, S, _ = linalg.svd(P[k])
    phi = U[:, 0]
    return f[k], phi, S[0] / S[1]


def n_eff(phi):
    a = np.abs(phi) ** 2
    return float(a.sum() ** 2 / (a ** 2).sum())


def main():
    print(f"{'Setup':<26} {'Dai':<12} {'f [Hz]':>8} {'n_kenh':>7} "
          f"{'N_eff':>7} {'N_eff/n':>8} {'SV1/SV2':>8}")
    print("-" * 82)
    for rel, var, label in CASES:
        x = sio.loadmat(os.path.join(DATA, *rel.split("/")))[var].astype(float)
        # dai cao can giu tan so lay mau cao hon -> decimate it hon
        for tag, band in BANDS:
            dec = (4, 8) if band[1] < 5 else (4, 4)      # 51.6 Hz hoac 103.2 Hz
            y, fs = preprocess(x, FS_RAW, dec_factors=dec, f_high=0.3)
            r = shape_at_peak(y, fs, band)
            if r is None:
                continue
            f, phi, svr = r
            nch = len(phi)
            ne = n_eff(phi)
            print(f"{label:<26} {tag:<12} {f:8.3f} {nch:7d} {ne:7.2f} {ne/nch:8.2f} {svr:8.1f}")
        print()


if __name__ == '__main__':
    main()
