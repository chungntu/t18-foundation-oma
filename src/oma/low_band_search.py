"""
Tim mode duoi 0.5 Hz - vung chua duoc khao sat.

Ly do: neu mode uon thap ton tai thi thiet ke soft-stiff dat no vao khoang
1P (~0.2 Hz) va 3P (~0.6 Hz), tuc quanh 0.3 Hz. Dai tim kiem truoc day bat
dau tu 0.5 Hz nen KHONG THE thay no.

De phan giai 0.3 Hz can cua so tre >= 2/0.3 = 6.7 s. Giam tan so lay mau
them mot buoc nua (1651 -> 12.9 Hz) va dung i = 150 cho tau = 11.6 s.

Chay:  python oma/low_band_search.py
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy import signal, linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # src/
from paths import DATA, RESULTS, FIGURES
from oma_lib import preprocess, ssi_cov, filter_poles, stable_poles, cluster_modes

FS_RAW = 1651.0
CASES = [
    ("dot1/xl_thangdung.mat", "xl_thangdung", "Vertical, campaign 1"),
    ("dot1/xl_ngang1.mat", "xl_ngang1", "Lateral 1, campaign 1"),
    ("dot2/xl_ngang.mat", "xl_ngang", "Lateral, campaign 2"),
    ("dot2/xl_td2.mat", "xl_td2", "Vertical d2ke, campaign 2"),
]

for rel, var, label in CASES:
    x = sio.loadmat(os.path.join(DATA, *rel.split("/")))[var].astype(float)
    y, fs = preprocess(x, FS_RAW, dec_factors=(4, 8, 4), f_high=0.12)
    dt = 1.0 / fs
    i_block = 150
    print(f"\n{'='*72}\n{label}")
    print(f"  fs = {fs:.2f} Hz, {y.shape[0]} mau = {y.shape[0]*dt/60:.1f} phut, "
          f"tau = {i_block*dt:.1f} s ({i_block*dt*0.3:.1f} chu ky @ 0.3 Hz)")
    print("="*72)

    f, P = signal.csd(y[:, :, None], y[:, None, :], fs=fs, nperseg=2048, axis=0)
    sv1 = np.array([linalg.svdvals(P[k])[0] for k in range(len(f))])
    m = (f >= 0.15) & (f <= 0.9)
    idx, props = signal.find_peaks(np.log(sv1[m]), prominence=0.8)
    if len(idx):
        order = np.argsort(props['prominences'])[::-1][:6]
        print("  dinh FDD trong 0.15-0.9 Hz:", np.round(np.sort(f[m][idx][order]), 3))
    else:
        print("  dinh FDD trong 0.15-0.9 Hz: khong co dinh noi bat")

    res = ssi_cov(y, dt, i_block, list(range(2, 62, 2)))
    res = filter_poles(res, (0.18, 0.95), xi_range=(0.002, 0.15))
    modes = cluster_modes(res, stable_poles(res), min_count=6)
    if modes:
        for mo in modes:
            print(f"  SSI: f = {mo['f']:.3f} Hz, xi = {100*mo['xi']:.2f} %, "
                  f"n_cuc = {mo['count']}, MPC = {mo['mpc']:.2f}")
    else:
        print("  SSI trong 0.18-0.95 Hz: KHONG co cum nao dat tieu chi")
