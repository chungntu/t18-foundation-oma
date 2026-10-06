"""
Thi nghiem giai thua 2x2: tach rieng anh huong cua CUA SO TRE va cua BO LOC.

Phan bien (dung) chi ra rang thi nghiem hai cau hinh khong tach duoc hai
nguyen nhan, vi cau hinh A doi dong thoi ca hai. O day doi tung yeu to mot.

                      loc thong cao 5 Hz      loc thong cao 0.3 Hz
    tau = 0.091 s           A                        C
    tau = 2.33 s            D                        B

  - Neu chi BO LOC gay ra ket qua: C se tim lai duoc dai 1 Hz.
  - Neu CUA SO TRE cung la rang buoc doc lap: C van khong tim duoc,
    va chi B moi tim duoc.

Chay:  python oma/configuration_experiment.py
"""

import os, sys
import numpy as np
import scipy.io as sio
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # src/
from paths import DATA, RESULTS, FIGURES
from oma_lib import ssi_cov, filter_poles, stable_poles, cluster_modes, preprocess

FS_RAW = 1651.0
FILE, VAR, LABEL = "dot2/xl_ngang.mat", "xl_ngang", "Lateral setup, campaign 2"


def prepare(x, decimate, f_high):
    if decimate:
        y, fs = preprocess(x, FS_RAW, dec_factors=(4, 8), f_high=f_high)
    else:
        y = signal.detrend(x, axis=0, type='constant')
        b, a = signal.butter(4, f_high / (FS_RAW / 2), btype='high')
        y = signal.filtfilt(b, a, y, axis=0)
        fs = FS_RAW
    return y, fs


CONFIGS = [
    # ten, decimate, loc thong cao [Hz], i, dai tim mode
    ("A", False, 5.0, 150, (0.5, 800.0)),
    ("C", False, 0.3, 150, (0.5, 800.0)),
    ("D", True, 5.0, 120, (0.5, 25.0)),
    ("B", True, 0.3, 120, (0.5, 25.0)),
]


def main():
    x = sio.loadmat(os.path.join(DATA, *FILE.split("/")))[VAR].astype(float)
    print(f"Ban ghi: {LABEL}  ({x.shape[0]} mau x {x.shape[1]} kenh)\n")

    summary = []
    for name, dec, fh, i_block, f_range in CONFIGS:
        y, fs = prepare(x, dec, fh)
        dt = 1.0 / fs
        tau = i_block * dt
        res = ssi_cov(y, dt, i_block, list(range(2, 82, 2)))
        res = filter_poles(res, f_range, xi_range=(0.00001, 0.15))
        modes = cluster_modes(res, stable_poles(res), min_count=8)

        low = [m for m in modes if m['f'] < 5.0]
        harm = [m for m in modes
                if m['f'] > 40 and abs(m['f'] - 50 * round(m['f'] / 50)) / (50 * round(m['f'] / 50)) < 0.002
                and m['xi'] < 0.0005]

        print("=" * 78)
        print(f"CAU HINH {name}:  fs = {fs:7.2f} Hz | loc thong cao {fh:4.1f} Hz | "
              f"i = {i_block} | tau = {tau:.3f} s ({tau*1.0:.2f} chu ky @ 1 Hz)")
        print("=" * 78)
        print(f"  tong so mode: {len(modes)}")
        if low:
            print("  mode duoi 5 Hz:  " + "  ".join(f"{m['f']:.3f} Hz (xi {100*m['xi']:.2f}%)" for m in low))
        else:
            print("  mode duoi 5 Hz:  KHONG CO")
        if harm:
            print(f"  cuc trung boi so 50 Hz, can < 0.05%: {len(harm)} cuc  -> "
                  + ", ".join(f"{m['f']:.2f}" for m in harm))
        rng = (f"{min(m['f'] for m in modes):.1f} - {max(m['f'] for m in modes):.1f} Hz"
               if modes else "-")
        print(f"  dai tan so cac mode: {rng}\n")
        summary.append((name, fs, fh, tau, len(modes), len(low), len(harm), rng))

    print("=" * 78)
    print("TONG HOP")
    print("=" * 78)
    print(f"{'Cau hinh':>9} {'fs [Hz]':>9} {'loc [Hz]':>9} {'tau [s]':>9} "
          f"{'n mode':>7} {'<5 Hz':>7} {'harmonic':>9}")
    for s in summary:
        print(f"{s[0]:>9} {s[1]:>9.1f} {s[2]:>9.1f} {s[3]:>9.3f} {s[4]:>7d} {s[5]:>7d} {s[6]:>9d}")
    print("\nDoc bang: neu C (bo loc da noi ra, nhung cua so tre van ngan) VAN khong")
    print("tra ve mode duoi 5 Hz, thi cua so tre la rang buoc doc lap voi bo loc.")


if __name__ == '__main__':
    main()
