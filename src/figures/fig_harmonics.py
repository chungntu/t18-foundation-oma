"""
Hinh: pho tin hieu do, doi chieu cac cuc ma CAU HINH A tra ve
(xem oma/naive_configuration.py) voi boi so cua tan so luoi dien 50 Hz.

Chay:  python oma/fig_harmonics.py
"""

import os
import sys
import numpy as np
import scipy.io as sio
from scipy import signal
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # src/
from paths import DATA, RESULTS, FIGURES, save_figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = FIGURES
FS = 1651.0

FILE, VAR = "dot2/xl_ngang.mat", "xl_ngang"
TITLE = "Lateral setup, campaign 2"

# cuc do CAU HINH A tra ve (khong decimate, i = 150, loc thong cao 5 Hz)
POLES_A = [57.557, 80.854, 84.186, 87.571, 99.803, 100.129, 105.124, 108.100,
           118.236, 122.897, 136.307, 150.239, 189.857, 200.209, 250.171,
           300.377, 321.489, 400.537, 604.527, 700.666]
# trong do, cac cuc nam trong 0.2 % cua mot boi so 50 Hz va can < 0.05 %
POLES_A_HARMONIC = [100.129, 150.239, 200.209, 250.171, 300.377, 400.537, 700.666]

# cuc do CAU HINH B tra ve
POLES_B = [1.052, 1.215, 1.611]


def main():
    x = sio.loadmat(os.path.join(DATA, *FILE.split("/")))[VAR].astype(float)
    x = signal.detrend(x, axis=0)
    f, P = signal.welch(x, fs=FS, nperseg=2 ** 16, axis=0)
    Pm = P.mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    for kpanel, (ax, (lo, hi, sub)) in enumerate(zip(axes, [
            (0.3, 25, "structural band, 0.3–25 Hz"),
            (25, 800, "band from which configuration A draws its poles, 25–800 Hz")])):
        m = (f >= lo) & (f <= hi)
        ax.semilogy(f[m], Pm[m], lw=0.7, color="#1f4e79", zorder=3)

        for k in range(1, 17):
            if lo <= 50 * k <= hi:
                ax.axvline(50 * k, color="#00a000", ls=":", lw=1.0, zorder=1)
        for fr in POLES_A:
            if lo <= fr <= hi:
                ax.axvline(fr, color="#c00000", ls="--", lw=0.8, zorder=2)
        for fr in POLES_B:
            if lo <= fr <= hi:
                ax.axvline(fr, color="#7030a0", ls="-", lw=1.4, zorder=2)

        ax.set_xlabel("Frequency [Hz]", fontsize=9)
        ax.set_ylabel("PSD, channel average", fontsize=9)
        ax.text(0.0, 1.015, "(%s)" % "ab"[kpanel], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")
        ax.set_xlim(lo, hi)
        ax.tick_params(labelsize=8)

    axes[0].plot([], [], color="#7030a0", lw=1.4, label="poles returned by configuration B")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].plot([], [], color="#c00000", ls="--", lw=0.8, label="poles returned by configuration A")
    axes[1].plot([], [], color="#00a000", ls=":", lw=1.0, label="multiples of 50 Hz")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "fig10_configurations.png")
    save_figure(fig, p)
    print("[luu]", p)


if __name__ == '__main__':
    main()
