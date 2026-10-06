"""
Hai viec phan bien yeu cau:

  PHAN 1. Dinh nguong cho tieu chi N_eff bang mo phong.
     - H0 "cuc bo": chi mot kenh chua tin hieu dai hep, cac kenh con lai la nhieu
     - H1 "toan cuc": moi kenh chia se mot tin hieu chung theo dang co duong nut
     Tu hai phan bo suy ra nguong va ty le bao dong gia.

  PHAN 2. Kiem dinh thay doi tan so o DON VI CAU HINH DO,
     thay cho viec gop tat ca cac cuc lam mau doc lap.

Chay:  python oma/criterion_calibration.py
"""

import itertools
import math
import numpy as np
from scipy import signal, linalg

RNG = np.random.default_rng(20260927)

FS = 51.59          # Hz, sau khi giam tan so lay mau
N_SAMP = 28000      # xap xi ban ghi ngan nhat
NPERSEG = 8192
F0 = 1.2            # Hz, tan so tin hieu mo phong
N_TRIAL = 400


def n_eff(phi):
    a = np.abs(phi) ** 2
    return float(a.sum() ** 2 / (a ** 2).sum())


def narrowband(n_samp, fs, f0, zeta=0.02):
    """Tin hieu dai hep: nhieu trang qua bo loc cong huong bac 2."""
    w = RNG.standard_normal(n_samp)
    wn = f0 / (fs / 2)
    b, a = signal.iirpeak(wn, Q=1 / (2 * zeta))
    return signal.lfilter(b, a, w)


def neff_from_series(y, fs, band=(1.0, 1.4)):
    f, P = signal.csd(y[:, :, None], y[:, None, :], fs=fs, nperseg=NPERSEG, axis=0)
    sv1 = np.array([linalg.svdvals(P[k])[0] for k in range(len(f))])
    m = (f >= band[0]) & (f <= band[1])
    k = np.where(m)[0][np.argmax(sv1[m])]
    U, _, _ = linalg.svd(P[k])
    return n_eff(U[:, 0])


def simulate(nch, weights, snr):
    """weights = None -> H0 (chi kenh 0 co tin hieu); nguoc lai H1 voi dang cho truoc."""
    s = narrowband(N_SAMP, FS, F0)
    s = s / s.std()
    y = RNG.standard_normal((N_SAMP, nch))
    if weights is None:
        y[:, 0] += snr * s
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.abs(w).max()
        y += snr * np.outer(s, w)
    return y


def part1():
    print("=" * 76)
    print("PHAN 1. NGUONG CHO TIEU CHI N_eff")
    print("=" * 76)
    # dang lac co duong nut: hai goc nguoc pha, tam gan bang khong
    shapes = {5: [1.0, 0.35, -0.27, -0.25, -0.05],
              6: [1.0, 0.35, -0.27, -0.25, -0.05, 0.30]}
    for nch in (5, 6):
        for snr in (3.0, 10.0):
            h0 = np.array([neff_from_series(simulate(nch, None, snr), FS)
                           for _ in range(N_TRIAL)])
            h1 = np.array([neff_from_series(simulate(nch, shapes[nch], snr), FS)
                           for _ in range(N_TRIAL)])
            # nguong: bach phan vi 95 cua H0
            thr = np.quantile(h0, 0.95)
            power = (h1 > thr).mean()
            print(f"\n  n_kenh = {nch}, SNR = {snr:.0f}")
            print(f"    H0 cuc bo   : N_eff trung vi {np.median(h0):.2f}, "
                  f"bach phan vi 95 = {thr:.2f}   (N_eff/n = {thr/nch:.2f})")
            print(f"    H1 toan cuc : N_eff trung vi {np.median(h1):.2f}, "
                  f"bach phan vi 5  = {np.quantile(h1,0.05):.2f}")
            print(f"    => nguong N_eff/n = {thr/nch:.2f} cho bao dong gia 5 %, "
                  f"do manh = {100*power:.0f} %")

    print("\n  Doi chieu voi so lieu thuc (N_eff/n):")
    print("    dai 13-24 Hz : 0.20 0.20 0.20 0.61 0.20 0.17 0.61")
    print("    dai 1.0-1.4  : 0.31 0.94 0.20 0.60 0.83 0.40 0.49")


def part2():
    print()
    print("=" * 76)
    print("PHAN 2. THAY DOI TAN SO, KIEM DINH O DON VI CAU HINH DO")
    print("=" * 76)
    c1 = {"Vertical": [1.1734], "Lateral 1": [1.2258, 1.2536],
          "Lateral 2": [1.3217], "Three-channel": [1.2370, 1.3307]}
    c2 = {"Lateral": [1.0520, 1.2146], "Vertical d1": [1.1465],
          "Vertical d2ke": [1.0735, 1.1985, 1.2305]}
    a = np.array([np.mean(v) for v in c1.values()])
    b = np.array([np.mean(v) for v in c2.values()])

    print(f"  dot 1, trung binh tung setup: {np.round(a,4)}   -> {a.mean():.4f} Hz")
    print(f"  dot 2, trung binh tung setup: {np.round(b,4)}   -> {b.mean():.4f} Hz")
    print(f"  chenh lech: {100*(b.mean()-a.mean())/a.mean():+.2f} %")
    print(f"  tach bach hoan toan? min(dot1) = {a.min():.4f} > max(dot2) = {b.max():.4f}  "
          f"-> {a.min() > b.max()}")

    obs = abs(a.mean() - b.mean())
    allv = np.concatenate([a, b])
    cnt = tot = 0
    for idx in itertools.combinations(range(len(allv)), len(a)):
        g1 = allv[list(idx)]
        g2 = np.delete(allv, list(idx))
        tot += 1
        if abs(g1.mean() - g2.mean()) >= obs - 1e-12:
            cnt += 1
    p_min = 2 / math.comb(len(allv), len(a))
    print(f"\n  kiem dinh hoan vi hai phia: p = {cnt}/{tot} = {cnt/tot:.4f}")
    print(f"  p nho nhat kha di voi n = {len(a)} va {len(b)}: {p_min:.4f}")
    print("  => phan tach hoan toan, nhung khong dat muc 0.05 THUAN TUY vi co mau.")


if __name__ == '__main__':
    part1()
    part2()
