"""
Dinh nguong va kiem do nhay cua tieu chi N_eff (muc 5.1 bai bao).

  PHAN 1. Dinh nguong bang mo phong.
     - H0 "cuc bo": chi mot kenh chua tin hieu dai hep, cac kenh con lai la nhieu
     - H1 "toan cuc": moi kenh chia se mot tin hieu chung theo dang co duong nut
     Tu hai phan bo suy ra nguong va ty le bao dong gia.

  PHAN 2. Do nhay cua N_eff voi sai lech he so khuech dai giua cac kenh
     (he so chuan cam bien trong log MACEC lech nhau toi 3 %) va voi muc
     nhieu khac nhau giua cac kenh.

Chay:  python oma/criterion_calibration.py
"""

import numpy as np
from scipy import signal, linalg

RNG = np.random.default_rng(20260927)

FS = 51200 / 31 / 32  # Hz = 51.61, sau khi giam tan so lay mau
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
    print("PHAN 2. DO NHAY CUA N_eff VOI HE SO KHUECH DAI VA MUC NHIEU")
    print("=" * 76)
    shape = np.array([1.0, 0.35, -0.27, -0.25, -0.05])
    base = n_eff(shape)
    # (a) he so khuech dai ngau nhien trong +-3 %, ap dung truc tiep len dang mode
    g = RNG.uniform(0.97, 1.03, size=(20000, 5))
    ne = np.array([n_eff(shape * gi) for gi in g])
    print(f"  (a) dang co duong nut, N_eff = {base:.3f}; he so lech +-3 %:"
          f" N_eff {ne.min():.3f} - {ne.max():.3f} (lech toi da {100*np.abs(ne/base-1).max():.1f} %)")
    # (b) H0 va H1 khi muc nhieu cac kenh khac nhau toi 10 lan
    for nch, w in ((5, shape),):
        h0, h1 = [], []
        for _ in range(N_TRIAL):
            lev = np.exp(RNG.uniform(0, np.log(10), nch))
            y0 = simulate(nch, None, 3.0) * lev
            y1 = simulate(nch, w, 3.0) * lev
            h0.append(neff_from_series(y0, FS))
            h1.append(neff_from_series(y1, FS))
        h0, h1 = np.array(h0), np.array(h1)
        print(f"  (b) n = {nch}, nhieu kenh lech toi 10 lan, SNR = 3:")
        print(f"      H0: bach phan vi 95 = {np.quantile(h0, .95):.2f};"
              f"  H1: bach phan vi 5 = {np.quantile(h1, .05):.2f};"
              f"  ty le H1 duoi 1.25 = {100*(h1 < 1.25).mean():.1f} %;"
              f"  ty le H0 tren 1.25 = {100*(h0 > 1.25).mean():.1f} %")


if __name__ == '__main__':
    part1()
    part2()
