"""
Pipeline dung chung cho ca hai dot do: tien xu ly -> SSI-COV -> FDD -> hinh + bang.

Dung boi run_dot1.py va run_dot2.py de hai dot duoc xu ly HOAN TOAN giong nhau
(dieu kien bat buoc de so sanh tan so truoc/sau khi lap thap).
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oma_lib import (preprocess, ssi_cov, filter_poles, stable_poles,
                     cluster_modes, fdd, peak_pick, mac, harmonic_indicator)

# --- tham so xu ly, DUNG CHUNG cho moi dot do -------------------------------
FS_RAW = 51200 / 31            # Hz = 1651.61; wf_increment = 31/51200 s trong metadata TDMS
DEC = (4, 8)                 # 1651 / 32 = 51.59 Hz
F_HIGH = 0.3                 # Hz, loc thong cao
F_RANGE = (0.5, 25.0)        # dai tim mode
I_BLOCK = 120                # cua so tre = 2.33 s
ORDERS = list(range(2, 82, 2))
XI_RANGE = (0.001, 0.15)
MIN_COUNT = 8


def analyze(path, var, label):
    """Doc mot setup, tien xu ly, SSI-COV + FDD. Tra ve dict ket qua."""
    print(f"\n{'=' * 72}\n{label}  ({os.path.basename(path)})\n{'=' * 72}")
    raw = sio.loadmat(path)[var].astype(float)
    print(f"  tho    : {raw.shape[0]} mau x {raw.shape[1]} kenh "
          f"= {raw.shape[0] / FS_RAW / 60:.1f} phut @ {FS_RAW:.0f} Hz")

    y, fs = preprocess(raw, FS_RAW, dec_factors=DEC, f_high=F_HIGH)
    dt = 1.0 / fs
    print(f"  sau xl : {y.shape[0]} mau @ {fs:.2f} Hz, cua so tre = {I_BLOCK * dt:.2f} s")

    res = ssi_cov(y, dt, I_BLOCK, ORDERS)
    res = filter_poles(res, F_RANGE, xi_range=XI_RANGE)
    stab = stable_poles(res)
    modes = cluster_modes(res, stab, min_count=MIN_COUNT)

    f_fdd, sv = fdd(y, fs, nperseg=4096)
    fpk, _ = peak_pick(f_fdd, sv[:, 0], F_RANGE)

    for m in modes:
        m['kurt'] = harmonic_indicator(y, fs, m['f'])
        m['fdd'] = float(fpk[np.argmin(np.abs(fpk - m['f']))]) if len(fpk) else np.nan

    print(f"\n  {'f [Hz]':>9} {'xi [%]':>8} {'sd_f':>7} {'n_cuc':>6} "
          f"{'MPC':>6} {'MPD':>6} {'kurt':>6} {'f_FDD':>8}")
    for m in modes:
        print(f"  {m['f']:9.3f} {100 * m['xi']:8.2f} {m['f_std']:7.4f} {m['count']:6d} "
              f"{m['mpc']:6.3f} {m['mpd']:6.1f} {m['kurt']:6.2f} {m['fdd']:8.3f}")
    print(f"\n  Dinh FDD: {np.round(fpk, 3)}")

    return dict(label=label, var=var, y=y, fs=fs, res=res, stab=stab,
                modes=modes, f_fdd=f_fdd, sv=sv, fpk=fpk, raw_shape=raw.shape)


# --- ve hinh ----------------------------------------------------------------

def plot_stabilization(r, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for n in sorted(r['res']):
        f, s = r['res'][n]['f'], r['stab'][n]
        ax.plot(f[~s], [n] * int((~s).sum()), '.', color='#bbbbbb', ms=4)
        ax.plot(f[s], [n] * int(s.sum()), '+', color='#c00000', ms=7, mew=1.2)

    ax2 = ax.twinx()
    m = (r['f_fdd'] >= F_RANGE[0]) & (r['f_fdd'] <= F_RANGE[1])
    ax2.semilogy(r['f_fdd'][m], r['sv'][m, 0], color='#1f4e79', lw=0.9, alpha=.8)
    ax2.set_ylabel('First singular value of the PSD matrix (FDD)', fontsize=9, color='#1f4e79')

    for mo in r['modes']:
        ax.axvline(mo['f'], color='#008000', ls='--', lw=0.8)
        ax.text(mo['f'], max(ORDERS) * 1.02, f"{mo['f']:.2f}", rotation=90,
                fontsize=7, ha='center', va='bottom', color='#008000')

    ax.set_xlim(*F_RANGE)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Model order')
    ax.set_title(f"SSI-COV stabilisation diagram — {r['label']}\n"
                 "(red +: stable pole; grey dot: unstable pole; green dashed: selected mode)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fdd(results, path, fmax=25):
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 2.8 * len(results)))
    for ax, r in zip(np.atleast_1d(axes), results):
        m = (r['f_fdd'] >= 0.3) & (r['f_fdd'] <= fmax)
        ax.semilogy(r['f_fdd'][m], r['sv'][m, 0], lw=.9, color='#1f4e79', label='SV1')
        ax.semilogy(r['f_fdd'][m], r['sv'][m, 1], lw=.6, color='#7f7f7f', alpha=.7, label='SV2')
        for mo in r['modes']:
            ax.axvline(mo['f'], color='#c00000', ls='--', lw=.8)
        ax.set_title(f"{r['label']} — FDD, structural band 0.3-{fmax} Hz", fontsize=9)
        ax.set_xlabel('Frequency [Hz]', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- xuat bang --------------------------------------------------------------

def write_csv(results, path):
    """Ghi bang ket qua. Dung module csv de nhan setup co dau phay duoc trich dan dung."""
    import csv
    cols = ["setup", "mode", "f_Hz", "f_std_Hz", "xi_percent", "xi_std_percent",
            "n_cuc_on_dinh", "bac", "MPC", "MPD_deg", "kurtosis", "f_FDD_Hz"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            for k, m in enumerate(r["modes"], 1):
                w.writerow([r["label"], k, f"{m['f']:.4f}", f"{m['f_std']:.4f}",
                            f"{100 * m['xi']:.3f}", f"{100 * m['xi_std']:.3f}",
                            m["count"], m["order"], f"{m['mpc']:.4f}",
                            f"{m['mpd']:.2f}", f"{m['kurt']:.3f}", f"{m['fdd']:.4f}"])
    print()
    print("[luu]", path)


def mac_matrix(m1, m2):
    return np.array([[mac(a['phi'], b['phi']) for b in m2] for a in m1])


def print_mac(r1, r2, path=None):
    M = mac_matrix(r1['modes'], r2['modes'])
    print(f"\n{'=' * 72}\nMAC: {r1['label']}  vs  {r2['label']}\n{'=' * 72}")
    if M.size == 0:
        print("  (khong co mode de so sanh)")
        return M
    print("        " + " ".join(f"{m['f']:7.2f}" for m in r2['modes']))
    for i, a in enumerate(r1['modes']):
        print(f"{a['f']:7.2f} " + " ".join(f"{v:7.3f}" for v in M[i]))
    if path:
        np.savetxt(path, M, delimiter=',', fmt='%.4f')
    return M
