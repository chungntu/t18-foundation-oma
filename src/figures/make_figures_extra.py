"""
Tam hinh bo sung cho bai bao (hinh 3, 4, 5, 6, 8, 9, 12, 13).
Bon hinh con lai do make_figures.py va fig_harmonics.py sinh ra.

  fig03_sensor_layout.png   - so do bo tri cam bien tren dai            (2.3)
  fig04_record_quality.png  - ban ghi tieu bieu: chuoi thoi gian + pho  (2.3)
  fig05_mechanisms.png      - so do hai co che lu ngang va lac          (3.1)
  fig06_criterion.png       - tieu chi khong thu nguyen                 (3.2)
  fig08_stabilisation.png   - bieu do on dinh SSI-COV kem FDD           (4.1)
  fig09_participation.png   - so kenh huy dong hieu dung, hai dai tan   (4.2)
  fig12_low_band.png        - tim kiem duoi 0.5 Hz                      (4.2)
  fig13_frequency_shift.png - tan so theo cau hinh do, hai dot          (5.3)

Chay:  python src/figures/make_figures_extra.py
"""

import os
import sys
import numpy as np
import scipy.io as sio
from scipy import signal, linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "oma"))
from paths import DATA, FIGURES, save_figure
from foundation_stiffness import (RINGS, D_CAP, D_OUT, batter_angle,
                                  dimensionless_ratio, rock_sway_ratio)
from oma_lib import (preprocess, ssi_cov, filter_poles, stable_poles,
                     cluster_modes, fdd)

FS_RAW = 1651.0
BLUE, RED, GREY = "#1f4e79", "#c00000", "0.55"

NODES = {1: (0.0, 0.0), 2: (-5.0, 0.0), 3: (-5.0, 5.0), 4: (0.0, 5.0), 5: (-2.5, 2.5)}
# huong do cua tung kenh trong tung cau hinh, doc tu log MACEC
DIR_VERTICAL = {1: "V", 2: "V", 3: "V", 4: "V", 5: "V"}
DIR_LATERAL = {1: "H", 2: "H", 3: "H", 4: "V", 5: "H"}


# ============================================================ HINH 3
def fig_sensor_layout():
    """(a) mat bang: kenh 1-5 tren mat dai.  (b) mat dung: kenh 6 tren thap,
    cao hon mat dai 2 m -- tren mat bang no se de len kenh 5 nen phai tach ra."""
    CAP_BOT, CAP_TOP = 5.435, 9.435
    R_TOWER = 2.25                      # ban kinh chan thap, ve minh hoa
    H6 = 2.0                            # kenh 6 cao hon mat dai 2 m

    # panel (b) rong gap ~1.4 lan chieu cao cua no; dat width_ratios theo do
    # thi hai panel cao bang nhau khi ca hai deu aspect="equal"
    fig, (axp, axe) = plt.subplots(
        1, 2, figsize=(12.6, 5.4), gridspec_kw=dict(width_ratios=[1.0, 1.40]))

    # ---------------------------------------------------------- (a) mat bang
    R = D_CAP / 2
    axp.add_patch(Circle((0, 0), R, fc="#f6f6f6", ec="k", lw=1.5, zorder=0))
    for r, n, nb in RINGS:
        axp.add_patch(Circle((0, 0), r, fc="none", ec="0.8", lw=0.7,
                             ls=(0, (5, 4)), zorder=1))
        for k in range(n):
            t = 2 * np.pi * k / n
            axp.add_patch(Circle((r * np.cos(t), r * np.sin(t)), D_OUT / 2,
                                 fc="white", ec="0.7", lw=0.8, zorder=2))

    cx, cy = NODES[5]
    Q = {k: (x - cx, y - cy) for k, (x, y) in NODES.items()}
    axp.plot([Q[k][0] for k in (1, 2, 3, 4, 1)],
             [Q[k][1] for k in (1, 2, 3, 4, 1)], color=BLUE, lw=2.0, zorder=4)
    for nd in (1, 2, 3, 4, 5):
        x, y = Q[nd]
        axp.plot(x, y, "o", mfc="white", mec=BLUE, mew=2.0, ms=13, zorder=5)
        axp.text(x, y, str(nd), ha="center", va="center", fontsize=8.5,
                 color=BLUE, zorder=6)

    axp.annotate("", xy=(2.5, 0), xytext=(-2.5, 0),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.0))
    axp.text(0, -0.75, "5 m", ha="center", fontsize=8, color=BLUE)
    axp.set_xlim(-R - 0.8, R + 0.8)
    axp.set_ylim(-R - 1.0, R + 0.8)
    axp.set_aspect("equal")
    axp.axis("off")
    axp.text(0.0, 1.0, "(a)", transform=axp.transAxes, fontsize=10,
             fontweight="bold", va="bottom", ha="left")

    # ---------------------------------------------------------- (b) mat dung
    axe.add_patch(Rectangle((-R, CAP_BOT), D_CAP, CAP_TOP - CAP_BOT,
                            fc="#e9e9e9", ec="k", lw=1.4, zorder=1))
    # thap ve dang cat ngang: hai thanh ben, khong co nap, ky hieu cat o dau
    YT = CAP_TOP + 7.4
    axe.fill_between([-R_TOWER, R_TOWER], CAP_TOP, YT, color="#f6f6f6", zorder=1)
    for sgn in (-1, 1):
        axe.plot([sgn * R_TOWER] * 2, [CAP_TOP, YT], color="0.45", lw=1.2,
                 zorder=2)
    xb = np.linspace(-R_TOWER, R_TOWER, 9)
    yb = YT + 0.42 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0])
    axe.plot(xb, yb, color="0.45", lw=1.2, zorder=3)
    axe.text(0, YT + 1.2, "tower", ha="center", fontsize=8.5, color="0.35")

    # dau coc duoi dai, ve dung duong kinh D_OUT va cat ngang o cuoi:
    # coc that dai 46 m nen khong the ve tron trong khung nay
    L_STUB = 5.0
    for r, nb in ((9.0, 5.0), (6.0, 7.0), (3.0, None)):
        t = 0.0 if nb is None else np.arctan(1.0 / nb)
        for sgn in (-1, 1):
            head = np.array([sgn * r, CAP_BOT])
            d = np.array([sgn * np.sin(t), -np.cos(t)])     # doc truc coc
            n = np.array([np.cos(t), sgn * np.sin(t)])      # phap tuyen
            a = head + 0.5 * D_OUT * n
            b = head - 0.5 * D_OUT * n
            axe.add_patch(Polygon([a, b, b + L_STUB * d, a + L_STUB * d],
                                  closed=True, fc="#dcdcdc", ec="0.45",
                                  lw=0.9, zorder=0))
            # ky hieu cat vuong goc voi truc coc
            m = head + L_STUB * d
            zz = np.array([m + (u * 0.5 * D_OUT) * n + v * 0.28 * d
                           for u, v in ((-1, 0), (-0.5, 1), (0, -1),
                                        (0.5, 1), (1, 0))])
            axe.plot(zz[:, 0], zz[:, 1], color="0.45", lw=1.0, zorder=1)

    # kenh 1-4 chieu len mat dung tai +-2.5 m, kenh 5 o tam
    for x, lab in ((-2.5, "1, 2"), (2.5, "3, 4")):
        axe.plot(x, CAP_TOP, "o", mfc="white", mec=BLUE, mew=1.8, ms=8, zorder=5)
        axe.text(x, CAP_TOP - 0.85, lab, ha="center", fontsize=8, color=BLUE)
    axe.plot(0, CAP_TOP, "o", mfc="white", mec=BLUE, mew=2.0, ms=13, zorder=5)
    axe.text(0, CAP_TOP, "5", ha="center", va="center", fontsize=8.5,
             color=BLUE, zorder=6)

    # kenh 6 tren thanh thap
    axe.plot(R_TOWER, CAP_TOP + H6, "o", mfc="white", mec=BLUE, mew=2.0,
             ms=13, zorder=5)
    axe.text(R_TOWER, CAP_TOP + H6, "6", ha="center", va="center",
             fontsize=8.5, color=BLUE, zorder=6)
    axe.annotate("", xy=(R_TOWER + 1.9, CAP_TOP + H6),
                 xytext=(R_TOWER + 1.9, CAP_TOP),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.0))
    axe.text(R_TOWER + 2.2, CAP_TOP + H6 / 2, "2 m", va="center",
             fontsize=8, color=BLUE)

    for y in (CAP_TOP, CAP_BOT):
        axe.plot([-R, R + 1.3], [y, y], color="0.6", lw=0.6, ls=(0, (4, 3)),
                 zorder=0)
        axe.text(-R - 0.4, y, "%+.3f" % y, ha="right", va="center", fontsize=7.5,
                 color="0.3")
    axe.annotate("", xy=(R, CAP_BOT - 5.8), xytext=(-R, CAP_BOT - 5.8),
                 arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
    axe.text(0, CAP_BOT - 6.6, "20.5 m", ha="center", fontsize=8, color="0.3")

    axe.set_xlim(-R - 2.6, R + 2.6)
    axe.set_ylim(CAP_BOT - 7.2, CAP_TOP + 10.2)
    axe.set_aspect("equal")
    axe.axis("off")
    axe.text(0.0, 1.0, "(b)", transform=axe.transAxes, fontsize=10,
             fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    _save(fig, "fig03_sensor_layout.png")


# ============================================================ HINH 4
def fig_record_quality():
    x = sio.loadmat(os.path.join(DATA, "dot2", "xl_ngang.mat"))["xl_ngang"].astype(float)
    x = signal.detrend(x, axis=0, type="constant")
    ch = int(np.argmax(np.abs(x).max(axis=0) / x.std(axis=0)))
    t = np.arange(x.shape[0]) / FS_RAW
    peak = np.abs(x[:, ch]).max() / x[:, ch].std()

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.4))

    # chuan hoa theo do lech chuan cua chinh kenh do -> truc doc doc duoc bang sigma
    v = x[:, ch] / x[:, ch].std()
    CLIP = 14.0
    axes[0].plot(t, v, lw=0.35, color=BLUE)
    axes[0].axhline(8, color=RED, lw=0.8, ls="--")
    axes[0].axhline(-8, color=RED, lw=0.8, ls="--")
    axes[0].set_ylim(-CLIP, CLIP)

    # su kien lon nhat vuot khoi khung: danh dau thay vi de no ep het phan con lai
    ipk = int(np.argmax(np.abs(v)))
    tpk = ipk / FS_RAW
    axes[0].annotate(f"{np.abs(v[ipk]):.0f}$\\sigma$",
                     xy=(tpk, CLIP), xytext=(tpk - 55, CLIP * 0.78),
                     ha="right", fontsize=8.5, color=RED,
                     arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0,
                                     shrinkA=2, shrinkB=0))
    axes[0].set_ylabel("acceleration / $\\sigma$")
    axes[0].set_xlabel("Time [s]", fontsize=9)
    axes[0].set_xlim(t[0], t[-1])

    # RMS truot -> tinh khong dung
    win = int(20 * FS_RAW)
    nseg = x.shape[0] // win
    tt = (np.arange(nseg) + 0.5) * win / FS_RAW
    for c in range(x.shape[1]):
        rms = np.array([x[i*win:(i+1)*win, c].std() for i in range(nseg)])
        axes[1].plot(tt, rms / rms.mean(), lw=1.0, label=f"ch {c+1}")
    axes[1].set_ylabel("RMS / mean RMS")
    axes[1].set_xlabel("Time [s]", fontsize=9)
    axes[1].legend(fontsize=7, ncol=6, loc="upper left")
    axes[1].set_xlim(t[0], t[-1])

    f, P = signal.welch(x, fs=FS_RAW, nperseg=2**16, axis=0)
    axes[2].semilogy(f, P.mean(axis=1), lw=0.6, color=BLUE)
    for k in range(1, 17):
        axes[2].axvline(50 * k, color="#00a000", ls=":", lw=0.8)
    axes[2].axvspan(0.2, 5, color=RED, alpha=.12)
    axes[2].set_xlim(0, 800)
    axes[2].set_xlabel("Frequency [Hz]", fontsize=9)
    axes[2].set_ylabel("PSD")

    for k, a in enumerate(axes):
        a.text(0.0, 1.015, "(%s)" % "abc"[k], transform=a.transAxes,
               fontsize=10, fontweight="bold", va="bottom", ha="left")
        a.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, "fig04_record_quality.png")


def _kurt(v):
    v = v - v.mean()
    return np.mean(v ** 4) / np.mean(v ** 2) ** 2


# ============================================================ HINH 5
def fig_mechanisms():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for k, (ax, mode) in enumerate(zip(axes, ("sway", "rock"))):
        _draw_frame(ax, mode, "ab"[k])
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig05_mechanisms.png")


def _draw_frame(ax, mode, panel):
    W, Hc, Lp, rake = 10.0, 1.6, 9.0, 5.0
    ax.add_patch(Rectangle((-W/2, 0), W, Hc, fc="0.8", ec="k", lw=1.3, zorder=3))
    for x0 in (-4.0, -2.0, 2.0, 4.0):
        dx = np.sign(x0) * Lp / rake
        ax.plot([x0, x0 + dx], [0, -Lp], color="k", lw=2.2, zorder=2, solid_capstyle="butt")
    ax.plot([-W/2 - 1.2, W/2 + 1.2], [-Lp, -Lp], color="0.4", lw=1.0, ls="--")

    d = 1.1
    if mode == "sway":
        ax.add_patch(Rectangle((-W/2 + d, 0), W, Hc, fc="none", ec=BLUE, lw=1.3,
                               ls=(0, (4, 3)), zorder=4))
        ax.add_patch(FancyArrow(0, Hc + 0.9, d, 0, width=0.09, head_width=0.5,
                                head_length=0.45, fc=BLUE, ec=BLUE,
                                length_includes_head=True, zorder=5))
        ax.text(d/2, Hc + 1.6, "translation $u$", ha="center", fontsize=9.5, color=BLUE)
        sign = {-4.0: "T", -2.0: "T", 2.0: "C", 4.0: "C"}
        title = ("Sway\nresisted by the axial stiffness of the inclined piles,\n"
                 "$k_{lat} = k_{ax}\\,\\Sigma\\sin^2\\alpha + (12EI/L^3)\\,\\Sigma\\cos^2\\alpha$")
    else:
        th = 0.10
        for xx, yy in [(-W/2, 0), (W/2, 0), (W/2, Hc), (-W/2, Hc)]:
            pass
        corners = np.array([[-W/2, 0], [W/2, 0], [W/2, Hc], [-W/2, Hc]])
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        ax.add_patch(Polygon((rot @ corners.T).T, closed=True, fc="none", ec=BLUE,
                             lw=1.3, ls=(0, (4, 3)), zorder=4))
        ax.annotate("", xy=(2.0, Hc + 1.6), xytext=(-2.0, Hc + 0.6),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4,
                                    connectionstyle="arc3,rad=0.35"))
        ax.text(0, Hc + 2.2, "rotation $\\theta$", ha="center", fontsize=9.5, color=BLUE)
        sign = {-4.0: "C", -2.0: "C", 2.0: "T", 4.0: "T"}
        title = ("Rocking\nresisted by the same axial stiffness acting on the lever arms,\n"
                 "$k_{\\theta} = k_{ax}\\,\\Sigma r^2\\cos^2\\alpha$")

    for x0, s in sign.items():
        col = RED if s == "T" else "#006000"
        ax.text(x0 + np.sign(x0) * 1.35, -Lp / 2, s, fontsize=11, color=col,
                ha="center", va="center", fontweight="bold")

    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-Lp - 2.2, Hc + 3.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.0, 1.015, "(%s)" % panel, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


# ============================================================ HINH 6
def fig_criterion():
    d = dimensionless_ratio()
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    a = np.linspace(np.radians(1.5), np.radians(60), 400)
    for shape, col in zip([0.8, 1.0, 1.295, 1.6, 2.0],
                          ["#c6dbef", "#6baed6", BLUE, "#2171b5", "#08306b"]):
        lab = f"$r_{{rms}}/k_g$ = {shape:.2f}" + ("  (T18)" if abs(shape - d['shape']) < 1e-3 else "")
        ax.semilogy(np.degrees(a), shape / np.tan(a), color=col, lw=2.0 if "T18" in lab else 1.3,
                    label=lab)

    ax.axhline(1.0, color="k", lw=1.0)
    ax.text(24, 1.06, "rocking = sway", fontsize=8.5, ha="center")
    ax.plot(d['alpha_crit_deg'], 1.0, 'o', color=RED, ms=7, zorder=5)
    ax.annotate(f"$\\alpha_{{crit}}$ = {d['alpha_crit_deg']:.0f}°",
                xy=(d['alpha_crit_deg'], 1.0), xytext=(d['alpha_crit_deg'] - 3, 0.55),
                fontsize=9, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))

    ax.axvspan(np.degrees(np.arctan(1/12)), np.degrees(np.arctan(1/4)),
               color="#00a000", alpha=.10)
    exact = rock_sway_ratio()['ratio_no_bending']
    ax.plot(np.degrees(np.arctan(1 / 5.94)), exact, '*', color=RED, ms=14, zorder=6)
    ax.text(np.degrees(np.arctan(1 / 5.94)) + 1.2, exact, f"  T18: {exact:.2f}",
            fontsize=9, color=RED, va="center")

    ax.set_xlabel("Pile angle of inclination from vertical, $\\alpha$ [degrees]")
    ax.set_ylabel("$f_{rock} / f_{sway}$")
    ax.set_xlim(0, 60)
    ax.set_ylim(0.4, 60)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    _save(fig, "fig06_criterion.png")


# ============================================================ HINH 8
def fig_stabilisation():
    y, fs = preprocess(sio.loadmat(os.path.join(DATA, "dot1", "xl_thangdung.mat"))
                       ["xl_thangdung"].astype(float), FS_RAW, dec_factors=(4, 8), f_high=0.3)
    dt, orders = 1 / fs, list(range(2, 82, 2))
    res = filter_poles(ssi_cov(y, dt, 120, orders), (0.5, 25.0), xi_range=(0.001, 0.15))
    stab = stable_poles(res)
    modes = cluster_modes(res, stab, min_count=8)
    f_fdd, sv = fdd(y, fs, nperseg=4096)

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for n in sorted(res):
        f, s = res[n]['f'], stab[n]
        ax.plot(f[~s], [n] * int((~s).sum()), '.', color="#bbbbbb", ms=4)
        ax.plot(f[s], [n] * int(s.sum()), '+', color=RED, ms=7, mew=1.2)
    ax2 = ax.twinx()
    m = (f_fdd >= 0.5) & (f_fdd <= 25)
    ax2.semilogy(f_fdd[m], sv[m, 0], color=BLUE, lw=0.9, alpha=.85)
    ax2.set_ylabel("First singular value of the PSD matrix", fontsize=9, color=BLUE)
    for mo in modes:
        ax.axvline(mo['f'], color="#008000", ls="--", lw=0.8)
    ax.set_xlim(0.5, 25)
    ax.set_ylim(0, 84)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Model order")
    fig.tight_layout()
    _save(fig, "fig08_stabilisation.png")


# ============================================================ HINH 9
# gia tri tu src/oma/participation.py
PART = [("Vertical, C1", 5, 1.55, 1.00), ("Lateral 1, C1", 5, 4.69, 1.00),
        ("Lateral 2, C1", 5, 1.00, 1.01), ("Three-channel, C1", 3, 1.79, 1.84),
        ("Lateral, C2", 6, 5.00, 1.17), ("Vertical d1, C2", 6, 2.41, 1.00),
        ("Vertical d2ke, C2", 6, 2.95, 3.64)]


def fig_participation():
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    idx = np.arange(len(PART))
    w = 0.38
    low = [p[2] for p in PART]
    high = [p[3] for p in PART]
    ax.bar(idx - w/2, low, w, color=BLUE, label="1.0–1.4 Hz band")
    ax.bar(idx + w/2, high, w, color="#bbbbbb", edgecolor="0.5", label="13–24 Hz band")
    ax.axhline(1.25, color=RED, lw=1.4, ls="--",
               label="rejection threshold $N_{eff}$ = 1.25")
    ax.set_xticks(idx)
    ax.set_xticklabels(["%s  (n=%d)" % (p[0], p[1]) for p in PART],
                       fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("Effective channel count $N_{eff}$")
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    _save(fig, "fig09_participation.png")


# ============================================================ HINH 12
def fig_low_band():
    cases = [("dot1/xl_thangdung.mat", "xl_thangdung", "Vertical, campaign 1", BLUE),
             ("dot1/xl_ngang1.mat", "xl_ngang1", "Lateral 1, campaign 1", "#6baed6"),
             ("dot2/xl_ngang.mat", "xl_ngang", "Lateral, campaign 2", RED),
             ("dot2/xl_td2.mat", "xl_td2", "Vertical d2ke, campaign 2", "#e07070")]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for rel, var, lab, col in cases:
        x = sio.loadmat(os.path.join(DATA, *rel.split("/")))[var].astype(float)
        y, fs = preprocess(x, FS_RAW, dec_factors=(4, 8, 4), f_high=0.12)
        f, P = signal.csd(y[:, :, None], y[:, None, :], fs=fs, nperseg=2048, axis=0)
        sv1 = np.array([linalg.svdvals(P[k])[0] for k in range(len(f))])
        m = (f >= 0.15) & (f <= 1.6)
        ax.semilogy(f[m], sv1[m] / sv1[m].max(), lw=1.2, color=col, label=lab)

    ax.axvspan(1.05, 1.33, color=BLUE, alpha=.13,
               label="admitted band, 1.05\u20131.33 Hz")
    ax.axvspan(0.18, 0.26, color="#00a000", alpha=.16,
               label="wave peak, campaign 1 (0.21 Hz)")
    ax.axvspan(0.28, 0.36, color="#f0a000", alpha=.22,
               label="wave peak, campaign 2 (0.32 Hz)")
    ax.axvline(0.5, color="k", lw=1.0, ls="--",
               label="lower edge of the analysis band")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("First singular value, normalised")
    ax.set_xlim(0.15, 1.6)
    ax.legend(fontsize=7.8, loc="lower right", ncol=2, framealpha=0.92)
    ax.grid(alpha=.25, which="major")
    fig.tight_layout()
    _save(fig, "fig12_low_band.png")


# ============================================================ HINH 13
C1 = {"Vertical": [1.1734], "Lateral 1": [1.2258, 1.2536],
      "Lateral 2": [1.3217], "Three-channel": [1.2370, 1.3307]}
C2 = {"Lateral": [1.0520, 1.2146], "Vertical d1": [1.1465],
      "Vertical d2ke": [1.0735, 1.1985, 1.2305]}


def fig_frequency_shift():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for i, (name, vals) in enumerate(C1.items()):
        ax.plot(vals, [i] * len(vals), 'o', mfc="none", mec=BLUE, ms=6)
        ax.plot(np.mean(vals), i, 's', color=BLUE, ms=9)
        ax.text(1.36, i, name, fontsize=8.5, va="center", color=BLUE)
    off = len(C1) + 0.6
    for i, (name, vals) in enumerate(C2.items()):
        ax.plot(vals, [off + i] * len(vals), 'o', mfc="none", mec=RED, ms=6)
        ax.plot(np.mean(vals), off + i, 's', color=RED, ms=9)
        ax.text(1.36, off + i, name, fontsize=8.5, va="center", color=RED)

    a = np.array([np.mean(v) for v in C1.values()])
    b = np.array([np.mean(v) for v in C2.values()])
    ax.axvline(a.mean(), color=BLUE, lw=1.2, ls="--")
    ax.axvline(b.mean(), color=RED, lw=1.2, ls="--")
    ax.axvspan(b.max(), a.min(), color="0.85", zorder=0)
    ax.annotate("", xy=(b.mean(), -0.75), xytext=(a.mean(), -0.75),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
    ax.text((a.mean() + b.mean()) / 2, -0.62,
            f"{100*(b.mean()-a.mean())/a.mean():+.1f} %", ha="center", fontsize=9)

    ax.plot([], [], 'o', mfc="none", mec="0.4", label="individual pole")
    ax.plot([], [], 's', color="0.4", label="setup mean")
    ax.set_yticks([])
    ax.set_ylim(-1.5, off + len(C2) - 0.3)
    ax.set_xlim(1.02, 1.47)
    ax.set_xlabel("Frequency [Hz]")
    ax.text(1.04, len(C1) - 0.5, "campaign 1\nbefore tower", fontsize=9, color=BLUE)
    ax.text(1.04, off + len(C2) - 1.2, "campaign 2\nafter tower", fontsize=9, color=RED)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    _save(fig, "fig13_frequency_shift.png")


def _save(fig, name):
    p = os.path.join(FIGURES, name)
    save_figure(fig, p)
    plt.close(fig)
    print("[luu]", name)


if __name__ == '__main__':
    fig_sensor_layout()
    fig_mechanisms()
    fig_criterion()
    fig_participation()
    fig_frequency_shift()
    fig_record_quality()
    fig_stabilisation()
    fig_low_band()
