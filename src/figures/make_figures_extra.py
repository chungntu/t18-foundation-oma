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
from paths import DATA, FIGURES, RESULTS, save_figure
from foundation_stiffness import RINGS, D_CAP, D_OUT, batter_angle
from oma_lib import (preprocess, ssi_cov, filter_poles, stable_poles,
                     cluster_modes, fdd, band_poles_by_setup)

FS_RAW = 51200 / 31            # Hz = 1651.61; wf_increment = 31/51200 s trong metadata TDMS
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4))
    for k, (ax, mode) in enumerate(zip(axes, ("focal", "rock"))):
        _draw_frame(ax, mode, "ab"[k])
    fig.tight_layout()
    _save(fig, "fig05_mechanisms.png")


def _rot(pts, c, th):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (R @ (pts - c).T).T + c


def _draw_frame(ax, mode, panel):
    W, Hc, Lp = 10.0, 1.6, 9.0
    PILES = [(-4.0, 5.0), (-1.5, None), (1.5, None), (4.0, 5.0)]   # (x dau coc, do xien)
    zf = 4.0 * 5.0                                                  # tieu diem r cot(alpha)
    ax.add_patch(Rectangle((-W/2, 0), W, Hc, fc="0.8", ec="k", lw=1.3, zorder=3))
    for x0, n in PILES:
        dx = 0.0 if n is None else np.sign(x0) * Lp / n
        ax.plot([x0, x0 + dx], [0, -Lp], color="k", lw=2.2, zorder=2, solid_capstyle="butt")
    ax.plot([-W/2 - 1.2, W/2 + 1.2], [-Lp, -Lp], color="0.4", lw=1.0, ls="--")
    corners = np.array([[-W/2, 0], [W/2, 0], [W/2, Hc], [-W/2, Hc]])

    if mode == "focal":
        for x0, n in PILES:
            if n is not None:
                ax.plot([x0, 0], [0, zf], color=GREY, lw=0.9, ls=(0, (5, 3)), zorder=1)
        c, th = np.array([0.0, zf]), -0.055
        ax.plot(*c, 'o', color=RED, ms=7, zorder=5)
        ax.text(0.5, zf, "F, where the pile axes meet\n$z_f = r\\cot\\alpha$",
                fontsize=13, color=RED, va="center")
        ax.annotate("", xy=_rot(np.array([[0.0, zf - 4.0]]), c, -0.25)[0],
                    xytext=_rot(np.array([[0.0, zf - 4.0]]), c, 0.25)[0],
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3,
                                    connectionstyle="arc3,rad=0.2"))
        ax.text(1.4, Hc + 1.2, "nearly a translation at the cap,\nresisted mainly by pile bending",
                fontsize=13, color=BLUE)
        ylim = (-Lp - 1.5, zf + 2.0)
    else:
        c, th = np.array([0.0, 0.0]), 0.10
        ax.plot(*c, 'o', color=RED, ms=6, zorder=6)
        ax.annotate("", xy=(2.0, Hc + 1.6), xytext=(-2.0, Hc + 0.6),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4,
                                    connectionstyle="arc3,rad=0.35"))
        ax.text(0, Hc + 2.3, "rotation $\\varphi$ about a centre near the pile heads,\n"
                "resisted by the axial pile stiffness", ha="center", fontsize=13, color=BLUE)
        for x0, s in ((-4.0, "C"), (-1.5, "C"), (1.5, "T"), (4.0, "T")):
            col = RED if s == "T" else "#006000"
            ax.text(x0 + np.sign(x0) * 1.35, -Lp / 2, s, fontsize=14, color=col,
                    ha="center", va="center", fontweight="bold")
        ylim = (-Lp - 1.5, Hc + 4.5)

    ax.add_patch(Polygon(_rot(corners, c, th), closed=True, fc="none", ec=BLUE,
                         lw=1.3, ls=(0, (4, 3)), zorder=4))
    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.0, 1.0, "(%s)" % panel, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")


# ============================================================ HINH 6
def fig_criterion():
    import rigid_cap_6dof as rc
    import foundation_stiffness as fsm
    Le = np.linspace(fsm.L_FREE + 1 * fsm.D_OUT, fsm.L_FREE + 14 * fsm.D_OUT, 40)
    K_LO, K_HI = 2.5e8, 5.8e8
    S = {k: [rc.summary(k, L) for L in Le] for k in (K_LO, K_HI)}

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for k, ls in ((K_LO, "-"), (K_HI, "--")):
        lab = f"$k_{{ax}}$ = {k:.1e} N/m"
        a.semilogy(Le, [s["f1"] for s in S[k]], ls, color=BLUE, lw=1.8, label="lowest lateral pair, " + lab)
        a.semilogy(Le, [s["f_rock"] for s in S[k]], ls, color="#7f3f00", lw=1.5, label="rocking pair, " + lab)
        a.semilogy(Le, [s["f_ver"] for s in S[k]], ls, color="0.6", lw=1.0, label="vertical, " + lab)
    a.semilogy(Le, [s["f_tor"] for s in S[K_LO]], ":", color="#006000", lw=2.0,
               label="torsion (pile bending only)")
    a.axhspan(1.05, 1.33, color=RED, alpha=.12, zorder=0)
    a.text(Le[-1], 1.38, "measured band 1.05-1.33 Hz", ha="right", fontsize=8.5, color=RED)
    a.set_xlabel("Effective fixity length $L_{eff}$ [m]  "
                 f"(free length in water {fsm.L_FREE:.2f} m)")
    a.set_ylabel("Natural frequency [Hz]")
    a.set_ylim(0.4, 20)
    a.grid(alpha=.3, which="both")
    a.legend(fontsize=7.4, loc="upper right")

    for d, ls in ((rc.FRAME_HALF, "-"), (rc.FRAME_HALF * np.sqrt(2), "--")):
        for k, col in ((K_LO, BLUE), (K_HI, "#08306b")):
            b.semilogy(Le, [rc.vh_ratio(s["zc1"], d) for s in S[k]], ls, color=col, lw=1.6)
            b.semilogy(Le, [rc.vh_ratio(s["zc2"], d) for s in S[k]], ls, color="#7f3f00", lw=1.3)
    b.plot([], [], "-", color=BLUE, label="lowest lateral pair")
    b.plot([], [], "-", color="#7f3f00", label="rocking pair")
    b.plot([], [], "-", color="0.4", label="$d$ = 2.5 m")
    b.plot([], [], "--", color="0.4", label="$d$ = 3.5 m")
    b.axhspan(0.08, 0.16, color=RED, alpha=.15, zorder=0)
    b.text(Le[0] + 0.3, 0.17, "measured, lateral setup of campaign 2", fontsize=8.5, color=RED)
    zf = max(rc.focal_heights().values())
    b.text(Le[-1], 0.8 * rc.vh_ratio(zf, rc.FRAME_HALF), f"bound from $z_f$ = {zf:.0f} m, $d$ = 2.5 m",
           ha="right", va="top", fontsize=8)
    b.axhline(rc.vh_ratio(zf, rc.FRAME_HALF), color="0.3", lw=0.8, ls=":")
    b.set_xlabel("Effective fixity length $L_{eff}$ [m]")
    b.set_ylabel("Vertical / horizontal amplitude on the cap top")
    b.set_ylim(0.01, 3)
    b.grid(alpha=.3, which="both")
    b.legend(fontsize=8, loc="center right")
    for ax, p in ((a, "a"), (b, "b")):
        ax.text(0.0, 1.01, f"({p})", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
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
               label="peak, campaign 1 (0.21 Hz)")
    ax.axvspan(0.28, 0.36, color="#f0a000", alpha=.22,
               label="peak, campaign 2 (0.32 Hz)")
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
C1, C2 = band_poles_by_setup(RESULTS)


# huong do: V = moi kenh dung; H = kenh ngang (kenh 4 dung o dot 2); M = hon hop
DIRECTION = {"Vertical": "V", "Lateral 1": "H", "Lateral 2": "H", "Three-channel": "M",
             "Lateral": "H", "Vertical d1": "V", "Vertical d2ke": "V"}


def fig_frequency_shift():
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    groups = [("V", "vertical setups"), ("H", "lateral setups"), ("M", "mixed directions")]
    y, yt, yl = 0.0, [], []
    for code, title in groups:
        ax.text(1.025, y - 0.1, title, fontsize=9, fontweight="bold", va="center")
        y -= 0.6
        for camp, d, col in ((1, C1, BLUE), (2, C2, RED)):
            for name, vals in d.items():
                if DIRECTION[name] != code:
                    continue
                ax.plot(vals, [y] * len(vals), 'o', mfc="none", mec=col, ms=6)
                ax.plot(np.mean(vals), y, 's', color=col, ms=8)
                yt.append(y)
                yl.append(f"C{camp}  {name}")
                y -= 0.6
        y -= 0.5
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8.5)
    for t, lab in zip(ax.get_yticklabels(), yl):
        t.set_color(BLUE if lab.startswith("C1") else RED)
    ax.plot([], [], 'o', mfc="none", mec="0.4", label="individual pole")
    ax.plot([], [], 's', color="0.4", label="setup mean")
    ax.set_xlim(1.02, 1.36)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylim(y + 0.6, 0.4)
    ax.legend(fontsize=8.5, loc="upper right")
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
