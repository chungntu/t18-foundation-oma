"""
Ve cac hinh cua bai bao (tru hinh 5, xem fig_harmonics.py).

  fig1_pile_layout.png   - mat bang 34 coc PHC D1000, ba vong, coc xien      (muc 4.1)
  fig2_borehole.png      - mat cat mong va dia tang DH9, kem SPT             (muc 4.2)
  fig3_sensitivity.png   - do nhay tan so lu ngang theo L_eff va k_ax        (muc 5.5)
  fig4_mode_shape.png    - dang bien dang do duoc, ba cau hinh               (muc 8.2)

So hinh theo dung THU TU XUAT HIEN trong bai.

Nguon hinh hoc va dia chat: Tap 4.2 Ban ve thiet ke mong Turbine (10/04/2021),
trang 41 (mat bang coc) va trang 32 (to rieng T18 + ho khoan DH9).

Chay:  python src/figures/make_figures.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # src/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
from paths import FIGURES, save_figure
from foundation_stiffness import (RINGS, D_CAP, L_FREE, D_OUT, batter_angle,
                                  frequencies, F_MEAS_C1, F_MEAS_C2)

# ---------------------------------------------------------------- dia chat DH9
# (do sau day lop tinh tu day bun -6.900 [m], mo ta, mau)
LAYERS = [
    (2.60,  "(SC) clayey SAND\nmedium dense", "#e8d9a0"),
    (16.00, "(CH) fat CLAY\nVERY SOFT", "#b9c9b0"),
    (18.50, "(CL)s lean CLAY with sand\nvery stiff", "#cdbb96"),
    (22.70, "(SC) clayey SAND\nmedium dense", "#e8d9a0"),
    (30.50, "(SC) clayey SAND\nmedium dense", "#e8d9a0"),
    (44.90, "(CH) fat CLAY\nvery stiff", "#a9bda0"),
    (55.40, "(CH)s fat CLAY with sand\nvery stiff", "#a9bda0"),
]
SPT = [(2, 10), (4, 1), (6, 1), (8, 1), (10, 2), (12, 1), (14, 2), (16, 15),
       (18, 17), (20, 26), (22, 26), (24, 15), (26, 18), (28, 25), (30, 27),
       (32, 23), (34, 27), (36, 28), (38, 30), (40, 31), (42, 29), (44, 28),
       (46, 22), (48, 27)]

MUD = -6.900
CAP_BOT, CAP_TOP = 5.435, 9.435
PILE_TIP = -40.5

# ----------------------------------------------- dang mode do duoc (check_rocking)
MODE_V_C1 = ("Vertical setup, campaign 1  (1.134 Hz)",
             [(1, 1.000, -0.0), (2, 0.351, 9.8), (3, 0.267, 179.1),
              (4, 0.248, -177.0), (5, 0.048, 156.5)])
MODE_V_C2 = ("Vertical d2ke setup, campaign 2  (1.228 Hz)",
             [(1, 0.031, -141.5), (2, 1.000, 0.0), (3, 0.205, 1.1),
              (4, 0.837, 175.2), (5, 0.183, -2.4)])
MODE_L_C2 = ("Lateral setup, campaign 2  (1.203 Hz)",
             [(1, 0.970, -1.7), (2, 1.000, -0.0), (3, 0.975, 3.2),
              (4, 0.081, 143.3), (5, 0.927, 1.3)])
VERTICAL_CHANNEL_IN_LATERAL = 4

NODES = {1: (0.0, 0.0), 2: (-5.0, 0.0), 3: (-5.0, 5.0), 4: (0.0, 5.0), 5: (-2.5, 2.5)}


# =============================================================== HINH 1
def fig_pile_layout():
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    R_cap = D_CAP / 2
    ax.add_patch(Circle((0, 0), R_cap, fc="#f2f2f2", ec="k", lw=1.6, zorder=0))

    styles = {None: ("#1f4e79", "vertical"),
              7.0: ("#c00000", "inclined 7:1"),
              5.0: ("#008000", "inclined 5:1")}
    seen = set()
    for r, n, nb in RINGS:
        col, lab = styles[nb]
        ax.add_patch(Circle((0, 0), r, fc="none", ec="0.55", lw=0.8, ls=(0, (6, 4)), zorder=1))
        step = 360.0 / n
        for k in range(n):
            th = np.radians(k * step)
            x, y = r * np.cos(th), r * np.sin(th)
            ax.add_patch(Circle((x, y), D_OUT / 2, fc="white", ec=col, lw=1.5, zorder=3))
            if nb is not None:
                L = 1.15
                ax.add_patch(FancyArrow(x, y, L * np.cos(th), L * np.sin(th),
                                        width=0.045, head_width=0.32, head_length=0.38,
                                        fc=col, ec=col, length_includes_head=True, zorder=2))
            if lab not in seen:
                ax.plot([], [], 'o', mfc='white', mec=col, mew=1.5, ms=8,
                        label=f"{lab} — {n} piles at R = {r:.0f} m, every {step:.0f}°")
                seen.add(lab)

    for r in (3, 6, 9):
        th = np.radians(110)
        ax.text(r * np.cos(th), r * np.sin(th), f"R{r*1000:.0f}",
                fontsize=8, color="0.35", ha="center", va="center", rotation=20,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), zorder=4)
    ax.annotate("", xy=(-R_cap, -R_cap - 1.2), xytext=(R_cap, -R_cap - 1.2),
                arrowprops=dict(arrowstyle="<->", lw=1.0))
    ax.text(0, -R_cap - 1.8, f"pile cap $\\varnothing$ {D_CAP*1000:.0f} mm",
            ha="center", fontsize=9)

    ax.set_xlim(-R_cap - 2.6, R_cap + 2.6)
    ax.set_ylim(-R_cap - 3.2, R_cap + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=8.5, frameon=False)
    fig.tight_layout()
    p = os.path.join(FIGURES, "fig01_pile_layout.png")
    save_figure(fig, p)
    plt.close(fig)
    print("[luu]", p)


# =============================================================== HINH 2
def fig_borehole():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 7.6),
                                   gridspec_kw=dict(width_ratios=[2.05, 1.0]), sharey=True)
    top = CAP_TOP + 2
    bot = MUD - LAYERS[-1][0]

    prev = 0.0
    for d_bot, name, col in LAYERS:
        z0, z1 = MUD - prev, MUD - d_bot
        axL.add_patch(Rectangle((-21, z1), 42, z0 - z1, fc=col, ec="0.4", lw=0.6, zorder=0))
        axL.text(12.0, (z0 + z1) / 2, name, fontsize=7.4, va="center", ha="left", zorder=7,
                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
        axL.text(-20.4, (z0 + z1) / 2, f"{d_bot - prev:.2f} m", fontsize=7, va="center",
                 ha="left", color="0.25", zorder=5)
        prev = d_bot

    axL.add_patch(Rectangle((-21, MUD), 42, 0 - MUD, fc="#cfe4f2", ec="none", alpha=.75, zorder=1))
    axL.axhline(0, color="#1f4e79", lw=1.0, zorder=2)
    axL.text(-20.4, 0.5, "sea water level  +0.000", fontsize=7.5, color="#1f4e79", zorder=7,
             bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
    axL.axhline(MUD, color="0.2", lw=1.2, zorder=2)
    axL.text(-20.4, MUD + 0.6, f"mud line  {MUD:+.3f}", fontsize=7.5, color="0.2", zorder=5)

    axL.add_patch(Rectangle((-D_CAP / 2, CAP_BOT), D_CAP, CAP_TOP - CAP_BOT,
                            fc="0.75", ec="k", lw=1.2, zorder=4))
    axL.text(0, (CAP_BOT + CAP_TOP) / 2, f"pile cap  $\\varnothing${D_CAP} m × 4 m",
             ha="center", va="center", fontsize=8.5, zorder=6)
    axL.text(D_CAP / 2 + 0.3, CAP_TOP, f"{CAP_TOP:+.3f}", fontsize=7.5, va="center")
    axL.text(D_CAP / 2 + 0.3, CAP_BOT, f"{CAP_BOT:+.3f}", fontsize=7.5, va="center")

    for x0, rake in [(-9, 5.0), (-6, 7.0), (-3, None), (3, None), (6, 7.0), (9, 5.0)]:
        dx = 0.0 if rake is None else np.sign(x0) * (CAP_BOT - PILE_TIP) / rake
        axL.plot([x0, x0 + dx], [CAP_BOT, PILE_TIP], color="k", lw=2.0, zorder=3,
                 solid_capstyle="butt")
    axL.annotate("", xy=(-1.3, CAP_BOT), xytext=(-1.3, MUD),
                 arrowprops=dict(arrowstyle="<->", lw=1.1, color="#c00000"))
    axL.text(-1.05, (CAP_BOT + MUD) / 2, f"{L_FREE:.2f} m of pile\nfree-standing in water",
             fontsize=8, color="#c00000", va="center")
    axL.text(0, PILE_TIP - 1.8, f"pile tip  $\\approx${PILE_TIP:+.1f}", ha="center", fontsize=7.5)

    axL.set_xlim(-21, 21)
    axL.set_ylim(bot, top)
    axL.set_ylabel("Elevation [m]")
    axL.set_xticks([])
    for s in ("top", "right", "bottom"):
        axL.spines[s].set_visible(False)

    dd = np.array([d for d, _ in SPT])
    nn = np.array([n for _, n in SPT])
    axR.plot(nn, MUD - dd, "o-", color="#1f4e79", ms=3.5, lw=1.1)
    axR.axvspan(0, 4, color="#c00000", alpha=.10)
    axR.set_xlim(0, 40)
    axR.set_xlabel("SPT $N$ [blows/30 cm]")
    axR.grid(alpha=.3)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    fig.tight_layout()
    p = os.path.join(FIGURES, "fig02_foundation_section.png")
    save_figure(fig, p)
    plt.close(fig)
    print("[luu]", p)


# =============================================================== HINH 3
def fig_sensitivity():
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    Le = np.linspace(L_FREE + 2 * D_OUT, L_FREE + 14 * D_OUT, 120)
    for k_ax, col in zip([2e8, 3e8, 5e8, 8e8], ["#9ecae1", "#4292c6", "#1f4e79", "#08306b"]):
        f = [frequencies(k_ax, L)['f_sway'] for L in Le]
        ax.plot(Le, f, color=col, lw=1.8, label=f"$k_{{ax}}$ = {k_ax:.0e} N/m per pile")

    ax.axhspan(1.05, 1.33, color="#c00000", alpha=.12, zorder=0)
    ax.axhline(F_MEAS_C1, color="#c00000", lw=1.4, ls="--")
    ax.axhline(F_MEAS_C2, color="#c00000", lw=1.0, ls=":")
    ax.text(Le[-1], F_MEAS_C1 + 0.03, f"campaign 1 mean {F_MEAS_C1} Hz",
            ha="right", fontsize=8.5, color="#c00000")
    ax.text(Le[-1], F_MEAS_C2 - 0.09, f"campaign 2 mean {F_MEAS_C2} Hz",
            ha="right", fontsize=8.5, color="#c00000")

    ax.set_xlabel("Effective fixity length of the pile, $L_{eff}$ [m]   "
                  f"(free length in water = {L_FREE:.2f} m)")
    ax.set_ylabel("Predicted sway frequency [Hz]")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(FIGURES, "fig07_sensitivity.png")
    save_figure(fig, p)
    plt.close(fig)
    print("[luu]", p)


# =============================================================== HINH 4
def _panel_frame(ax):
    xs = [NODES[k][0] for k in (1, 2, 3, 4, 1)]
    ys = [NODES[k][1] for k in (1, 2, 3, 4, 1)]
    ax.plot(xs, ys, color="0.6", lw=1.0, zorder=1)
    ax.set_xlim(-7.0, 2.2)
    ax.set_ylim(-2.0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_mode_shape():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9))
    for k, (ax, (title, data), kind) in enumerate(zip(
            axes, [MODE_V_C1, MODE_V_C2, MODE_L_C2], ["v", "v", "h"])):
        _panel_frame(ax)
        for nd, mag, ph in data:
            x, y = NODES[nd]
            s = mag * np.cos(np.radians(ph))
            is_vert = (kind == "v") or (nd == VERTICAL_CHANNEL_IN_LATERAL)
            col = "#c00000" if s < 0 else "#1f4e79"
            L = 1.9 * s
            if is_vert:
                if abs(L) > 0.04:
                    ax.add_patch(FancyArrow(x, y, 0, L, width=0.08, head_width=0.42,
                                            head_length=0.36, fc=col, ec=col,
                                            length_includes_head=True, zorder=3))
            else:
                ax.add_patch(FancyArrow(x, y, L, 0, width=0.08, head_width=0.42,
                                        head_length=0.36, fc=col, ec=col,
                                        length_includes_head=True, zorder=3))
            ax.plot(x, y, 'o', mfc="white", mec="k", ms=7, zorder=4)
            ax.text(x + 0.22, y - 0.55, f"{nd}: {s:+.2f}", fontsize=7.6, zorder=5)
        ax.text(0.0, 1.015, "(%s)" % "abc"[k], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    axes[0].plot([-6.4, 1.4], [2.5, 2.5], ls=(0, (5, 4)), color="0.35", lw=1.0)
    axes[0].text(1.5, 2.5, "nodal line", fontsize=8, color="0.35", va="center")
    # duong nut di qua nut 1 (0,0) va nut 3 (-5,5), vuong goc voi truc 2-4
    axes[1].plot([-6.3, 1.3], [6.3, -1.3], ls=(0, (5, 4)), color="0.35", lw=1.0)
    axes[1].text(-6.3, 6.6, "nodal line", fontsize=8, color="0.35", va="bottom")

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p = os.path.join(FIGURES, "fig11_mode_shape.png")
    save_figure(fig, p)
    plt.close(fig)
    print("[luu]", p)


if __name__ == '__main__':
    fig_pile_layout()
    fig_borehole()
    fig_sensitivity()
    fig_mode_shape()
