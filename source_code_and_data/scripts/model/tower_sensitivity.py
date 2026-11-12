"""
Do nhay cua tam xoay mode ngang co ban voi su co mat cua thap (Supplement S7).

Thap khong co thong so trong ho so, nen KHONG mo hinh hoa thap cu the. Thay vao
do, mot mode uon cua thap duoc thay bang mot dao dong tu mot bac tu do:
    khoi luong modal m_t, tan so rieng (ngam chan) f_t, dat o cao do z_t so voi
    dau coc; lo xo k_t = m_t (2 pi f_t)^2 noi m_t voi diem cua vat cung dai o z_t.
Chuyen vi ngang cua diem do: Ux + Ry * z_t  (dai cung, mat x-z).
He 7 bac tu do: 6 cua dai + chuyen vi ngang tuyet doi x_t cua m_t.

Hai gioi han:
    f_t << f_1 : khoi luong thap bi cach ly, thap chi con la lo xo mem k_t;
    f_t >> f_1 : thap di theo dai nhu vat cung, khoi luong m_t dat o z_t.

Dau ra: tam xoay z_c va ty so dung/ngang tai mat dai cua mode "do dai chi phoi"
(mode trong mat x-z, duoi 3 Hz, co ty le dong nang cua dai lon nhat).

Chay:  python scripts/model/tower_sensitivity.py
"""

import os
import sys
import csv
import numpy as np
from scipy import linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import foundation_stiffness as fs
import rigid_cap_6dof as rc
from paths import RESULTS

K_AX = 3.5e8                         # N/m, giua khoang uoc luong tu ho khoan
L_EFF = fs.L_FREE + 6.0 * fs.D_OUT   # m, giua khoang suy nguoc (4.2-8.4 D)
D_SENSOR = 2.5 * np.sqrt(2)          # m, goc khung do cach truc xoay xa nhat

M_T = (100e3, 200e3, 400e3, 800e3)             # kg, khoi luong modal
Z_T = (30.0, 60.0, 90.0)                       # m tren dau coc
F_T = (0.2, 0.32, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 1e3)   # Hz; 1e3 = thap cung


def cap_mode(m_t, z_t, f_t):
    K6 = rc.stiffness(K_AX, L_EFF)
    M6 = rc.mass(L_EFF)
    k_t = m_t * (2 * np.pi * f_t) ** 2
    b = np.zeros(7)
    b[0], b[4], b[6] = 1.0, z_t, -1.0          # Ux + Ry z_t - x_t
    K = np.zeros((7, 7))
    K[:6, :6] = K6
    K += k_t * np.outer(b, b)
    M = np.zeros((7, 7))
    M[:6, :6] = M6
    M[6, 6] = m_t
    w2, V = linalg.eigh(K, M)
    best = None
    for j in range(7):
        f = np.sqrt(max(w2[j], 0)) / (2 * np.pi)
        phi = V[:, j]
        cap = phi[:6]
        in_plane = abs(cap[0]) + abs(cap[4]) > 10 * (abs(cap[1]) + abs(cap[3]) + abs(cap[2]) + abs(cap[5]))
        if not in_plane or f > 3.0:
            continue
        e_cap = cap @ M6 @ cap                 # ty le dong nang cua dai (V chuan hoa theo M)
        if best is None or e_cap > best["e_cap"]:
            zc = -cap[0] / cap[4] if abs(cap[4]) > 1e-14 else np.inf
            best = dict(f=f, zc=zc, e_cap=e_cap)
    return best


def main():
    base = rc.summary(K_AX, L_EFF)
    vh0 = D_SENSOR / (base["zc1"] - rc.H_TOP)
    print(f"Khong thap: f1 = {base['f1']:.3f} Hz, z_c = {base['zc1']:.1f} m, dung/ngang = {vh0:.3f}")
    print(f"Tieu diem: 42.0 va 45.0 m; can hinh hoc d/(z_f - h) = {D_SENSOR / (42.0 - rc.H_TOP):.3f}")
    rows = []
    print(f"\n{'m_t [t]':>8} {'z_t [m]':>8} {'f_t [Hz]':>9} {'f [Hz]':>7} {'z_c [m]':>8} {'dung/ngang':>11} {'dai %':>6}")
    for m_t in M_T:
        for z_t in Z_T:
            for f_t in F_T:
                r = cap_mode(m_t, z_t, f_t)
                vh = D_SENSOR / abs(r["zc"] - rc.H_TOP)
                rows.append(dict(m_t_tonne=m_t / 1e3, z_t_m=z_t, f_t_Hz=f_t, f_Hz=round(r["f"], 4),
                                 zc_m=round(r["zc"], 2), vh=round(vh, 4),
                                 cap_energy_fraction=round(r["e_cap"], 3)))
                print(f"{m_t/1e3:8.0f} {z_t:8.0f} {f_t:9.2f} {r['f']:7.3f} {r['zc']:8.1f} {vh:11.3f} {100*r['e_cap']:6.0f}")
    out = os.path.join(RESULTS, "tables", "table_tower_sensitivity.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("[luu]", out)

    def rng(sel):
        v = [r for r in rows if sel(r)]
        return (min(r["zc_m"] for r in v), max(r["zc_m"] for r in v),
                min(r["vh"] for r in v), max(r["vh"] for r in v),
                min(r["f_Hz"] for r in v), max(r["f_Hz"] for r in v))

    for name, sel in (("f_t <= 0.5 Hz", lambda r: r["f_t_Hz"] <= 0.5),
                      ("0.8 <= f_t <= 2 Hz", lambda r: 0.8 <= r["f_t_Hz"] <= 2.0),
                      ("f_t >= 3 Hz", lambda r: r["f_t_Hz"] >= 3.0)):
        a = rng(sel)
        print(f"  {name:>20}: z_c {a[0]:7.1f} .. {a[1]:7.1f} m; dung/ngang {a[2]:.3f} .. {a[3]:.3f}; f {a[4]:.3f} .. {a[5]:.3f} Hz")


if __name__ == "__main__":
    main()
