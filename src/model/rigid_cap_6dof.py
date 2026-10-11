"""
Mo hinh dai cung 6 bac tu do tren nhom coc xien (muc 4 cua bai bao).

Moi coc la mot phan tu dam 3D noi cung vao dai o dau coc va ngam tai diem
ngam hieu dung, cach dau coc L_eff do theo truc coc:
    - doc truc   : k_ax (do cung dau coc, gom doan tu do va doan ngam)
    - uon        : phan tu dam ngam hai dau, EI, chieu dai L_eff
    - xoan       : GJ / L_eff
Chuyen vi dau coc = U + Phi x p_i (dai cung), xoay dau coc = Phi.

Goc toa do O: tam dai, o cao do day dai (dau coc), z huong len.

Khac voi mo hinh khong ghep cu (PT 1-4 ban truoc), o day giu day du so hang
ghep k_u_phi giua tinh tien ngang va xoay, va bac tu do xoan.

Hai ket qua khong phu thuoc dat nen:
  1. Khi bo qua uon coc, K = k_ax * G (G chi phu thuoc hinh hoc), nen moi ty so
     tan so va moi dang mode khong phu thuoc k_ax.
  2. Khoi ngang-lac cua G co mot tri rieng gan bang 0: xoay quanh "tieu diem"
     noi cac truc coc cat nhau. Do cung cua mode thap nhat vi vay den tu uon coc,
     con vi tri tam xoay cua no bi hinh hoc ghim gan tieu diem.

Chay:  python src/model/rigid_cap_6dof.py
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
from paths import RESULTS

G_PILE = fs.E_PILE / 2.4          # Pa, nu = 0.2
J_PILE = 2 * fs.I_PILE            # m4, mo-men quan tinh cuc tiet dien vanh khuyen
H_TOP = fs.T_CAP                  # m, mat tren dai so voi dau coc
FRAME_HALF = 2.5                  # m, nua canh khung do 5 x 5 m


def skew(r):
    x, y, z = r
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def piles():
    """Vi tri dau coc p_i va vec-to don vi truc coc a_i (tu dau coc xuong mui)."""
    out = []
    for r, n, nb in fs.RINGS:
        al = fs.batter_angle(nb)
        for k in range(n):
            th = 2 * np.pi * k / n
            p = np.array([r * np.cos(th), r * np.sin(th), 0.0])
            a = np.array([np.sin(al) * np.cos(th), np.sin(al) * np.sin(th), -np.cos(al)])
            out.append((p, a))
    return out


def local_frame(a):
    """He truc dia phuong (x' = truc coc, y', z') dang ma tran hang."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(ref, a)
    y /= np.linalg.norm(y)
    z = np.cross(a, y)
    return np.vstack([a, y, z])


def head_stiffness_local(k_ax, L, EI, with_bending=True):
    """Do cung dau coc trong he dia phuong, DOF (u, v, w, rx, ry, rz)."""
    k = np.zeros((6, 6))
    k[0, 0] = k_ax
    if with_bending:
        k[3, 3] = G_PILE * J_PILE / L
        a, b, c = 12 * EI / L ** 3, 6 * EI / L ** 2, 4 * EI / L
        k[1, 1] = k[2, 2] = a              # v (mat x'-y'), w (mat x'-z')
        k[1, 5] = k[5, 1] = b              # v - rz
        k[2, 4] = k[4, 2] = -b             # w - ry
        k[4, 4] = k[5, 5] = c
    return k


def stiffness(k_ax, L_eff, EI=fs.EI, with_bending=True):
    """Ma tran do cung 6x6 cua dai, DOF q = (Ux, Uy, Uz, Rx, Ry, Rz) tai O."""
    K = np.zeros((6, 6))
    kl = head_stiffness_local(k_ax, L_eff, EI, with_bending)
    for p, a in piles():
        R = local_frame(a)
        R6 = linalg.block_diag(R, R)
        T = np.eye(6)
        T[0:3, 3:6] = -skew(p)            # u_dau = U + Phi x p = U - S(p) Phi
        B = R6 @ T
        K += B.T @ kl @ B
    return K


def rigid_mass(m, c, Ic):
    """Ma tran khoi luong 6x6 tai O cua vat cung khoi luong m, trong tam c."""
    S = skew(c)
    M = np.zeros((6, 6))
    M[0:3, 0:3] = m * np.eye(3)
    M[0:3, 3:6] = -m * S
    M[3:6, 0:3] = m * S
    M[3:6, 3:6] = Ic + m * (c @ c * np.eye(3) - np.outer(c, c))
    return M


def mass(L_eff, pile_fraction=13 / 35):
    """Dai cung + phan khoi luong coc va nuoc kem theo tham gia, gop o dau coc.

    pile_fraction: ty le khoi luong (coc tren L_eff + nuoc kem tren doan tu do)
    tham gia dao dong; 13/35 la khoi luong tuong duong cua dam ngam - dau truot.
    0 = chi co dai; 1 = gop toan bo.
    """
    p = fs.cap_mass_inertia()
    R = fs.D_CAP / 2
    m_cap = p['m_cap']
    Ixx = m_cap * (3 * R ** 2 + fs.T_CAP ** 2) / 12
    Izz = m_cap * R ** 2 / 2
    M = rigid_mass(m_cap, np.array([0, 0, fs.T_CAP / 2]), np.diag([Ixx, Ixx, Izz]))
    m_one = pile_fraction * (fs.A_PILE * fs.RHO_C * L_eff
                             + fs.RHO_W * np.pi / 4 * fs.D_OUT ** 2 * fs.L_FREE)
    for pos, _ in piles():
        M += rigid_mass(m_one, pos, np.zeros((3, 3)))
    return M


def classify(phi):
    """Gan nhan cho mot vec-to rieng dua tren thanh phan troi."""
    e = np.abs(phi) / np.abs(phi).max()
    ux, uy, uz, rx, ry, rz = e
    if rz > 0.5 and max(ux, uy, uz) < 0.1 and max(rx, ry) < 0.1:
        return "torsion"
    if uz > 0.5 and max(ux, uy) < 0.1:
        return "vertical"
    return "lateral"


def rotation_centre(phi):
    """Cao do tam xoay z_c (so voi dau coc) cua mot mode ngang-lac.

    u(z) = Ux + Ry z  (mat x-z),  v(z) = Uy - Rx z  (mat y-z).
    """
    ux, uy, _, rx, ry, _ = phi
    if abs(ux) + abs(ry) >= abs(uy) + abs(rx):
        return -ux / ry if abs(ry) > 1e-14 else np.inf
    return uy / rx if abs(rx) > 1e-14 else np.inf


def modes(k_ax, L_eff, pile_fraction=13 / 35, with_bending=True, M_extra=None):
    K = stiffness(k_ax, L_eff, with_bending=with_bending)
    M = mass(L_eff, pile_fraction)
    if M_extra is not None:
        M = M + M_extra
    w2, V = linalg.eigh(K, M)
    out = []
    for j in range(6):
        phi = V[:, j]
        kind = classify(phi)
        zc = rotation_centre(phi) if kind == "lateral" else np.nan
        out.append(dict(f=np.sqrt(max(w2[j], 0)) / (2 * np.pi), kind=kind, zc=zc, phi=phi))
    return out


def summary(k_ax, L_eff, pile_fraction=13 / 35):
    """Tan so va tam xoay cua: cap ngang thu nhat, xoan, dung, cap lac."""
    ms = modes(k_ax, L_eff, pile_fraction)
    lat = sorted([m for m in ms if m["kind"] == "lateral"], key=lambda m: m["f"])
    tor = [m for m in ms if m["kind"] == "torsion"]
    ver = [m for m in ms if m["kind"] == "vertical"]
    return dict(f1=lat[0]["f"], zc1=lat[0]["zc"], f_rock=lat[-1]["f"], zc2=lat[-1]["zc"],
                f_tor=tor[0]["f"] if tor else np.nan,
                f_ver=ver[0]["f"] if ver else np.nan,
                ratio=lat[-1]["f"] / lat[0]["f"])


def focal_heights():
    """Cao do noi cac truc coc moi vong cat truc dung qua tam: r / tan(alpha)."""
    return {r: r / np.tan(fs.batter_angle(nb)) for r, n, nb in fs.RINGS if nb}


def axial_block():
    """Khoi ngang-lac (Ux, Ry) cua K khi chi co do cung doc truc, chia cho k_ax."""
    G = stiffness(1.0, 1.0, with_bending=False)
    B = G[np.ix_([0, 4], [0, 4])]
    w, V = linalg.eigh(B)
    zc = [-V[0, j] / V[1, j] for j in range(2)]
    coup = abs(B[0, 1]) / np.sqrt(B[0, 0] * B[1, 1])
    return B, w, zc, coup


def vh_ratio(zc, d):
    """Ty so bien do dung / ngang tai mot diem tren mat dai, cach truc xoay d."""
    return d / abs(H_TOP - zc)


def invert_L_eff(f_target, k_ax, pile_fraction=13 / 35):
    """Chieu dai ngam hieu dung L_eff sao cho mode ngang thu nhat bang f_target."""
    from scipy.optimize import brentq
    g = lambda Le: summary(k_ax, Le, pile_fraction)["f1"] - f_target
    return brentq(g, fs.L_FREE + 0.5, fs.L_FREE + 30 * fs.D_OUT)


def fixity_depth_dr1965(G_scale=1.0, depth=5.0):
    """Do sau ngam duoi day bun theo Davisson & Robinson (1965), dat mo-dun khong doi:
        L_s = 1.4 R,  R = (EI / k_h)^(1/4),  k_h ~ E_s = 3 G (set khong thoat nuoc)
    G lay trung binh theo chieu day tren `depth` met dau tien duoi day bun
    (Ohsaki & Iwasaki, bien dang nho), nhan G_scale.
    """
    z, acc = 0.0, 0.0
    for t, n in fs.EMBEDDED_LAYERS:
        dz = min(t, depth - z)
        if dz <= 0:
            break
        acc += dz * fs.g0_ohsaki_iwasaki(n)
        z += dz
    G = G_scale * acc / depth
    R = (fs.EI / (3 * G)) ** 0.25
    return 1.4 * R, G


K_AX_RANGE = (1.5e8, 2.5e8, 3.5e8, 6.0e8)              # N/m, muc 6.2
L_EFF_MULT = (3, 5, 8, 12)                              # L_eff = L_free + m D
FRACTIONS = (0.0, 13 / 35, 1.0)


def main():
    B, w, zc_ax, coup = axial_block()
    print("=" * 74)
    print("1. CHI CO DO CUNG DOC TRUC (K = k_ax G): khoi ngang-lac (Ux, Ry)")
    print("=" * 74)
    print(f"  G_uu = {B[0,0]:.4f}   G_u_phi = {B[0,1]:.3f} m   G_phi_phi = {B[1,1]:.1f} m2")
    print(f"  he so ghep |G_u_phi| / sqrt(G_uu G_phi_phi) = {coup:.3f}")
    print(f"  tri rieng (/k_ax): {w[0]:.4f}  {w[1]:.1f}")
    print(f"  tam xoay hai vec-to rieng: z = {zc_ax[0]:.1f} m  va  {zc_ax[1]:.2f} m")
    print("  tieu diem tung vong (r cot alpha):",
          ", ".join(f"R={r:.0f} m -> {h:.1f} m" for r, h in focal_heights().items()))

    rows = []
    print("\n" + "=" * 74)
    print("2. MO HINH DAY DU (doc truc + uon + xoan), quet k_ax, L_eff, khoi luong coc")
    print("=" * 74)
    print(f"  {'k_ax':>8} {'L_eff':>6} {'frac':>5} {'f1':>6} {'zc1':>6} {'f_tor':>6}"
          f" {'f_ver':>6} {'f_rock':>7} {'zc2':>6} {'ratio':>6} {'v/h':>6}")
    for k in K_AX_RANGE:
        for m in L_EFF_MULT:
            Le = fs.L_FREE + m * fs.D_OUT
            for fr in FRACTIONS:
                s = summary(k, Le, fr)
                vh = vh_ratio(s["zc1"], FRAME_HALF)
                rows.append(dict(k_ax=k, L_eff=round(Le, 3), pile_fraction=round(fr, 4),
                                 f1=round(s["f1"], 4), zc1=round(s["zc1"], 2),
                                 f_torsion=round(s["f_tor"], 4), f_vertical=round(s["f_ver"], 4),
                                 f_rock=round(s["f_rock"], 4), zc2=round(s["zc2"], 3),
                                 ratio=round(s["ratio"], 3), vh_at_2p5m=round(vh, 4)))
                print(f"  {k:8.1e} {Le:6.2f} {fr:5.2f} {s['f1']:6.3f} {s['zc1']:6.1f}"
                      f" {s['f_tor']:6.3f} {s['f_ver']:6.2f} {s['f_rock']:7.2f}"
                      f" {s['zc2']:6.2f} {s['ratio']:6.2f} {vh:6.3f}")
    out = os.path.join(RESULTS, "tables", "table3_coupled_model.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("[luu]", out)

    def rng(key):
        v = [r[key] for r in rows]
        return min(v), max(v)

    print("\n  Pham vi tren toan bo quet:")
    for key in ("f1", "zc1", "f_torsion", "f_rock", "zc2", "ratio", "vh_at_2p5m"):
        lo, hi = rng(key)
        print(f"    {key:>11}: {lo:8.3f} - {hi:8.3f}")
    tf = [r["f_torsion"] / r["f1"] for r in rows]
    print(f"    f_tor / f1 : {min(tf):8.3f} - {max(tf):8.3f}")

    print("\n" + "=" * 74)
    print("3. SUY NGUOC L_eff TU TAN SO DO (k_ax lay tu uoc luong thuan, muc 6.2)")
    print("=" * 74)
    # trung binh tung setup dot 1, BO Lateral 2 (truot yeu cau 1, muc 5.2)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "oma"))
    from oma_lib import band_poles_by_setup
    c1, _ = band_poles_by_setup(RESULTS)
    means = [float(np.mean(v)) for name, v in c1.items() if name != "Lateral 2" and v]
    targets = (round(min(means), 4), round(float(np.mean(means)), 4), round(max(means), 4))
    print(f"  setup mean dot 1 (bo Lateral 2): {np.round(means, 4)}")
    inv_rows = []
    print(f"  {'f [Hz]':>7} {'k_ax':>8} {'frac':>5} {'L_eff':>7} {'(L_eff-L_free)/D':>17}")
    for f_t in targets:
        for k in (2.5e8, 5.8e8):
            for fr in (0.0, 1.0):
                Le = invert_L_eff(f_t, k, fr)
                inv_rows.append(dict(f_target=f_t, k_ax=k, pile_fraction=fr, L_eff=round(Le, 3),
                                     depth_to_fixity_D=round((Le - fs.L_FREE) / fs.D_OUT, 2)))
                print(f"  {f_t:7.4f} {k:8.1e} {fr:5.2f} {Le:7.2f} {(Le - fs.L_FREE) / fs.D_OUT:17.2f}")
    out = os.path.join(RESULTS, "tables", "table8_L_eff_inversion.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(inv_rows[0].keys()))
        wr.writeheader()
        wr.writerows(inv_rows)
    print("[luu]", out)

    # do nhay cua f1 theo k_ax o L_eff co dinh
    Le = fs.L_FREE + 3 * fs.D_OUT
    a, b = summary(1.5e8, Le)["f1"], summary(6.0e8, Le)["f1"]
    print(f"\n  L_eff = {Le:.2f} m: k_ax tang 4 lan (1.5e8 -> 6e8) lam f1 tang {100*(b/a-1):.1f} %")

    print("\n  Uoc luong thuan do sau ngam, Davisson & Robinson (1965):")
    for depth in (3.0, 5.0):
        for s in (0.5, 1.0):
            Ls, G = fixity_depth_dr1965(s, depth)
            print(f"    G trung binh {depth:.0f} m dau = {G/1e6:5.1f} MPa (G/G0 = {s}):"
                  f"  L_s = {Ls:.2f} m = {Ls / fs.D_OUT:.1f} D")


if __name__ == "__main__":
    main()
