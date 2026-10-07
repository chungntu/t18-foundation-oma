"""
He so hieu qua nhom coc, tinh DOC LAP voi so lieu do, bang he so tuong tac
doc truc cua Randolph & Wroth (1979):

    alpha_ij = ln(r_m / s_ij) / ln(r_m / r_0),   r_m = 2.5 * L_emb * (1 - nu)

voi s_ij la khoang cach hai dau coc; alpha_ii = 1; alpha = 0 khi s >= r_m.

Voi dai tuyet doi cung, chuyen vi doc truc cac coc bi RANG BUOC theo dang
chuyen vi cua dai. Goi A la ma tran tuong tac va v la vec-to chuyen vi doc
truc don vi cua tung coc ung voi mot co che, thi he so hieu qua la

    eta = (v^T A^{-1} v) / (v^T v)

Ba co che duoc xet:
  - deu      : moi coc lun nhu nhau              v_i = 1
  - lu ngang : dai tinh tien ngang u             v_i = sin(alpha_i) cos(theta_i)
  - lac      : dai xoay mot goc                  v_i = r_i cos(alpha_i) cos(theta_i)

LUU Y quan trong: ca hai co che lu ngang va lac deu cho dang chuyen vi doc
truc PHAN DOI XUNG theo phuong vi (ty le cos theta), khong phai mot doi xung
mot phan doi xung nhu truc giac ban dau.

Chay:  python model/group_interaction.py
"""

import numpy as np
from numpy.linalg import inv

RINGS = [(3.0, 4, None), (6.0, 12, 7.0), (9.0, 18, 5.0)]
L_EMB = 33.6      # m, doan coc ngam
NU = 0.3
R0 = 0.5          # m, ban kinh coc


def pile_table():
    """Toa do dau coc, phuong vi, goc xien."""
    x, y, th, al, rr = [], [], [], [], []
    for r, n, nb in RINGS:
        a = 0.0 if nb is None else np.arctan(1.0 / nb)
        for k in range(n):
            t = 2 * np.pi * k / n
            x.append(r * np.cos(t)); y.append(r * np.sin(t))
            th.append(t); al.append(a); rr.append(r)
    return (np.array(x), np.array(y), np.array(th), np.array(al), np.array(rr))


def interaction_matrix(x, y):
    n = len(x)
    r_m = 2.5 * L_EMB * (1 - NU)
    denom = np.log(r_m / R0)
    S = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 1.0
            elif S[i, j] < r_m:
                A[i, j] = np.log(r_m / S[i, j]) / denom
    return A, S, r_m


def eta(A_inv, v):
    v = np.asarray(v, dtype=float)
    return float((v @ A_inv @ v) / (v @ v))


def group_kax(A, v, k_soil, k_free):
    """Do cung doc truc hieu dung moi coc trong nhom, theo co che v.

    Tuong tac coc-dat-coc chi tac dong len doan NGAM; doan tu do trong nuoc
    co mem rieng, khong tuong tac. Ma tran mem dau coc:
        F = I / k_free + A / k_soil
    va do cung hieu dung moi coc = v^T F^{-1} v / v^T v.
    (Nhan eta vao do cung noi tiep ca coc la SAI: no tinh ca doan tu do
    nhu the doan do cung tuong tac qua dat.)
    """
    v = np.asarray(v, dtype=float)
    F = np.eye(len(v)) / k_free + A / k_soil
    return float(v @ inv(F) @ v / (v @ v))


def main():
    x, y, th, al, rr = pile_table()
    A, S, r_m = interaction_matrix(x, y)
    A_inv = inv(A)
    n = len(x)

    off = S[~np.eye(n, dtype=bool)]
    print("=" * 74)
    print("NHOM COC")
    print("=" * 74)
    print(f"  so coc = {n},  r_m = 2.5*L_emb*(1-nu) = {r_m:.1f} m,  r_0 = {R0} m")
    print(f"  khoang cach dau coc: min {off.min():.2f} m, trung vi {np.median(off):.2f} m, "
          f"max {off.max():.2f} m")
    print(f"  he so tuong tac alpha: max (ngoai duong cheo) = "
          f"{A[~np.eye(n,dtype=bool)].max():.3f}")

    v_unif = np.ones(n)
    v_sway = np.sin(al) * np.cos(th)
    v_rock = rr * np.cos(al) * np.cos(th)

    e_u, e_s, e_r = eta(A_inv, v_unif), eta(A_inv, v_sway), eta(A_inv, v_rock)

    print()
    print("=" * 74)
    print("HE SO HIEU QUA NHOM THEO TUNG CO CHE")
    print("=" * 74)
    print(f"  lun deu (doi xung)            eta = {e_u:.3f}")
    print(f"  lu ngang qua coc xien         eta = {e_s:.3f}")
    print(f"  lac quanh truc ngang          eta = {e_r:.3f}")
    print()
    print("=" * 74)
    print("DO CUNG DOC TRUC MOI COC: UOC LUONG THUAN (muc 6.2 bai bao)")
    print("=" * 74)
    import foundation_stiffness as fs
    k_free, fwd = fs.kax_from_soil()
    N_av = sum(t * n for t, n in fs.EMBEDDED_LAYERS) / fs.L_EMB
    print(f"  N trung binh doan ngam = {N_av:.1f};  EA/L_free = {k_free:.3e} N/m")
    print(f"  {'G/G0':>5} {'G_avg MPa':>9} {'muL':>5} {'k_ngam':>10} {'k_co_lap':>10} {'k_nhom':>10}")
    for s, r in fwd.items():
        k_g = group_kax(A, v_sway, r['k_soil'], k_free)
        print(f"  {s:5.1f} {r['G_avg']/1e6:9.1f} {r['muL']:5.2f} {r['k_soil']:10.2e}"
              f" {r['k_iso']:10.2e} {k_g:10.2e}")


if __name__ == '__main__':
    main()
