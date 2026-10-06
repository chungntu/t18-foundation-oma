"""
Uoc luong tan so dao dong cua mong T18 tu NGUYEN LY DAU, dung so lieu ban ve.

Nguon so lieu (Tap 4.2 - Ban ve thiet ke mong Turbine, 10/04/2021):
  - trang 41: MAT BANG BO TRI PHC D1000 - 34 COC
        vong trong  R = 3.0 m,  4 coc,  goc 90 deg, coc thang dung
        vong giua   R = 6.0 m, 12 coc,  goc 30 deg, coc xien 7:1
        vong ngoai  R = 9.0 m, 18 coc,  goc 20 deg, coc xien 5:1
        duong kinh dai 20.5 m
  - trang 32 (T18): dinh dai +9.435, day dai +5.435 (dai day 4 m),
        day bun -6.900  => 12.335 m coc tu do trong nuoc,
        chieu dai coc 46 m, 34 coc PHC D1000,
        dia chat (ho khoan DH9): 2.6 m cat set chat vua; 13.4 m SET BEO RAT MEM
        (SPT N = 0..2); duoi do la cac lop cung hon (N = 17..35).

Muc tieu:
  1. Kiem tra co che nao chi phoi tan so: LAC tren do cung doc truc coc,
     hay LAC LU NGANG tren do cung ngang cua nhom coc.
  2. Hieu chinh nguoc do cung doc truc coc tu tan so DO DUOC.

LUU Y: day la uoc luong bac nhat. Cac gia thiet phai xac nhan lai bang
ban ve chi tiet coc va bao cao dia chat (xem cuoi file).

Chay:  python model/foundation_stiffness.py
"""

import numpy as np

# ----------------------------------------------------------------- hinh hoc
RINGS = [  # (ban kinh [m], so coc, do xien n trong "n:1"; None = thang dung)
    (3.0, 4, None),
    (6.0, 12, 7.0),
    (9.0, 18, 5.0),
]
D_CAP = 20.5          # m, duong kinh dai
T_CAP = 4.0           # m, chieu day dai
L_FREE = 12.335       # m, tu day dai +5.435 den day bun -6.900
L_PILE = 46.0         # m, chieu dai coc tinh toan

# ------------------------------------------------- tiet dien coc (GIA THIET)
D_OUT = 1.000         # m, PHC D1000
T_WALL = 0.130        # m, GIA THIET - can xac nhan bang ban ve chi tiet coc
E_PILE = 40e9         # Pa, GIA THIET cho be tong PHC cuong do cao (~C80)
RHO_C = 2500.0        # kg/m3
RHO_W = 1025.0        # kg/m3

D_IN = D_OUT - 2 * T_WALL
A_PILE = np.pi / 4 * (D_OUT ** 2 - D_IN ** 2)
I_PILE = np.pi / 64 * (D_OUT ** 4 - D_IN ** 4)
EA = E_PILE * A_PILE
EI = E_PILE * I_PILE

# ------------------------------------------------------- tan so do duoc (OMA)
F_MEAS_C1 = 1.257     # Hz, trung binh nhom mode toan cuc, dot 1 (chua co thap)
F_MEAS_C2 = 1.153     # Hz, dot 2 (da co thap)


def batter_angle(n):
    """Goc nghieng so voi phuong thang dung cua coc xien n:1 [rad]."""
    return 0.0 if n is None else np.arctan(1.0 / n)


def cap_mass_inertia():
    """Khoi luong va mo-men quan tinh khoi cua dai + coc (khong ke thap)."""
    V_cap = np.pi / 4 * D_CAP ** 2 * T_CAP
    m_cap = V_cap * RHO_C
    R = D_CAP / 2
    # quay quanh mot duong kinh qua trong tam dia tron day t
    J_cap = m_cap * (3 * R ** 2 + T_CAP ** 2) / 12

    m_pile_each = A_PILE * L_PILE * RHO_C
    n_pile = sum(n for _, n, _ in RINGS)
    m_piles = m_pile_each * n_pile
    J_piles = sum(m_pile_each * n * r ** 2 for r, n, _ in RINGS)

    # khoi luong nuoc kem theo tren doan coc ngap (added mass he so 1.0)
    m_added = RHO_W * np.pi / 4 * D_OUT ** 2 * L_FREE * n_pile

    return dict(m_cap=m_cap, J_cap=J_cap, m_piles=m_piles, J_piles=J_piles,
                m_added=m_added, V_cap=V_cap, n_pile=n_pile)


def geometry_sums():
    """Cac tong hinh hoc dung cho do cung lac va ngang."""
    s_rot = 0.0      # sum r^2 cos^2(alpha)  -> lac tu do cung doc truc
    s_lat_ax = 0.0   # sum sin^2(alpha)      -> ngang tu do cung doc truc (coc xien)
    s_lat_b = 0.0    # sum cos^2(alpha)      -> ngang tu uon coc
    for r, n, nb in RINGS:
        a = batter_angle(nb)
        s_rot += n * r ** 2 * np.cos(a) ** 2
        s_lat_ax += n * np.sin(a) ** 2
        s_lat_b += n * np.cos(a) ** 2
    return s_rot, s_lat_ax, s_lat_b


def k_lateral_bending(L_eff):
    """Do cung ngang cua MOT coc coi nhu dam ngam hai dau, chieu dai hieu dung L_eff."""
    return 12 * EI / L_eff ** 3


def frequencies(k_ax, L_eff, m_extra=0.0, J_extra=0.0):
    """Tan so lac va tan so lu ngang, voi do cung doc truc mot coc k_ax [N/m]."""
    p = cap_mass_inertia()
    s_rot, s_lat_ax, s_lat_b = geometry_sums()

    M = p['m_cap'] + p['m_piles'] + p['m_added'] + m_extra
    J = p['J_cap'] + p['J_piles'] + J_extra

    k_theta = k_ax * s_rot
    k_lat = k_ax * s_lat_ax + k_lateral_bending(L_eff) * s_lat_b

    f_rock = np.sqrt(k_theta / J) / (2 * np.pi)
    f_sway = np.sqrt(k_lat / M) / (2 * np.pi)
    return dict(M=M, J=J, k_theta=k_theta, k_lat=k_lat,
                k_lat_from_batter=k_ax * s_lat_ax,
                k_lat_from_bending=k_lateral_bending(L_eff) * s_lat_b,
                f_rock=f_rock, f_sway=f_sway)


def solve_kax_for_frequency(f_target, L_eff):
    """Tim k_ax sao cho tan so lu ngang bang f_target (giai truc tiep)."""
    p = cap_mass_inertia()
    s_rot, s_lat_ax, s_lat_b = geometry_sums()
    M = p['m_cap'] + p['m_piles'] + p['m_added']
    k_need = (2 * np.pi * f_target) ** 2 * M
    k_ax = (k_need - k_lateral_bending(L_eff) * s_lat_b) / s_lat_ax
    return k_ax, k_need


def rock_sway_ratio(L_eff=None):
    """Ty so f_lac / f_lu_ngang.

    Neu bo qua do cung uon coc, k_ax TRIET TIEU:
        f_rock/f_sway = sqrt( (sum r^2 cos^2 a / J) / (sum sin^2 a / M) )
    tuc chi phu thuoc HINH HOC va PHAN BO KHOI LUONG, khong phu thuoc dat nen.
    """
    p = cap_mass_inertia()
    s_rot, s_lat_ax, s_lat_b = geometry_sums()
    M = p['m_cap'] + p['m_piles'] + p['m_added']
    J = p['J_cap'] + p['J_piles']

    r_no_bending = np.sqrt((s_rot / J) / (s_lat_ax / M))
    out = dict(ratio_no_bending=r_no_bending, M=M, J=J,
               s_rot=s_rot, s_lat_ax=s_lat_ax, s_lat_b=s_lat_b)
    if L_eff is not None:
        out['ratio_with_bending'] = {}
        for k_ax in [2e8, 3e8, 5e8, 8e8, 2e9]:
            r = frequencies(k_ax, L_eff)
            out['ratio_with_bending'][k_ax] = r['f_rock'] / r['f_sway']
    return out


def main():
    p = cap_mass_inertia()
    s_rot, s_lat_ax, s_lat_b = geometry_sums()

    print("=" * 74)
    print("1. SO LIEU DAN XUAT TU BAN VE")
    print("=" * 74)
    print(f"  Tiet dien coc PHC D1000 (vach {T_WALL*1000:.0f} mm, GIA THIET):")
    print(f"     A  = {A_PILE:.4f} m2      I  = {I_PILE:.5f} m4")
    print(f"     EA = {EA:.3e} N        EI = {EI:.3e} N.m2")
    print(f"  Dai: the tich {p['V_cap']:.0f} m3, khoi luong {p['m_cap']/1e6:.2f} x10^6 kg")
    print(f"  {p['n_pile']} coc: khoi luong {p['m_piles']/1e6:.2f} x10^6 kg, "
          f"nuoc kem theo {p['m_added']/1e6:.2f} x10^6 kg")
    print(f"  Tong hinh hoc:  sum r^2cos^2a = {s_rot:.1f} m2   "
          f"sum sin^2a = {s_lat_ax:.3f}   sum cos^2a = {s_lat_b:.2f}")
    print(f"  Chieu dai coc tu do trong nuoc: {L_FREE:.2f} m")

    print("\n" + "=" * 74)
    print("2. CO CHE NAO CHI PHOI? (thu voi k_ax = 5e8 N/m, L_eff = L_free + 5D)")
    print("=" * 74)
    L_eff = L_FREE + 5 * D_OUT
    r = frequencies(5e8, L_eff)
    print(f"  Khoi luong dao dong M = {r['M']/1e6:.2f} x10^6 kg")
    print(f"  Quan tinh quay     J = {r['J']:.3e} kg.m2")
    print(f"  Do cung lac    k_th = {r['k_theta']:.3e} N.m/rad  ->  f_lac  = {r['f_rock']:6.2f} Hz")
    print(f"  Do cung ngang  k_lat= {r['k_lat']:.3e} N/m      ->  f_ngang = {r['f_sway']:6.2f} Hz")
    print(f"     trong do do coc xien dong gop {r['k_lat_from_batter']:.3e} N/m "
          f"({100*r['k_lat_from_batter']/r['k_lat']:.0f} %)")
    print(f"     uon coc dong gop            {r['k_lat_from_bending']:.3e} N/m "
          f"({100*r['k_lat_from_bending']/r['k_lat']:.0f} %)")
    print(f"\n  => f_lac ({r['f_rock']:.1f} Hz) cao hon han f_do (~1.2 Hz):")
    print( "     tan so do duoc KHONG phai lac tren do cung doc truc coc.")
    print( "     Co che chi phoi la LU NGANG cua nhom coc, trong do coc xien")
    print( "     dong gop phan lon do cung.")

    print("\n" + "=" * 74)
    print("3. DO NHAY THEO CHIEU DAI NGAM HIEU DUNG VA DO CUNG DOC TRUC")
    print("=" * 74)
    print(f"  {'L_eff [m]':>10} " + " ".join(f"{k:>9.0e}" for k in [2e8, 3e8, 5e8, 8e8]))
    print(f"  {'':>10} " + " ".join(f"{'f [Hz]':>9}" for _ in range(4)))
    for Le in [L_FREE + 3 * D_OUT, L_FREE + 5 * D_OUT, L_FREE + 8 * D_OUT, L_FREE + 12 * D_OUT]:
        row = [frequencies(k, Le)['f_sway'] for k in [2e8, 3e8, 5e8, 8e8]]
        print(f"  {Le:>10.1f} " + " ".join(f"{v:>9.2f}" for v in row))

    print("\n" + "=" * 74)
    print("4. HIEU CHINH NGUOC TU TAN SO DO DUOC (dot 1, chua co thap)")
    print("=" * 74)
    for Le, tag in [(L_FREE + 3 * D_OUT, "ngam nong"),
                    (L_FREE + 5 * D_OUT, "trung binh"),
                    (L_FREE + 8 * D_OUT, "ngam sau")]:
        k_ax, k_need = solve_kax_for_frequency(F_MEAS_C1, Le)
        print(f"  L_eff = {Le:5.1f} m ({tag:10s}): de f = {F_MEAS_C1} Hz thi can "
              f"k_ax = {k_ax:.3e} N/m/coc")
    k_ax_mid, _ = solve_kax_for_frequency(F_MEAS_C1, L_FREE + 5 * D_OUT)
    print(f"\n  De so sanh, do cung dan hoi thuan tuy cua doan coc tu do:")
    print(f"     EA/L_free = {EA/L_FREE:.3e} N/m")
    print(f"     k_ax hieu chinh / (EA/L_free) = {k_ax_mid/(EA/L_FREE):.2f}")

    print("\n" + "=" * 74)
    print("5. DOI CHIEU")
    print("=" * 74)
    print(f"  Do duoc, dot 1 (chua thap) : {F_MEAS_C1:.3f} Hz")
    print(f"  Do duoc, dot 2 (co thap)   : {F_MEAS_C2:.3f} Hz")
    print( "  Uoc luong nguyen ly dau    : xem bang muc 3 - nam trong cung khoang")
    print( "  => uoc luong nguyen ly dau bao trum ca hai gia tri do duoc;")
    print( "     xem muc 3 ve do nhay theo gia thiet.")

    print("\n" + "=" * 74)
    print("6. SUY NGUOC KHOI LUONG THAP THAM GIA DAO DONG")
    print("=" * 74)
    print("  Gia thiet: do cung ngang KHONG doi giua hai dot (chi them khoi luong).")
    print("  Khi do  M2/M1 = (f1/f2)^2 .")
    M1 = p['m_cap'] + p['m_piles'] + p['m_added']
    ratio = (F_MEAS_C1 / F_MEAS_C2) ** 2
    dM = (ratio - 1) * M1
    print(f"     M1 (dai + coc + nuoc kem theo) = {M1/1e6:.2f} x10^6 kg = {M1/1e3:.0f} tan")
    print(f"     (f1/f2)^2 = ({F_MEAS_C1}/{F_MEAS_C2})^2 = {ratio:.4f}")
    print(f"     => khoi luong tang them  dM = {dM/1e3:.0f} tan")
    print("  Doi chieu: thap + nacelle + rotor cua tua-bin co cong suat nay thuong")
    print("  vao khoang 500-700 tan. Neu dM suy ra lon hon dang ke thi hoac do cung")
    print("  cung thay doi, hoac mot phan chenh lech la do dieu kien moi truong.")
    print("  => day la mot phep kiem tra doc lap, KHONG can mo hinh dat nen.")

    print("\n" + "=" * 74)
    print("CAN XAC NHAN DE CHOT SO")
    print("=" * 74)
    print("  - Chieu day vach coc PHC D1000 va cap do ben be tong (ban ve chi tiet coc)")
    print("  - Bao cao dia chat day du: suc khang cat khong thoat nuoc theo do sau,")
    print("    de tinh do cung doc truc coc thay vi gia thiet")
    print("  - Khoi luong thap + nacelle + rotor cho trang thai dot 2")
    print("  - Khoi luong thuc cua dai (co cot thep, long neo) thay vi be tong thuan")


if __name__ == '__main__':
    main()


# ============================================================================
# TRA LOI PHAN BIEN: hieu ung nhom coc, va uoc luong k_ax doc lap
# ============================================================================

# SPT trung binh theo tung lop trong doan coc ngam (tu ho khoan DH9)
# (chieu day [m], N trung binh)
EMBEDDED_LAYERS = [(2.6, 10.0), (13.4, 1.4), (2.5, 16.0),
                   (4.2, 26.0), (7.8, 21.0), (3.1, 27.0)]
L_EMB = sum(t for t, _ in EMBEDDED_LAYERS)


def group_effect_bounds(eta_ratio_range=(0.5, 2.0)):
    """Ty so f_rock/f_sway khi hai co che co he so hieu qua nhom KHAC nhau.

    Phan bien dung: k_ax khong triet tieu neu he so hieu qua nhom cho lac
    (eta_rock) khac cho lu ngang (eta_sway). Khi do
        f_rock/f_sway = 7.69 * sqrt(eta_rock/eta_sway)
    """
    base = rock_sway_ratio()['ratio_no_bending']
    lo = base * np.sqrt(eta_ratio_range[0])
    hi = base * np.sqrt(eta_ratio_range[1])
    return base, lo, hi


def g0_ohsaki_iwasaki(N):
    """Mo-dun cat bien dang nho tu SPT, Ohsaki & Iwasaki (1973):
    G0 = 1200 N^0.8 [t/m2], tu tham do dia chan, ap dung cho moi loai dat.
    Tra ve Pa.
    """
    return 1200.0 * N ** 0.8 * 9.80665e3


def kax_soil_rw1978(G_scale=1.0, nu=0.3):
    """Do cung dau coc cua DOAN COC NGAM, Randolph & Wroth (1978), ke ca do
    nen dan hoi cua coc va mui coc (khong dung gioi han coc cung 2*pi*G*L/zeta,
    vi coc dai L/r0 = 67 co muL ~ 3-4.5).

    G(z) lay theo Ohsaki & Iwasaki tung lop, nhan G_scale (1.0 = bien dang nho,
    phu hop dao dong moi truong; 0.5 = can duoi cho phan tan cua tuong quan va
    xao tron set mem khi ha coc).
        P/(w G_L r0) = [4/((1-nu)xi) + (2 pi rho/zeta)(tanh muL/muL)(L/r0)]
                       / [1 + (1/(pi lam)) (4/((1-nu)xi)) (tanh muL/muL)(L/r0)]
    """
    r0 = D_OUT / 2
    G_avg = G_scale * sum(t * g0_ohsaki_iwasaki(n) for t, n in EMBEDDED_LAYERS) / L_EMB
    G_L = G_scale * g0_ohsaki_iwasaki(EMBEDDED_LAYERS[-1][1])   # lop o mui coc
    G_b = G_L                                                      # duoi mui: lay bang G_L
    rho, xi = G_avg / G_L, G_L / G_b
    r_m = (0.25 + xi * (2.5 * rho * (1 - nu) - 0.25)) * L_EMB
    zeta = np.log(r_m / r0)
    lam = (EA / (np.pi * r0 ** 2)) / G_L                           # coc dac tuong duong
    muL = np.sqrt(2.0 / (zeta * lam)) * L_EMB / r0
    t = np.tanh(muL) / muL
    base = 4.0 / ((1 - nu) * xi)
    k = G_L * r0 * (base + 2 * np.pi * rho / zeta * t * L_EMB / r0) \
        / (1 + base * t * L_EMB / r0 / (np.pi * lam))
    return k, dict(G_avg=G_avg, G_L=G_L, rho=rho, zeta=zeta, lam=lam, muL=muL)


def kax_from_soil(G_scales=(0.5, 1.0), nu=0.3):
    """Uoc luong DOC LAP do cung doc truc MOT COC CO LAP tu ho so dia chat:
    doan ngam (Randolph & Wroth 1978) noi tiep doan tu do EA/L_free.
    Tuong quan SPT-G0 RAT THO - ket qua chi dung o muc bac do lon.
    Do cung nhom (co tuong tac coc-dat-coc) tinh trong group_interaction.py.
    """
    k_free = EA / L_FREE
    out = {}
    for s in G_scales:
        k_soil, p = kax_soil_rw1978(s, nu)
        out[s] = dict(k_soil=k_soil, k_iso=1.0 / (1.0 / k_soil + 1.0 / k_free), **p)
    return k_free, out


if __name__ != '__main__':
    pass


def dimensionless_ratio():
    """Dang khong thu nguyen cua tieu chi (muc 3.3 cua bai bao).

    Voi nhom coc co MOT goc xien alpha:
        sum r^2 cos^2(a) = cos^2(a) * sum r^2 = cos^2(a) * n * r_rms^2
        sum sin^2(a)     = n * sin^2(a)
    nen
        f_rock/f_sway = sqrt( (M/J) * sum r^2cos^2a / sum sin^2a )
                      = (r_rms / k_g) * cot(alpha)
    voi r_rms = sqrt(sum r^2 / n) va k_g = sqrt(J/M).

    Hai co che bang nhau khi tan(alpha) = r_rms / k_g.
    """
    p = cap_mass_inertia()
    M = p['m_cap'] + p['m_piles'] + p['m_added']
    J = p['J_cap'] + p['J_piles']
    k_g = np.sqrt(J / M)
    n = sum(nn for _, nn, _ in RINGS)
    r_rms = np.sqrt(sum(nn * r ** 2 for r, nn, _ in RINGS) / n)
    alpha_crit = np.arctan(r_rms / k_g)
    return dict(k_g=k_g, r_rms=r_rms, shape=r_rms / k_g,
                alpha_crit_deg=np.degrees(alpha_crit),
                ratio=lambda a_rad: (r_rms / k_g) / np.tan(a_rad))
