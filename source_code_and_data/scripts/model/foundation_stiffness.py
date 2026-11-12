"""
So lieu hinh hoc, tiet dien, khoi luong va do cung doc truc coc cua mong T18.

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

Nguon so lieu (Tap 4.1 - Thuyet minh, va Tap 4.3 - Bang tinh, 20/03/2021):
  - Tap 4.1 muc III.8: be tron duong kinh 20.5 m, tong chieu day 3 m, trong do
        1 m phia tren vat tu mep vao den co mong; co mong duong kinh 6 m, day 1 m
        (tong chieu cao 4 m, khop voi +5.435 .. +9.435).
  - Tap 4.3 (kiem toan lun, mong 34 coc): h1 = 4 m, Vfound = 830.131 m3,
        gamma_c = 25 kN/m3, Ffound = 20753 kN; coc PHC D1000 loai C, vach 130 mm,
        Ac = 0.355 m2.
  - Tap 4.1 (vat lieu): be tong coc PHC C80, mo-dun dan hoi 42933 MPa.
  - Tap 4.1 muc III.6: thap cao 88 m, khoi luong tap trung dinh thap 256312 kg;
        Tap 4.3: tai dung tai dinh mong, to hop gan nhu vinh vien, Fz = 4681 kN.

Module nay cung cap so lieu cho mo hinh dai cung 6 bac tu do (rigid_cap_6dof.py)
va uoc luong THUAN do cung doc truc mot coc tu ho so dia chat (muc 6.2 bai bao).

Chay:  python scripts/model/foundation_stiffness.py
"""

import numpy as np

# ----------------------------------------------------------------- hinh hoc
RINGS = [  # (ban kinh [m], so coc, do xien n trong "n:1"; None = thang dung)
    (3.0, 4, None),
    (6.0, 12, 7.0),
    (9.0, 18, 5.0),
]
D_CAP = 20.5          # m, duong kinh dai
T_CAP = 4.0           # m, tong chieu cao dai ke ca co mong (+5.435 .. +9.435)
T_SLAB = 2.0          # m, phan be day deu
T_TAPER = 1.0         # m, phan vat tu mep be vao co mong
D_NECK = 6.0          # m, duong kinh co mong
T_NECK = 1.0          # m, chieu cao co mong
V_CAP_DESIGN = 830.131  # m3, the tich be mong theo bang tinh (Tap 4.3)
L_FREE = 12.335       # m, tu day dai +5.435 den day bun -6.900
L_PILE = 46.0         # m, chieu dai coc tinh toan

# ------------------------------------------- tiet dien coc (theo ho so thiet ke)
D_OUT = 1.000         # m, PHC D1000 loai C
T_WALL = 0.130        # m, Tap 4.3: "wall thickness 130mm", Din = 740 mm
E_PILE = 42.933e9     # Pa, Tap 4.1: be tong coc PHC C80, mo-dun dan hoi 42933 MPa
RHO_C = 2500.0        # kg/m3
RHO_W = 1025.0        # kg/m3

D_IN = D_OUT - 2 * T_WALL
A_PILE = np.pi / 4 * (D_OUT ** 2 - D_IN ** 2)
I_PILE = np.pi / 64 * (D_OUT ** 4 - D_IN ** 4)
EA = E_PILE * A_PILE
EI = E_PILE * I_PILE

# SPT trung binh theo tung lop trong doan coc ngam (tu ho khoan DH9)
# (chieu day [m], N trung binh)
EMBEDDED_LAYERS = [(2.6, 10.0), (13.4, 1.4), (2.5, 16.0),
                   (4.2, 26.0), (7.8, 21.0), (3.1, 27.0)]
L_EMB = sum(t for t, _ in EMBEDDED_LAYERS)


def batter_angle(n):
    """Goc nghieng so voi phuong thang dung cua coc xien n:1 [rad]."""
    return 0.0 if n is None else np.arctan(1.0 / n)


def cap_radius(z):
    """Ban kinh dai o cao do z tren day dai: be day deu, phan vat, co mong."""
    R, r = D_CAP / 2, D_NECK / 2
    if z <= T_SLAB:
        return R
    if z <= T_SLAB + T_TAPER:
        return R + (r - R) * (z - T_SLAB) / T_TAPER
    return r


def cap_body(n=4000):
    """The tich, cao do trong tam va mo-men quan tinh (tren don vi khoi luong)
    cua dai, tich phan theo cac lat tron mong. Hinh dang lay theo Tap 4.1."""
    dz = T_CAP / n
    z = (np.arange(n) + 0.5) * dz
    r = np.array([cap_radius(v) for v in z])
    dV = np.pi * r ** 2 * dz
    V = dV.sum()
    zc = (dV * z).sum() / V
    izz = (dV * r ** 2 / 2).sum() / V                           # quanh truc dung
    ixx = (dV * (r ** 2 / 4 + (z - zc) ** 2)).sum() / V          # quanh truc ngang qua trong tam
    return V, zc, ixx, izz


def cap_mass_inertia():
    """Khoi luong dai, coc va nuoc kem theo (khong ke thap).

    Khoi luong dai lay theo the tich trong bang tinh thiet ke (830.131 m3);
    trong tam va mo-men quan tinh lay theo hinh dang (be + phan vat + co mong),
    the tich hinh hoc cua hinh dang nay la 840 m3, lech 1.2 %.
    """
    V_shape, z_cg, ixx, izz = cap_body()
    V_cap = V_CAP_DESIGN
    m_cap = V_cap * RHO_C
    m_pile_each = A_PILE * L_PILE * RHO_C
    n_pile = sum(n for _, n, _ in RINGS)
    m_added = RHO_W * np.pi / 4 * D_OUT ** 2 * L_FREE * n_pile
    return dict(m_cap=m_cap, m_piles=m_pile_each * n_pile, m_added=m_added,
                V_cap=V_cap, n_pile=n_pile, V_shape=V_shape, z_cg=z_cg,
                Ixx_cg=m_cap * ixx, Izz=m_cap * izz)


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


def pile_bending_frequency(L, end="fixed-fixed"):
    """Tan so uon thu nhat cua MOT coc, khoi luong gom coc va nuoc kem theo [Hz]."""
    beta_L = {"fixed-fixed": 4.730, "fixed-pinned": 3.927}[end]
    m = A_PILE * RHO_C + RHO_W * np.pi / 4 * D_OUT ** 2
    return beta_L ** 2 / (2 * np.pi * L ** 2) * np.sqrt(EI / m)


def main():
    p = cap_mass_inertia()
    print("=" * 74)
    print("SO LIEU DAN XUAT TU BAN VE")
    print("=" * 74)
    print(f"  Tiet dien coc PHC D1000 (vach {T_WALL*1000:.0f} mm, E = {E_PILE/1e9:.2f} GPa, theo ho so thiet ke):")
    print(f"     A  = {A_PILE:.4f} m2      I  = {I_PILE:.5f} m4")
    print(f"     EA = {EA:.3e} N        EI = {EI:.3e} N.m2")
    print(f"  Dai: the tich {p['V_cap']:.0f} m3 (hinh dang: {p['V_shape']:.0f} m3), "
          f"khoi luong {p['m_cap']/1e6:.3f} x10^6 kg, trong tam {p['z_cg']:.2f} m tren day dai")
    print(f"       Ixx (qua trong tam) = {p['Ixx_cg']:.3e} kg.m2, Izz = {p['Izz']:.3e} kg.m2")
    print(f"  {p['n_pile']} coc: khoi luong {p['m_piles']/1e6:.2f} x10^6 kg, "
          f"nuoc kem theo {p['m_added']/1e6:.2f} x10^6 kg")
    print("\n  Tieu diem tung vong coc xien, r cot(alpha):")
    for r, n, nb in RINGS:
        if nb:
            print(f"     R = {r:.0f} m, {nb:.0f}:1  ->  {r / np.tan(batter_angle(nb)):.1f} m tren dau coc")

    print("\n" + "=" * 74)
    print("DO CUNG DOC TRUC MOT COC CO LAP, UOC LUONG THUAN TU SPT (muc 6.2)")
    print("=" * 74)
    k_free, fwd = kax_from_soil()
    print(f"  EA/L_free = {k_free:.3e} N/m")
    for s, r in fwd.items():
        print(f"  G = {s:.1f} G0: G_avg = {r['G_avg']/1e6:.1f} MPa, muL = {r['muL']:.2f}, "
              f"k_ngam = {r['k_soil']:.2e}, k_co_lap = {r['k_iso']:.2e} N/m")

    print("\n  Tan so uon thu nhat cua mot coc (L_eff suy nguoc 14.2-17.0 m, muc 6.2):")
    for L in (14.2, 17.0):
        print(f"    L = {L:.1f} m: ngam-ngam {pile_bending_frequency(L):.1f} Hz,"
              f" ngam-khop {pile_bending_frequency(L, 'fixed-pinned'):.1f} Hz")


if __name__ == '__main__':
    main()
