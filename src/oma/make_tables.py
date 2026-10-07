"""
Sinh so lieu cho cac bang cua bai bao.

  Bang 2 - thong ke chat luong ban ghi tho, ca bay cau hinh
  Bang 4 - cac cuc nhan dang duoc theo cau hinh
  Bang 5 - quan he pha giua cac goc doi dien va MAC giua cac cau hinh
  Bang 9 - trung binh theo cau hinh va muc thay doi giua hai dot

Ket qua ghi ra results/tables/*.csv va in ra man hinh.

Chay:  python src/oma/make_tables.py
"""

import os
import sys
import csv
import numpy as np
import scipy.io as sio
from scipy import signal, linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from paths import DATA, RESULTS
from oma_lib import preprocess, mac, band_poles_by_setup

FS_RAW = 51200 / 31            # Hz = 1651.61; wf_increment = 31/51200 s trong metadata TDMS
OUT = os.path.join(RESULTS, "tables")
os.makedirs(OUT, exist_ok=True)

SETUPS = [
    ("dot1/xl_thangdung.mat", "xl_thangdung", 1, "Vertical"),
    ("dot1/xl_ngang1.mat", "xl_ngang1", 1, "Lateral 1"),
    ("dot1/xl_ngang2.mat", "xl_ngang2", 1, "Lateral 2"),
    ("dot1/xl_ga.mat", "xl_ga", 1, "Three-channel"),
    ("dot2/xl_ngang.mat", "xl_ngang", 2, "Lateral"),
    ("dot2/xl_td1.mat", "xl_td1", 2, "Vertical d1"),
    ("dot2/xl_td2.mat", "xl_td2", 2, "Vertical d2ke"),
]


def kurt(v):
    v = v - v.mean()
    return float(np.mean(v ** 4) / np.mean(v ** 2) ** 2)


def _write(name, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[luu]", name)


# ---------------------------------------------------------------- BANG 2
def table2():
    rows = []
    for rel, var, camp, name in SETUPS:
        x = sio.loadmat(os.path.join(DATA, *rel.split("/")))[var].astype(float)
        x = signal.detrend(x, axis=0, type="constant")
        n, nch = x.shape
        sd = x.std(axis=0)
        peak = float((np.abs(x).max(axis=0) / sd).max())
        n8 = int(max((np.abs(x[:, c]) > 8 * sd[c]).sum() for c in range(nch)))
        seg = np.array_split(x, 10, axis=0)
        rms = np.array([s.std(axis=0) for s in seg])
        drift = float((rms.max(axis=0) / np.maximum(rms.min(axis=0), 1e-12)).max())
        spread = float(sd.max() / sd.min())
        rows.append(dict(campaign=camp, setup=name, channels=nch, samples=n,
                         minutes=round(n / FS_RAW / 60, 1),
                         peak_over_sigma=round(peak),
                         max_kurtosis=round(max(kurt(x[:, c]) for c in range(nch))),
                         samples_over_8sigma=n8,
                         rms_drift=round(drift, 1),
                         channel_spread=round(spread, 1)))
    _write("table2_records.csv", rows)
    print()
    print("=== BANG 2  thong ke ban ghi ===")
    hdr = "{:>2} {:<15}{:>3}{:>10}{:>6}{:>9}{:>8}{:>7}{:>10}{:>10}"
    print(hdr.format("C", "setup", "ch", "samples", "min", "peak/sd",
                     "kurt", ">8sd", "RMSdrift", "spread"))
    for r in rows:
        print(hdr.format(r["campaign"], r["setup"], r["channels"], r["samples"],
                         r["minutes"], r["peak_over_sigma"], r["max_kurtosis"],
                         r["samples_over_8sigma"], r["rms_drift"], r["channel_spread"]))


# ---------------------------------------------------------------- BANG 4
def table4():
    rows = []
    for camp, fn in ((1, "dot1/modes_dot1.csv"), (2, "dot2/modes_dot2.csv")):
        p = os.path.join(RESULTS, *fn.split("/"))
        with open(p, encoding="utf-8") as fh:
            for d in csv.DictReader(fh):
                fv = float(d["f_Hz"])
                rows.append(dict(campaign=camp, setup=d["setup"], f_Hz=fv,
                                 xi_pct=float(d["xi_percent"]),
                                 n_poles=int(d["n_cuc_on_dinh"]),
                                 MPC=float(d["MPC"]),
                                 band="low" if fv < 5 else "high"))
    _write("table4_poles.csv", rows)
    print()
    print("=== BANG 4  cac cuc nhan dang duoc ===")
    for camp in (1, 2):
        print("  -- campaign {} --".format(camp))
        setups = list(dict.fromkeys(r["setup"] for r in rows if r["campaign"] == camp))
        for s in setups:
            for band in ("low", "high"):
                g = [r for r in rows if r["setup"] == s and r["band"] == band]
                txt = "  ".join("{:.3f} ({:.2f})".format(r["f_Hz"], r["xi_pct"]) for r in g)
                print("    {:<26} {:<5} {}".format(s if band == "low" else "", band,
                                                   txt if txt else "-"))


# ---------------------------------------------------------------- BANG 5
PHASE_CASES = [
    ("dot1/xl_thangdung.mat", "xl_thangdung", "Vertical, C1", [1, 2, 3, 4, 5], "V"),
    ("dot2/xl_td1.mat", "xl_td1", "Vertical d1, C2", [1, 2, 3, 4, 5, 6], "V"),
    ("dot2/xl_td2.mat", "xl_td2", "Vertical d2ke, C2", [1, 2, 3, 4, 5, 7], "V"),
    ("dot2/xl_ngang.mat", "xl_ngang", "Lateral, C2", [1, 2, 3, 4, 5, 6], "H"),
]


def shape_at_peak(y, fs, band=(1.00, 1.40), nperseg=8192):
    f, P = signal.csd(y[:, :, None], y[:, None, :], fs=fs, nperseg=nperseg, axis=0)
    sv1 = np.array([linalg.svdvals(P[k])[0] for k in range(len(f))])
    m = (f >= band[0]) & (f <= band[1])
    k = np.where(m)[0][np.argmax(sv1[m])]
    U, _, _ = linalg.svd(P[k])
    phi = U[:, 0]
    j = int(np.argmax(np.abs(phi)))
    return float(f[k]), phi / phi[j]


def table5():
    rows = {}
    shapes = {}
    for rel, var, name, nodes, kind in PHASE_CASES:
        x = sio.loadmat(os.path.join(DATA, *rel.split("/")))[var].astype(float)
        y, fs = preprocess(x, FS_RAW, dec_factors=(4, 8), f_high=0.3)
        fpk, phi = shape_at_peak(y, fs)
        shapes[name] = phi
        idx = {nd: i for i, nd in enumerate(nodes)}
        d13 = float(np.degrees(np.angle(phi[idx[1]] / phi[idx[3]])))
        d24 = float(np.degrees(np.angle(phi[idx[2]] / phi[idx[4]])))
        corners = [abs(phi[idx[n]]) for n in (1, 2, 3, 4)]
        rows[name] = dict(setup=name, direction=kind, f_Hz=round(fpk, 3),
                          phase_1_3_deg=round(d13, 1), phase_2_4_deg=round(d24, 1),
                          centre_amp=round(float(abs(phi[idx[5]])), 3),
                          mean_corner_amp=round(float(np.mean(corners)), 3))
    _write("table5_phase.csv", list(rows.values()))
    print()
    print("=== BANG 5  quan he pha ===")
    hdr = "{:<20}{:>4}{:>9}{:>11}{:>11}{:>9}{:>9}"
    print(hdr.format("setup", "dir", "f [Hz]", "1-3 [deg]", "2-4 [deg]", "centre", "corners"))
    for r in rows.values():
        print(hdr.format(r["setup"], r["direction"], r["f_Hz"], r["phase_1_3_deg"],
                         r["phase_2_4_deg"], r["centre_amp"], r["mean_corner_amp"]))
    # Hai dai luong KHAC NHAU, deu can bao cao:
    #  (a) MAC giua dang mode SSI cua hai cum TUONG UNG  -> cung mot nhanh
    #  (b) MAC giua vec-to ky di FDD tai dinh pho TROI cua tung setup
    #      -> moi setup troi mot thanh vien khac cua cap mode gan nhau
    a = shapes["Vertical d1, C2"]
    b = shapes["Vertical d2ke, C2"]
    print()
    print("  MAC (b) vec-to FDD tai dinh troi cua tung setup, 5 kenh chung = {:.3f}"
          .format(mac(a[:5], b[:5])))
    print("     td1 troi o {:.3f} Hz, td2 troi o {:.3f} Hz - hai thanh vien khac nhau"
          .format(rows["Vertical d1, C2"]["f_Hz"], rows["Vertical d2ke, C2"]["f_Hz"]))
    print("  MAC (a) dang mode SSI cua hai cum tuong ung, 5 kenh chung = 0.967")
    print("     (tinh trong run_dot2.py; 6 kenh cho 0.966, chenh lech khong dang ke)")


# ---------------------------------------------------------------- BANG 9
C1, C2 = band_poles_by_setup(RESULTS)


# huong do: V = moi kenh dung; H = kenh ngang (kenh 4 dung o dot 2, chua ghi nhan o dot 1);
# M = hon hop (ga.mat: kenh "dung", "ngang1", "ngang2")
DIRECTION = {"Vertical": "V", "Lateral 1": "H", "Lateral 2": "H", "Three-channel": "M",
             "Lateral": "H", "Vertical d1": "V", "Vertical d2ke": "V"}


def table9():
    rows = []
    for camp, d in ((1, C1), (2, C2)):
        for name, v in d.items():
            rows.append(dict(campaign=camp, setup=name, direction=DIRECTION[name],
                             n_poles=len(v), poles=" ".join("{:.3f}".format(x) for x in v),
                             setup_mean=round(float(np.mean(v)), 4)))
    _write("table9_shift.csv", rows)
    print()
    print("=== BANG 9  trung binh theo cau hinh ===")
    for r in rows:
        print("  C{}  {:<15} {} {:<24} -> {}".format(r["campaign"], r["setup"], r["direction"],
                                                     r["poles"], r["setup_mean"]))
    print()
    print("  So sanh CUNG HUONG do (dot 2 so voi dot 1):")
    for code in ("V", "H"):
        for n1, v1 in C1.items():
            if DIRECTION[n1] != code:
                continue
            for n2, v2 in C2.items():
                if DIRECTION[n2] != code:
                    continue
                a, b = np.mean(v1), np.mean(v2)
                print("    {}  {:<14} -> {:<14} {:+6.2f} %".format(code, n1, n2, 100 * (b - a) / a))


if __name__ == "__main__":
    table2()
    table4()
    table5()
    table9()
