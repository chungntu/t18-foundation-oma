"""Chay lai toan bo phan tich va hinh cua bai bao, theo dung thu tu.

    T18_DATA=/path/to/Dien_gio_TraVinh  python run_all.py

T18_DATA tro toi thu muc chua dot1/ va dot2/ (xem src/paths.py). Neu khong
dat, du lieu duoc tim o data/Dien_gio_TraVinh trong repo.
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    "src/oma/run_dot1.py",                 # nhan dang, dot 1 (4 setup)
    "src/oma/run_dot2.py",                 # nhan dang, dot 2 (3 setup)
    "src/oma/participation.py",            # so kenh hieu dung N_eff
    "src/oma/configuration_experiment.py", # thi nghiem 2x2: cua so hiep phuong sai vs loc
    "src/oma/low_band_search.py",          # tim mode thap duoi 0.5 Hz
    "src/oma/criterion_calibration.py",    # nguong N_eff bang mo phong
    "src/oma/make_tables.py",              # du lieu cac bang
    "src/model/foundation_stiffness.py",   # tieu chi hinh hoc, do cung
    "src/model/group_interaction.py",      # he so hieu qua nhom, do cung thuan
    "src/figures/make_figures.py",         # hinh 1, 2, 7, 11
    "src/figures/make_figures_extra.py",   # hinh 3-6, 8, 9, 12, 13
    "src/figures/fig_harmonics.py",        # hinh 10
]


def main():
    from src.paths import DATA
    if not os.path.isdir(os.path.join(DATA, "dot1")):
        sys.exit(f"Khong thay du lieu o {DATA}. Dat bien moi truong T18_DATA.")
    t0 = time.time()
    for step in STEPS:
        print(f"\n===== {step}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, step)], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"LOI o buoc {step} (ma {r.returncode})")
    print(f"\n[xong] {len(STEPS)} buoc, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
