"""Chay lai toan bo phan tich va hinh cua bai bao, theo dung thu tu.

    python source_code_and_data/scripts/run_all.py

Ban ghi do duoc tim o source_code_and_data/data/raw (chua dot1/ va dot2/).
Dat bien moi truong T18_DATA de tro toi mot thu muc khac (xem paths.py).
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    "oma/run_campaign_1.py",                 # nhan dang, dot 1 (4 setup)
    "oma/run_campaign_2.py",                 # nhan dang, dot 2 (3 setup)
    "oma/participation.py",                    # so kenh hieu dung N_eff
    "oma/configuration_experiment.py", # thi nghiem 2x2: cua so hiep phuong sai vs loc
    "oma/low_band_search.py",          # tim mode thap duoi 0.5 Hz
    "oma/criterion_calibration.py",    # nguong N_eff bang mo phong, do nhay
    "oma/make_tables.py",              # du lieu cac bang
    "oma/rotation_centre_test.py",     # ty so dung/ngang, mode xoan (muc 6.1)
    "oma/segment_uncertainty.py",      # bat dinh tan so theo ba doan ban ghi
    "model/foundation_stiffness.py",   # so lieu hinh hoc, do cung doc truc thuan
    "model/rigid_cap_6dof.py",         # dai cung 6 bac tu do, tieu diem, suy nguoc L_eff
    "model/group_interaction.py",      # he so hieu qua nhom, do cung thuan
    "model/tower_sensitivity.py",      # do nhay tam xoay voi thap (Supplement S7)
    "figures/make_figures.py",         # pile_layout, foundation_section, mode_shape
    "figures/make_figures_extra.py",   # cac hinh con lai
    "figures/fig_harmonics.py",        # processing_configurations
]


def main():
    from paths import DATA
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
