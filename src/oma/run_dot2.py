"""
Nhan dang modal DOT 2 (sau khi lap thap) - dung dung pipeline nhu dot 1.

Khac biet cot loi so voi logfile.m cua nhom (MACEC):
  - CO decimate 1651 -> 51.6 Hz truoc khi SSI   (nhom: khong decimate)
  - Loc thong cao 0.3 Hz                        (nhom: 5 Hz, co luc 100 Hz)
  - Cua so tre i*dt = 2.33 s                    (nhom: 150/1651 = 0.09 s)
=> nhan dang duoc mode ket cau 0.5-25 Hz thay vi bam vao harmonic 50 Hz.

Chay:  python oma/run_dot2.py
"""

import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # src/
from paths import DATA, RESULTS, FIGURES

from pipeline import (analyze, plot_stabilization, plot_fdd, write_csv, print_mac)

SETDIR = os.path.join(DATA, "dot2")
OUT = os.path.join(RESULTS, "dot2")

SETUPS = [
    ("xl_ngang.mat", "xl_ngang", "Lateral, campaign 2"),
    ("xl_td1.mat", "xl_td1", "Vertical d1, campaign 2"),
    ("xl_td2.mat", "xl_td2", "Vertical d2ke, campaign 2"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    results = [analyze(os.path.join(SETDIR, f), v, l) for f, v, l in SETUPS]

    for r in results:
        plot_stabilization(r, os.path.join(OUT, f"stab_{r['var']}.png"))
    plot_fdd(results, os.path.join(OUT, "fdd_dot2.png"))
    write_csv(results, os.path.join(OUT, "modes_dot2.csv"))

    r1 = next(r for r in results if r['var'] == 'xl_td1')
    r2 = next(r for r in results if r['var'] == 'xl_td2')
    print_mac(r1, r2, os.path.join(OUT, "MAC_td1_td2.csv"))

    print(f"\n[xong] ket qua trong {OUT}")


if __name__ == '__main__':
    main()
