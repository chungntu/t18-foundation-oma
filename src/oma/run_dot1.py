"""
Nhan dang modal DOT 1 (truoc khi lap thap) - dung dung pipeline nhu dot 2.

Chay:  python oma/run_dot1.py
"""

import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # src/
from paths import DATA, RESULTS, FIGURES

from pipeline import (analyze, plot_stabilization, plot_fdd, write_csv, print_mac)

SETDIR = os.path.join(DATA, "dot1")
OUT = os.path.join(RESULTS, "dot1")

SETUPS = [
    ("xl_thangdung.mat", "xl_thangdung", "Vertical, campaign 1"),
    ("xl_ngang1.mat", "xl_ngang1", "Lateral 1, campaign 1"),
    ("xl_ngang2.mat", "xl_ngang2", "Lateral 2, campaign 1"),
    ("xl_ga.mat", "xl_ga", "Three-channel setup, campaign 1"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    results = [analyze(os.path.join(SETDIR, f), v, l) for f, v, l in SETUPS]

    for r in results:
        plot_stabilization(r, os.path.join(OUT, f"stab_{r['var']}.png"))
    plot_fdd(results, os.path.join(OUT, "fdd_dot1.png"))
    write_csv(results, os.path.join(OUT, "modes_dot1.csv"))

    # cac setup 5 kenh cung phuong ngang -> so sanh dang mode
    r1 = next(r for r in results if r['var'] == 'xl_ngang1')
    r2 = next(r for r in results if r['var'] == 'xl_ngang2')
    print_mac(r1, r2, os.path.join(OUT, "MAC_ngang1_ngang2.csv"))

    print(f"\n[xong] ket qua trong {OUT}")


if __name__ == '__main__':
    main()
