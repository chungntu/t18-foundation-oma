"""Duong dan chuan cua du an. Moi script deu lay duong dan tu day.

Ban ghi do co the nam ngoai repo: dat bien moi truong T18_DATA tro toi thu
muc chua dot1/ va dot2/ (vd. T18_DATA=/path/to/Dien_gio_TraVinh).
"""

import os

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)

DATA = os.environ.get("T18_DATA",
                      os.path.join(ROOT, "data", "Dien_gio_TraVinh"))  # ban ghi do (khong day len git)
DRAWINGS = os.path.join(ROOT, "data", "drawings")        # ban ve thiet ke (khong day len git)
RESULTS = os.path.join(ROOT, "results")                  # CSV va hinh chan doan
FIGURES = os.path.join(ROOT, "paper", "figures")         # hinh dung trong bai bao

for _d in (RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)


def save_figure(fig, path_png, dpi=200):
    """Luu hinh bai bao: PNG (xem nhanh) va PDF vector (dung trong LaTeX va
    nop tap chi; ban TIFF 600 dpi duoc dung tu PDF khi dong goi)."""
    fig.savefig(path_png, dpi=dpi)
    fig.savefig(os.path.splitext(path_png)[0] + ".pdf")
