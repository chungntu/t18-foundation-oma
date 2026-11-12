"""Duong dan chuan cua du an. Moi script deu lay duong dan tu day.

Bo cuc thu muc bai bao (paper_folder_and_file_naming_rule.md):
    <paper root>/manuscript/figures          hinh dung trong bai bao
    <paper root>/source_code_and_data/scripts   thu muc nay
    <paper root>/source_code_and_data/results   CSV va hinh chan doan
    <paper root>/source_code_and_data/data/raw  ban ghi do (dot1/, dot2/), khong day len git
    <paper root>/reference_materials/design_documents   ho so thiet ke, khong day len git

Ban ghi do co the nam ngoai du an: dat bien moi truong T18_DATA tro toi thu
muc chua dot1/ va dot2/.
"""

import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS)                            # source_code_and_data/
PAPER_ROOT = os.path.dirname(ROOT)

DATA = os.environ.get("T18_DATA", os.path.join(ROOT, "data", "raw"))
DRAWINGS = os.path.join(PAPER_ROOT, "reference_materials", "design_documents")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(PAPER_ROOT, "manuscript", "figures")

for _d in (RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)


def save_figure(fig, path_png, dpi=200):
    """Luu hinh bai bao: PNG (xem nhanh) va PDF vector (dung trong LaTeX va
    nop tap chi; ban TIFF 600 dpi duoc dung tu PDF khi dong goi)."""
    fig.savefig(path_png, dpi=dpi)
    fig.savefig(os.path.splitext(path_png)[0] + ".pdf")
