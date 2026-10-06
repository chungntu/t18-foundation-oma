"""Tao lai danh muc tai lieu trong paper.tex tu references.bib.

Kieu Springer basic theo huong dan cua Journal of Civil Structural Health
Monitoring, danh so theo thu tu trich dan lan dau:
    Author AB, Author CD (Year) Title. Journal Vol(No):pages. https://doi.org/...

    python paper/make_references.py
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "paper.tex"
BIB = HERE / "references.bib"


def parse_bib(text):
    """Tra ve {key: (type, {field: value})}; gia tri giu nguyen ma LaTeX."""
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        typ, key, i = m.group(1).lower(), m.group(2), m.end()
        fields, depth, start = {}, 1, i
        # tim dau } dong cua muc
        j = i
        while depth and j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        body = text[start:j - 1]
        for f in re.finditer(r"(\w+)\s*=\s*", body):
            name, k = f.group(1).lower(), f.end()
            if k < len(body) and body[k] == "{":
                d, q = 1, k + 1
                while d and q < len(body):
                    d += {"{": 1, "}": -1}.get(body[q], 0)
                    q += 1
                fields.setdefault(name, body[k + 1:q - 1].strip())
            else:
                v = re.match(r"[^,\n]+", body[k:])
                fields.setdefault(name, v.group(0).strip() if v else "")
        entries[key] = (typ, fields)
    return entries


def strip_case_braces(s):
    """Bo ngoac bao ve chu hoa {SSI} -> SSI, giu ngoac cua lenh dau {\\'a}."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\{([^{}\\]*)\}", r"\1", s)
    return " ".join(s.split())


def initials(first):
    out = []
    for part in first.replace(".", " ").split():
        sub = []
        for piece in part.split("-"):
            m = re.match(r"(\{\\[^{}]*\}|\{\\.\w\}|[A-Z])", piece)
            if m:
                sub.append(m.group(1))
        if sub:
            out.append("-".join(sub))
    return "".join(out)


def sp_name(n):
    n = n.strip()
    if n.startswith("{") and n.endswith("}") and "," not in n:
        return n[1:-1]                      # tac gia tap the
    if "," not in n:
        parts = n.split()
        return f"{' '.join(parts[1:]) or parts[0]} {initials(parts[0]) if len(parts) > 1 else ''}".strip()
    last, first = [x.strip() for x in n.split(",", 1)]
    return f"{strip_case_braces(last)} {initials(first)}".strip()


def authors(f):
    if "author" not in f:
        return f.get("organization", "").strip("{}")
    return ", ".join(sp_name(a) for a in re.split(r"\s+and\s+", f["author"]))


def doi_url(f):
    if "doi" in f:
        return " \\url{https://doi.org/" + f["doi"] + "}"
    if "url" in f:
        return " \\url{" + f["url"] + "}"
    return ""


def sp_entry(typ, f):
    au, year = authors(f), f.get("year", "")
    title = strip_case_braces(f.get("title", "")).rstrip(".")
    note = f" {strip_case_braces(f['note'])}" if "note" in f else ""
    if typ == "article":
        vol = f.get("volume", "")
        num = f"({f['number']})" if "number" in f else ""
        pages = f":{f['pages']}" if "pages" in f else ""
        src = f"{strip_case_braces(f.get('journal', ''))} {vol}{num}{pages}".strip()
        return f"{au} ({year}) {title}. {src}{note}.{doi_url(f)}"
    if typ in ("inproceedings", "inbook", "incollection"):
        where = strip_case_braces(f.get("booktitle", ""))
        pub = f". {strip_case_braces(f['publisher'])}" if typ != "inproceedings" and "publisher" in f else ""
        addr = f", {f['address']}" if "address" in f else ""
        pp = f", pp {f['pages']}" if "pages" in f and not f["pages"].startswith("V") else ""
        art = f", {f['pages']}" if "pages" in f and f["pages"].startswith("V") else ""
        return f"{au} ({year}) {title}. In: {where}{addr}{pub}{pp}{art}.{doi_url(f)}"
    if typ == "book":
        addr = f", {f['address']}" if "address" in f else ""
        return f"{au} ({year}) {title}. {strip_case_braces(f.get('publisher', ''))}{addr}.{doi_url(f)}"
    if typ == "techreport":
        num = f" {f['number']}" if "number" in f else ""
        addr = f", {f['address']}" if "address" in f else ""
        return (f"{au} ({year}) {title}. {f.get('type', 'Report')}{num}, "
                f"{strip_case_braces(f['institution'])}{addr}.{doi_url(f)}")
    how = f" {strip_case_braces(f['howpublished'])}." if "howpublished" in f else ""
    return f"{au} ({year}) {title}.{how}{doi_url(f)}"


def bibliography(tex, entries):
    order = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", tex.split("\\begin{thebibliography}")[0]):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in order:
                order.append(k)
    missing = [k for k in order if k not in entries]
    if missing:
        raise SystemExit(f"thieu trong references.bib: {missing}")
    items = "\n\n".join(f"\\bibitem{{{k}}}\n{sp_entry(*entries[k])}" for k in order)
    return f"\\begin{{thebibliography}}{{99}}\n\n{items}\n\n\\end{{thebibliography}}\n", len(order)


def main():
    tex = TEX.read_text(encoding="utf-8")
    bib, n = bibliography(tex, parse_bib(BIB.read_text(encoding="utf-8")))
    if "\\begin{thebibliography}" in tex:
        tex = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}\n", lambda _: bib, tex, flags=re.S)
    else:
        tex = tex.replace("\\end{document}", bib + "\n\\end{document}")
    TEX.write_text(tex, encoding="utf-8", newline="\n")
    print(f"ok: {n} tai lieu")


if __name__ == "__main__":
    main()
