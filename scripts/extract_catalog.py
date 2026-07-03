"""Rebuilds data/catalog.json from the source catalog dump.

The catalog was provided as an 8-"page"-labeled but actually 242-page PDF --
a browser's built-in JSON viewer, printed to PDF. Two artifacts of that path
needed fixing before the text is valid JSON:

1. The viewer's floating "Pretty print" toolbar button is a fixed-position
   overlay that lands on top of the first content line of every single page.
   Naive text extraction interleaves its characters into the real content
   (e.g. "Pretty print" + "Mid-Professional" -> "pMriidn-tProfessional").
   Fix: pdfplumber exposes each character's exact y-position ("top"); the
   overlay consistently sits ~1.5pt below the real content line on every
   page, so we drop chars at that offset before extracting text.

2. The viewer soft-wraps long string values across visual lines without
   inserting an actual newline into the string. Printing to PDF turns those
   soft-wraps into literal '\\n' characters, which breaks JSON parsing (a
   raw newline inside a string is illegal). Fix: re-scan the extracted text
   tracking whether we're inside a JSON string (respecting `\\"` escapes);
   any newline encountered while inside a string is a wrap artifact and gets
   replaced with a single space instead of removed outright, since the
   original text had a word-boundary space there.

Run: python scripts/extract_catalog.py <source.pdf> data/catalog.json
"""
import json
import sys

import pdfplumber

OVERLAY_TOP = 30.8  


def _keep(obj) -> bool:
    if obj.get("object_type") != "char":
        return True
    return round(obj["top"], 1) != OVERLAY_TOP


def extract_text(pdf_path: str) -> str:
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            filtered = page.filter(_keep)
            pages_text.append(filtered.extract_text() or "")
    return "\n".join(pages_text)


def fix_wrapped_strings(text: str) -> str:
    out = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
            elif ch == "\\":
                out.append(ch)
                escape = True
            elif ch == '"':
                in_string = False
                out.append(ch)
            elif ch == "\n":
                out.append(" ")  
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def main(src: str, dst: str) -> None:
    raw = extract_text(src)
    fixed = fix_wrapped_strings(raw)
    data = json.loads(fixed)  
    ok = [d for d in data if d.get("status") == "ok"]
    print(f"parsed {len(data)} records, {len(ok)} with status=ok")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python scripts/extract_catalog.py <source.pdf> <output.json>")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2])
