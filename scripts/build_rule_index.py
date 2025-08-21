# scripts/build_rule_index.py
import re, json, sys
from pathlib import Path
import pdfplumber

# ---- Adjust if your PDF is elsewhere ----
RULES_PDF = Path("dataset/docs/FSAE_Rules_2024_V1.pdf")

# Matches things like: V.1  / V.1.2  / GR.6.4.1  / T.7.7.1a
RULE_ID_RE = re.compile(r'^(?:[A-Z]{1,3})\.(?:\d+)(?:\.\d+)*(?:[a-z])?', re.M)

def extract_rules(pdf_path: Path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    text = "\n".join(pages)

    # Find all rule start positions
    starts = [m.start() for m in RULE_ID_RE.finditer(text)]
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(text)
        chunk = text[s:e].strip()
        # Split first line into id + body
        first_line = chunk.splitlines()[0]
        rid = RULE_ID_RE.match(first_line).group(0)
        body = first_line[len(rid):].strip()
        # Append rest of lines as part of the same rule text
        rest = "\n".join(chunk.splitlines()[1:]).strip()
        full_text = (body + ("\n" + rest if rest else "")).strip()
        # collapse internal whitespace
        full_text_oneline = re.sub(r'\s+', ' ', full_text).strip()
        blocks.append({"id": rid, "text": full_text_oneline})
    return blocks

def main():
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else RULES_PDF
    rules = extract_rules(pdf)
    out = Path("outputs/rule_index.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out} with {len(rules)} rules.")

if __name__ == "__main__":
    main()
