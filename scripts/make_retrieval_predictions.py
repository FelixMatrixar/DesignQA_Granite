# scripts/make_retrieval_predictions.py
import re
import json
from pathlib import Path
import pandas as pd

QA_CSV    = Path("dataset/rule_extraction/rule_retrieval_qa.csv")
INDEX     = Path("outputs/rule_index.json")
RULE_TEXT = Path("outputs/rulebook_fulltext.txt")  # saved by your index builder
OUT_CSV   = Path("outputs/retrieval_my.csv")

# --- ID Canonicalization ------------------------------------------------------

def canon_id(rid: str) -> str:
    """
    Canonicalize a rule id:
      - trim, lowercase
      - ensure a dot after the leading letters (e.g., 'v1' -> 'v.1')
      - treat any of [space . - –] between numeric parts as a dot
      - collapse duplicate dots, trim trailing dots
    """
    rid = (rid or "").strip().lower()
    # ensure dot after alpha head if followed by digit (v1 -> v.1, ev2 -> ev.2)
    rid = re.sub(r"^([a-z]{1,3})(?=\d)", r"\1.", rid)
    # unify separators around dots/dashes/spaces: "v - 1 . 2" -> "v.1.2"
    rid = re.sub(r"\s*[\.\-–]\s*", ".", rid)
    rid = re.sub(r"\s+", ".", rid)  # leftover naked spaces between numbers
    rid = rid.replace("..", ".")
    rid = rid.strip(".")
    return rid

# Accept "rule v.1.2", "rule v1.2a", or even just "v 1 - 2 a"
RULE_ID_IN_QUESTION = re.compile(
    r"""
    (?:\brules?\s*[:\-]?\s*)?                 # optional 'rule'/'rules'
    (?P<rid>                                  # capture id
        [A-Z]{1,3}                            # head (V, EV, GR, etc.)
        (?:[\s.\-–]?\d+){1,8}                 # one or more numeric parts
        [a-z]?                                # optional letter suffix
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

ANY_RULE_ID = re.compile(
    r"""
    (?P<rid>
        [A-Z]{1,3}
        (?:[\s.\-–]?\d+){1,8}
        [a-z]?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# --- Index I/O ----------------------------------------------------------------

def load_index():
    data = json.loads(INDEX.read_text(encoding="utf-8"))
    # Build canonical map {canon_id -> text}
    id2text = {}
    for d in data:
        k = canon_id(str(d.get("id", "")))
        if not k:
            continue
        # keep first, append later duplicates (just in case)
        if k in id2text:
            id2text[k] = (id2text[k] + " " + (d.get("text") or "")).strip()
        else:
            id2text[k] = (d.get("text") or "").strip()
    ids_sorted = sorted(id2text.keys(), key=len, reverse=True)
    return id2text, ids_sorted

# --- ID extraction & variants -------------------------------------------------

def extract_rule_id(q: str) -> str:
    m = RULE_ID_IN_QUESTION.search(q or "")
    if not m:
        # fallback: detect any plausible rule id even without the word 'rule'
        m = ANY_RULE_ID.search(q or "")
    raw = (m.group("rid") if m else "").strip()
    return canon_id(raw)

def variants(rid: str):
    """Yield relaxed variants for matching."""
    rid = canon_id(rid)
    if not rid:
        return
    yield rid
    # v.1.2a -> v.1.2
    m = re.match(r"^([a-z]{1,3}\.(?:\d+(?:\.\d+)*))(?:[a-z])$", rid)
    if m:
        yield m.group(1)
    # collapse accidental double dots (paranoia)
    yield rid.replace("..", ".")
    # de-dot after head: v.1.2 -> v1.2 (some indices might store that)
    yield re.sub(r"^([a-z]{1,3})\.", r"\1", rid)

def progressive_backoff(rid: str):
    """Backoff: ev.7.2.1 -> ev.7.2 -> ev.7 -> ev."""
    rid = canon_id(rid)
    if not rid:
        return
    parts = rid.split(".")
    for k in range(len(parts), 0, -1):
        yield ".".join(parts[:k])

# --- Flexible fallback search in fulltext ------------------------------------

def make_flexible_base_pattern(rid: str) -> str:
    """
    Convert canonical rid (e.g., 'v.1.2a') into a regex that tolerates
    spaces/dots/dashes between parts and optional suffix.
    """
    rid = canon_id(rid)
    base = re.escape(rid)
    # allow spaces/dots/dashes between parts instead of literal '.'
    base = base.replace(r"\.", r"[\s\.\-–]*")
    return base

def regex_find_in_text(rid: str, text: str) -> str:
    """
    Last resort: pull the first line after a lenient appearance of the ID.
    Returns body (without the ID), or "" if not found.
    """
    if not text or not rid:
        return ""
    base = make_flexible_base_pattern(rid)
    # allow an extra letter suffix and optional punctuation before the body
    pat = re.compile(rf"(^|\n)\s*{base}[a-z]?\s*[:\)\.\-–]?\s+([^\n]+)", re.IGNORECASE)
    m = pat.search(text)
    if not m:
        return ""
    return (m.group(2) or "").strip()

# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    id2text, ids = load_index()
    fulltext = RULE_TEXT.read_text(encoding="utf-8") if RULE_TEXT.exists() else ""

    df = pd.read_csv(QA_CSV)
    preds, misses = [], []

    for i, q in enumerate(df["question"]):
        rid = extract_rule_id(str(q))
        pred = ""

        # 1) exact & small variants
        for v in variants(rid):
            pred = id2text.get(v, "")
            if pred:
                break

        # 2) prefix match (first rule whose id startswith rid)
        if not pred and rid:
            for key in ids:
                if key.startswith(rid):
                    pred = id2text[key]
                    break

        # 3) progressive backoff on the ID segments
        if not pred and rid:
            for back in progressive_backoff(rid):
                # exact first
                pred = id2text.get(back, "")
                if pred:
                    break
                # then any key that startswith the shorter id
                for key in ids:
                    if key.startswith(back):
                        pred = id2text[key]
                        break
                if pred:
                    break

        # 4) final fallback: regex search in raw text
        if not pred and fulltext and rid:
            body = regex_find_in_text(rid, fulltext)
            pred = body

        if not pred:
            misses.append((i, rid, q))
        preds.append(pred)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["model_prediction"] = preds
    out[["ground_truth", "model_prediction"]].to_csv(OUT_CSV, index=False)

    print(f"Wrote {OUT_CSV} • missing={len(misses)}")
    for i, rid, q in misses[:10]:
        print(f"[MISS] row={i} rule_id='{rid}'")
