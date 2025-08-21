# scripts/make_compilation_predictions_llm.py
# Model-driven "Rule Compilation" with retrieval + LLM re-ranking/selection.
# Output CSV: columns [ground_truth, model_prediction]; model_prediction is
# a plain comma-separated string: e.g., "IN.8.3.2, IN.8.3.3, IN.8.3.4, IN.8.4"
#
# Runtime: local Ollama HTTP API only (no OpenAI, no dotenv).
# Prereq:
#   1) ollama pull granite:2b        # or another Granite tag you prefer
#   2) ollama serve                  # run server (PowerShell/CMD)
#   3) python scripts/make_compilation_predictions_llm.py

import os, re, json, math, collections, time
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import requests

# ---------- Paths ----------
QA_CSV  = Path("dataset/rule_extraction/rule_compilation_qa.csv")
INDEX   = Path("outputs/rule_index.json")   # list of {"id": "...", "title": "...", "text": "..."}
OUT_CSV = Path("outputs/compilation_granite_ollama.csv")

# ---------- Config ----------
CANDIDATE_TOPK = 320
FAMILY_KEEP_FRAC = 0.20
ADD_ANCESTORS = True
MAX_OUTPUT_IDS = 350
CONTEXT_PER_RULE_CHARS = 180
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800

# Ollama settings (edit if needed)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "granite3.2-vision:2b"

# ---------- Text utils ----------
ID_RX = re.compile(r"^(?P<fam>[A-Z]{1,2})\.(?P<num>.+)$")
VALID_ID_CHUNK = re.compile(r"\b([A-Z]{1,2}(?:\.[0-9A-Za-z]+)+)\b")

def clean(s: str) -> str:
    s = (s or "")
    s = s.replace("\u00ad","").replace("\ufb01","fi").replace("\ufb02","fl")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def tokenise(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9+/\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split() if len(t) > 2]

def rule_family(rule_id: str) -> str:
    m = ID_RX.match(rule_id)
    return m.group("fam") if m else ""

def natural_key(rule_id: str):
    m = ID_RX.match(rule_id)
    fam = m.group("fam") if m else ""
    nums = []
    if m:
        for piece in m.group("num").split("."):
            try:
                nums.append(int(piece))
            except:
                nums.append(piece)
    return (fam, nums)

def ancestors(rule_id: str) -> List[str]:
    parts = rule_id.split(".")
    if len(parts) <= 2:
        return []
    out = set()
    for i in range(2, len(parts)):
        out.add(".".join(parts[:i]))
    return [x for x in out if x != rule_id]

# ---------- Load rules ----------
def load_rules() -> List[Dict]:
    rules = json.loads(INDEX.read_text(encoding="utf-8"))
    norm = []
    for r in rules:
        rid = r.get("id") or r.get("rule_id") or ""
        title = r.get("title","")
        text = r.get("text","")
        if not ID_RX.match(rid):
            continue
        field = " ".join([rid.replace("."," "), title, text])
        norm.append({
            "id": rid,
            "title": clean(title),
            "text": clean(text),
            "doc_tokens": tokenise(field)
        })
    return norm

# ---------- Tiny BM25 ----------
class BM25:
    def __init__(self, docs: List[List[str]]):
        self.N = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_lens) / max(1, self.N)) if self.N else 0.0
        self.df = collections.Counter()
        for d in docs:
            for t in set(d):
                self.df[t] += 1
        self.idf = {t: math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1) for t in self.df}
        self.docs = docs
        self.k1, self.b = 1.2, 0.75

    def score(self, q: List[str], idx: int) -> float:
        if not self.docs: return 0.0
        d = self.docs[idx]
        dl = self.doc_lens[idx]
        tf = collections.Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            num = tf[t] * (self.k1 + 1)
            den = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * (num / den)
        return s

# ---------- Question parsing ----------
TERM_PATTERNS = [
    r"relevant to\s+(.+?)\?\s*answer",
    r"relevant to\s+(.+?)\s*(?:\?|$)",
    r"related to\s+(.+?)\s*(?:\?|$)",
    r"about\s+(.+?)\s*(?:\?|$)",
]
SYNONYMS = {
    "accumulator": ["battery", "cells", "pack", "ams"],
    "impact attenuator": ["crash attenuator", "energy absorber", "ia"],
    "roll hoop": ["roll bar", "main hoop", "front hoop", "rear hoop"],
    "shutdown circuit": ["sdc", "master switch", "kill switch", "e stop", "estop"],
    "insulation monitoring device": ["imd"],
    "tyre": ["tire", "tires", "tyres"],
    "restrictor": ["intake restrictor"],
    "fire extinguisher": ["extinguisher"],
}

def extract_term(question: str) -> str:
    for pat in TERM_PATTERNS:
        m = re.search(pat, question, flags=re.I)
        if m:
            term = m.group(1)
            term = re.sub(r"^['\"“”‘’\s]+|['\"“”‘’\s]+$", "", term)
            return term
    m = re.search(r"(.+?)\?", question)
    return m.group(1).strip() if m else question.strip()

def expand_query(term: str) -> List[str]:
    term_l = term.lower().strip()
    toks = tokenise(term_l)
    for k, vs in SYNONYMS.items():
        if k in term_l:
            toks.extend(tokenise(" ".join(vs)))
    return sorted(set(toks))

def gen_query_variants(term: str) -> List[str]:
    base = term.strip()
    toks = expand_query(base)
    variants = set()
    variants.add(base)
    variants.add(" ".join(toks))
    for t in list(toks)[:6]:
        if len(t) > 3 and t.endswith("s"):
            variants.add(" ".join([w if w != t else t[:-1] for w in toks]))
        elif len(t) > 3:
            variants.add(" ".join([w if w != t else (t + "s") for w in toks]))
    variants |= {v.replace("-", " ").replace("/", " ") for v in list(variants)}
    return [v for v in variants if v]

# ---------- Retrieval ----------
def retrieve_candidates(rules: List[Dict], question: str) -> Tuple[List[Dict], set]:
    term = extract_term(question)
    variants = gen_query_variants(term)
    if not variants:
        return [], set()

    bm = BM25([r["doc_tokens"] for r in rules])

    max_scores = {}
    for v in variants[:6]:
        q = tokenise(v)
        if not q:
            continue
        for i in range(len(rules)):
            s = bm.score(q, i)
            if s > 0 and (i not in max_scores or s > max_scores[i]):
                max_scores[i] = s

    if not max_scores:
        return [], set()

    ranked = sorted(max_scores.items(), key=lambda x: -x[1])[:min(CANDIDATE_TOPK, len(max_scores))]
    pool = [rules[i] for i, _ in ranked]

    fam_counts = collections.Counter(rule_family(r["id"]) for r in pool)
    total = len(pool)
    top2 = {f for f, _ in fam_counts.most_common(2)}
    fam_keep = {f for f, c in fam_counts.items() if c >= FAMILY_KEEP_FRAC * total} | top2

    filtered = [r for r in pool if rule_family(r["id"]) in fam_keep]
    return (filtered if filtered else pool), fam_keep

# ---------- Prompting ----------
PROMPT_SYSTEM = (
    "You are a meticulous rules librarian for Formula SAE regulations. "
    "Given a user question and a list of CANDIDATE RULES (each with an ID and content), "
    "return ONLY the IDs of rules that are relevant. "
    "Do NOT invent new IDs. Prefer specific leaf rules but also include parent rules if all children are relevant. "
    "Output JSON array of strings, e.g., [\"T.7\", \"T.7.1\", \"T.7.1.3\"]."
)

PROMPT_USER_TEMPLATE = """Question:
{question}

Candidate rules (choose only from these IDs):
{catalog}

Instructions:
- Return a JSON array of rule IDs (strings) chosen ONLY from the candidate IDs shown.
- You may also use wildcard per-family (e.g., "EV.7.*") if MANY children apply; I will expand that later.
- Be inclusive but avoid obviously unrelated sections.
JSON:"""

def make_catalog(cands: List[Dict]) -> str:
    lines = []
    for r in cands:
        preview = (r["title"] + " — " if r["title"] else "") + r["text"][:CONTEXT_PER_RULE_CHARS]
        preview = preview.replace("\n", " ")
        lines.append(f"- {r['id']}: {preview}")
    return "\n".join(lines)

# ---------- Ollama backend ----------
def call_ollama(prompt: str, model: str = OLLAMA_MODEL, temperature: float = LLM_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }
    try:
        r = requests.post(url, json=data, timeout=600)
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"[Ollama] HTTP error: {e}\nPastikan `ollama serve` berjalan dan model '{model}' sudah dipull.")

def llm_pick_ids(question: str, candidates: List[Dict]) -> List[str]:
    catalog = make_catalog(candidates)
    user = PROMPT_USER_TEMPLATE.format(question=question.strip(), catalog=catalog)
    full = f"System:\n{PROMPT_SYSTEM}\n\nUser:\n{user}"
    out = call_ollama(full)
    return parse_ids_from_text(out, set(r["id"] for r in candidates))

def llm_pick_ids_multi(question: str, candidates: List[Dict], n: int = 3, temp: float = 0.2) -> List[str]:
    votes = collections.Counter()
    for _ in range(n):
        ids = llm_pick_ids(question, candidates)  # temp handled in call_ollama if needed
        for rid in ids:
            votes[rid] += 1
    keep = [rid for rid, c in votes.items() if c >= 2]
    if not keep:
        keep = [rid for rid, _ in votes.most_common(20)]
    return keep

# ---------- Output parsing ----------
def expand_wildcards_global(chosen: List[str], all_ids: set, allowed_fams: set) -> List[str]:
    out = []
    for x in chosen:
        if x.endswith(".*"):
            prefix = x[:-2]
            fam = rule_family(prefix)
            if allowed_fams and fam not in allowed_fams:
                continue
            matches = [rid for rid in all_ids if rid == prefix or rid.startswith(prefix + ".")]
            out.extend(matches)
        else:
            out.append(x)
    seen, ordered = set(), []
    for rid in out:
        if rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    return ordered

from collections import defaultdict
def build_children_map(all_ids: List[str]) -> Dict[str, List[str]]:
    kids = defaultdict(list)
    for rid in all_ids:
        parts = rid.split(".")
        for i in range(2, len(parts)):
            parent = ".".join(parts[:i])
            child  = ".".join(parts[:i+1])
            kids[parent].append(child)
    return kids

def complete_siblings(chosen: List[str], all_ids: set, abs_thresh: int = 3, frac_thresh: float = 0.6) -> List[str]:
    children = build_children_map(sorted(all_ids, key=natural_key))
    chosen_set = set(chosen)
    changed = True
    while changed:
        changed = False
        for parent, kids in children.items():
            k = [c for c in kids if c in chosen_set]
            if len(k) >= abs_thresh or (len(kids) >= 2 and len(k) / max(1, len(kids)) >= frac_thresh):
                add = [c for c in kids if c not in chosen_set]
                if add:
                    chosen_set.update(add)
                    chosen_set.add(parent)
                    changed = True
    return sorted(chosen_set, key=natural_key)

def parse_ids_from_text(text: str, candidate_set: set) -> List[str]:
    text = text.strip()
    ids = []
    m = re.search(r"\[[^\]]*\]", text, flags=re.S)
    blob = m.group(0) if m else text
    try:
        parsed = json.loads(blob)
        if isinstance(parsed, dict) and "ids" in parsed:
            ids = parsed["ids"]
        elif isinstance(parsed, list):
            ids = parsed
    except Exception:
        wilds = re.findall(r"\b([A-Z]{1,2}(?:\.[0-9A-Za-z]+)+\.\*)\b", text)
        ids = [m.group(1) for m in VALID_ID_CHUNK.finditer(text)]
        ids.extend(wilds)

    seen, out = set(), []
    for x in ids:
        x = x.strip().strip("\"'` ,")
        if not x:
            continue
        if x.endswith(".*") or x in candidate_set:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out

# ---------- Per-question pipeline ----------
def predict_for_question(rules: List[Dict], question: str) -> List[str]:
    cands, fam_keep = retrieve_candidates(rules, question)
    if not cands:
        return []

    cand_ids = set(r["id"] for r in cands)
    all_ids  = set(r["id"] for r in rules)

    chosen_raw = llm_pick_ids_multi(question, cands, n=3, temp=0.2)
    chosen_expanded = expand_wildcards_global(chosen_raw, all_ids, fam_keep)

    kept, seen = [], set()
    for rid in chosen_expanded:
        fam = rule_family(rid)
        if (rid in cand_ids) or (rid in all_ids and (not fam_keep or fam in fam_keep)):
            if rid not in seen:
                seen.add(rid)
                kept.append(rid)

    if not kept:
        kept = [r["id"] for r in cands[:30]]

    if ADD_ANCESTORS:
        with_anc = set(kept)
        for rid in list(kept):
            for par in ancestors(rid):
                if par in all_ids:
                    with_anc.add(par)
        kept = list(with_anc)

    kept = complete_siblings(kept, all_ids, abs_thresh=3, frac_thresh=0.6)
    kept = sorted(kept, key=natural_key)[:MAX_OUTPUT_IDS]
    return kept

# ---------- Main ----------
if __name__ == "__main__":
    rules = load_rules()
    if not rules:
        raise SystemExit(f"No rules loaded from {INDEX}.")
    df = pd.read_csv(QA_CSV)
    preds = []
    for q in df["question"].tolist():
        try:
            ids = predict_for_question(rules, q)
        except Exception as e:
            print(f"[warn] prediction error: {e}")
            ids = []
        preds.append(", ".join(ids))
        # (Optional) small pause for local server breathing room
        time.sleep(0.05)

    out = df.copy()
    out["model_prediction"] = preds
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out[["ground_truth","model_prediction"]].to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
