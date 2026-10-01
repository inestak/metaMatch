#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import zipfile, json, shutil, sys
import os

ROOT = Path(os.environ.get("BIOML_WORK_ROOT", Path(__file__).resolve().parent / "bioml2026_work"))
CONS = Path(os.environ.get(
    "BIOML_CONSENSUS_ROOT",
    ROOT / "outputs_bioml2026_union403" / "FULL_ENTITY_ALIGNABILITY_FILTER",
))

SELECTED = {
    "NCIT-DOID": {
        "tier": "SAFE",
        "vote": 3,
        "variant": "greedy1to1",
        "input": CONS / "NCIT-DOID" / "consensus" / "SAFE" / "vote_ge_3_greedy1to1.tsv",
        "selection_basis": "best historical pseudo-test proxy F1 among candidates above hidden gold reference",
        "pseudo_f1": 0.4652987326493663,
        "expected_predictions": 2082,
    },
    "SNOMED-FMA": {
        "tier": "SAFE",
        "vote": 8,
        "variant": "greedy1to1",
        "input": CONS / "SNOMED-FMA" / "consensus" / "SAFE" / "vote_ge_3_greedy1to1.tsv",
        "selection_basis": "best historical pseudo-test proxy F1 among candidates above hidden gold reference",
        "pseudo_f1": 0.3952773336612963,
        "expected_predictions": 4575,
    },
    "SNOMED-NCIT": {
        "tier": "SAFE",
        "vote": 7,
        "variant": "greedy1to1",
        "input": CONS / "SNOMED-NCIT" / "consensus" / "SAFE" / "vote_ge_3_greedy1to1.tsv",
        "selection_basis": "no exact 2025 pseudo-test available; SAFE chosen by cross-pair consistency and 7/9 as natural above-gold operating point",
        "pseudo_f1": None,
        "expected_predictions": 10003,
    },
}

OUT = CONS / "FINAL_OAEI_2026"
TSV_DIR = OUT / "tsv"
ZIP_PATH = OUT / "BIOML2026_FINAL_SUBMISSION.zip"
MANIFEST = OUT / "FINAL_SELECTION_MANIFEST.json"

SRC_ALIASES = ["SrcEntity", "src_iri", "source", "src"]
TGT_ALIASES = ["TgtEntity", "tgt_iri", "target", "tgt"]

def first_existing(cols, aliases):
    lower = {str(c).lower(): c for c in cols}
    for a in aliases:
        if a in cols:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None

def load_and_normalize(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    src = first_existing(df.columns, SRC_ALIASES)
    tgt = first_existing(df.columns, TGT_ALIASES)
    if src is None or tgt is None:
        raise ValueError(f"{path}: cannot identify source/target columns. Columns={list(df.columns)}")
    out = pd.DataFrame({
        "SrcEntity": df[src].astype(str),
        "TgtEntity": df[tgt].astype(str),
        "Relation": "=",
    })
    out = out.dropna(subset=["SrcEntity", "TgtEntity"])
    out = out.drop_duplicates(["SrcEntity", "TgtEntity"], keep="first").reset_index(drop=True)
    return out

OUT.mkdir(parents=True, exist_ok=True)
TSV_DIR.mkdir(parents=True, exist_ok=True)

manifest = {
    "submission_name": "BioML2026 final selected consensus",
    "selection": {},
    "warning": (
        "NCIT-DOID and SNOMED-FMA operating points were selected using historical 2025 pseudo-test gold. "
        "SNOMED-NCIT had no exact 2025 pseudo-test and was selected by cross-pair consistency."
    ),
}

final_paths = {}

for pair, cfg in SELECTED.items():
    df = load_and_normalize(cfg["input"])
    out_tsv = TSV_DIR / f"{pair}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    final_paths[pair] = out_tsv

    manifest["selection"][pair] = {
        "tier": cfg["tier"],
        "vote_rule": f">={cfg['vote']}/9",
        "variant": cfg["variant"],
        "input_path": str(cfg["input"]),
        "output_path": str(out_tsv),
        "n_predictions": int(len(df)),
        "expected_predictions_from_consensus": cfg["expected_predictions"],
        "pseudo_f1": cfg["pseudo_f1"],
        "selection_basis": cfg["selection_basis"],
    }

    if len(df) != cfg["expected_predictions"]:
        print(
            f"WARNING {pair}: expected {cfg['expected_predictions']} predictions, "
            f"but normalized output has {len(df)}."
        )

with open(MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for pair in ["NCIT-DOID", "SNOMED-FMA", "SNOMED-NCIT"]:
        z.write(final_paths[pair], arcname=f"{pair}.tsv")

print("=" * 100)
print("FINAL OAEI SUBMISSION")
print("=" * 100)
for pair in ["NCIT-DOID", "SNOMED-FMA", "SNOMED-NCIT"]:
    m = manifest["selection"][pair]
    print(
        f"{pair:14s} "
        f"{m['tier']:4s} "
        f"{m['vote_rule']:6s} "
        f"{m['variant']:11s} "
        f"n={m['n_predictions']:6d} "
        f"pseudo_f1={m['pseudo_f1']}"
    )

print()
print("ZIP:", ZIP_PATH)
print("MANIFEST:", MANIFEST)
print()
print("ZIP CONTENT:")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    for info in z.infolist():
        print(f"  {info.filename:20s} {info.file_size:10d} bytes")

# Strict validation
for pair, p in final_paths.items():
    d = pd.read_csv(p, sep="\t", dtype=str)
    assert list(d.columns) == ["SrcEntity", "TgtEntity", "Relation"], (pair, list(d.columns))
    assert d["Relation"].eq("=").all(), pair
    assert not d[["SrcEntity","TgtEntity"]].duplicated().any(), pair

print("\nVALIDATION OK")
