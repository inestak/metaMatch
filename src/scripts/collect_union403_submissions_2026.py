#!/usr/bin/env python3
"""Collecte les trois soumissions Union403 sans consulter le Gold caché."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_f1(root: Path, pair: str, mode: str) -> float:
    path = root / pair / mode / "public_valid_selection/selection_results.json"
    payload = json.loads(path.read_text())
    return float(payload["selected"]["f1"])


def verify_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    expected = ["SrcEntity", "TgtEntity", "Relation"]
    if list(frame.columns) != expected:
        raise ValueError(f"Colonnes invalides dans {path}: {list(frame.columns)}")
    if frame.duplicated(["SrcEntity", "TgtEntity"]).any():
        raise ValueError(f"Paires dupliquées dans {path}")
    if frame.SrcEntity.duplicated().any() or frame.TgtEntity.duplicated().any():
        raise ValueError(f"La sortie n'est pas 1:1: {path}")
    if not frame.Relation.eq("=").all():
        raise ValueError(f"Relation autre que '=' dans {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.project_root / "outputs_bioml2026_union403"
    output = args.output_dir or args.project_root / "oaei_bioml2026_union403_ready"
    output.mkdir(parents=True, exist_ok=True)

    snomed_modes = ["classes", "classes_properties"]
    snomed_scores = {mode: valid_f1(root, "SNOMED-NCIT", mode) for mode in snomed_modes}
    selected_snomed_mode = max(snomed_scores, key=snomed_scores.get)
    selected = {
        "NCIT-DOID": "classes",
        "SNOMED-FMA": "classes",
        "SNOMED-NCIT": selected_snomed_mode,
    }

    manifest = {
        "protocol": "Union403 XGBoost BioML 2026",
        "hidden_test_access": "never",
        "snomed_ncit_public_valid_f1": snomed_scores,
        "files": [],
    }
    for pair, mode in selected.items():
        source = root / pair / mode / "final_refit/submission.tsv"
        frame = verify_submission(source)
        target = output / f"{pair}_submission.tsv"
        shutil.copy2(source, target)
        manifest["files"].append({
            "pair": pair,
            "entity_mode": mode,
            "alignment_count": len(frame),
            "sha256": sha256(target),
            "path": str(target),
        })

    (output / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

