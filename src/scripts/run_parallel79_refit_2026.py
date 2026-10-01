#!/usr/bin/env python3
"""Apply a frozen public-valid protocol to a saved 79-feature refit MetaSpace."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--input-subdir", required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument(
        "--entity-mode",
        choices=["classes", "classes_and_properties"],
        default="classes",
    )
    parser.add_argument("--submission-suffix", required=True)
    args = parser.parse_args()

    selection = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    selected = selection["selected"]
    mode = str(selected["post_filter_mode"])
    threshold = float(selected["threshold"])

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.scripts.experiment_saved_metaspace",
            "--pair",
            args.pair,
            "--task-type",
            "equiv",
            "--input-subdir",
            args.input_subdir,
            "--output-subdir",
            args.output_subdir,
            "--experiment",
            "model",
            "--model-family",
            "xgboost",
            "--feature-mode",
            "base79",
            "--folds",
            "3",
            "--fixed-post-filter-mode",
            mode,
            "--fixed-threshold",
            str(threshold),
            "--no-test-evaluation",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.scripts.finalize_base79_submission_2026",
            "--pair",
            args.pair,
            "--input-subdir",
            args.output_subdir,
            "--entity-mode",
            args.entity_mode,
            "--submission-suffix",
            args.submission_suffix,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
