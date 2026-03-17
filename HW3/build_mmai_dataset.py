#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd


QUESTION = "What type of mixed-integer linear program is represented in the image?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=str,
        required=True,
        help="Directory created by milp_evolve_tab1_processor_gurobi.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="mmai-data",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of examples",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into mmai-data/images instead of symlinking",
    )
    parser.add_argument(
        "--skip-missing-answer",
        action="store_true",
        help="Skip rows with missing generator_class",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    manifest_path = processed_dir / "manifest.csv"
    out_dir = Path(args.out_dir)
    images_out = out_dir / "images"
    train_jsonl = out_dir / "train-data.jsonl"
    test_jsonl = out_dir / "test-data.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find manifest: {manifest_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    required_cols = {"image_path", "milp_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    if args.limit is not None:
        df = df.head(args.limit)

    records = []
    n_skipped = 0

    for _, row in df.iterrows():
        image_path = row["image_path"]
        milp_name = row["milp_name"]

        if pd.isna(image_path) or not Path(image_path).exists():
            n_skipped += 1
            continue

        if pd.isna(milp_name) or str(milp_name).strip() == "":
            if args.skip_missing_answer:
                n_skipped += 1
                continue
            milp_name = "UNKNOWN"

        src = Path(image_path)
        dst = images_out / src.name

        # Avoid duplicate filename collisions
        if dst.exists() and dst.resolve() != src.resolve():
            stem = dst.stem
            suffix = dst.suffix
            k = 1
            while True:
                candidate = images_out / f"{stem}_{k}{suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                k += 1

        if not dst.exists():
            if args.copy_images:
                shutil.copy2(src, dst)
            else:
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst)

        records.append(
            {
                "image": f"images/{dst.name}",
                "question": QUESTION,
                "answer": f"this is an instance of a {milp_name} problem",
            }
        )

    rng = random.Random(42)
    rng.shuffle(records)
    n_total = len(records)
    n_train = int(round(n_total * 0.9))
    train_records = records[:n_train]
    test_records = records[n_train:]

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for record in train_records:
            f.write(json.dumps(record) + "\n")

    with open(test_jsonl, "w", encoding="utf-8") as f:
        for record in test_records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(train_records)} examples to {train_jsonl}")
    print(f"Wrote {len(test_records)} examples to {test_jsonl}")
    print(f"Skipped {n_skipped} rows")
    print(f"Images directory: {images_out}")


if __name__ == "__main__":
    main()