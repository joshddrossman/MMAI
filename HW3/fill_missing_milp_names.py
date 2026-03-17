#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "microsoft/MILP-Evolve"
REPO_TYPE = "dataset"


def is_missing(x) -> bool:
    if x is None:
        return True
    if pd.isna(x):
        return True
    return str(x).strip() == ""


def extract_generator_class_name(py_path: Path) -> Optional[str]:
    """
    Extract the most likely MILP generator class from a Python file.

    Priority:
    1. Class instantiated inside `if __name__ == "__main__":`
    2. Class with generator-like methods such as `generate_instance` or `solve`
    3. First top-level class whose name is not a common helper name
    4. Fallback: first top-level class
    """
    source = py_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(py_path))

    top_level_classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if not top_level_classes:
        return None

    class_names = {cls.name for cls in top_level_classes}

    # 1) look for instantiation in __main__
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            is_main_guard = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            )
            if is_main_guard:
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Name) and func.id in class_names:
                            return func.id
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Name) and func.id in class_names:
                            return func.id

    # 2) score classes by likely generator methods
    preferred_methods = {
        "generate_instance",
        "generate",
        "solve",
        "build_model",
        "generate_graph",
    }

    scored = []
    for cls in top_level_classes:
        method_names = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
        score = len(preferred_methods.intersection(method_names))
        scored.append((score, cls.name))

    scored.sort(reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]

    # 3) skip obvious helper names if possible
    helper_names = {"Graph", "Node", "Edge", "Utils", "Helper", "Container"}
    for cls in top_level_classes:
        if cls.name not in helper_names:
            return cls.name

    # 4) fallback
    return top_level_classes[0].name


def download_code_file(code_repo_path: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=code_repo_path,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument(
        "--cache-dir",
        default="hf_code_cache",
        help="Local cache directory for downloaded .py files",
    )
    parser.add_argument(
        "--update-generator-class-too",
        action="store_true",
        help="Also overwrite/fill generator_class with the recovered milp_name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without saving",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    cache_dir = Path(args.cache_dir)

    df = pd.read_csv(manifest_path)

    if "milp_name" not in df.columns:
        df["milp_name"] = pd.NA

    if "code_repo_path" not in df.columns:
        raise ValueError("Manifest must contain a 'code_repo_path' column.")

    num_missing = sum(is_missing(x) for x in df["milp_name"])
    print(f"Rows with missing milp_name: {num_missing}")

    updated = 0
    failed = 0

    for idx, row in df.iterrows():
        if not is_missing(row["milp_name"]):
            continue

        code_repo_path = row["code_repo_path"]
        if is_missing(code_repo_path):
            print(f"[row {idx}] missing code_repo_path; skipping")
            failed += 1
            continue

        try:
            py_path = download_code_file(str(code_repo_path), cache_dir)
            extracted = extract_generator_class_name(py_path)

            if extracted is None or str(extracted).strip() == "":
                print(f"[row {idx}] could not extract class from {code_repo_path}")
                failed += 1
                continue

            print(f"[row {idx}] milp_name <- {extracted}")
            df.at[idx, "milp_name"] = extracted

            if args.update_generator_class_too:
                if "generator_class" not in df.columns:
                    df["generator_class"] = pd.NA
                df.at[idx, "generator_class"] = extracted

            updated += 1

        except Exception as e:
            print(f"[row {idx}] error processing {code_repo_path}: {e}")
            failed += 1

    print(f"Updated: {updated}")
    print(f"Failed: {failed}")

    if args.dry_run:
        print("Dry run only; manifest not saved.")
        return

    df.to_csv(manifest_path, index=False)
    print(f"Saved updated manifest to {manifest_path}")


if __name__ == "__main__":
    main()