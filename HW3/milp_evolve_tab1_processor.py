#!/usr/bin/env python3
"""
milp_evolve_tab1_processor_gurobi.py

Downloads the first n MILP-Evolve tab1 compressed instances from Hugging Face,
matches each archive to its generating Python file, extracts the first top-level
class name from that generator script, reads the .mps inside the .tar.gz using
Gurobi, and builds an image-based representation of the MILP.

Outputs:
- manifest.csv                         summary table
- images/<instance_stem>.png          rendered image representation
- tensors/<instance_stem>.npz         numeric arrays used to create the image

Example:
    python milp_evolve_tab1_processor_gurobi.py --n 1000 --out-dir milp_evolve_tab1_processed

Dependencies:
    pip install huggingface_hub numpy matplotlib scipy gurobipy
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import math
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from huggingface_hub import HfApi, hf_hub_download

import gurobipy as gp
from gurobipy import GRB


# =========================
# Configuration
# =========================

REPO_ID = "microsoft/MILP-Evolve"
REPO_TYPE = "dataset"
TAB1_INSTANCES_PATH = "instances/tab1_compressed"
TAB1_CODE_PATH = "milp_code/evolve_tab1/code"
MAX_FILE_SIZE_MB_DEFAULT = 500


# =========================
# Utilities
# =========================

def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def safe_log1p_scale(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def zscore_or_zero(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma


def compress_to_unit_interval(x: np.ndarray) -> np.ndarray:
    z = zscore_or_zero(x)
    return 0.5 * (np.tanh(z) + 1.0)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =========================
# Hugging Face helpers
# =========================

def list_tab1_archives(max_file_size_mb: Optional[float] = None) -> List[str]:
    """
    List tab1 .tar.gz files from the HF dataset repo, optionally skipping files
    larger than max_file_size_mb.

    This version is robust to different Hugging Face API return shapes.
    """
    api = HfApi()

    # Ask for the full repo file list first; this is the most reliable part.
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    archives = [
        f for f in files
        if f.startswith(f"{TAB1_INSTANCES_PATH}/") and f.endswith(".tar.gz")
    ]
    archives = sorted(archives, key=lambda x: natural_key(Path(x).name))

    if max_file_size_mb is None:
        return archives

    max_bytes = int(max_file_size_mb * 1024 * 1024)

    # Build a size lookup using repo tree metadata
    size_by_path: Dict[str, Optional[int]] = {}
    try:
        infos = api.list_repo_tree(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo=TAB1_INSTANCES_PATH,
            recursive=True,
            expand=True,
        )

        for info in infos:
            # Handle both object-like and dict-like entries
            if isinstance(info, dict):
                path = info.get("path") or info.get("name")
                size = info.get("size")
                info_type = info.get("type")
            else:
                path = getattr(info, "path", None) or getattr(info, "name", None)
                size = getattr(info, "size", None)
                info_type = getattr(info, "type", None)

            if path is None:
                continue
            if info_type is not None and info_type != "file":
                continue

            size_by_path[path] = size
    except Exception as exc:
        print(f"WARNING: Could not retrieve remote file sizes from HF metadata: {exc}")
        print("Proceeding without pre-download size filtering.")
        return archives

    kept: List[str] = []
    skipped_large = 0
    skipped_unknown = 0

    for path in archives:
        size = size_by_path.get(path)

        if size is None:
            # If HF does not provide size metadata for a file, keep it rather than
            # accidentally dropping everything.
            skipped_unknown += 1
            kept.append(path)
            continue

        if size > max_bytes:
            skipped_large += 1
            print(
                f"Skipping large file: {Path(path).name} "
                f"({size / (1024 * 1024):.1f} MB > {max_file_size_mb:.1f} MB)"
            )
            continue

        kept.append(path)

    print(
        f"Kept {len(kept)} tab1 archives with size <= {max_file_size_mb:.1f} MB; "
        f"skipped {skipped_large} larger files; "
        f"{skipped_unknown} had unknown size and were kept."
    )

    return kept


def paired_code_path_for_archive(archive_repo_path: str) -> str:
    archive_name = Path(archive_repo_path).name
    if not archive_name.endswith(".tar.gz"):
        raise ValueError(f"Unexpected archive name: {archive_name}")
    stem = archive_name[:-7]  # remove .tar.gz
    return f"{TAB1_CODE_PATH}/{stem}.py"


def download_hf_file(repo_path: str, local_dir: Path) -> Path:
    ensure_dir(local_dir)
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_path,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


# =========================
# Python generator class extraction
# =========================

def extract_first_top_level_class_name(py_path: Path) -> Optional[str]:
    source = py_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(py_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node.name
    return None


# =========================
# Archive helpers
# =========================

def extract_mps_bytes_from_tar_gz(tar_gz_path: Path) -> bytes:
    """
    Extract the first .mps or .mps.gz file from a tar.gz archive.
    Returns decompressed raw .mps bytes.
    """
    with tarfile.open(tar_gz_path, mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if not members:
            raise ValueError(f"No files found in archive: {tar_gz_path}")

        preferred = None
        for suffix in (".mps", ".mps.gz"):
            for m in members:
                if m.name.endswith(suffix):
                    preferred = m
                    break
            if preferred is not None:
                break
        if preferred is None:
            preferred = members[0]

        extracted = tf.extractfile(preferred)
        if extracted is None:
            raise ValueError(f"Unable to extract {preferred.name} from {tar_gz_path}")

        data = extracted.read()
        if preferred.name.endswith(".gz"):
            data = gzip.decompress(data)
        return data


# =========================
# Gurobi extraction
# =========================

@dataclass
class ColumnBound:
    lower: float
    upper: float
    is_integer: bool
    is_binary: bool
    is_semi_cont: bool = False
    is_semi_int: bool = False


@dataclass
class ParsedMILP:
    name: str
    row_names: List[str]
    row_types: List[str]         # L, G, E
    row_rhs: np.ndarray
    col_names: List[str]
    obj_coeffs: np.ndarray
    A_row: np.ndarray
    A_col: np.ndarray
    A_val: np.ndarray
    bounds: List[ColumnBound]
    obj_sense: str               # "min" or "max"


def load_mps_with_gurobi(mps_bytes: bytes) -> ParsedMILP:
    """
    Read raw .mps bytes using Gurobi and extract matrix/objective/metadata.
    """
    with tempfile.NamedTemporaryFile(suffix=".mps", delete=True) as tmp:
        tmp.write(mps_bytes)
        tmp.flush()

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        try:
            model = gp.read(tmp.name, env=env)
        finally:
            pass

        model.setParam("OutputFlag", 0)
        model.update()

        name = model.ModelName or "UNKNOWN"

        vars_ = model.getVars()
        conss = model.getConstrs()

        col_names = [v.VarName for v in vars_]
        row_names = [c.ConstrName for c in conss]

        row_rhs = np.array([float(c.RHS) for c in conss], dtype=float)
        row_types = [
            "L" if c.Sense == "<" else "G" if c.Sense == ">" else "E"
            for c in conss
        ]

        obj_coeffs = np.array([float(v.Obj) for v in vars_], dtype=float)
        obj_sense = "min" if model.ModelSense == GRB.MINIMIZE else "max"

        A = model.getA()
        if not sp.isspmatrix(A):
            A = sp.csr_matrix(A)
        A = A.tocoo()

        A_row = A.row.astype(np.int32)
        A_col = A.col.astype(np.int32)
        A_val = A.data.astype(float)

        bounds: List[ColumnBound] = []
        for v in vars_:
            lb = float(v.LB)
            ub = float(v.UB)

            # normalize "infinite" values
            if lb <= -GRB.INFINITY / 2:
                lb = -math.inf
            if ub >= GRB.INFINITY / 2:
                ub = math.inf

            vtype = v.VType
            is_binary = vtype == GRB.BINARY
            is_integer = vtype in {GRB.BINARY, GRB.INTEGER}
            is_semi_cont = vtype == GRB.SEMICONT
            is_semi_int = vtype == GRB.SEMIINT

            bounds.append(
                ColumnBound(
                    lower=lb,
                    upper=ub,
                    is_integer=is_integer,
                    is_binary=is_binary,
                    is_semi_cont=is_semi_cont,
                    is_semi_int=is_semi_int,
                )
            )

        env.close()

        return ParsedMILP(
            name=name,
            row_names=row_names,
            row_types=row_types,
            row_rhs=row_rhs,
            col_names=col_names,
            obj_coeffs=obj_coeffs,
            A_row=A_row,
            A_col=A_col,
            A_val=A_val,
            bounds=bounds,
            obj_sense=obj_sense,
        )


# =========================
# Feature engineering
# =========================

ROW_TYPE_TO_ID = {"E": 0, "L": 1, "G": 2}
VAR_TYPE_TO_ID = {
    "continuous": 0,
    "integer": 1,
    "binary": 2,
    "semi_cont": 3,
    "semi_int": 4,
}
BOUND_TYPE_TO_ID = {
    "free": 0,
    "lower_only": 1,
    "upper_only": 2,
    "box": 3,
    "fixed": 4,
    "binary": 5,
}


def row_degree(m: int, A_row: np.ndarray) -> np.ndarray:
    deg = np.zeros(m, dtype=float)
    if A_row.size:
        counts = np.bincount(A_row, minlength=m)
        deg[:len(counts)] = counts
    return deg


def col_degree(n: int, A_col: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=float)
    if A_col.size:
        counts = np.bincount(A_col, minlength=n)
        deg[:len(counts)] = counts
    return deg


def infer_var_type(bound: ColumnBound) -> str:
    if bound.is_binary:
        return "binary"
    if bound.is_semi_int:
        return "semi_int"
    if bound.is_semi_cont:
        return "semi_cont"
    if bound.is_integer:
        return "integer"
    return "continuous"


def infer_bound_type(bound: ColumnBound) -> str:
    if bound.is_binary:
        return "binary"
    lo, up = bound.lower, bound.upper
    if np.isneginf(lo) and np.isposinf(up):
        return "free"
    if np.isfinite(lo) and np.isposinf(up):
        return "lower_only"
    if np.isneginf(lo) and np.isfinite(up):
        return "upper_only"
    if np.isfinite(lo) and np.isfinite(up):
        if abs(lo - up) < 1e-12:
            return "fixed"
        return "box"
    return "box"


@dataclass
class MILPImageTensors:
    center: np.ndarray
    left_sidebar: np.ndarray
    top_sidebar: np.ndarray


def build_image_tensors(parsed: ParsedMILP, max_dim: Optional[int] = None) -> MILPImageTensors:
    """
    center: binary sparsity pattern of A (1 if nonzero, 0 otherwise)
    left sidebar per row: [constraint sense, RHS scale, row degree]
    top sidebar per col: [var type, obj sign, obj magnitude, col degree, bound type]
    """
    m = len(parsed.row_names)
    n = len(parsed.col_names)

    if max_dim is not None:
        keep_m = min(m, max_dim)
        keep_n = min(n, max_dim)

        # Binary sparsity, not coefficient magnitudes
        center = np.zeros((keep_m, keep_n), dtype=np.float32)
        nz_mask = (parsed.A_row < keep_m) & (parsed.A_col < keep_n)
        rr = parsed.A_row[nz_mask]
        cc = parsed.A_col[nz_mask]
        center[rr, cc] = 1.0

        row_types = parsed.row_types[:keep_m]
        row_rhs_vals = parsed.row_rhs[:keep_m]
        row_deg_vals = row_degree(m, parsed.A_row)[:keep_m]

        obj = parsed.obj_coeffs[:keep_n]
        col_deg_vals = col_degree(n, parsed.A_col)[:keep_n]
        bounds = parsed.bounds[:keep_n]
    else:
        # Binary sparsity, not coefficient magnitudes
        center = np.zeros((m, n), dtype=np.float32)
        center[parsed.A_row, parsed.A_col] = 1.0

        row_types = parsed.row_types
        row_rhs_vals = parsed.row_rhs
        row_deg_vals = row_degree(m, parsed.A_row)

        obj = parsed.obj_coeffs
        col_deg_vals = col_degree(n, parsed.A_col)
        bounds = parsed.bounds

    row_type_ids = np.array([ROW_TYPE_TO_ID.get(t, -1) for t in row_types], dtype=float)
    rhs_scaled = compress_to_unit_interval(safe_log1p_scale(row_rhs_vals))
    row_deg_scaled = compress_to_unit_interval(safe_log1p_scale(row_deg_vals))

    row_type_scaled = row_type_ids.copy()
    if row_type_scaled.size > 0:
        max_id = max(ROW_TYPE_TO_ID.values())
        row_type_scaled = row_type_scaled / max(max_id, 1)

    left_sidebar = np.stack(
        [row_type_scaled, rhs_scaled, row_deg_scaled],
        axis=1,
    ).astype(np.float32)

    var_types = [infer_var_type(b) for b in bounds]
    var_type_ids = np.array([VAR_TYPE_TO_ID[v] for v in var_types], dtype=float)
    max_vid = max(VAR_TYPE_TO_ID.values())
    var_type_scaled = var_type_ids / max(max_vid, 1)

    obj_sign = np.zeros_like(obj)
    obj_sign[obj > 0] = 1.0
    obj_sign[obj < 0] = -1.0
    obj_sign_scaled = 0.5 * (obj_sign + 1.0)

    obj_mag_scaled = compress_to_unit_interval(safe_log1p_scale(np.abs(obj)))
    col_deg_scaled = compress_to_unit_interval(safe_log1p_scale(col_deg_vals))

    bound_types = [infer_bound_type(b) for b in bounds]
    bound_type_ids = np.array([BOUND_TYPE_TO_ID[b] for b in bound_types], dtype=float)
    max_bid = max(BOUND_TYPE_TO_ID.values())
    bound_type_scaled = bound_type_ids / max(max_bid, 1)

    top_sidebar = np.stack(
        [var_type_scaled, obj_sign_scaled, obj_mag_scaled, col_deg_scaled, bound_type_scaled],
        axis=0,
    ).astype(np.float32)

    return MILPImageTensors(
        center=center,
        left_sidebar=left_sidebar,
        top_sidebar=top_sidebar,
    )


# =========================
# Rendering
# =========================

def _block_reduce_2d(arr: np.ndarray, out_h: int, out_w: int, mode: str = "mean") -> np.ndarray:
    """
    Resize a 2D array to (out_h, out_w) using block reduction.
    - mode='max'  : preserves sparse nonzero structure well
    - mode='mean' : smoother for sidebars
    Works for both downsampling and modest upsampling.
    """
    arr = np.asarray(arr, dtype=np.float32)
    H, W = arr.shape

    if out_h <= 0 or out_w <= 0:
        raise ValueError("Output height/width must be positive.")
    if H == 0 or W == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)
    if H == out_h and W == out_w:
        return arr.copy()

    out = np.zeros((out_h, out_w), dtype=np.float32)

    row_edges = np.linspace(0, H, out_h + 1)
    col_edges = np.linspace(0, W, out_w + 1)

    row_starts = np.floor(row_edges[:-1]).astype(int)
    row_ends = np.ceil(row_edges[1:]).astype(int)
    col_starts = np.floor(col_edges[:-1]).astype(int)
    col_ends = np.ceil(col_edges[1:]).astype(int)

    for i in range(out_h):
        r0 = min(row_starts[i], H - 1)
        r1 = max(r0 + 1, min(row_ends[i], H))
        for j in range(out_w):
            c0 = min(col_starts[j], W - 1)
            c1 = max(c0 + 1, min(col_ends[j], W))
            block = arr[r0:r1, c0:c1]

            if mode == "max":
                out[i, j] = float(np.max(block))
            elif mode == "mean":
                out[i, j] = float(np.mean(block))
            else:
                raise ValueError(f"Unknown reduction mode: {mode}")

    return out


def _choose_sidebar_sizes(
    image_size: int,
    left_features: int,
    top_features: int,
) -> tuple[int, int]:
    """
    Choose visible but not overly large sidebar sizes in pixels.
    """
    left_w = max(8, min(image_size // 8, 3 * left_features))
    top_h = max(8, min(image_size // 8, 3 * top_features))
    return left_w, top_h

def render_milp_image(
    tensors: MILPImageTensors,
    out_path: Path,
    image_size: int = 384,
) -> None:
    """
    Render a composite MILP image to a fixed square output size.

    Key design choices:
    - center uses binary sparsity and max-pooling-based downsampling
    - sidebars use mean pooling
    - final PNG is saved directly at exactly image_size x image_size
      without an additional matplotlib interpolation step
    """
    center = np.asarray(tensors.center, dtype=np.float32)
    left = np.asarray(tensors.left_sidebar, dtype=np.float32)
    top = np.asarray(tensors.top_sidebar, dtype=np.float32)

    H, W = center.shape
    H_left, F_row = left.shape
    F_col, W_top = top.shape

    if H != H_left:
        raise ValueError("Left sidebar height must match center height")
    if W != W_top:
        raise ValueError("Top sidebar width must match center width")

    left_w, top_h = _choose_sidebar_sizes(
        image_size=image_size,
        left_features=F_row,
        top_features=F_col,
    )

    center_h = image_size - top_h
    center_w = image_size - left_w

    if center_h <= 0 or center_w <= 0:
        raise ValueError("image_size is too small relative to sidebar sizes")

    # Binary center + max pooling preserves sparse structure
    center_binary = (center != 0).astype(np.float32)
    center_img = _block_reduce_2d(center_binary, center_h, center_w, mode="max")

    # Sidebars: mean pooling is fine
    left_img = _block_reduce_2d(left, center_h, left_w, mode="mean")
    top_img = _block_reduce_2d(top, top_h, center_w, mode="mean")

    composite = np.full((image_size, image_size), 0.5, dtype=np.float32)
    composite[top_h:, :left_w] = left_img
    composite[:top_h, left_w:] = top_img
    composite[top_h:, left_w:] = center_img

    # Save exact pixels, no axes, no margins, no title
    plt.imsave(out_path, composite, cmap="gray", vmin=0.0, vmax=1.0)


# =========================
# Main workflow
# =========================

def process_one_instance(
    archive_repo_path: str,
    image_dir: Path,
    tensor_dir: Path,
    max_dim: Optional[int],
    image_size: int,
) -> Dict[str, object]:
    code_repo_path = paired_code_path_for_archive(archive_repo_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_local = download_hf_file(archive_repo_path, tmp_path)
        code_local = download_hf_file(code_repo_path, tmp_path)

        generator_class = extract_first_top_level_class_name(code_local)
        mps_bytes = extract_mps_bytes_from_tar_gz(archive_local)
        parsed = load_mps_with_gurobi(mps_bytes)
        tensors = build_image_tensors(parsed, max_dim=max_dim)

    stem = Path(archive_repo_path).name[:-7]
    image_path = image_dir / f"{stem}.png"
    tensor_path = tensor_dir / f"{stem}.npz"

    render_milp_image(
        tensors,
        out_path=image_path,
        image_size=image_size,
    )

    np.savez_compressed(
        tensor_path,
        center=tensors.center,
        left_sidebar=tensors.left_sidebar,
        top_sidebar=tensors.top_sidebar,
        row_names=np.array(parsed.row_names, dtype=object),
        row_types=np.array(parsed.row_types, dtype=object),
        col_names=np.array(parsed.col_names, dtype=object),
        obj_coeffs=parsed.obj_coeffs,
        A_row=parsed.A_row,
        A_col=parsed.A_col,
        A_val=parsed.A_val,
        row_rhs=parsed.row_rhs,
        generator_class="" if generator_class is None else generator_class,
        obj_sense=parsed.obj_sense,
    )

    num_rows = len(parsed.row_names)
    num_cols = len(parsed.col_names)
    nnz = len(parsed.A_val)

    n_bin = sum(b.is_binary for b in parsed.bounds)
    n_int = sum((b.is_integer and not b.is_binary) for b in parsed.bounds)
    n_cont = num_cols - n_bin - n_int

    return {
        "instance_name": stem,
        "archive_repo_path": archive_repo_path,
        "code_repo_path": code_repo_path,
        "generator_class": generator_class,
        "mps_name": parsed.name,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "num_nnz": nnz,
        "n_bin": n_bin,
        "n_int": n_int,
        "n_cont": n_cont,
        "image_path": str(image_path),
        "tensor_path": str(tensor_path),
        "archive_local_path": None,
        "code_local_path": None,
    }


def write_manifest(rows: List[Dict[str, object]], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _np_scalar_to_py(value: np.ndarray) -> object:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_existing_manifest_row(
    *,
    stem: str,
    archive_repo_path: str,
    image_path: Path,
    tensor_path: Path,
) -> Optional[Dict[str, object]]:
    try:
        data = np.load(tensor_path, allow_pickle=True)
    except Exception as exc:
        print(f"  WARNING: Failed to load existing tensors for {stem}: {exc}")
        return None

    row_names = data.get("row_names")
    col_names = data.get("col_names")
    A_val = data.get("A_val")

    num_rows = int(len(row_names)) if row_names is not None else None
    num_cols = int(len(col_names)) if col_names is not None else None
    num_nnz = int(len(A_val)) if A_val is not None else None

    generator_class = _np_scalar_to_py(data.get("generator_class"))
    obj_sense = _np_scalar_to_py(data.get("obj_sense"))

    return {
        "instance_name": stem,
        "archive_repo_path": archive_repo_path,
        "code_repo_path": paired_code_path_for_archive(archive_repo_path),
        "generator_class": generator_class,
        "mps_name": None,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "num_nnz": num_nnz,
        "n_bin": None,
        "n_int": None,
        "n_cont": None,
        "image_path": str(image_path),
        "tensor_path": str(tensor_path),
        "archive_local_path": None,
        "code_local_path": None,
        "obj_sense": obj_sense,
    }


def parse_instance_index(stem: str) -> Optional[int]:
    match = re.search(r"milp_(\d+)", stem)
    if match is None:
        return None
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process MILP-Evolve tab1 instances into image representations using Gurobi.")
    parser.add_argument("--n", type=int, default=1000, help="Number of instances to process.")
    parser.add_argument("--out-dir", type=str, default="milp_evolve_tab1_processed", help="Output directory.")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="Optional max rows/cols for center image crop (top-left).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Final exported image size in pixels (square).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Deprecated: existing outputs are always reused when present.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing at the first instance with index >= this value.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=MAX_FILE_SIZE_MB_DEFAULT,
        help="Skip remote .tar.gz files larger than this many MB before downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    tensor_dir = out_dir / "tensors"
    ensure_dir(out_dir)
    ensure_dir(image_dir)
    ensure_dir(tensor_dir)

    print("Listing tab1 archives from Hugging Face...")
    archives = list_tab1_archives(max_file_size_mb=args.max_file_size_mb)
    if len(archives) == 0:
        raise RuntimeError("No tab1 archives found.")

    selected = archives[:args.n]
    print(f"Found {len(archives)} tab1 archives. Processing first {len(selected)}.")

    manifest_rows: List[Dict[str, object]] = []

    for idx, archive_repo_path in enumerate(selected, start=1):
        stem = Path(archive_repo_path).name[:-7]
        instance_index = parse_instance_index(stem)
        image_path = image_dir / f"{stem}.png"
        tensor_path = tensor_dir / f"{stem}.npz"

        if instance_index is not None and instance_index < args.start_index:
            if image_path.exists() and tensor_path.exists():
                existing_row = load_existing_manifest_row(
                    stem=stem,
                    archive_repo_path=archive_repo_path,
                    image_path=image_path,
                    tensor_path=tensor_path,
                )
                if existing_row is not None:
                    manifest_rows.append(existing_row)
            continue

        if image_path.exists() and tensor_path.exists():
            print(f"[{idx}/{len(selected)}] Using existing {stem}")
            existing_row = load_existing_manifest_row(
                stem=stem,
                archive_repo_path=archive_repo_path,
                image_path=image_path,
                tensor_path=tensor_path,
            )
            if existing_row is not None:
                manifest_rows.append(existing_row)
                continue
            print(f"  WARNING: Reprocessing {stem} due to load failure.")

        try:
            print(f"[{idx}/{len(selected)}] Processing {archive_repo_path}")
            row = process_one_instance(
                archive_repo_path=archive_repo_path,
                image_dir=image_dir,
                tensor_dir=tensor_dir,
                max_dim=args.max_dim,
                image_size=args.image_size,
            )
            manifest_rows.append(row)
        except Exception as e:
            print(f"  ERROR on {archive_repo_path}: {e}")
            manifest_rows.append({
                "instance_name": stem,
                "archive_repo_path": archive_repo_path,
                "code_repo_path": paired_code_path_for_archive(archive_repo_path),
                "generator_class": None,
                "mps_name": None,
                "num_rows": None,
                "num_cols": None,
                "num_nnz": None,
                "n_bin": None,
                "n_int": None,
                "n_cont": None,
                "image_path": str(image_path),
                "tensor_path": str(tensor_path),
                "archive_local_path": None,
                "code_local_path": None,
                "error": str(e),
            })

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    print(f"Done. Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()