#!/usr/bin/env python3
import argparse
import gzip
import pickle
from pathlib import Path
import tarfile

import pandas as pd
import torch
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
except Exception as e:
    Data = None


_DYNAMIC_CLASS_CACHE = {}


class _Placeholder:
    pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("gap_data"):
            key = (module, name)
            if key not in _DYNAMIC_CLASS_CACHE:
                _DYNAMIC_CLASS_CACHE[key] = type(name, (_Placeholder,), {})
            return _DYNAMIC_CLASS_CACHE[key]
        return super().find_class(module, name)


def load_pkl_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def to_pyg_data(obj, add_node_type: bool = True):
    if Data is None:
        raise ImportError("torch_geometric is required to build PyG Data objects.")

    store = getattr(obj, "_store", None)
    if store is None:
        raise ValueError("Object does not expose a _store attribute.")

    x_rows = store.get("x_rows")
    x_cols = store.get("x_cols")
    edge_index = store.get("edge_index_rowcols")
    edge_vals = store.get("edge_vals_rowcols")

    if x_rows is None or x_cols is None or edge_index is None:
        raise ValueError("Missing required tensors (x_rows, x_cols, edge_index_rowcols).")

    # Sanitize any non-finite values early
    x_rows = torch.nan_to_num(x_rows, nan=0.0, posinf=0.0, neginf=0.0)
    x_cols = torch.nan_to_num(x_cols, nan=0.0, posinf=0.0, neginf=0.0)
    if edge_vals is not None:
        edge_vals = torch.nan_to_num(edge_vals, nan=0.0, posinf=0.0, neginf=0.0)

    num_rows = x_rows.size(0)
    row_dim = x_rows.size(1)
    col_dim = x_cols.size(1)
    feat_dim = max(row_dim, col_dim)

    if row_dim < feat_dim:
        x_rows = F.pad(x_rows, (0, feat_dim - row_dim))
    if col_dim < feat_dim:
        x_cols = F.pad(x_cols, (0, feat_dim - col_dim))

    if add_node_type:
        row_type = torch.zeros((x_rows.size(0), 2), device=x_rows.device, dtype=x_rows.dtype)
        col_type = torch.zeros((x_cols.size(0), 2), device=x_cols.device, dtype=x_cols.dtype)
        row_type[:, 0] = 1
        col_type[:, 1] = 1
        x_rows = torch.cat([x_rows, row_type], dim=1)
        x_cols = torch.cat([x_cols, col_type], dim=1)

    x = torch.cat([x_rows, x_cols], dim=0)

    edge_index = edge_index.clone()
    edge_index[1] = edge_index[1] + num_rows

    edge_attr = None
    if edge_vals is not None:
        if edge_vals.dim() == 2 and edge_vals.size(1) >= 1:
            edge_attr = edge_vals[:, 0:1]
        else:
            edge_attr = edge_vals.view(-1, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def parse_index(file_path: Path, prefix: str, suffix: str):
    name = file_path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    idx_str = name[len(prefix) : -len(suffix)]
    if not idx_str.isdigit():
        return None
    return int(idx_str)


def parse_milp_name(milp_name: str):
    if not milp_name.startswith("milp_"):
        return None, None
    parts = milp_name.split("-", maxsplit=1)
    if len(parts) != 2:
        return None, None
    id_part = parts[0].replace("milp_", "")
    if not id_part.isdigit():
        return None, None
    return int(id_part), parts[1]


def build_dataframe(
    data_root: Path,
    conv_root: Path,
    attributes_csv: Path | None = Path("language_data_example_tab1_attributes.csv"),
) -> pd.DataFrame:
    rows = []
    missing_conv = 0
    missing_desc = 0
    missing_data = 0
    missing_attr = 0

    attr_map = {}
    if attributes_csv:
        attr_df = pd.read_csv(attributes_csv)
        attr_map = {
            (int(r.milp_id), str(r.source)): r.generator_class
            for r in attr_df.itertuples(index=False)
        }

    for data_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        milp_id, source = parse_milp_name(data_dir.name)
        conv_dir = conv_root / data_dir.name
        if not conv_dir.exists():
            missing_conv += 1
            continue

        data_files = {}
        for f in data_dir.glob("data_*.pkl.gz"):
            idx = parse_index(f, "data_", ".pkl.gz")
            if idx is not None:
                data_files[idx] = f

        desc_files = {}
        for f in conv_dir.glob("desc_*.txt"):
            idx = parse_index(f, "desc_", ".txt")
            if idx is not None:
                desc_files[idx] = f

        for idx in sorted(set(data_files) | set(desc_files)):
            data_path = data_files.get(idx)
            desc_path = desc_files.get(idx)

            if data_path is None:
                missing_data += 1
                continue
            if desc_path is None:
                missing_desc += 1
                continue

            data_obj = load_pkl_gz(data_path)
            description = desc_path.read_text(encoding="utf-8").strip()

            generator_class = None
            if milp_id is not None and source is not None and attr_map:
                generator_class = attr_map.get((milp_id, source))
                if generator_class is None:
                    missing_attr += 1

            rows.append(
                {
                    "milp_name": data_dir.name,
                    "milp_id": milp_id,
                    "source": source,
                    "index": idx,
                    "data": data_obj,
                    "pyg_data": to_pyg_data(data_obj),
                    "description": description,
                    "generator_class": generator_class,
                    "data_path": str(data_path),
                    "desc_path": str(desc_path),
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["missing_conv_dirs"] = missing_conv
    df.attrs["missing_desc_files"] = missing_desc
    df.attrs["missing_data_files"] = missing_data
    df.attrs["missing_attr_rows"] = missing_attr
    return df


def build_pyg_dataset(
    data_root: Path,
    conv_root: Path,
    attributes_csv: Path | None = Path("language_data_example_tab1_attributes.csv"),
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 256,
) -> dict:
    """
    Build a compact PyTorch-serializable dataset with PyG Data objects and text embeddings.
    Returns a dict with keys: data_list, classes, class_to_idx.
    """
    from text_embedding import embed_texts_pretrained

    df = build_dataframe(data_root, conv_root, attributes_csv)
    df = df.dropna(subset=["generator_class"]).reset_index(drop=True)

    texts = df["description"].fillna("").astype(str).tolist()
    text_embs = embed_texts_pretrained(
        texts=texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    classes = sorted(df["generator_class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = torch.tensor([class_to_idx[c] for c in df["generator_class"].tolist()])

    data_list = []
    for i, row in df.iterrows():
        g = row["pyg_data"]
        if isinstance(g, str):
            g = to_pyg_data(load_pkl_gz(Path(row["data_path"])))
        g.y = labels[i]
        g.text_emb = text_embs[i].unsqueeze(0)
        g.description = row["description"]
        data_list.append(g)

    return {
        "data_list": data_list,
        "classes": classes,
        "class_to_idx": class_to_idx,
    }


def build_pyg_dataset_from_hf(
    repo_id: str = "microsoft/MILP-Evolve",
    tar_filename: str = "language_data_example.tar.gz",
    local_dir: Path = Path("hf_milp_evolve"),
    extract_subdir: str = "extracted",
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 256,
) -> dict:
    """
    Download the HF dataset tarball and build the PyG dataset payload.
    The .pkl.gz files are handled by load_pkl_gz during processing.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ImportError("Please install huggingface_hub to download the dataset.") from e

    local_dir = Path(local_dir)
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[f"**/{tar_filename}"],
    )

    tar_path = local_dir / tar_filename
    if not tar_path.exists():
        matches = list(local_dir.rglob(tar_filename))
        if matches:
            tar_path = matches[0]
        else:
            try:
                from huggingface_hub import hf_hub_download
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not find {tar_filename} under {local_dir}"
                ) from e
            tar_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=tar_filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                )
            )
    extract_dir = local_dir / extract_subdir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    data_root = extract_dir / "language_data_example/tab1/data"
    conv_root = extract_dir / "language_data_example/tab1/conv"
    attr_csv = "language_data_example_tab1_attributes.csv"

    return build_pyg_dataset(
        data_root,
        conv_root,
        attr_csv,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Combine tab1 MILP data with language descriptions."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("language_data_example/tab1/data"),
        help="Path to tab1 data directory.",
    )
    parser.add_argument(
        "--conv-root",
        type=Path,
        default=Path("language_data_example/tab1/conv"),
        help="Path to tab1 conv directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save dataframe (pickle).",
    )

    args = parser.parse_args()
    df = build_dataframe(args.data_root, args.conv_root)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(args.output)

    print(f"Rows: {len(df)}")
    print(
        "Missing conv dirs: {0}, missing desc files: {1}, missing data files: {2}".format(
            df.attrs.get("missing_conv_dirs", 0),
            df.attrs.get("missing_desc_files", 0),
            df.attrs.get("missing_data_files", 0),
        )
    )

    df.to_csv("language_data_example_tab1.csv", index=False)


if __name__ == "__main__":
    main()
