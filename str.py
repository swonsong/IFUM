import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import re
import esm

# load 
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    esm_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esm_model = esm_model.to(device).eval()    
    return esm_model, alphabet


@torch.no_grad()
def sequence_to_pdb(esm_model, alphabet, seq, device, chunk_size=None):
    if chunk_size is not None:
        esm_model.trunk.set_chunk_size(chunk_size)
    inputs = alphabet.encode(seq)
    inputs = inputs.unsqueeze(0).to(device)
    outputs = esm_model(inputs)
    pdb_str = outputs["pdb_str"][0]
    mean_plddt = outputs["plddt"].mean().item()
    return pdb_str, mean_plddt


def out_name(raw_name):
    base = re.sub(r"\.(cif|pdb)$", "", raw_name)
    return base + ".pdb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--chunk_size", type=int, default=64,
        help="chunking for memory efficiency. 0 disables chunking.",
    )
    parser.add_argument(
        "--max_len", type=int, default=400,
        help="Sequences longer than this will be skipped for memory protection.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv, index_col=0)
    esm_model, alphabet = load_model()
    chunk_size = args.chunk_size if args.chunk_size > 0 else None

    skipped, failed = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        name = out_name(str(row["name"]))
        out_path = os.path.join(args.out_dir, name)
        if os.path.exists(out_path) and not args.overwrite:
            continue

        seq = str(row["aa_seq"]).strip().upper()
        if len(seq) > args.max_len:
            skipped.append(row["name"])
            continue

        try:
            pdb_str, mean_plddt = sequence_to_pdb(esm_model, alphabet, seq, device, chunk_size)
        except RuntimeError as e:
            print(f"[WARN] failed on {row['name']} ({len(seq)} aa): {e}")
            failed.append(row["name"])
            torch.cuda.empty_cache()
            continue

        with open(out_path, "w") as f:
            f.write(pdb_str)

    print(f"Done. skipped(too long)={len(skipped)} failed={len(failed)}")
    if skipped:
        print("Skipped:", skipped[:20], "..." if len(skipped) > 20 else "")
    if failed:
        print("Failed:", failed[:20], "..." if len(failed) > 20 else "")


if __name__ == "__main__":
    main()

"""
python prepare_pdb.py \
    --csv data/rocklin.csv \
    --out_dir pdbs/ \
    --chunk_size 64 \
    --max_len 400
"""