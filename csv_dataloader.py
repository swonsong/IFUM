'''pseudocode for dataloader to apply megascale & MGnify stability dataset to IFUM'''

'''
---datasets---
megascale:
    (processed) K50 dG dataset.csv:
        name[0], aa_seq[27], deltaG[22], mut_type[28], +a
    alphafold .PDB(WT):
        ATOM      1  N   VAL A   1       3.830   6.584  12.265  1.00 67.14           N  
        # if 3d atom coordinates between WT and mut doesnt that differ, use pdb data to every megascale proteins?
        # then apply esmfold to MGnify dataset only

MGnify stability.csv(seq):
    name, AA seq, deltaG, +a
'''
'''
---models---
esmif: backbone atom coordinate -> (structure embedding?) -> aa sequence
prott5: aa sequence -> LM embedding(seq embedding)

esmfold: aa seq->language model feature(internal repr.: pair rep, seq rep)->3d atomic coordinate
'''

import pandas as pd
import torch
from glob import glob
import esm
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import os
import warnings
import time
import typing as T
import logging
from timeit import default_timer as timer
import gc
import json

# Logger Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%y/%m/%d %H:%M:%S")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
warnings.filterwarnings('ignore')

'''
parse csv directory(input) & pdb directory(output)
for csv files in csv directory:
    get [name, aa_seq, deltaG]columns
    concat
replace sequences(U,Z,O) to X in concatenated dataframe
remove duplicated sequences
change format(dataframe to List[Tuple[str, str]])
apply "create_batched_sequence_datasets()"
run esmfold rowwise for concatenated dataframe
write [name].pdb file, 3d atom coordinate
'''

def get_args():
    parser = argparse.ArgumentParser(description='Generate PDB files from CSV directory')
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing .csv files') # megascale & mgnify csv files
    parser.add_argument('--pdb_dir', type=str, required=True, help='Output directory for .pdb files') # fixed atom 3D coordinate
    return parser.parse_args()

def process_csv_files(csv_dir):
    csv_files = glob(os.path.join(csv_dir, "*.csv"))
    processed_csv = pd.DataFrame(columns=['name','aa_seq','deltaG'])
    
    for csv_file in csv_files:
        try:
            file = pd.read_csv(csv_file)
        except Exception as e:
            logger.error(f"Failed to read CSV {csv_file}: {e}")
            continue
        logger.info(f"Processing CSV file: {csv_file}")

        if 'name' not in file.columns or 'aa_seq' not in file.columns:
            logger.warning(f"CSV file {csv_file} is missing required columns 'name' or 'aa_seq'. Found: {list(df.columns)}")
            continue

        processed_csv = pd.concat([processed_csv, file['name','aa_seq','deltaG']], ignore_index=True)

        def clean_seq(input_seq:str):
            input_seq = input_seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            return input_seq
        processed_csv.apply(clean_seq(), axis=1)
        processed_csv = processed_csv.drop_duplicates(subset=['aa_seq'])
    return processed_csv

def create_batched_sequence_datasets(
    sequences: T.List[T.Tuple[str, str]], 
    max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    """Batches sequences to avoid OOM during inference."""
    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)
    yield batch_headers, batch_sequences

def run_esmfold(input_csv, out_dir, device, num_recycles=None, max_tokens_per_batch=1024, chunk_size=None):
    """Runs ESMFold prediction on a processed csv file"""
    logger.info(f"Reading sequences from {input_csv}")
    all_sequences = list(zip(input_csv['name'], input_csv['aa_seq']))
    logger.info(f"Loaded {len(all_sequences)} sequences.")
    
    logger.info("Loading ESMFold model...")
    model = esm.pretrained.esmfold_v1().to(device)
    model.eval()
    if chunk_size is not None:
        model.set_chunk_size(chunk_size)
        
    logger.info("Starting Predictions using ESMFold")
    batched_sequences = create_batched_sequence_datasets(all_sequences, max_tokens_per_batch)
    num_completed, num_sequences = 0, len(all_sequences)
    
    for headers, sequences in batched_sequences:
        start = timer()
        try:
            output = model.infer(sequences, num_recycles=num_recycles)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA OOM on a batch of size {len(sequences)}. Try lowering --max-tokens-per-batch.")
                continue
            raise
        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        tottime = timer() - start
        for header, seq, pdb_string, mean_plddt, ptm in zip(headers, sequences, pdbs, output["mean_plddt"], output["ptm"]):
            output_file = out_dir / f"{header}.pdb"
            output_file.write_text(pdb_string)
            num_completed += 1
            logger.info(f"Predicted structure for {header} (L={len(seq)}, pLDDT={mean_plddt:.1f}, pTM={ptm:.3f}) in {tottime/len(headers):0.1f}s. ({num_completed}/{num_sequences})")
    logger.info("ESMFold predictions finished.")
    del model
    # clean up
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

def main():
    args = get_args()
    os.makedirs(args.csv_dir, exist_ok=True)
    os.makedirs(args.pdb_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("--- Processing CSV files ---")
    input_csv = process_csv_files(args.csv_dir)
    
    # run ESMFold
    logger.info("--- Running ESMFold prediction ---")
    run_esmfold(
        input_csv=input_csv,
        out_dir=args.pdb_dir,
        device=device,
        num_recycles=args.num_recycles,
        max_tokens_per_batch=args.max_tokens_per_batch,
        chunk_size=args.chunk_size
    )
        
    logger.info("Pipeline completed successfully!")

if __name__ == '__main__':
    main()

'''
python csv_dataloader.py --csv_dir [path to csv files] --pdb_dir [path to pdb files: output directory]
'''