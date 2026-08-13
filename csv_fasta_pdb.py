"""
Generate PDB files from CSV directory
arg parsing(csv_dir, fasta_dir, pdb_dir) -> read csv files and extract sequence, metadata(split, dG) by each row -> run ESMFold prediction on the sequence -> save PDB files to pdb_dir
"""
import torch
from glob import glob
import esm
from esm.data import read_fasta
import esm.inverse_folding
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
import pandas as pd

# Logger Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%y/%m/%d %H:%M:%S")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description='Generate PDB files from CSV directory')
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing .csv files')
    parser.add_argument('--fasta_dir', type=str, required=True, help='Output directory for .fasta files')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Output directory for .pdb files')
    # parser.add_argument('--num_recycles', type=int, default=None, help='Number of recycles for ESMFold')
    # parser.add_argument('--max_tokens_per_batch', type=int, default=1024, help='Maximum tokens per batch')
    # parser.add_argument('--chunk_size', type=int, default=None, help='Chunk size for ESMFold (helps with OOM)')
    return parser.parse_args()

def process_csv_files(csv_dir, fasta_dir):
    csv_files = glob(os.path.join(csv_dir, "*.csv"))
    metadata = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logger.error(f"Failed to read CSV {csv_file}: {e}")
            continue
        logger.info(f"Processing CSV file: {csv_file}")

        if 'name' not in df.columns or 'aa_seq' not in df.columns:
            logger.warning(f"CSV file {csv_file} is missing required columns 'name' or 'aa_seq'. Found: {list(df.columns)}")
            continue

        for row in df.iterrows():
            p_name = str(row['name'])
            # remove common extensions from sequence header if they exist (e.g. .cif, .pdb)
            p_name = os.path.splitext(p_name)[0]
            seq = str(row['aa_seq']).strip()
                
            fasta_path = os.path.join(fasta_dir, f"{p_name}.fasta")

            with open(fasta_path, 'w') as f:
                f.write(f">{p_name}\n{seq}\n")
                
                split = row['split']
                dG = row['deltaG']
                metadata[p_name] = {
                    'split': split,
                    'deltaG': float(dG)
                }

            logger.info(f"Wrote FASTA for {p_name} to {fasta_path}")
        
    # Save the accumulated metadata dictionary to metadata.json in fasta_dir
    metadata_path = os.path.join(fasta_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata dict to {metadata_path}")
    return metadata

def create_batched_sequence_datasest(
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

def run_esmfold(seq_path, out_dir, device, num_recycles=None, max_tokens_per_batch=1024, chunk_size=None):
    """Runs ESMFold prediction on a fasta file."""
    logger.info(f"Reading sequences from {seq_path}")
    all_sequences = sorted(read_fasta(seq_path), key=lambda header_seq: len(header_seq[1]))
    logger.info(f"Loaded {len(all_sequences)} sequences.")
    
    logger.info("Loading ESMFold model...")
    model = esm.pretrained.esmfold_v1().to(device)
    model.eval()
    if chunk_size is not None:
        model.set_chunk_size(chunk_size)
        
    logger.info("Starting Predictions using ESMFold")
    batched_sequences = create_batched_sequence_datasest(all_sequences, max_tokens_per_batch)
    num_completed, num_sequences = 0, len(all_sequences)
    
    for headers, seqs in batched_sequences:
        for header, seq in zip(headers, seqs):
            clean_header = os.path.splitext(header)[0]
            pdb_path = os.path.join(out_dir, f"{clean_header}.pdb")
            
            # Check if prediction already exists to enable resume-on-failure
            if os.path.exists(pdb_path):
                logger.info(f"Structure for {clean_header} already exists. Skipping.")
                num_completed += 1
                continue
                
            try:
                start_time = timer()
                with torch.no_grad():
                    kwargs = {}
                    if num_recycles is not None:
                        kwargs['num_recycles'] = num_recycles
                        
                    if device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            pdb_string = model.infer_pdb(seq, **kwargs)
                    else:
                        pdb_string = model.infer_pdb(seq, **kwargs)
                
                with open(pdb_path, "w") as f:
                    f.write(pdb_string)
                
                end_time = timer()
                num_completed += 1
                logger.info(f"Predicted structure for {clean_header} in {end_time - start_time:.2f}s ({num_completed}/{num_sequences})")
            except Exception as e:
                logger.error(f"Error predicting structure for {clean_header}: {e}")
                
            # Perform clean up
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

def main():
    args = get_args()
    os.makedirs(args.csv_dir, exist_ok=True)
    os.makedirs(args.fasta_dir, exist_ok=True)
    os.makedirs(args.pdb_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Process CSV files to generate FASTA files and metadata
    logger.info("--- Step 1: Processing CSV files and writing FASTA files ---")
    process_csv_files(args.csv_dir, args.fasta_dir)
    
    # 2. Run ESMFold prediction on each generated FASTA file
    logger.info("--- Step 2: Running ESMFold prediction on FASTA files ---")
    fasta_files = glob(os.path.join(args.fasta_dir, "*.fasta"))
    if not fasta_files:
        logger.warning(f"No FASTA files found in {args.fasta_dir}. Pipeline exiting.")
        return
        
    for fasta_file in fasta_files:
        run_esmfold(
            seq_path=fasta_file,
            out_dir=args.pdb_dir,
            device=device,
            num_recycles=args.num_recycles,
            max_tokens_per_batch=args.max_tokens_per_batch,
            chunk_size=args.chunk_size
        )
        
    logger.info("Pipeline completed successfully!")

if __name__ == '__main__':
    main()