import pandas as pd
import torch
from torch import nn
from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    EsmFold2Model,
    LigandInput,
    Modification,
    ProteinInput,
    StructurePredictionInput,
)
# from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2Model
from glob import glob
from tqdm import tqdm
import argparse
import sys
import os
from pathlib import Path
import warnings
import time
import typing as T
import logging
from timeit import default_timer as timer
import gc

'''Logger Setup'''
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%y/%m/%d %H:%M:%S")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description='Generate .cif/.pdb files from CSV directory')
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing .csv files')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Output directory for .cif/.pdb files')
    parser.add_argument('--num_recycles', type=int, default=None, help='Number of recycles for ESMFold')
    parser.add_argument('--chunk_size', type=int, default=None, help='Chunk size for ESMFold optimization')
    parser.add_argument('--max_tokens_per_batch', type=int, default=1024, help='Max tokens per batch')
    return parser.parse_args()

def dna_to_protein(dna_sequence:str):
    codon_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
    } # *: Stop Codons
    
    dna_sequence = dna_sequence.upper()
    protein_sequence = []

    for i in range(0, len(dna_sequence) - 2, 3):
        codon = dna_sequence[i:i+3]
        amino_acid = codon_table.get(codon, "X")
        protein_sequence.append(amino_acid)

    return "".join(protein_sequence)

def clean_seq(input_seq:str):
    input_seq = input_seq.upper().replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    return input_seq

def process_csv(csv_dir):
    mega1 = glob(os.path.join(csv_dir, "Tsuboyama2023_Dataset1_20230416.csv"))
    mega2 = glob(os.path.join(csv_dir, "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"))
    mgnify = glob(os.path.join(csv_dir, "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv"))
    processed_csv = pd.DataFrame(columns=['name','aa_seq','dG', 'WT_name', 'mut_type'])
    
    '''prep'''
    condition = mega1['name'].str.contains('scramble') & (mega1['deltaG'] <= 0.5)
    mega1 = mega1.loc[condition, ['name', 'dna_seq', 'deltaG']].copy()
    
    mega1['dna_seq'] = mega1['dna_seq'].str.apply(dna_to_protein)
    mega1.rename(columns={'dna_seq':'aa_seq', 'deltaG':'dG'}, inplace=True)

    mega1['WT_name'] = mega1['name']
    mega1['mut_type'] = 'wt'
    
    mega2 = mega2[['name', 'aa_seq', 'dG_ML', 'WT_name', 'mut_type']]
    mega2['dG_ML'] = pd.to_numeric(mega2['dG_ML'], errors='coerce')
    mega2 = mega2.dropna(subset=['dG_ML'])
    mega2.rename(columns={'dG_ML':'dG'})
    
    mgnify

    processed_csv = pd.concat([processed_csv, mega1, mega2, mgnify], ignore_index=True)
    processed_csv['name'] = processed_csv['name'].str.replace('|', ':', regex=False) # for EA|run*
    processed_csv['aa_seq'] = processed_csv['aa_seq'].str.apply(clean_seq)

    return processed_csv

def esm_run_prep(input_csv, pdb_dir):
    '''input_csv = processed_csv, pdb_dir = args.pdb_dir'''
    pdb_files = glob(os.path.join(pdb_dir, "*.pdb")) + glob(os.path.join(pdb_dir, "*.cif"))
    pdb_baseid = set(os.path.basename(f).split(".cif")[0] for f in pdb_files)

    input_csv = input_csv[input_csv['mut_type'] == 'wt']
    input_csv = input_csv.sort_values(by='name').drop_duplicates(subset=['aa_seq'], keep='first')
    
    skip = input_csv['WT_name'].isin(pdb_baseid)
    input_csv = input_csv.loc[~skip, ['name', 'aa_seq']].copy()

    return input_csv

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
    """Runs ESMFold2 prediction on a processed csv file"""
    logger.info(f"Reading sequences from {input_csv}")
    
    all_sequences = list(zip(input_csv['name'], input_csv['aa_seq']))
    logger.info(f"Loaded {len(all_sequences)} sequences.")
    
    logger.info("Loading ESMFold model...")
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device=str(device)).eval()
    # model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device_map='auto').eval()

    if chunk_size is not None:
        model.set_chunk_size(chunk_size)
        
    logger.info("Starting Predictions using ESMFold")
    batched_sequences = create_batched_sequence_datasets(all_sequences, max_tokens_per_batch)
    num_completed, num_sequences = 0, len(all_sequences)
    
    for headers, sequences in batched_sequences:
        for idx, (header, seq) in enumerate(zip(headers, sequences)):
            start = timer()
            try:
                spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
                loops = num_recycles if num_recycles is not None else 20
                
                result = ESMFold2InputBuilder().fold(
                    model, spi, num_loops=loops, num_sampling_steps=100, num_diffusion_samples=1, seed=0
                    )
                # pdb_string = model.infer_protein_as_pdb(seq, num_loops=loops, num_sampling_steps=100)

                tottime = timer() - start
                output_file = Path(out_dir) / f"{header}.cif"
                with open(output_file, "w") as f:
                    f.write(result.complex.to_mmcif())
                    # f.write(pdb_string)
                    
                num_completed += 1
                logger.info(f"Predicted structure for {header} in {tottime/len(headers):0.1f}s. ({num_completed}/{num_sequences})")

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"CUDA OOM on sequence '{header}' (Length: {len(seq)}). Skipping this sequence.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

    logger.info("ESMFold predictions finished.")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

def main():
    args = get_args()
    os.makedirs(args.csv_dir, exist_ok=True)
    os.makedirs(args.pdb_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # apply DDP?
    logger.info(f"Using device: {device}")
    
    logger.info("--- Processing CSV files ---")
    processed_csv = process_csv(args.csv_dir)
    processed_csv_path = os.path.join(args.pdb_dir, "processed.csv")
    processed_csv.to_csv(processed_csv_path)
    logger.info(f"processed csv data saved to {processed_csv_path}")

    esm_run_csv = esm_run_prep(processed_csv, args.pdb_dir)
    # run ESMFold
    logger.info("--- Running ESMFold prediction ---")
    run_esmfold(
        input_csv=esm_run_csv,
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
python csv_dataloader.py --csv_dir [path to csv files] --pdb_dir [path to cif/pdb files: output directory] --max_tokens_per_batch [] --chunk_size [] --num_recycle
'''
