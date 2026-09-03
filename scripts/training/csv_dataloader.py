'''
# pseudocode
parse csv directory(input) & pdb directory(output)
for csv files in csv directory:
    get [name, aa_seq, deltaG]columns
    if [aa_seq]column not exist: convert [dna_seq]column to [aa_seq]column
    concat
replace sequences(U,Z,O) to X in concatenated dataframe
remove duplicated sequences by [aa_seq]
change format(dataframe to List[Tuple[str, str]])
apply "create_batched_sequence_datasets()"
run esmfold rowwise for concatenated dataframe
write [name].pdb file, 3d atom coordinate
write dG.csv file, [name, deltaG]columns in pdb directory
'''

import pandas as pd
import torch
from torch import nn
import esm
# ---ESMFold2 ESMC---
# from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

# ---ESMFold2 folding---
# from esm.models.esmfold2 import (
#     # DNAInput,
#     ESMFold2InputBuilder,
#     EsmFold2Model,
#     # LigandInput,
#     Modification,
#     ProteinInput,
#     StructurePredictionInput,
# )

# ---ESMFold2 single folding(esm/models/esmfold2/model.py)---
from esm.models.esmfold2 import EsmFold2Model

from glob import glob
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
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing .csv files') # megascale & mgnify csv files
    parser.add_argument('--pdb_dir', type=str, required=True, help='Output directory for .pdb files') # fixed atom 3D coordinate
    parser.add_argument('--num_recycles', type=int, default=None, help='Number of recycles for ESMFold')
    parser.add_argument('--chunk_size', type=int, default=None, help='Chunk size for ESMFold optimization')
    parser.add_argument('--max_tokens_per_batch', type=int, default=1024, help='Max tokens per batch')
    return parser.parse_args()

def dna_to_protein(dna_sequence):
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
        amino_acid = codon_table.get(codon, "X") # "X" for unknown/incomplete codons
        protein_sequence.append(amino_acid)
            
    return "".join(protein_sequence)

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
        if 'name' not in file.columns or ('aa_seq' not in file.columns and 'dna_seq' not in file.columns):
            logger.warning(f"CSV file {csv_file} is missing required columns 'name' or 'aa_seq'/'dna_seq'. Found: {list(file.columns)}")
            continue

        file['name'] = file['name'].str.replace('.', '_', regex=False)
        if 'aa_seq' not in file.columns:
            file['aa_seq'] = file['dna_seq'].apply(dna_to_protein)
        if 'deltaG' not in file.columns:
            file['deltaG'] = None

        processed_csv = pd.concat([processed_csv, file[['name', 'aa_seq', 'deltaG']]], ignore_index=True)

    def clean_seq(input_seq:str):
        input_seq = input_seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        return input_seq
    processed_csv['aa_seq'] = processed_csv['aa_seq'].apply(clean_seq)
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
    # ---ESMFold2 ESMC---
    # model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-68", device="cuda").eval()
    # tokenizer = EsmcTokenizer()
    
    
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device="cuda").eval()

    if chunk_size is not None:
        model.set_chunk_size(chunk_size)
        
    logger.info("Starting Predictions using ESMFold")
    batched_sequences = create_batched_sequence_datasets(all_sequences, max_tokens_per_batch)
    num_completed, num_sequences = 0, len(all_sequences)
    
    for headers, sequences in batched_sequences:
        start = timer()
        try:
            # ---ESMFold2 ESMC---
            # inputs = tokenizer(sequences, return_tensors="pt", padding=True)
            # inputs = {k: v.to(model.device) for k, v in inputs.items()}
            # with torch.inference_mode():
            #     output = model(**inputs)

            # ---ESMFold2 single folding(esm/models/esmfold2/model.py)---
            model = EsmFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()

            # ---ESMFold1(original IFUM)---
            # output = model.infer(sequences)

            # ---ESMFold2 folding---
            # spi = StructurePredictionInput(
            #     sequences=[
            #         ProteinInput(id="A", sequence=sequences)
            #     ]
            # )

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA OOM on a batch of size {len(sequences)}. Try lowering --max-tokens-per-batch.")
                continue
            raise

        # ---ESMFold2 single folding(esm/models/esmfold2/model.py)---
        pdbs = model.infer_protein_as_pdb(sequences)
        
        # ---ESMFold1(original IFUM)---
        # output = {key: value.cpu() for key, value in output.items()}
        # pdbs = model.output_to_pdb(output)

        # ---ESMFold2 folding---        
        # result = ESMFold2InputBuilder().fold(
        #     model, spi, num_loops=20, num_sampling_steps=100, num_diffusion_samples=1, seed=0
        # )

        tottime = timer() - start
        
        # ---ESMFold2 ESMC---
        # print(f"pLDDT mean: {float(result.plddt.mean()):.3f}, pTM: {float(result.ptm):.3f}, ipTM: {float(result.iptm):.3f}")
        
        for header, seq, pdb_string, mean_plddt, ptm in zip(headers, sequences, pdbs, 
                                                            # output["mean_plddt"], output["ptm"]
                                                            ):
            output_file = Path(out_dir) / f"{header}.pdb"
            output_file.write_text(pdb_string)
            num_completed += 1
            # logger.info(f"Predicted structure for {header} (L={len(seq)}, pLDDT={mean_plddt:.1f}, pTM={ptm:.3f}) in {tottime/len(headers):0.1f}s. ({num_completed}/{num_sequences})")
            logger.info(f"Predicted structure for {header} (L={len(seq)}) in {tottime/len(headers):0.1f}s. ({num_completed}/{num_sequences})")
    logger.info("ESMFold predictions finished.")
    # clean up
    del model
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
    
    dG_csv = input_csv.drop(columns=['aa_seq']).set_index('name')
    dG_csv_path = os.path.join(args.pdb_dir, "dG.csv")
    dG_csv.to_csv(dG_csv_path)
    logger.info(f"dG data saved to {dG_csv_path}")

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