"""
Preprocess VCF files into numpy arrays for machine learning.
"""

import argparse
from pathlib import Path
import numpy as np
import allel
from typing import List

# Number of SNPs per chromosome
CHROMOSOME_SNP_COUNTS = {
    1: 6468094, 2: 7142354, 3: 5853017, 4: 5785195, 5: 5374295,
    6: 5086995, 7: 4785234, 8: 4652396, 9: 3624329, 10: 4111690,
    11: 4192831, 12: 4007427, 13: 3055900, 14: 2857858, 15: 2618903,
    16: 2697949, 17: 2746760, 18: 2267185, 19: 1832506, 20: 1812841,
    21: 1105538, 22: 1103547
}

class VCFProcessor:
    """Process VCF files into numpy arrays suitable for machine learning."""
    
    def __init__(self, max_snps_per_chrom: int = 100000):
        self.max_snps = max_snps_per_chrom
    
    def process_vcf(self, vcf_path: Path) -> np.ndarray:
        """
        Process a single VCF file into a binary matrix.
        
        Args:
            vcf_path: Path to VCF file
            
        Returns:
            numpy array of shape (n_samples, n_variants) containing binary SNP data
        """
        chrom = int(vcf_path.stem.split('_')[1].split('.')[0])
        total_snps = CHROMOSOME_SNP_COUNTS[chrom]
        
        print(f"\nProcessing chromosome {chrom} ({total_snps:,} SNPs)")
        
        class ProgressWriter:
            def __init__(self, total):
                self.total = total
                self.current = 0
                
            def write(self, message):
                if "rows in" in message:
                    try:
                        self.current = int(message.split()[1])
                        percent = (self.current / self.total) * 100
                        print(f"\rProgress: {self.current:,}/{self.total:,} SNPs ({percent:.1f}%)", end="", flush=True)
                    except:
                        pass
            
            def flush(self):
                pass
        
        # Read VCF and convert to binary matrix
        callset = allel.read_vcf(
            str(vcf_path), 
            fields=['samples', 'calldata/GT'],
            log=ProgressWriter(total_snps)
        )
        
        gt = allel.GenotypeArray(callset['calldata/GT'])
        matrix = gt.to_n_alt().T
        matrix = (matrix > 0).astype(np.uint8)
        
        print(f"Selecting top {self.max_snps} SNPs by variance...")
        if matrix.shape[1] > self.max_snps:
            variance = np.var(matrix, axis=0)
            top_indices = np.argsort(variance)[-self.max_snps:]
            matrix = matrix[:, top_indices]
            
        print(f"\nFinal matrix shape: {matrix.shape}")
        return matrix

def combine_chromosomes(matrices: List[np.ndarray]) -> np.ndarray:
    """Combine multiple chromosome matrices into one."""
    return np.concatenate(matrices, axis=1)

def main():
    parser = argparse.ArgumentParser(description="Preprocess VCF files into numpy arrays")
    parser.add_argument(
        "-c", 
        "--chromosomes", 
        type=int, 
        nargs="+", 
        default=[22],
        help="Chromosome numbers to process (default: [22])"
    )
    parser.add_argument(
        "--max-snps", 
        type=int, 
        default=100000,
        help="Maximum SNPs to keep per chromosome"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force reprocessing even if preprocessed files exist"
    )
    args = parser.parse_args()
    
    download_dir = Path("data/downloads")
    preprocess_dir = Path("data/preprocessed")
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    
    processor = VCFProcessor(max_snps_per_chrom=args.max_snps)
    matrices = []
    
    # Process individual chromosomes
    for chrom in args.chromosomes:
        if not 1 <= chrom <= 22:
            print(f"Invalid chromosome number {chrom}, skipping")
            continue
            
        vcf_path = download_dir / f"chromosome_{chrom}.vcf.gz"
        out_path = preprocess_dir / f"chromosome_{chrom}.npz"
        
        if not vcf_path.exists():
            print(f"VCF file not found: {vcf_path}")
            continue
        
        if out_path.exists() and not args.force:
            print(f"Preprocessed file already exists for chromosome {chrom}, loading: {out_path}")
            data = np.load(out_path)
            matrices.append(data['matrix'])
            continue
            
        matrix = processor.process_vcf(vcf_path)
        np.savez_compressed(out_path, matrix=matrix)
        matrices.append(matrix)
    
    # Combine chromosomes if multiple were processed
    if len(matrices) > 1:
        combined = combine_chromosomes(matrices)
        chrom_string = "-".join(str(c) for c in sorted(args.chromosomes))
        out_path = preprocess_dir / f"chromosome_{chrom_string}.npz"
        
        if not out_path.exists() or args.force:
            np.savez_compressed(out_path, matrix=combined)
            print(f"\nSaved combined matrix with shape {combined.shape} to {out_path}")
        else:
            print(f"\nCombined matrix file already exists: {out_path}")

if __name__ == "__main__":
    main()