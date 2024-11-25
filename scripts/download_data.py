"""
Download VCF files from the 1000 Genomes Project FTP server.
"""

import argparse
from pathlib import Path
import urllib.request

VCF_URL_TEMPLATE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr<NUMBER>.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

def download_chromosome(number: int, output_dir: Path):
    """
    Download a single chromosome VCF file.
    
    Args:
        number: Chromosome number (1-22)
        output_dir: Directory to save the downloaded file
    """
    url = VCF_URL_TEMPLATE.replace("<NUMBER>", str(number))
    output_path = output_dir / f"chromosome_{number}.vcf.gz"
    
    if output_path.exists():
        print(f"Chromosome {number} VCF already exists at {output_path}")
        return
        
    print(f"Downloading chromosome {number} VCF from {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Successfully downloaded {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download 1000 Genomes Project VCF files")
    parser.add_argument(
        "-c", 
        "--chromosomes", 
        type=int, 
        nargs="+", 
        default=[22],
        help="Chromosome numbers to download (default: [22])"
    )
    args = parser.parse_args()

    download_dir = Path("data/downloads")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    for chrom in args.chromosomes:
        if not 1 <= chrom <= 22:
            print(f"Invalid chromosome number {chrom}, skipping")
            continue
        download_chromosome(chrom, download_dir)

if __name__ == "__main__":
    main()