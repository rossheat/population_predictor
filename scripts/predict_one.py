"""
Make predictions for individual samples using a trained super-population model.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from train_test_model import PopulationPredictor, SUPER_POPULATIONS

def load_model(model_path: Path, input_dim: int) -> PopulationPredictor:
    """Load the trained model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PopulationPredictor(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    return model

def predict_sample(model: PopulationPredictor, sample_data: np.ndarray) -> dict:
    """
    Make prediction for a single sample.
    
    Args:
        model: Trained PopulationPredictor model
        sample_data: Numpy array of shape (n_variants,) containing binary SNP data
        
    Returns:
        Dictionary containing predicted super-population and probabilities
    """
    device = next(model.parameters()).device
    
    # Prepare input
    x = torch.FloatTensor(sample_data).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
    
    # Convert to probabilities dict
    probs_dict = {
        pop: probabilities[0][idx].item() 
        for pop, idx in SUPER_POPULATIONS.items()
    }
    
    # Get predicted population
    predicted_pop = [k for k, v in SUPER_POPULATIONS.items() if v == pred_idx][0]
    
    return {
        'predicted_population': predicted_pop,
        'probabilities': probs_dict
    }

def load_sample_data(npz_path: Path, sample_idx: int) -> np.ndarray:
    """Load genetic data for a single sample from preprocessed NPZ file."""
    data = np.load(npz_path)
    matrix = data['matrix']
    
    if sample_idx >= len(matrix):
        raise ValueError(f"Sample index {sample_idx} out of range (max: {len(matrix)-1})")
        
    return matrix[sample_idx]

def main():
    parser = argparse.ArgumentParser(description="Predict super-population for individual samples")
    parser.add_argument(
        "--model", 
        type=Path,
        required=True,
        help="Path to trained model file (.pt)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to preprocessed data file (.npz)"
    )
    parser.add_argument(
        "-s",
        "--sample-idx",
        type=int,
        required=True,
        help="Index of the sample to predict"
    )
    args = parser.parse_args()

    # Ensure model and data match
    model_name = args.model.stem  # e.g., "model_chr21-22"
    data_name = args.data.stem    # e.g., "chromosome_21-22"
    
    chrom_nums_model = model_name.split('chr')[1]
    chrom_nums_data = data_name.split('chromosome_')[1]
    
    if chrom_nums_model != chrom_nums_data:
        raise ValueError(
            f"Model and data chromosomes don't match!\n"
            f"Model was trained on chromosomes {chrom_nums_model}\n"
            f"Data is from chromosomes {chrom_nums_data}"
        )
    
    # Load sample data
    print(f"Loading sample data from {args.data}")
    sample_data = load_sample_data(args.data, args.sample_idx)
    
    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(args.model, input_dim=len(sample_data))
    
    # Make prediction
    print("\nMaking prediction...")
    result = predict_sample(model, sample_data)
    
    # Print results
    print(f"\nPredicted super-population: {result['predicted_population']}")
    print("\nProbabilities:")
    for pop, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"{pop}: {prob:.4f}")

if __name__ == "__main__":
    main()