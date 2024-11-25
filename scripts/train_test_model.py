"""
Train and test a neural network model for super-population prediction using genetic data.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

SUPER_POPULATIONS = {
   'EUR': 0,  # European
   'EAS': 1,  # East Asian
   'AFR': 2,  # African
   'SAS': 3,  # South Asian
   'AMR': 4   # American
}

class GenomesDataset(Dataset):
   """Dataset class for genetic data."""
   
   def __init__(self, matrix, labels):
       self.matrix = torch.FloatTensor(matrix)
       self.labels = torch.LongTensor(labels)
   
   def __len__(self):
       return len(self.labels)
   
   def __getitem__(self, idx):
       return self.matrix[idx], self.labels[idx]

class PopulationPredictor(nn.Module):
   """Neural network model for super-population prediction."""
   
   def __init__(self, input_dim, hidden_dim=256, n_heads=4, dropout=0.3):
       super().__init__()
       
       self.dense1 = nn.Sequential(
           nn.Linear(input_dim, hidden_dim),
           nn.ReLU(),
           nn.Dropout(dropout)
       )
       
       self.attention = nn.MultiheadAttention(
           embed_dim=hidden_dim,
           num_heads=n_heads,
           dropout=dropout,
           batch_first=True
       )
       
       self.layer_norm = nn.LayerNorm(hidden_dim)
       
       self.classifier = nn.Sequential(
           nn.Linear(hidden_dim, hidden_dim//2),
           nn.ReLU(),
           nn.Dropout(dropout),
           nn.Linear(hidden_dim//2, len(SUPER_POPULATIONS))
       )
   
   def forward(self, x):
       x = self.dense1(x)
       x = x.unsqueeze(1)  # Add sequence dimension
       x, _ = self.attention(x, x, x)
       x = x.squeeze(1)
       x = self.layer_norm(x)
       return self.classifier(x)

class PlotManager:
   def __init__(self, save_dir: Path):
       plt.ion()  # Enable interactive mode
       self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15,5))
       self.save_dir = save_dir
       self.history = {
           'train_loss': [], 'val_loss': [],
           'train_acc': [], 'val_acc': []
       }
       
   def update(self, metrics):
       for k, v in metrics.items():
           self.history[k].append(v)
           
       self.ax1.clear()
       self.ax2.clear()
       
       # Plot losses
       self.ax1.plot(self.history['train_loss'], label='Train Loss')
       self.ax1.plot(self.history['val_loss'], label='Val Loss')
       self.ax1.set_title('Training and Validation Loss')
       self.ax1.set_xlabel('Epoch')
       self.ax1.set_ylabel('Loss')
       self.ax1.legend()
       
       # Plot accuracies
       self.ax2.plot(self.history['train_acc'], label='Train Acc')
       self.ax2.plot(self.history['val_acc'], label='Val Acc')
       self.ax2.set_title('Training and Validation Accuracy')
       self.ax2.set_xlabel('Epoch')
       self.ax2.set_ylabel('Accuracy (%)')
       self.ax2.legend()
       
       plt.tight_layout()
       self.fig.canvas.draw()
       self.fig.canvas.flush_events()
   
   def save(self, chrom_string):
       plt.savefig(self.save_dir / f'training_history_chr{chrom_string}.png')
       plt.close()

def load_and_validate_data(chromosomes):
   """Load and validate data for training."""
   
   preprocessed_dir = Path("data/preprocessed")
   chrom_string = "-".join(str(c) for c in sorted(chromosomes))
   data_path = preprocessed_dir / f"chromosome_{chrom_string}.npz"
   
   if not data_path.exists():
       raise FileNotFoundError(f"Preprocessed data not found: {data_path}")
   
   data = np.load(data_path)
   matrix = data['matrix']
   
   panel_path = Path("data/population_labels.panel")
   panel = pd.read_csv(panel_path, sep='\t')
   
   if len(panel["sample"]) != panel["sample"].nunique():
       raise ValueError("Duplicate sample IDs found in panel file")
   
   if panel['super_pop'].isna().any():
       raise ValueError("Missing super-population labels found")
       
   if not all(pop in SUPER_POPULATIONS for pop in panel['super_pop'].unique()):
       raise ValueError("Unknown super-population labels found")
       
   if len(matrix) != len(panel):
       raise ValueError(f"Matrix rows ({len(matrix)}) don't match number of individuals ({len(panel)})")
   
   # Convert super population codes to numeric labels
   labels = np.array([SUPER_POPULATIONS[p] for p in panel['super_pop']])
   
   return matrix, labels, panel

def prepare_dataloaders(matrix, labels, panel, batch_size=32):
   """Prepare train/validation/test dataloaders with stratification."""
   
   # First split: 80% train, 20% temp
   train_indices, temp_indices = train_test_split(
       range(len(labels)),
       test_size=0.2,
       stratify=labels,
       random_state=42
   )
   
   # Second split: 50% validation, 50% test from temp
   val_indices, test_indices = train_test_split(
       temp_indices,
       test_size=0.5,
       stratify=labels[temp_indices],
       random_state=42
   )
   
   # Verify split sizes
   expected_train_size = int(0.8 * len(matrix))
   expected_val_test_size = int(0.1 * len(matrix))
   
   assert abs(len(train_indices) - expected_train_size) <= 1, "Unexpected train set size"
   assert abs(len(val_indices) - expected_val_test_size) <= 1, "Unexpected validation set size"
   assert abs(len(test_indices) - expected_val_test_size) <= 1, "Unexpected test set size"
   
   # Verify no overlap between splits
   assert len(set(train_indices) & set(val_indices)) == 0, "Overlap between train and validation sets"
   assert len(set(train_indices) & set(test_indices)) == 0, "Overlap between train and test sets"
   assert len(set(val_indices) & set(test_indices)) == 0, "Overlap between validation and test sets"
   
   train_dataset = GenomesDataset(matrix[train_indices], labels[train_indices])
   val_dataset = GenomesDataset(matrix[val_indices], labels[val_indices])
   test_dataset = GenomesDataset(matrix[test_indices], labels[test_indices])
   
   print("\nDistribution of super-populations in splits:")
   total_samples = len(panel)
   
   for split_name, indices in [("Train", train_indices), 
                             ("Validation", val_indices), 
                             ("Test", test_indices)]:
       print(f"\n{split_name} set distribution:")
       print(f"{'Super-pop':<8} {'Count':>8} {'Percentage':>12}")
       print("-" * 32)
       
       split_labels = labels[indices]
       unique, counts = np.unique(split_labels, return_counts=True)
       dist = dict(zip(unique, counts))
       
       split_total = len(indices)
       for pop_code, count in sorted(dist.items()):
           pop_name = [k for k, v in SUPER_POPULATIONS.items() if v == pop_code][0]
           percentage = (count / split_total) * 100
           print(f"{pop_name:<8} {count:>8} {percentage:>11.1f}%")
       
       print(f"\nTotal {split_name.lower()} samples: {split_total}")
       print(f"Percentage of total: {(split_total/total_samples)*100:.1f}%")
   
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size)
   test_loader = DataLoader(test_dataset, batch_size=batch_size)
   
   return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, plot_manager, epochs=10):
   """Train the model and return best state dict."""
   
   criterion = nn.CrossEntropyLoss()
   optimizer = Adam(model.parameters(), lr=0.001)
   scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
   
   best_val_acc = 0
   best_model = None
   
   for epoch in range(epochs):
       model.train()
       train_loss = 0
       train_correct = 0
       train_total = 0
       
       pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
       for batch_x, batch_y in pbar:
           batch_x, batch_y = batch_x.to(device), batch_y.to(device)
           
           optimizer.zero_grad()
           output = model(batch_x)
           loss = criterion(output, batch_y)
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
           _, predicted = output.max(1)
           train_total += batch_y.size(0)
           train_correct += predicted.eq(batch_y).sum().item()
           
           pbar.set_postfix({'loss': f'{loss.item():.4f}'})
       
       train_loss = train_loss / len(train_loader)
       train_acc = 100. * train_correct / train_total
       
       model.eval()
       val_loss = 0
       val_correct = 0
       val_total = 0
       
       with torch.no_grad():
           for batch_x, batch_y in val_loader:
               batch_x, batch_y = batch_x.to(device), batch_y.to(device)
               
               output = model(batch_x)
               loss = criterion(output, batch_y)
               val_loss += loss.item()
               
               _, predicted = output.max(1)
               val_total += batch_y.size(0)
               val_correct += predicted.eq(batch_y).sum().item()
       
       val_loss = val_loss / len(val_loader)
       val_acc = 100. * val_correct / val_total
       
       scheduler.step(val_acc)
       
       plot_manager.update({
           'train_loss': train_loss,
           'train_acc': train_acc,
           'val_loss': val_loss,
           'val_acc': val_acc
       })
       
       print(f"\nEpoch {epoch+1}/{epochs}")
       print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
       print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
       print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
       
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           best_model = model.state_dict()
           print(f"New best validation accuracy: {val_acc:.2f}%")
   
   return best_model

def test_model(model, test_loader, device):
   """Test the model and print results."""
   
   model.eval()
   correct = 0
   total = 0
   predictions = []
   true_labels = []
   
   with torch.no_grad():
       for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
           batch_x, batch_y = batch_x.to(device), batch_y.to(device)
           
           outputs = model(batch_x)
           _, predicted = outputs.max(1)
           
           total += batch_y.size(0)
           correct += predicted.eq(batch_y).sum().item()
           
           predictions.extend(predicted.cpu().numpy())
           true_labels.extend(batch_y.cpu().numpy())
   
   accuracy = 100. * correct / total
   print(f'\nOverall Test Accuracy: {accuracy:.2f}%')
   
   print("\nPer-population Test Accuracies:")
   print(f"{'Population':<8} {'Accuracy':>10}")
   print("-" * 20)
   
   for pop_name, pop_code in SUPER_POPULATIONS.items():
       mask = np.array(true_labels) == pop_code
       if mask.sum() > 0:
           pop_acc = 100. * np.sum(np.array(predictions)[mask] == pop_code) / mask.sum()
           print(f"{pop_name:<8} {pop_acc:>9.2f}%")
   
   return accuracy

def main():
   parser = argparse.ArgumentParser(description="Train and test super-population prediction model")
   parser.add_argument(
       "-c", 
       "--chromosomes", 
       type=int, 
       nargs="+", 
       default=[22],
       help="Chromosome numbers to use (default: [22])"
   )
   parser.add_argument(
       "-e", 
       "--epochs", 
       type=int, 
       default=10,
       help="Number of training epochs"
   )
   parser.add_argument(
       "--batch-size", 
       type=int, 
       default=32,
       help="Batch size for training"
   )
   args = parser.parse_args()
   
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   models_dir = Path("models")
   models_dir.mkdir(exist_ok=True)
   
   plot_manager = PlotManager(save_dir=models_dir)
   
   matrix, labels, panel = load_and_validate_data(args.chromosomes)
   train_loader, val_loader, test_loader = prepare_dataloaders(
       matrix, 
       labels,
       panel,
       batch_size=args.batch_size
   )
   
   input_dim = matrix.shape[1]
   model = PopulationPredictor(input_dim=input_dim).to(device)
   
   best_model = train_model(
       model,
       train_loader,
       val_loader,
       device,
       plot_manager,
       epochs=args.epochs
   )
   
   chrom_string = "-".join(str(c) for c in sorted(args.chromosomes))
   model_path = models_dir / f'model_chr{chrom_string}.pt'
   torch.save(best_model, model_path)
   plot_manager.save(chrom_string)
   
   model.load_state_dict(best_model)
   test_model(model, test_loader, device)

if __name__ == "__main__":
   main()