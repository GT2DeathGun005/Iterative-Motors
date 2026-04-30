"""
Modulo di Behavioral Cloning (Imitation Learning)
Addestra una Policy Network utilizzando le dimostrazioni umane raccolte in data_collection.py.
Ottimizzato con split di validazione, early stopping e salvataggio dei pesi per il warm start del SAC.
"""

import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class TorcsHDF5Dataset(Dataset):
    """Dataset personalizzato per leggere in modo efficiente i chunk HDF5."""
    def __init__(self, file_path: str):
        super().__init__()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File dataset non trovato: {file_path}")
            
        self.file_path = file_path
        # Apriamo in read-only. Usiamo in-memory loading per dataset piccoli (es. < 1-2GB)
        # per evitare colli di bottiglia I/O.
        with h5py.File(self.file_path, 'r') as h5f:
            self.states = torch.tensor(h5f['states'][:], dtype=torch.float32)
            self.actions = torch.tensor(h5f['actions'][:], dtype=torch.float32)
            
        self.length = self.states.shape[0]
        
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx]

class PolicyNetwork(nn.Module):
    """
    Rete Neurale Actor (Policy) che mappa lo stato nell'azione continua.
    Architettura feed-forward ottimizzata per state vectors continui.
    Output: [steering, accel, brake, gear]
    """
    def __init__(self, state_dim: int = 29, action_dim: int = 4, hidden_size: int = 256):
        super(PolicyNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Tanh comprime l'output tra -1 e 1
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calcola l'azione. NOTA: L'output di Tanh è in [-1, 1].
        L'azione reale richiede un mapping dipendente dall'asse (es: accel [0,1], steer [-1,1]).
        Nel behavior cloning mappiamo semplicemente sul target, ma il wrapper RL dovrà scalare l'azione.
        """
        return self.net(state)

class BehaviorCloningTrainer:
    def __init__(self, model: nn.Module, dataset: TorcsHDF5Dataset, 
                 batch_size: int = 128, val_split: float = 0.2, 
                 lr: float = 3e-4, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Validation split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for states, targets in self.train_loader:
            states, targets = states.to(self.device), targets.to(self.device)
            
            # Map targets to [-1, 1] for Tanh if they are [0, 1]
            # Assumiamo che il dataset abbia:
            # - steer: [-1, 1]
            # - accel: [0, 1] -> scalato a [-1, 1]
            # - brake: [0, 1] -> scalato a [-1, 1]
            # - gear: da normalizzare
            # Per semplicità, qui l'MSE agisce direttamente sui target come sono,
            # ma è meglio avere i target normalizzati in [-1, 1].
            # Lo facciamo qui al volo:
            targets_norm = targets.clone()
            targets_norm[:, 1] = targets[:, 1] * 2.0 - 1.0  # accel da [0,1] a [-1,1]
            targets_norm[:, 2] = targets[:, 2] * 2.0 - 1.0  # brake da [0,1] a [-1,1]
            # Normalizziamo gear (0..6) tra -1 e 1
            targets_norm[:, 3] = (targets[:, 3] / 3.0) - 1.0 
            
            self.optimizer.zero_grad()
            predictions = self.model(states)
            loss = self.criterion(predictions, targets_norm)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for states, targets in self.val_loader:
                states, targets = states.to(self.device), targets.to(self.device)
                
                targets_norm = targets.clone()
                targets_norm[:, 1] = targets[:, 1] * 2.0 - 1.0
                targets_norm[:, 2] = targets[:, 2] * 2.0 - 1.0
                targets_norm[:, 3] = (targets[:, 3] / 3.0) - 1.0 
                
                predictions = self.model(states)
                loss = self.criterion(predictions, targets_norm)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)

    def train(self, max_epochs: int = 200, patience: int = 15, checkpoint_path: str = "bc_policy.pth"):
        print(f"Inizio training Behavior Cloning su {self.device}...")
        
        for epoch in range(max_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1:03d}/{max_epochs} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  -> Model improved. Saved checkpoint to {checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping triggerato all'epoca {epoch+1}. Miglior Val Loss: {self.best_val_loss:.5f}")
                    break

def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning for TORCS agent")
    parser.add_argument("--dataset", type=str, default="human_expert.h5", help="Path to HDF5 dataset")
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="bc_policy.pth", help="Output model weights file")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Caricamento dataset {args.dataset}...")
    dataset = TorcsHDF5Dataset(args.dataset)
    print(f"Dataset caricato: {len(dataset)} campioni totali.")
    
    # State_dim e action_dim devono corrispondere ai dati salvati
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]
    print(f"Dimensioni rilevate - State: {state_dim}, Action: {action_dim}")
    
    model = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
    
    trainer = BehaviorCloningTrainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
    
    trainer.train(max_epochs=args.epochs, checkpoint_path=args.output)
    print("Addestramento Behavior Cloning completato.")

if __name__ == "__main__":
    main()
