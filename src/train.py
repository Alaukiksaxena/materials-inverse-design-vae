"""
Training script for Materials VAE

Author: Dr. Alaukik Saxena
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from typing import Dict, List

from vae_model import MaterialsVAE
from data_utils import create_dataloaders, generate_synthetic_materials_data


def train_epoch(
    model: MaterialsVAE,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    property_weight: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary of average losses
    """
    model.train()
    train_losses = {
        'loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'property_loss': []
    }
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, y_pred, mu, log_var, z = model(x)
        
        # Compute loss
        losses = model.loss_function(x, x_recon, y, y_pred, mu, log_var, property_weight)
        
        # Backward pass
        losses['loss'].backward()
        optimizer.step()
        
        # Record losses
        for key in train_losses:
            train_losses[key].append(losses[key].item())
    
    # Average losses
    avg_losses = {key: np.mean(values) for key, values in train_losses.items()}
    
    return avg_losses


def validate(
    model: MaterialsVAE,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    property_weight: float = 1.0
) -> Dict[str, float]:
    """
    Validate model.
    
    Returns:
        Dictionary of average validation losses
    """
    model.eval()
    val_losses = {
        'loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'property_loss': []
    }
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            x_recon, y_pred, mu, log_var, z = model(x)
            
            # Compute loss
            losses = model.loss_function(x, x_recon, y, y_pred, mu, log_var, property_weight)
            
            # Record losses
            for key in val_losses:
                val_losses[key].append(losses[key].item())
    
    # Average losses
    avg_losses = {key: np.mean(values) for key, values in val_losses.items()}
    
    return avg_losses


def train_vae(
    model: MaterialsVAE,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    property_weight: float = 1.0,
    device: str = 'cpu',
    save_dir: str = '../results'
) -> Dict[str, List[float]]:
    """
    Full training loop for VAE.
    
    Returns:
        Dictionary of training history
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'train_property_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_property_loss': []
    }
    
    best_val_loss = float('inf')
    
    print(f"Training VAE on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(n_epochs):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, property_weight)
        
        # Validate
        val_losses = validate(model, val_loader, device, property_weight)
        
        # Record history
        history['train_loss'].append(train_losses['loss'])
        history['train_recon_loss'].append(train_losses['recon_loss'])
        history['train_kl_loss'].append(train_losses['kl_loss'])
        history['train_property_loss'].append(train_losses['property_loss'])
        history['val_loss'].append(val_losses['loss'])
        history['val_recon_loss'].append(val_losses['recon_loss'])
        history['val_kl_loss'].append(val_losses['kl_loss'])
        history['val_property_loss'].append(val_losses['property_loss'])
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {train_losses['loss']:.4f} | Val Loss: {val_losses['loss']:.4f}")
            print(f"  Train Property Loss: {train_losses['property_loss']:.4f} | Val Property Loss: {val_losses['property_loss']:.4f}")
        
        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['loss'],
                'history': history
            }, os.path.join(save_dir, 'best_model.pt'))
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return history


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Validation', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(history['train_recon_loss'], label='Train', alpha=0.8)
    axes[0, 1].plot(history['val_recon_loss'], label='Validation', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL loss
    axes[1, 0].plot(history['train_kl_loss'], label='Train', alpha=0.8)
    axes[1, 0].plot(history['val_kl_loss'], label='Validation', alpha=0.8)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Property loss
    axes[1, 1].plot(history['train_property_loss'], label='Train', alpha=0.8)
    axes[1, 1].plot(history['val_property_loss'], label='Validation', alpha=0.8)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Property Prediction Loss')
    axes[1, 1].set_title('Property Prediction Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Configuration
    config = {
        'n_samples': 1000,
        'n_elements': 5,
        'latent_dim': 16,
        'hidden_dims': [128, 64, 32],
        'batch_size': 32,
        'n_epochs': 150,
        'learning_rate': 1e-3,
        'beta': 1.0,
        'property_weight': 1.0,
        'train_split': 0.8
    }
    
    # Set device
    # Device — supports CUDA (NVIDIA), MPS (Apple Silicon), or CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")



    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("\nGenerating synthetic materials data...")
    compositions, properties, element_names = generate_synthetic_materials_data(
        n_samples=config['n_samples'],
        n_elements=config['n_elements']
    )
    
    print(f"Generated {len(compositions)} samples")
    print(f"Elements: {element_names}")
    print(f"Property range: [{properties.min():.2f}, {properties.max():.2f}]")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        compositions,
        properties,
        batch_size=config['batch_size'],
        train_split=config['train_split']
    )
    
    # Initialize model
    model = MaterialsVAE(
        input_dim=config['n_elements'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        property_dim=1,
        beta=config['beta']
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Train model
    print("\nStarting training...")
    history = train_vae(
        model,
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        property_weight=config['property_weight'],
        device=device
    )
    
    # Plot training history
    os.makedirs('../figures', exist_ok=True)
    plot_training_history(history, save_path='../figures/training_history.png')
    
    # Save config and history
    os.makedirs('../results', exist_ok=True)
    with open('../results/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open('../results/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed successfully!")
