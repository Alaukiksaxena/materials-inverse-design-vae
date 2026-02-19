"""
Variational Autoencoder for Materials Inverse Design

This module implements a VAE that learns a latent representation of materials
compositions and can generate new candidates based on target properties.

Author: Dr. Alaukik Saxena
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class MaterialsVAE(nn.Module):
    """
    Variational Autoencoder for materials composition and property learning.
    
    Architecture:
    - Encoder: Maps composition features to latent distribution (mean, log_var)
    - Decoder: Reconstructs composition from latent representation  
    - Property predictor: Predicts target properties from latent space
    
    Args:
        input_dim: Dimensionality of input composition features
        latent_dim: Dimensionality of latent space
        hidden_dims: List of hidden layer dimensions
        property_dim: Number of properties to predict
        beta: Weight for KL divergence term (beta-VAE)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: list = None,
        property_dim: int = 1,
        beta: float = 1.0
    ):
        super(MaterialsVAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.property_dim = property_dim
        self.beta = beta
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Property predictor from latent space
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, property_dim)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to composition."""
        return self.decoder(z)
    
    def predict_property(self, z: torch.Tensor) -> torch.Tensor:
        """Predict properties from latent representation."""
        return self.property_predictor(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        properties = self.predict_property(z)
        return x_recon, properties, mu, log_var, z
    
    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        property_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss with property prediction."""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Property prediction loss (MSE)
        property_loss = F.mse_loss(y_pred, y_true, reduction='mean')
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + property_weight * property_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'property_loss': property_loss
        }
    
    def generate(self, z: torch.Tensor = None, n_samples: int = 1) -> torch.Tensor:
        """Generate new compositions from latent space."""
        if z is None:
            z = torch.randn(n_samples, self.latent_dim)
        
        with torch.no_grad():
            samples = self.decode(z)
        
        return samples
