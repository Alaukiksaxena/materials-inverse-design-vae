"""
Bayesian Optimization for Inverse Materials Design

This module implements Bayesian optimization to search the VAE latent space
for materials compositions with target properties.

Author: Dr. Alaukik Saxena
"""

import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Callable, Tuple, Optional


class BayesianOptimizer:
    """
    Bayesian Optimization for property-targeted inverse design in VAE latent space.
    
    Uses Gaussian Process surrogate model with acquisition functions to guide
    the search for materials with desired properties.
    """
    
    def __init__(
        self,
        vae_model: torch.nn.Module,
        latent_dim: int,
        acquisition_function: str = 'ei',
        xi: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Args:
            vae_model: Trained VAE model
            latent_dim: Dimensionality of latent space
            acquisition_function: 'ei' (Expected Improvement) or 'ucb' (Upper Confidence Bound)
            xi: Exploration-exploitation trade-off parameter
            device: 'cpu' or 'cuda'
        """
        self.vae_model = vae_model
        self.vae_model.eval()
        self.latent_dim = latent_dim
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.device = device
        
        # History of explored points
        self.Z_history = []
        self.y_history = []
    
    def predict_property(self, z: np.ndarray) -> Tuple[float, float]:
        """
        Predict property value and uncertainty for a latent point.
        
        Args:
            z: Latent vector [latent_dim]
            
        Returns:
            mean: Predicted property value
            std: Prediction uncertainty
        """
        z_tensor = torch.FloatTensor(z).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get property prediction
            prop_pred = self.vae_model.predict_property(z_tensor)
            
            # Simple uncertainty estimation using dropout at inference
            self.vae_model.train()
            n_samples = 20
            predictions = []
            for _ in range(n_samples):
                pred = self.vae_model.predict_property(z_tensor)
                predictions.append(pred.cpu().numpy())
            self.vae_model.eval()
            
            predictions = np.array(predictions)
            mean = predictions.mean()
            std = predictions.std()
        
        return mean, std
    
    def expected_improvement(
        self, 
        z: np.ndarray, 
        y_best: float
    ) -> float:
        """
        Expected Improvement acquisition function.
        
        Args:
            z: Latent point to evaluate
            y_best: Best property value observed so far
            
        Returns:
            Expected improvement value (negative for minimization)
        """
        mu, sigma = self.predict_property(z)
        
        if sigma == 0:
            return 0
        
        # For minimization (lower property values are better)
        improvement = y_best - mu - self.xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return -ei  # Negative for minimization
    
    def upper_confidence_bound(
        self, 
        z: np.ndarray, 
        kappa: float = 2.0
    ) -> float:
        """
        Upper Confidence Bound acquisition function.
        
        Args:
            z: Latent point to evaluate
            kappa: Exploration parameter
            
        Returns:
            UCB value (negative for minimization)
        """
        mu, sigma = self.predict_property(z)
        
        # For minimization (lower values are better)
        ucb = mu - kappa * sigma
        
        return ucb
    
    def optimize_acquisition(
        self, 
        y_best: float, 
        n_restarts: int = 10
    ) -> np.ndarray:
        """
        Optimize the acquisition function to find next point to evaluate.
        
        Args:
            y_best: Best property value observed so far
            n_restarts: Number of random restarts for optimization
            
        Returns:
            z_next: Next latent point to evaluate
        """
        best_z = None
        best_acq = float('inf')
        
        for _ in range(n_restarts):
            # Random starting point in latent space
            z0 = np.random.randn(self.latent_dim)
            
            # Define objective based on acquisition function
            if self.acquisition_function == 'ei':
                objective = lambda z: self.expected_improvement(z, y_best)
            else:  # ucb
                objective = lambda z: self.upper_confidence_bound(z)
            
            # Optimize
            result = minimize(
                objective,
                z0,
                method='L-BFGS-B',
                bounds=[(-3, 3)] * self.latent_dim
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_z = result.x
        
        return best_z
    
    def propose_candidates(
        self, 
        n_candidates: int = 5,
        target_property: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propose new candidate materials compositions.
        
        Args:
            n_candidates: Number of candidates to propose
            target_property: Target property value (if None, minimize)
            
        Returns:
            compositions: Proposed compositions [n_candidates, n_elements]
            properties: Predicted properties [n_candidates]
        """
        # Determine best value seen so far
        if len(self.y_history) > 0:
            y_best = min(self.y_history)
        else:
            y_best = float('inf') if target_property is None else target_property
        
        candidates_z = []
        candidates_prop = []
        
        for _ in range(n_candidates):
            # Find next point via acquisition function
            z_next = self.optimize_acquisition(y_best, n_restarts=20)
            
            # Predict property
            mu, std = self.predict_property(z_next)
            
            candidates_z.append(z_next)
            candidates_prop.append(mu)
            
            # Update history (simulate evaluation)
            self.Z_history.append(z_next)
            self.y_history.append(mu)
            
            # Update best
            if mu < y_best:
                y_best = mu
        
        # Decode latent points to compositions
        z_tensor = torch.FloatTensor(np.array(candidates_z)).to(self.device)
        
        with torch.no_grad():
            compositions = self.vae_model.decode(z_tensor).cpu().numpy()
        
        # Normalize compositions to sum to 1
        compositions = np.abs(compositions)
        compositions = compositions / compositions.sum(axis=1, keepdims=True)
        
        return compositions, np.array(candidates_prop)
    
    def grid_search_latent(
        self, 
        n_points_per_dim: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform grid search in latent space to visualize property landscape.
        
        Args:
            n_points_per_dim: Number of points per dimension
            
        Returns:
            Z: Grid of latent points
            properties_mean: Predicted property values
            properties_std: Prediction uncertainties
        """
        # Create grid for first two latent dimensions
        z1 = np.linspace(-3, 3, n_points_per_dim)
        z2 = np.linspace(-3, 3, n_points_per_dim)
        Z1, Z2 = np.meshgrid(z1, z2)
        
        # Initialize other dimensions to zero
        Z_grid = np.zeros((n_points_per_dim * n_points_per_dim, self.latent_dim))
        Z_grid[:, 0] = Z1.ravel()
        Z_grid[:, 1] = Z2.ravel()
        
        # Predict properties for all grid points
        properties_mean = []
        properties_std = []
        
        for z in Z_grid:
            mu, sigma = self.predict_property(z)
            properties_mean.append(mu)
            properties_std.append(sigma)
        
        properties_mean = np.array(properties_mean).reshape(n_points_per_dim, n_points_per_dim)
        properties_std = np.array(properties_std).reshape(n_points_per_dim, n_points_per_dim)
        
        return (Z1, Z2), properties_mean, properties_std
