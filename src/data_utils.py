"""
Data utilities for materials dataset handling.

Author: Dr. Alaukik Saxena
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class MaterialsDataset(Dataset):
    """PyTorch Dataset for materials compositions and properties."""
    
    def __init__(
        self,
        compositions: np.ndarray,
        properties: np.ndarray,
        composition_scaler: Optional[StandardScaler] = None,
        property_scaler: Optional[StandardScaler] = None
    ):
        """
        Args:
            compositions: Array of material compositions [n_samples, n_elements]
            properties: Array of material properties [n_samples, n_properties]
            composition_scaler: Optional pre-fitted scaler for compositions
            property_scaler: Optional pre-fitted scaler for properties
        """
        self.compositions = torch.FloatTensor(compositions)
        self.properties = torch.FloatTensor(properties)
        
        self.composition_scaler = composition_scaler
        self.property_scaler = property_scaler
    
    def __len__(self) -> int:
        return len(self.compositions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compositions[idx], self.properties[idx]


def load_materials_data(
    filepath: str,
    composition_cols: list,
    property_col: str,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Load materials data from CSV file.
    
    Args:
        filepath: Path to CSV file
        composition_cols: List of column names for composition elements
        property_col: Column name for target property
        normalize: Whether to normalize the data
        
    Returns:
        compositions: Composition array
        properties: Property array
        composition_scaler: Fitted scaler for compositions (if normalize=True)
        property_scaler: Fitted scaler for properties (if normalize=True)
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Extract compositions and properties
    compositions = df[composition_cols].values
    properties = df[[property_col]].values
    
    composition_scaler = None
    property_scaler = None
    
    if normalize:
        # Normalize compositions
        composition_scaler = StandardScaler()
        compositions = composition_scaler.fit_transform(compositions)
        
        # Normalize properties
        property_scaler = StandardScaler()
        properties = property_scaler.fit_transform(properties)
    
    return compositions, properties, composition_scaler, property_scaler


def create_dataloaders(
    compositions: np.ndarray,
    properties: np.ndarray,
    batch_size: int = 32,
    train_split: float = 0.8,
    composition_scaler: Optional[StandardScaler] = None,
    property_scaler: Optional[StandardScaler] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        compositions: Composition array
        properties: Property array
        batch_size: Batch size for DataLoader
        train_split: Fraction of data for training
        composition_scaler: Fitted composition scaler
        property_scaler: Fitted property scaler
        
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
    """
    # Split data
    n_train = int(len(compositions) * train_split)
    indices = np.random.permutation(len(compositions))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_dataset = MaterialsDataset(
        compositions[train_indices],
        properties[train_indices],
        composition_scaler,
        property_scaler
    )
    
    val_dataset = MaterialsDataset(
        compositions[val_indices],
        properties[val_indices],
        composition_scaler,
        property_scaler
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


def generate_synthetic_materials_data(
    n_samples: int = 1000,
    n_elements: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generate synthetic materials data for demonstration.
    
    Simulates alloy compositions (Fe, Ni, Co, Cr, Cu) and a target property
    (e.g., thermal expansion coefficient) with realistic correlations.
    
    Args:
        n_samples: Number of samples to generate
        n_elements: Number of elements in composition
        seed: Random seed
        
    Returns:
        compositions: Generated compositions [n_samples, n_elements]
        properties: Generated properties [n_samples, 1]
        element_names: Names of elements
    """
    np.random.seed(seed)
    
    element_names = ['Fe', 'Ni', 'Co', 'Cr', 'Cu'][:n_elements]
    
    # Generate random compositions that sum to 1
    compositions = np.random.dirichlet(np.ones(n_elements), size=n_samples)
    
    # Generate target property with nonlinear dependence on composition
    # Simulate low TEC for specific Fe-Ni ratios
    properties = []
    
    for comp in compositions:
        # Base property
        base_prop = 12.0  # Base thermal expansion coefficient
        
        # Invar effect for high Fe+Ni content
        fe_ni_content = comp[0] + comp[1]  # Fe + Ni
        if fe_ni_content > 0.7:
            invar_effect = -10.0 * (fe_ni_content - 0.7)
        else:
            invar_effect = 0
        
        # Co contribution (slightly lowers TEC)
        co_effect = -2.0 * comp[2] if n_elements > 2 else 0
        
        # Cr contribution (raises TEC slightly)
        cr_effect = 1.5 * comp[3] if n_elements > 3 else 0
        
        # Cu contribution (raises TEC)
        cu_effect = 3.0 * comp[4] if n_elements > 4 else 0
        
        # Add some noise
        noise = np.random.normal(0, 0.5)
        
        prop = base_prop + invar_effect + co_effect + cr_effect + cu_effect + noise
        prop = max(1.0, prop)  # Ensure positive TEC
        
        properties.append(prop)
    
    properties = np.array(properties).reshape(-1, 1)
    
    return compositions, properties, element_names


def denormalize_data(
    data: np.ndarray,
    scaler: StandardScaler
) -> np.ndarray:
    """
    Denormalize data using a fitted scaler.
    
    Args:
        data: Normalized data
        scaler: Fitted StandardScaler
        
    Returns:
        Denormalized data
    """
    if scaler is None:
        return data
    return scaler.inverse_transform(data)


def save_materials_data(
    filepath: str,
    compositions: np.ndarray,
    properties: np.ndarray,
    element_names: list,
    property_name: str = 'TEC'
):
    """
    Save materials data to CSV file.
    
    Args:
        filepath: Output CSV filepath
        compositions: Composition array
        properties: Property array
        element_names: Names of composition elements
        property_name: Name of target property
    """
    # Create DataFrame
    data = {}
    for i, elem in enumerate(element_names):
        data[elem] = compositions[:, i]
    data[property_name] = properties.ravel()
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
