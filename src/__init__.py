"""
Materials VAE Inverse Design Package

Author: Dr. Alaukik Saxena
"""

from .vae_model import MaterialsVAE
from .bayesian_optimization import BayesianOptimizer
from .data_utils import (
    MaterialsDataset,
    load_materials_data,
    create_dataloaders,
    generate_synthetic_materials_data,
    denormalize_data,
    save_materials_data
)

__all__ = [
    'MaterialsVAE',
    'BayesianOptimizer',
    'MaterialsDataset',
    'load_materials_data',
    'create_dataloaders',
    'generate_synthetic_materials_data',
    'denormalize_data',
    'save_materials_data'
]
