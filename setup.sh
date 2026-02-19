#!/bin/bash

# Quick setup and test script for Materials VAE Inverse Design
# Author: Dr. Alaukik Saxena

echo "========================================="
echo "Materials VAE Inverse Design - Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "Found: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate
echo "Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed"
echo ""

# Run a quick test
echo "========================================="
echo "Running quick test..."
echo "========================================="
cd src
python -c "
import torch
import numpy as np
from vae_model import MaterialsVAE
from data_utils import generate_synthetic_materials_data

print('Testing VAE model...')
model = MaterialsVAE(input_dim=5, latent_dim=16)
print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

print('\\nTesting data generation...')
compositions, properties, elements = generate_synthetic_materials_data(n_samples=100)
print(f'Generated {len(compositions)} samples')
print(f'Elements: {elements}')

print('\\nTesting forward pass...')
x = torch.FloatTensor(compositions)
x_recon, props, mu, log_var, z = model(x)
print(f'Forward pass successful')
print(f'Latent space shape: {z.shape}')

print('\\n' + '='*50)
print('ALL TESTS PASSED!')
print('='*50)
print('\\nReady to use! Try:')
print('  1. jupyter notebook notebooks/VAE_Inverse_Design_Demo.ipynb')
print('  2. python train.py')
"

cd ..

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the main demonstration notebook:"
echo "     jupyter notebook notebooks/VAE_Inverse_Design_Demo.ipynb"
echo ""
echo "  3. Or train from command line:"
echo "     cd src && python train.py"
echo ""
echo "========================================="
