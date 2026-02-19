# Quick Usage Guide


```bash
# 1. Setup
./setup.sh

# 2. View the main demonstration
source venv/bin/activate
jupyter notebook notebooks/VAE_Inverse_Design_Demo.ipynb
```

The Jupyter notebook provides a complete walkthrough with:
- Data generation and visualization
- VAE training and evaluation
- Latent space analysis
- Bayesian optimization for inverse design
- Results and discovered materials

### Project Structure

```
├── src/                    # Core implementation
│   ├── vae_model.py        # VAE architecture
│   ├── bayesian_optimization.py
│   ├── data_utils.py
│   └── train.py
├── notebooks/              # Interactive demonstration
│   └── VAE_Inverse_Design_Demo.ipynb  ← START HERE
├── data/                   # Sample datasets
├── requirements.txt        # Dependencies
└── README.md               # Full documentation
```

### Expected Runtime

- Setup: ~2 minutes
- Training (CPU): ~5 minutes (150 epochs)
- Training (GPU): ~1 minute
- Full notebook: ~10 minutes

### Technical Requirements

- Python 3.8+
- 4GB RAM minimum
- CPU sufficient (GPU optional)

### Contact

**Dr. Alaukik Saxena**  
ML Scientist @ Computomics GmbH  
Former Postdoc @ Max Planck Institute for Sustainable Materials
