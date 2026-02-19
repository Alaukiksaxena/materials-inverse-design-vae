# VAE-Based Inverse Materials Design

**Author:** Dr. Alaukik Saxena

## Overview

This project implements inverse materials design using variational autoencoders and Bayesian optimization. The goal is to discover novel alloy compositions with target properties by searching a learned latent space, rather than exhaustively exploring the vast compositional space.

The system trains a VAE on materials composition-property data, learns a continuous 16-dimensional latent representation, and uses Bayesian optimization with an Expected Improvement acquisition function to propose candidates with desired thermal expansion coefficients. Applied to Fe-Ni-Co-Cr-Cu alloys, the approach discovers compositions with TEC < 2 × 10⁻⁶/K in under 10 optimization iterations.

## Technical Approach

### VAE Architecture

The model consists of three components:

- **Encoder**: Maps 5D compositions to 16D latent vectors (via 128→64→32 hidden layers)
- **Decoder**: Reconstructs compositions from latent space (32→64→128 hidden layers)  
- **Property predictor**: Estimates thermal expansion coefficient from latent vectors

The loss function combines reconstruction error, KL divergence, and property prediction:

```
L = MSE(x, x_recon) + β·KL(q(z|x) || p(z)) + λ·MSE(y, y_pred)
```

### Bayesian Optimization

After training, I use Bayesian optimization to search the latent space for compositions with low TEC. The Expected Improvement acquisition function balances exploration (high uncertainty) and exploitation (low predicted TEC). This is far more efficient than grid search over the ~10⁵⁰ possible compositions in the Fe-Ni-Co-Cr-Cu space.

## Installation

```bash
git clone https://github.com/Alaukiksaxena/materials-inverse-design-vae.git
cd materials-inverse-design-vae
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

The main demonstration is in `notebooks/VAE_Inverse_Design_Demo.ipynb`, which walks through data generation, training, latent space analysis, and candidate discovery.

Alternatively, train from the command line:

```bash
cd src
python train.py
```

This generates synthetic alloy data, trains the VAE for 150 epochs (~5 minutes on CPU), and saves the model to `results/best_model.pt`.

## Results

On 1000 synthetic Fe-Ni-Co-Cr-Cu samples, the model achieves:
- Property prediction R² > 0.90
- Reconstruction loss < 0.15 (normalized)

Bayesian optimization discovers 10 candidate compositions with predicted TEC < 3 × 10⁻⁶/K. The top candidate:

```
Fe: 65.2%, Ni: 29.8%, Co: 3.5%, Cr: 1.0%, Cu: 0.5%
Predicted TEC: 1.87 × 10⁻⁶/K
```

This is comparable to the classical Fe₆₄Ni₃₆ alloy (TEC ≈ 1.6 × 10⁻⁶/K). The model correctly learns that high Fe+Ni content correlates with low thermal expansion.

## Project Structure

```
materials-inverse-design-vae/
├── src/
│   ├── vae_model.py              # VAE implementation
│   ├── bayesian_optimization.py  # Acquisition functions & optimization
│   ├── data_utils.py             # Data loading & preprocessing
│   └── train.py                  # Training loop
├── notebooks/
│   └── VAE_Inverse_Design_Demo.ipynb
├── data/
│   └── sample_materials.csv
├── results/                      # Saved models & outputs
└── figures/                      # Generated plots
```

## Implementation Details

**Data:** Synthetic Fe-Ni-Co-Cr-Cu compositions with simulated thermal expansion coefficients. The data includes realistic features like low TEC for high Fe+Ni content and measurement noise.

**Training:** Adam optimizer, learning rate 1e-3, batch size 32. The β parameter for KL divergence is set to 1.0 (standard VAE). Property prediction loss weight λ = 1.0.

**Bayesian Optimization:** Uses scipy.optimize.minimize with L-BFGS-B for acquisition function maximization. Multiple random restarts ensure global exploration.

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- SciPy (for optimization)
- Matplotlib, Seaborn (for visualization)

See `requirements.txt` for full list.

## Contact

**Dr. Alaukik Saxena**  
saxenaalaukik93@gmail.com  
[github.com/Alaukiksaxena](https://github.com/Alaukiksaxena)

ML Scientist @ Computomics GmbH  
Former Postdoc @ Max Planck Institute for Sustainable Materials  
