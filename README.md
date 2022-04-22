# sdr

sdr is a Python library for performing sufficient dimension reduction (SDR) on Gaussian mixture models. The paper explaining this novel method can be found [here](https://drive.google.com/file/d/1SZuspTafYE4jrQEYni4sw1y_1JdNhec7/view?usp=sharing).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install sdr.

```bash
pip install -U git+https://github.com/k-wib/sdr.git
```

## Usage example

```python
from sdr import *
import numpy as np
from sklearn import datasets

# Create dataset
X, y = datasets.make_friedman1(n_samples = 5000, n_features = 15, noise = 0)
X = X - np.mean(X, axis = 0)
Z = np.hstack((X, y.reshape(-1, 1)))

# Fit a GMM
gmm_fit = NormalMixFit(num_comps = [2,5,10,15], covariance_type = 'full', criterion = 'aic', random_state = 42)
gmm_fit.get_gmm(X, y)

# CATOC (Cayley transformation method with a fixed step size)
B = 2 * 10**5
p = 5
sdr_model_a = SDR(n = B, n_lb = B, p = p, eta = 10, epochs = 500, stoc = True, algo = 'cayley', early_stopping = 50)
sdr_model_a.fit(gmm_fit.fitted_gmm)
sdr_model_a.plot()

# CATOCA (Cayley transformation method where the step size is determined by the Armijo's rule)
armijo = CayleyArmijo(rho_1 = 0.1, rho_2 = 0.9, max_iter = 10)
sdr_model_b = SDR(n = B, n_lb = B, p = p, eta = 10, epochs = 500, stoc = True, algo = armijo, early_stopping = 50)
sdr_model_b.fit(gmm_fit.fitted_gmm)
sdr_model_b.plot()

# CATTON (Natural gradient method with a fixed step size, where the reduced dimension is determined adaptively)
sdr_model_c = DynamicSDR(n = B, n_lb = B, eta = 10, epochs = 500, stoc = True, algo = 'natgd', early_stopping = 50, loss_ratio = 0.5)
sdr_model_c.fit(gmm_fit.fitted_gmm)
sdr_model_c.plot()
```

## Reproducing the results

- The file [Sim_Gauss.Rmd](https://github.com/k-wib/sdr/blob/main/Sim_Gauss.Rmd) contains the code used to obtain Figures 1 and 3 in the paper.
- The file [Experiments.ipynb](https://github.com/k-wib/sdr/blob/main/Experiments.ipynb) contains the code used to obtain Figure 2 of Table 1 in the paper. In order to run this file, you should upload the notebook into Google Colab, and add [this folder](https://drive.google.com/drive/folders/11wCNmMTVS_sWRiNUoMz3a7Qw8mS-pZQ8?usp=sharing) into My Drive.
