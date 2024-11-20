import random
import numpy as np
from numpy.random import normal as noise
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as normal
from scipy.stats import norm
#import statsmodels.api as sm # OLS

import linear_model as lm
from linear_model import BayesianLinearModel

# 1.1
gen = np.random.default_rng(seed=42)
N = 5000

def simular(do_x=None):
    Z1s = gen.uniform(-3,3, size=N)
    if do_x is None:
        X1s = 1 + 3*Z1s + 2*Z1s**3 + gen.normal(size=N,scale=6)
    else:
        X1s = np.full(N, do_x)

    Y1s = -1 - 2*X1s + 6*Z1s**2 + gen.normal(size=N,scale=1)
    return Z1s, X1s, Y1s

def Y_doX(y, x, blm, dz=0.01):
  res = 0
  for z in np.arange(-4,4,dz):
    pz = normal(0,1).pdf(z)
    _, _, py_xz = blm.predict(
      X=np.array([1,x,z**2]).reshape(1,3),
      y=np.array(y),
      variance=True
    )

    res += pz * py_xz * dz

  return res

Z1s, X1s, Y1s = simular()

PHI1 = np.concatenate([
        np.ones(N).reshape(N, 1), # c_0
        X1s.reshape(N, 1), # c_x X
        (Z1s**2).reshape(N, 1), # c_z Zˆ2
    ],
    axis=1
)

blm1= BayesianLinearModel(basis=lambda x: x)
blm1.update(PHI1, Y1s.reshape(N,1))
# Obtenemos las estimaciones
mean1 = blm1.location
cov1 = blm1.dispersion
ev1 = blm1.evidence()
print(f'Efecto causal de X sobre Y: {mean1[1]}')
print(f'Evidencia: {ev1}')