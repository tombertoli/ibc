import numpy as np
from scipy.stats import multivariate_normal as normal
from linear_model import BayesianLinearModel
import matplotlib.pyplot as plt

gen = np.random.default_rng(seed=54)
N=10000

def simular(do_x=None):
  Z = gen.normal(size=N,scale=1)
  if do_x is None:
    X = Z**2 + gen.normal(size=N, scale=1)
  else:
    X = np.full(N,do_x)

  M = 2 * Z**2 + 10*X + gen.normal(size=N, scale=1)
  Y = -1 + 2 * M**2 + gen.normal(size=N, scale=1)
  return Z, X, M, Y

def Y_doX(y, x, blm, dz=0.01):
  res = 0
  for z in np.arange(-4,4,dz):
    pz = normal(0,1).pdf(z)
    _, _, py_xz = blm.predict(
      X=np.array([1,x**2,x*z**2, z**4]).reshape(1,4),
      y=np.array(y),
      variance=True
    )

    res += pz * py_xz * dz

  return res

Z, X, M, Y = simular()

PHI = np.concatenate([
  np.ones(N).reshape(N,1),
  (X**2).reshape(N,1),
  (X * Z**2).reshape(N,1),
  (Z**4).reshape(N,1)
], axis=1)

blm = BayesianLinearModel(basis=lambda x: x)
blm.update(PHI, Y.reshape(N,1))

print(f'Efecto causal de X**2 sobre Y: {mean1[1]}')
print(f"Evidencia: {blm.evidence()}")