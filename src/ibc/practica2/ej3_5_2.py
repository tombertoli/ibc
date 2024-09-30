import numpy as np
from scipy.stats import bernoulli, beta, multinomial
from math import prod
from functools import reduce

# 3.3

pc = [0,0.15,0.3,0.45,0.7]

s = [0.9, 0.6]

x = [0.95, 0.9]

# Modelo 1

x_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1]
x_probas = np.ones((2,len(x_vals))) / len(x_vals)

s_vals = [0.1, 0.3, 0.5, 0.6, 0.9, 1]
s_probas = np.ones((2,len(s_vals))) / len(s_vals)

pc_vals = [0, 0.15, 0.3, 0.45, 0.7, 0.9]
pc_probas = np.ones((5, len(pc_vals))) / len(pc_vals)

def ppc(pc):
  return pc_probas[pc[0], pc[1]]

N = 500

def pxs(ixs):
  return prod((x_probas[t, ix] for t, ix in enumerate(ixs)))

def pss(iss):
  return prod((s_probas[t, si] for t, si in enumerate(iss)))

def pdqxspc_M1(d,iss,ixs,ipc):
  qs = q_pcxs(ipc, ixs, iss) # Tiene probabilidad 1 siempre
  return pd_q(d,qs) * pxs(ixs) * pss(iss) * ppc(ipc)

def d_q(q):
  return multinomial(N, q)

def pd_q(d, q):
  return multinomial.pmf(d, N, q)

def q_pcxs(ipc, ixs, iss):
  pc = pc_vals[ipc[1]]
  xs = x_vals[ixs[0]], x_vals[ixs[1]]
  ss = s_vals[iss[0]], s_vals[iss[1]]

  return [
    pc * (1 - ss[0]) * (1 - ss[1]) + (1 - pc) * xs[0] * xs[1],
    pc * (1 - ss[0]) * ss[1] + (1 - pc) * xs[0] * (1 - xs[1]),
    pc * ss[0] * (1 - ss[1]) + (1 - pc) * (1 - xs[0]) * xs[1],
    pc * ss[0] * ss[1] + (1 - pc) * (1 - xs[0]) * (1 - xs[1]),
  ]

# Ej 3.4

def m1_MAP(datos):
  global x_probas
  global s_probas
  global pc_probas

  for c in range(datos.shape[0]):
    d = datos[c]
    pd = 0 # P(d)
    joint_x_d = np.zeros_like(x_probas) # P(d, x)
    joint_s_d = np.zeros_like(s_probas) # P(d, s)
    joint_pc_d = np.zeros_like(pc_vals) # P(d, pc)

    for hx0 in range(len(x_vals)):
      for hx1 in range(len(x_vals)):
        for hs0 in range(len(s_vals)):
          for hs1 in range(len(s_vals)):
            for hpc in range(len(pc_vals)):
              proba = pdqxspc_M1(d,(hs0,hs1),(hx0,hx1),(c,hpc))
              joint_x_d[0][hx0] += proba
              joint_x_d[1][hx1] += proba
              joint_s_d[0][hs0] += proba
              joint_s_d[1][hs1] += proba

              joint_pc_d[hpc] += proba
              pd += proba
        
    x_probas = joint_x_d / pd
    s_probas = joint_s_d / pd
    pc_probas[c] = joint_pc_d / pd

  return (
    x_probas,
    s_probas,
    pc_probas
  )

diagM1 = np.load('src/ibc/practica2/M1_diags.npy')
print(m1_MAP(diagM1))