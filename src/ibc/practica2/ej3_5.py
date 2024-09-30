import numpy as np
from scipy.stats import bernoulli, beta, multinomial

# 3.3

pc = [0,0.15,0.3,0.45,0.7]

s = [0.9, 0.6]

x = [0.95, 0.9]

# Modelo 0

x_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1]
x_probas = np.ones((2,len(x_vals))) / len(x_vals)

s_vals = [0.1, 0.3, 0.5, 0.6, 0.9, 1]
s_probas = np.ones((2,len(s_vals))) / len(s_vals)

pc_vals = [0, 0.15, 0.3, 0.45, 0.7, 0.9]
pc_probas = np.ones((5, len(pc_vals))) / len(pc_vals)

def px(ix):
  return x_probas[ix[0], ix[1]]

def ps(si):
  return s_probas[si[0], si[1]]

def ppc(ipc):
  return pc_probas[ipc[0], ipc[1]]

def pe_pc(e, ipc):
  return bernoulli.pmf(e, pc_vals[ipc[1]])

def pd_esxM0(d,e,si,ix):
  return (bernoulli.pmf(d, s_vals[si[1]]) ** e) * (bernoulli.pmf(d, 1-x_vals[ix[1]]) ** (1 - e))

def pdesxpc_M0(d,e,si,ix,ipc):
  return pd_esxM0(d,e,si,ix) * px(ix) * ps(si) * pe_pc(e,ipc) * ppc(ipc)

# Modelo 1

N = 500

def d_q(q):
  return multinomial(N, q)

def q_pcxs(pc, xs, ss):
  return [
    pc * (1 - ss[0]) * (1 - ss[1]) + (1 - pc) * xs[0] * xs[1],
    pc * (1 - ss[0]) * ss[1] + (1 - pc) * xs[0] * (1 - xs[1]),
    pc * ss[0] * (1 - ss[1]) + (1 - pc) * (1 - xs[0]) * xs[1],
    pc * ss[0] * ss[1] + (1 - pc) * (1 - xs[0]) * (1 - xs[1]),
  ]


# Generacion

gen = np.random.default_rng(42)

def gen_diagnosticos_M1():
  def gen_diagnostico(c):
    qs = q_pcxs(pc[c], x, s)
    return d_q(qs).rvs(random_state=gen)
    
  C = 5
  diags = np.empty((C, 4))

  for c in range(C):
    diags[c] = gen_diagnostico(c)

  return diags
  
diagM0 = np.load('src/ibc/practica2/M0_diags.npy')
#print(gen_diagnosticos_M1())

# Ej 3.4

def m0_MAP(datos):
  global x_probas
  global s_probas
  global pc_probas

  for c in range(datos.shape[0]):
    for n in range(datos.shape[1]):
      # P(e) ??
      for t in range(datos.shape[2]):
        diagnostico = datos[c,n,t]
        pd = 0 # P(d)
        joint_x_d = np.zeros_like(x_vals) # P(d, x)
        joint_s_d = np.zeros_like(s_vals) # P(d, s)
        joint_pc_d = np.zeros_like(pc_vals) # P(d, pc)

        for hx in range(len(x_vals)):
          for hs in range(len(s_vals)):
            for hpc in range(len(pc_vals)):
              for he in range(2):
                proba = pdesxpc_M0(diagnostico,he,(t, hs),(t, hx),(c, hpc))
                joint_x_d[hx] += proba
                joint_s_d[hs] += proba
                joint_pc_d[hpc] += proba
                pd += proba
        
        x_probas[t] = [
          # P(x | d) = P(d , x) / P(d)
          px / pd
          for px in joint_x_d
        ]

        s_probas[t] = [
          ps / pd
          for ps in joint_s_d
        ]

        pc_probas[c] = [
          ppc / pd
          for ppc in joint_pc_d
        ]

  print(x_probas)
  print(s_probas)
  print(pc_probas)

  return "hola"
  return {
    'x': max(zip(x_vals, x_probas), key=lambda x: x[1]),
    's': max(zip(s_vals, s_probas), key=lambda s: s[1]),
    'pc': max(zip(pc_vals, pc_probas), key=lambda s: s[1]),
  }

print(m0_MAP(diagM0))