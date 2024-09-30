import numpy as np
from scipy.stats import bernoulli, beta, multinomial
#from math import prod
#from functools import reduce

# 3.3

pc = [0,0.15,0.3,0.45,0.7]

s = [0.9, 0.6]

x = [0.95, 0.9]

# Modelo 0

x_vals = np.linspace(0,1,num=10)
x_probas = np.ones_like(x_vals) / len(x_vals)

def px(x):
  #return beta(1,1).pdf(x)
  return beta.pdf(x,1,1)

def ps(s):
  #return beta(1,1).pdf(s)
  return beta.pdf(s,1,1)

def ppc(pc):
  #return beta(1,1).pdf(pc)
  return beta.pdf(pc,1,1)

def pe_pc(e, pc):
  #return bernoulli(pc).pmf(e)
  return bernoulli.pmf(e, pc)

def pd_esxM0(d,e,s,x):
  #return (bernoulli(s).pmf(d) ** e) * (bernoulli(1-x).pmf(d) ** (1 - e))
  return (bernoulli.pmf(d, s) ** e) * (bernoulli.pmf(d, 1-x) ** (1 - e))

def pdesxpc_M0(d,e,s,x,pc):
  return pd_esxM0(d,e,s,x) * px(x) * ps(s) * pe_pc(e,pc) * ppc(pc)

# Modelo 1

N = 500

#def pd_q(d,q):
#  return multinomial(N, q).pmf(d)
#
## Esta mal, no deberia retornar una lista, sino una proba
#def pq_pcxs(qs,pc,xs,ss):
#  probas = [
#    pc * (1 - ss[0]) * (1 - ss[1]) + (1 - pc) * xs[0] * xs[1],
#    pc * (1 - ss[0]) * ss[1] + (1 - pc) * xs[0] * (1 - xs[1]),
#    pc * ss[0] * (1 - ss[1]) + (1 - pc) * (1 - xs[0]) * xs[1],
#    pc * ss[0] * ss[1] + (1 - pc) * (1 - xs[0]) * (1 - xs[1]),
#  ]
#  return [p == q for p, q in zip(probas, qs)]
#
#def pxs(xs):
#  return prod((px(x) for x in xs))
#
#def pss(ss):
#  return prod((ps(s) for s in ss))
#
#def pdqxspc_M1(d,q,ss,xs,pc):
#  return pd_q(d,q) * pq_pcxs(q,pc,x,s) * pxs(xs) * pss(ss) * ppc(pc)

# Modelo 1 bis

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

def gen_diagnosticos_M0():
  def gen_diagnostico(c,t):
    dprobas = [sum(pdesxpc_M0(hd, he, s[t], x[t], pc[c]) for he in range(2)) for hd in range(2)]

    return gen.choice(2, p=dprobas)

  N = 500 # 2 diagnosticos por persona
  C = 5
  diags = np.empty((C, N, 2))

  for c in range(C):
    for i in range(N):
      diags[c, i] = gen_diagnostico(c, 0), gen_diagnostico(c, 1)

  return diags

def gen_diagnosticos_M1():
  def gen_diagnostico(c):
    qs = q_pcxs(pc[c], x, s)
    return d_q(qs).rvs(random_state=gen)
    
  C = 5
  diags = np.empty((C, 4))

  for c in range(C):
    diags[c] = gen_diagnostico(c)

  return diags
  
#print(gen_diagnosticos_M1())

# Ej 3.4

def m0_MAP(datos):
  hxs = np.linspace(0,1,num=10)
  phxs = []

  for hx in hxs:
    log_likelihood = 1 # log P(D | X)

    for d in datos:
      d_proba = 0 # P(d, X) = P(d | X) * P(X)
      for hs in np.linspace(0,1,num=5):
        for hpc in np.linspace(0,1,num=5):
          for he in range(2):
            d_proba += pdesxpc_M0(d,he,hs,hx,hpc)

      log_likelihood += np.log(d_proba * len(hxs))

    phxs.append(log_likelihood)

  phxs = [np.exp(phx) for phx in phxs] # P(D | X)
  pDatos = sum(likelihood / len(hxs) for likelihood in phxs) # P(D)
  phxs = [phx / (len(hxs) * pDatos) for phx in phxs] # P(X | D)
  return list(zip(hxs, phxs))

if __name__ == '__main__':
  #diagM0 = gen_diagnosticos_M0()
  #np.save('M0_diags', diagM0)
  #print(m0_MAP(diagM0[0, :, 0].ravel()))

  diagM1 = gen_diagnosticos_M1()
  np.save('M1_diags', diagM1)