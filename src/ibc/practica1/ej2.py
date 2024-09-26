import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

gen = np.random.default_rng(42)

def gen_a_causa_b(gen):
  a = gen.choice([0,1])
  b = gen.choice([0, 1], p=[(1 - a) * 0.9 + 0.05, a * 0.9 + 0.05])
  return (a,b)

T = 16
datos = [ gen_a_causa_b(gen) for _ in range(T) ]

#print(datos)

def ab_likelihood(data):
  return reduce(lambda acc, e: acc + np.log10(0.5) + np.log10((1 - e[0]) * 0.9 + 0.05) + np.log10(e[0] * 0.9 + 0.05), data, 0)

def ba_likelihood(data):
  return reduce(lambda acc, e: acc + np.log10(0.5) + np.log10((1-e[1]) * 0.9 + 0.05) + np.log10(e[1] * 0.9 + 0.05), data, 0)

def nf(d):
  ls = [ab_likelihood(d), ba_likelihood(d)]
  return sum([10**x / len(ls) for x in ls])

def posterior(likelihood_function, t):
  conjunta = (10 ** likelihood_function(datos[:t])) / 2
  norm = nf(datos[:t])
  return conjunta / norm


ab = [ posterior(ab_likelihood, t) for t in range(T) ]
ba = [ posterior(ba_likelihood, t) for t in range(T) ]

plt.plot(ba)
plt.plot(ab)
plt.legend(['M0: Real', 'M1: Alternativo'])
plt.xlabel('Episodio')
plt.ylabel('P(Modelo | Datos)')
plt.show()