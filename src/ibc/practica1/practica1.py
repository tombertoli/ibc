import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd

gen = np.random.default_rng(42)

def gen_monty_hall_episode(gen):
  opts = list(range(3))
  c = gen.choice(opts)
  r = gen.choice(opts)
  opts.remove(r)

  if r != c:
    opts.remove(c)

  s = gen.choice(opts)

  return (c, s, r)

#T = 16
#datos = [ gen_monty_hall_episode(gen) for _ in range(T) ]

datos = pd.read_csv('NoMontyHall.csv').values
T = 60 #len(datos) - 1

#for (c, s, r) in datos:
#  assert c != s
#  assert r != s

def base_model_likelihood(data):
  return reduce(lambda acc, e: acc + np.log10((1/3) * (1/2) * (1/3)), data, 0)

def monty_hall_model_likelihood(data):
  return reduce(lambda acc, e: acc + np.log10((1/3) * (1/3) * (0 if e[0] == e[1] else (1/2) if e[0] == e[2] else 1)), data, 0)

print('base model likelihood:', base_model_likelihood(datos))
print('monty hall likeliehood:', monty_hall_model_likelihood(datos))

print('log bayes factor:', monty_hall_model_likelihood(datos) - base_model_likelihood(datos))

def typical_prediction(likelihood):
  return 10**(likelihood / (3 * T))

print('typical prediction (base):', typical_prediction(base_model_likelihood(datos)))
print('typical prediction (MH):', typical_prediction(monty_hall_model_likelihood(datos)))

def alt_model_likelihood(data):
  p = 0
  for (c, s, r) in data:
    if s == c:
      p += np.log10(0.5)
    else:
      p += 0.5 * np.log10(0.5) + 0.5 * np.log10(0.5 if c == r else 1)

    p += np.log10((1/3) * (1/3))

  return p

def nf(d):
  return sum([(10 ** base_model_likelihood(d)) / 3, (10 ** monty_hall_model_likelihood(d)) / 3, (10 ** alt_model_likelihood(d)) / 3])

def posterior(likelihood_function, t):
  conjunta = (10 ** likelihood_function(datos[:t])) / 3
  norm = nf(datos[:t])
  return conjunta / norm

mh = [ posterior(monty_hall_model_likelihood, t) for t in range(T) ]
bm = [ posterior(base_model_likelihood, t) for t in range(T) ]

am = [ posterior(alt_model_likelihood, t) for t in range(T) ]

plt.plot(bm)
plt.plot(mh)
plt.plot(am)
plt.legend(['M0: Base', 'M1: Monty Hall', 'M2: Alternative Model'])
plt.xlabel('Episodio')
plt.ylabel('P(Modelo | Datos)')
plt.show()