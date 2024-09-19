import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

gen = np.random.default_rng(seed=42)
beta = 0.05 ** 0.5

def gen_data(gen):
  x = gen.uniform(-0.5,0.5)
  y = gen.normal(loc=np.sin(2 * np.pi * x), scale=beta)
  return [x, y]

T = 20
datos = np.array([gen_data(gen) for _ in range(T)])

#plt.plot(np.linspace(-0.5, 0.5), [np.sin(2 * np.pi * x) for x in np.linspace(-0.5, 0.5)], )
#plt.scatter(x=datos[:, 0], y=datos[:, 1])
##plt.legend(['M0: Real', 'M1: Alternativo'])
#plt.xlabel('Episodio')
#plt.ylabel('P(Modelo | Datos)')
#plt.show()

ls = []

for i in range(10):
  X = np.column_stack([datos[:, 0] ** d for d in range(i + 1)])
  ols = sm.OLS(datos[:, 1], X)
  res = ols.fit()
  ls.append(res)

#plt.bar(list(range(10)), [np.exp(res.llf) for res in ls])
#plt.xticks(list(range(10)))
#plt.show()

def ds(x, n):
  return np.column_stack([x ** d for d in range(n + 1)]) 

#plt.plot(np.linspace(-0.5, 0.5), [np.sin(2 * np.pi * x) for x in np.linspace(-0.5, 0.5)], *[[l.predict(ds(x,n)) for x in np.linspace(-0.5, 0.5)] for n, l in enumerate(ls)])
plt.plot(np.linspace(-0.5, 0.5), [np.sin(2 * np.pi * x) for x in np.linspace(-0.5, 0.5)])
for i in range(10):
  plt.plot(np.linspace(-0.5, 0.5), [ls[i].predict(ds(x,i)) for x in np.linspace(-0.5, 0.5)])
plt.ylim(bottom=-1.5, top=1.5)
plt.show()