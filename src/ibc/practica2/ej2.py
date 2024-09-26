import numpy as np
import matplotlib.pyplot as plt

### Ejercicio 2

# 2.2
gen = np.random.default_rng(42)

qc = 3
qs = 1.2

def simular_apuestas(b,t):
    ts = gen.integers(2, size=t)
    return np.cumsum(np.log(b * qc * ts + (1 - b) * qs * (1 - ts)))

def simular_personas(N,T, /, b=0.5):
  personas = np.empty((N, T))
  for i in range(N):
      personas[i] = simular_apuestas(b, T)
      
  return personas

def ej22():
  T = 10
  N = 10000

  personas = simular_personas(N, T)
  plt.plot(np.mean(np.exp(personas), axis=0))
  plt.xlabel('Episodio (t)')
  plt.ylabel('Recursos promedio ($\\omega$)')
  plt.show() 

# 2.4
def ej24():
  b = 0.35
  T = 100
  N = 10

  personas = simular_personas(N, T,b=b)
  for p in personas:
    plt.plot(np.exp(p))
  
  plt.xlabel('Episodio (t)')
  plt.ylabel('Recursos ($\\omega$)')
  plt.show()

# 2.5
def w_t(b, ts):
    return (1/len(ts)) * (np.sum(ts) * np.log(b*qc) + np.sum(1-ts) * np.log((1-b) * qs))

def ej25():
  T = 5000
  ts = gen.integers(2, size=T)
  bs = np.linspace(0, 1, num=11, endpoint=False)[1:]

  for i in np.logspace(1, np.log10(T), num=10, dtype=int)[4:]:
    plt.plot(bs, [np.exp(w_t(b, ts[:i])) for b in bs], label=f'T = {i}')

  plt.title('Convergencia de Tasa de Crecimiento')
  plt.xlabel('Valor de $b$')
  plt.ylabel('Tasa de crecimiento')
  plt.legend()
  plt.show()

#ej25()
#ej24()
ej22()