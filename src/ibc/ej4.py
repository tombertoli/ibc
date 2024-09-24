import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, multivariate_normal
from ModeloLineal import posterior, log_evidence, likelihood

def modelo_base(X):
  base = np.ones((len(X),2))
  base[:, 1] = X['altura_madre']
  return base

def modelo_biologico(X):
  bio = np.ones((len(X), 3))
  bio[:, 1] = X['altura_madre']
  bio[:, 2] = X['sexo'] == 'F'
  return bio

def modelo_grupos(X):
  rand = np.ones((len(X), 3))
  rand[:, 1] = X['altura_madre']
  rand[:, 2] = X['id'] % (X['id'].max() / 2)
  return rand

X = pd.read_csv('alturas.csv')

base = modelo_base(X)
bio = modelo_biologico(X)
rand = modelo_grupos(X)

#plt.bar(['base', 'bio', 'rand'], [log_evidence(X['altura'], base), log_evidence(X['altura'], bio), log_evidence(X['altura'], rand)], color=plt.color_sequences['tab10'])
#plt.show()

def geom_mean(y, M):
  return np.exp(log_evidence(y, M) / len(y))

plt.bar(['base', 'bio', 'rand'], [geom_mean(X['altura'], base), geom_mean(X['altura'], bio), geom_mean(X['altura'], rand)], color=plt.color_sequences['tab10'])
plt.show()

def pM(m):
  return 1/3

def pDatos(y):
  modelos = [base, bio, rand]
  return sum((pDatos_M(y,m) * pM(m) for m in modelos))

def pDatos_M(y,m):
  return np.exp(log_evidence(y,m) / len(y))

def pM_Datos(y, m):
  return pDatos_M(y,m) * pM(m) / pDatos(y)

#print(pM_Datos(X['altura'], base))
#print(pM_Datos(X['altura'], bio))
#print(pM_Datos(X['altura'], rand))
plt.bar(['base', 'bio', 'rand'], [
  pM_Datos(X['altura'], base), 
  pM_Datos(X['altura'], bio),
  pM_Datos(X['altura'], rand)
], color=plt.color_sequences['tab10'])
plt.show()
