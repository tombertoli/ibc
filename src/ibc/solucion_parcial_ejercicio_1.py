import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def pr(r):
    return 1/3

def pc(c):
    return 1/3

def ps_rM0(s,r):
    return (s!=r) * 1/2

def ps_rcM1(s,r,c):
    if r!=c:
        return (s!=r) * (s!=c) * 1
    else:
        return (s!=r) * 1/2

def prcs_M(r,c,s,m):
    if m==0:
        return pr(r)*pc(c)*ps_rM0(s,r)
    elif m==1:
        return pr(r)*pc(c)*ps_rcM1(s,r,c)


H = np.arange(3)

def ps_cM(s,c,m): # P(s|c,M) = P(s,c|M)/P(c|M)
    num = 0 # P(s,c|M) = sum_r P(r,c,s|M)
    den = 0 # P(c|M) = sum_{rs} P(r,c,s|M)
    for hr in H:
        num += prcs_M(hr,c,s,m)
        for hs in H:
            den += prcs_M(hr,c,hs,m)
    return num/den

def pr_csM(r,c,s,m): # P(r|c,s,M) = P(r,c,s|M)/p(c,s|M)
    num = prcs_M(r,c,s,m)
    den = 0 # p(c,s|M) = sum_r P(r,c,s|M)
    for hr in H:
        den += prcs_M(hr,c,s,m)
    return num/den

def pEpisodio_M(c,s,r,m):
    if m < 2:
        return prcs_M(r,c,s,m)

    global pp_Datos
    tot = 0 
    for a in range(2):
        for p in range(len(pp_Datos)): 
            tot += prcsap_M(r,c,s,a,p)

    if m == 2:
        pp_Datos = [sum((prcsap_M(r,c,s,ha,ip) for ha in range(2))) / tot for ip in range(len(pp_Datos))]
    
    return tot

def simular(T=16,seed=0):
    np.random.seed(seed)
    Datos = []
    for t in range(T):
        r = np.random.choice(3, p=[pr(hr) for hr in H])
        c = np.random.choice(3, p=[pc(hc) for hc in H])
        s = np.random.choice(3, p=[ps_rcM1(hs,r,c) for hs in H])
        Datos.append((c,s,r))
    return Datos

T = 16
Datos = simular()

def secuencia_de_predicciones(Datos,m):
    global pp_Datos
    if m == 2:
        pp_Datos = (1/len(p_vals)) * np.ones(len(p_vals))

    pDatos_M = [1]
    for t in range(len(Datos)):
        c, s, r = Datos[t]

        pDatos_M.append(pEpisodio_M(c,s,r,m))
    return pDatos_M

def pDatos_M(Datos, m):
    return np.prod(secuencia_de_predicciones(Datos,m))

#print(pDatos_M(Datos, m=0)) # 8.234550899283273e-21
#print(pDatos_M(Datos, m=1)) # 3.372872048346429e-17

#
# 1.9
#

p_vals = np.linspace(0, 1, num=10)
pp_Datos = (1/len(p_vals)) * np.ones(len(p_vals))

def pa_p(a, ip): # P(a | p) = p
    if a == 0:
        return 1 - p_vals[ip]
    else: 
        return p_vals[ip]

def prcsap_M(r,c,s,a,ip):
    return pr(r) * pc(c) * (ps_rM0(s, r) ** (1 - a)) * (ps_rcM1(s, r, c) ** a) * pa_p(a, ip) * pp_Datos[ip]

def pDatos_M(Datos, m):
    return np.prod(secuencia_de_predicciones(Datos,m))

def load_datos():
    data = list(pd.read_csv('NoMontyHall.csv')[:2000].itertuples(index=False, name=None))
    return data

no_mh_datos = load_datos()

#print(pDatos_M(no_mh_datos, m=2))

# 1.5

log_evidencia_M0 = np.log10(pDatos_M(no_mh_datos, m=0))
log_evidencia_M1 = np.log10(pDatos_M(no_mh_datos, m=1))
log_evidencia_M2 = np.log10(pDatos_M(no_mh_datos, m=2))

def pM(m):
    return (1/3)

#pDatos_M0 = secuencia_de_predicciones(no_mh_datos,m=0)
#pDatos_M1 = secuencia_de_predicciones(no_mh_datos,m=1)
#pDatos_M2 = secuencia_de_predicciones(no_mh_datos,m=2)
#
#pDatosM = [np.cumprod(pDatos_M0) * pM(0), # P(Datos,M0)
#           np.cumprod(pDatos_M1) * pM(1), # P(Datos,M1)
#           np.cumprod(pDatos_M2) * pM(2)] # P(Datos,M2)
#           
#pDatos = sum(pDatosM)
#
## 1.7 Posterior
#
#pM_Datos = [pDatosM[0]/pDatos, # P(M0|Datos)
#            pDatosM[1]/pDatos, # P(M1|Datos)
#            pDatosM[2]/pDatos] # P(M2|Datos)

#plt.plot(pM_Datos[0], label="M0: Base")
#plt.plot(pM_Datos[1], label="M1: Monty Hall")
#plt.plot(pM_Datos[2], label="M2: Alternativo")
#plt.legend()
#plt.show()

# Practica 2
# 1.1

def posteriors(Datos):
    global pp_Datos
    pp_Datos = (1/len(p_vals)) * np.ones(len(p_vals))
    pa_Datos = (1/2) * np.ones(2)

    post_a = np.empty((len(Datos), 2))
    post_p = np.empty((len(Datos), len(pp_Datos)))

    pDatos_M = [1]
    for t in range(len(Datos)):
        c, s, r = Datos[t]
        pDatos_M.append(pEpisodio_M(c,s,r,2))

        pa_Datos = [sum((prcsap_M(r,c,s,a,hp) for hp in range(len(pp_Datos)))) / pDatos_M[-1] for a in range(2)]
        post_a[t] = pa_Datos

        pp_Datos = [sum((prcsap_M(r,c,s,ha,ip) for ha in range(2))) / pDatos_M[-1] for ip in range(len(pp_Datos))]
        post_p[t] = pp_Datos
    return post_a, post_p
    
#post_a, post_p = posteriors(no_mh_datos)
#for p in range(len(pp_Datos)):
#    plt.plot(range(len(no_mh_datos)), post_p[:, p], label=f"p = {p_vals[p]:.2f}")
#plt.legend()
#plt.show()
#
#plt.plot(range(len(no_mh_datos)), post_a[:, 0], label=f"a = {0}")
#plt.legend()
#plt.show()

# 1.4

def log_prediction_rate(Data, m_func):
    return np.sum([-np.log(m_func(c,s,r)) for c,s,r, in Data])

#print(log_prediction_rate(no_mh_datos, lambda c,s,r: pEpisodio_M(c,s,r,0)))
#print(log_prediction_rate(no_mh_datos, lambda c,s,r: pEpisodio_M(c,s,r,1)))

#pp_Datos = (1/len(p_vals)) * np.ones(len(p_vals))
#print(log_prediction_rate(no_mh_datos, lambda c,s,r: pEpisodio_M(c,s,r,2)))

def cross_entropy(r, m):
    tot = 0 
    for hc in range(3):
        for hs in range(3):
            for hr in range(3):
                if hs == hr: continue
                tot += r(hc,hs,hr) * (-np.log2(m(hc,hs,hr)))
    
    return tot

#pp_Datos = (1/len(p_vals)) * np.ones(len(p_vals))
#print(cross_entropy(lambda c,s,r: pEpisodio_M(c,s,r,2), lambda c,s,r: pEpisodio_M(c,s,r,2)))

### Ejercicio 2

# 2.2

T = 10
N = 10000
qc = 3
qs = 1.2

def simular_apuestas(b,t):
    ts = np.random.randint(0, 2, t)
    return np.cumsum(np.log(b * qc * ts + (1 - b) * qs * (1 - ts)))

#personas = np.empty((N, T))
#for i in range(N):
#    personas[i] = simular_apuestas(0.5, T)
#
#plt.plot(np.mean(personas, axis=0))
#plt.show()

# b = 0.35


#plt.plot(simular_apuestas(0.35, 1000))
#plt.show()

ts = np.random.randint(0,2,1000)

def w_t(b, ts):
    return [(b * qc) ** np.sum(ts[:i]) + ((1-b) * qs) ** np.sum(1 - ts[:i]) for i in range(len(ts))]

for b in np.linspace(0, 1, num=10):
    plt.plot(w_t(b, ts), label=f'b = {b}')

plt.yscale('log')
plt.legend()
plt.show()