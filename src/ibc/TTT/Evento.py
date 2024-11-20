# -*- coding: utf-8 -*-
import math
import numpy as np
from Gaussiana import Gaussian

BETA = 1.0 # El desvío estándar del desempeño: N(desempeño | mu=habilidad, sd=BETA).

class Evento(object):
    #
    # Constructor
    def __init__(self, equipos: list[list[Gaussian]], beta=BETA):
        # Ejemplo:
        # equipos = [ [priorA, priorB], [priorC]  ]
        # donde el el orden indica qué equipo ganó.
        self.equipos = equipos
        self.beta = beta
    #
    # Representación
    def __repr__(self):
        return f'{self.equipos}'
    #
    @property
    def desempeño_individuos(self):
        # Genera todos los desempeños sumando ruido a las habilidades
        res = []
        ruido = Gaussian(0,self.beta)
        for equipo in self.equipos:
            res.append([]) # Contenedor de equipo
            for habilidad in equipo:
                desempenio = habilidad + ruido
                res[-1].append(desempenio)
        return(res)
    #
    @property
    def desempeño_equipos(self):
        # Suma de los desempeños individuales
        res = []
        for equipo in self.desempeño_individuos:
            res.append(np.sum(equipo))
        return(res)
    #
    @property
    def diferencia_equipos(self):
        desempeño_equipos = self.desempeño_equipos
        return desempeño_equipos[0]-desempeño_equipos[1]
    #
    @property
    def likelihood_diferencia(self):
        evento = self
        #
        # Marginal de la diferencia: P(diferencia, resultado)
        marginal_diferencia = evento.diferencia_equipos > 0
        #
        # Likelihood = P(diferencia, resultado) / P(diferencia)
        return(marginal_diferencia / evento.diferencia_equipos)
    #
    @property
    def likelihood_equipos(self) -> list[Gaussian]:
        evento = self
        #
        # I(d = ta - tb) Diferencia entre los desempeños de los equipos
        ta, tb = evento.desempeño_equipos
        d = evento.likelihood_diferencia
        # ta = tb + d; tb = ta - d
        return([tb + d, ta - d ])
    #
    @property
    def likelihood_desempeño(self):
        evento = self
        desempeño_individuos = evento.desempeño_individuos
        desempeño_equipos = evento.desempeño_equipos
        likelihood_equipos = evento.likelihood_equipos
        # te = p1 + p2 + p3
        #        <=>
        # p1 = te - (p2 + p3)
        res: list[list[Gaussian]] = []
        for e in range(len(evento.equipos)):#e=0
            res.append([])
            for i in range(len(evento.equipos[e])):#i=0
                te = likelihood_equipos[e]
                te_sin_i = Gaussian(
                    desempeño_equipos[e].mu - desempeño_individuos[e][i].mu,
                    math.sqrt(desempeño_equipos[e].sigma**2 -  desempeño_individuos[e][i].sigma**2) )
                res[-1].append(te - te_sin_i)
        return res
    #
    @property
    def likelihood_habilidad(self):
        res: list[list[Gaussian]] = []
        ruido = Gaussian(0,self.beta)
        for equipo in self.likelihood_desempeño:
            res.append([])
            for lh_p in equipo:
                res[-1].append(lh_p - ruido)
        return res
    @property
    def likelihood(self):
        return self.likelihood_habilidad
    @property
    def posterior(self):
        evento = self
        likelihood = evento.likelihood_habilidad
        prior = evento.equipos
        res = []
        for e in range(len(prior)):
            res.append([])
            for i in range(len(prior[e])):
                res[-1].append(prior[e][i]*likelihood[e][i])
        return res


#priorA = Gaussian(3,1)
#priorB = Gaussian(2,1)
#priorC = Gaussian(6,1)
#Equipo1, Equipo2 = Evento([ [priorA, priorB], [priorC]]).posterior
#print(Equipo1)
#[N(mu=3.439, sigma=0.938), N(mu=2.439, sigma=0.938)]
#print(Equipo2)
#[N(mu=5.561, sigma=0.938)]

