# -*- coding: utf-8 -*-
from typing import Callable
from Gaussiana import Gaussian
from Habilidad import Habilidad
from Evento import Evento
from collections import defaultdict
import math
from copy import copy

class Historia(object):
    def __init__(self, eventos: list[list[list[str]]], priors=defaultdict(lambda: Gaussian(0.0, 3.0)) ):
        self.eventos = eventos
        self.priors = priors
        self.habilidades = [ [[Habilidad() for _ in equipo] for equipo in evento] for evento in eventos]
    #
    def __repr__(self):
        return f'Historia(Eventos={len(self.eventos)})'
    #
    def forward_propagation(self):
        ultimo_mensaje = copy(self.priors)

        for t, ts in enumerate(zip(self.eventos, self.habilidades)):
            priors_t = [
                [
                    ultimo_mensaje[nombre] * habilidad.backward
                    for nombre, habilidad in zip(*es)
                ] for es in zip(*ts)
            ]

            likelihood = Evento(priors_t).likelihood
            def actualizar_forward(nombre, li, h):
                h.forward = ultimo_mensaje[nombre]
                h.likelihood = li
                ultimo_mensaje[nombre] = h.forward_posterior
                return h

            self.actualizar_mensajes(t, likelihood, actualizar_forward)

    def backward_propagation(self):
        ultimo_mensaje = defaultdict(lambda: Gaussian(0.0, math.inf))

        for t, ts in reversed(list(enumerate(zip(self.eventos, self.habilidades)))):
            priors_t = [
                [
                    habilidad.forward * ultimo_mensaje[nombre]
                    for nombre, habilidad in zip(*es)
                ] for es in zip(*ts)
            ]

            likelihood = Evento(priors_t).likelihood

            def actualizar_backward(nombre, li, h):
                h.backward = ultimo_mensaje[nombre]
                h.likelihood = li
                ultimo_mensaje[nombre] = h.backward_posterior
                return h
            self.actualizar_mensajes(t, likelihood, actualizar_backward)

    def actualizar_mensajes(self, t, likelihood: list[list[Gaussian]], f: Callable[[str, Gaussian, Habilidad], Habilidad]):
        for e in range(len(self.eventos[t])):
            for i in range(len(self.eventos[t][e])):
                nombre = self.eventos[t][e][i]
                self.habilidades[t][e][i] = f(nombre, likelihood[e][i], self.habilidades[t][e][i])

    def posteriors(self):
        return [[[habilidad.posterior for habilidad in equipo] for equipo in evento] for evento in self.habilidades]


