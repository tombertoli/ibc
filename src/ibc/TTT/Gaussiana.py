# -*- coding: utf-8 -*-
"""
   TrueskillThroughTime.py
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   :copyright: (c) 2019-2024 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
"""

import math
from typing import Self
inf = math.inf
sqrt2 = math.sqrt(2); sqrt2pi = math.sqrt(2 * math.pi)
from scipy.stats import norm
from scipy.stats import truncnorm

__all__ = ['MU', 'SIGMA', 'Gaussian', 'N01', 'N00', 'Ninf', 'Nms']

MU = 0.0
SIGMA = 6.
PI = SIGMA**-2
TAU = PI * MU

class Gaussian(object):
    #
    # Constructor
    def __init__(self,mu=MU, sigma=SIGMA):
        if sigma >= 0.0:
            self.mu, self.sigma = mu, sigma
        else:
            raise ValueError(" sigma should be greater than 0 ")
    #
    # Iterador
    def __iter__(self):
        return iter((self.mu, self.sigma))
    #
    # Print
    def __repr__(self):
        return 'N(mu={:.3f}, sigma={:.3f})'.format(self.mu, self.sigma)
    #
    # pi = 1/sigma^2
    @property
    def pi(self):
        if self.sigma > 0.0:
            return(1/self.sigma**2)
        else:
            return inf
    #
    # tau = mu*pi
    @property
    def tau(self):
        if self.sigma > 0.0:
            return(self.mu*self.pi)
        else:
            return 0
    #
    # N > 0
    def __gt__(self, threshold):
        # Gaussiana truncada: N > 0.
        mu, sigma = self
        truncated_norm = truncnorm(
            (threshold - mu) / sigma, inf, loc=mu, scale=sigma)
        # Devolver la Gaussiana con misma media y desvío
        return(Gaussian(truncated_norm.mean(), truncated_norm.std()))
    #
    # N >= 0
    def __ge__(self, threshold):
        return self.__gt__(threshold)
    #
    # N + M
    def __add__(self, M):
        _mu = self.mu + M.mu
        _sigma = math.sqrt(self.sigma**2 + M.sigma**2)
        return(Gaussian(_mu, _sigma))
    #
    # N - M
    def __sub__(self, M):
        _mu = self.mu - M.mu
        _sigma = math.sqrt(self.sigma**2 + M.sigma**2)
        return(Gaussian(_mu, _sigma))
    #
    # N * M
    def __mul__(self, M: Self):
        if M.pi == 0:
            return self
        if self.pi == 0:
            return M
        _pi = (self.pi + M.pi)
        _sigma = _pi**-(1/2)
        _mu = (self.tau + M.tau)/_pi
        return(Gaussian(_mu,_sigma))
    def __rmul__(self, other):
        return self.__mul__(other)
    #
    # N / M
    def __truediv__(self, M):
        _pi = self.pi - M.pi
        _sigma = _pi**-(1/2)
        _mu = (self.tau - M.tau)/_pi
        return(Gaussian(_mu,_sigma))
    # Delta
    def delta(self, M):
        return abs(self.mu - M.mu) , abs(self.sigma - M.sigma)
    #
    def cdf(self, x):
        return norm(*self).cdf(x)
    # IsApprox
    def isapprox(self, M, tol=1e-4):
        return (abs(self.mu - M.mu) < tol) and (abs(self.sigma - M.sigma) < tol)


def suma(Ns):
    res = Gaussian(0,0) # neutro de la suma
    for N in Ns:
        res += res + N
    return(res)


N01 = Gaussian(0,1)
N00 = Gaussian(0,0)
Ninf = Gaussian(0,inf)
Nms = Gaussian(MU, SIGMA)





