import numpy as np
from .PolynomialSystem import PolynomialSystem
from ..distributions import ExponentialDistribution

class LaguerrePolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        """ Return normalized version of Laguerre polynomial system.
        
            Returns
            -------
            polysys: NormalizedPolynomials
                Wrapped normalized Laguerre polynomial system."""

        return self

    @staticmethod
    def recur_coeff(deg):

        n = np.arange(deg)

        r = np.column_stack(((2*n + 1)/(n+1), -1/(n + 1), n/(n + 1)))

        return r

    @staticmethod
    def sqnorm(self, n):

        nrm2 = np.ones_like(n)
        return nrm2

    @staticmethod
    def weighting_dist():

        dist = ExponentialDistribution(1)
        return dist
