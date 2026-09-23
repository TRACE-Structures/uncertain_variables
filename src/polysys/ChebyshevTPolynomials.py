import numpy as np
from .PolynomialSystem import PolynomialSystem
from .NormalizedPolynomials import NormalizedPolynomials
from ..distributions import SemiCircleDistribution

class ChebyshevTPolynomials(PolynomialSystem):
    """ Implementation of Chebyshev polynomials of the first kind."""
    
    @classmethod
    def normalized(self):
        """ Return normalized version of the first kind Chebyshev polynomial system.

            Returns
            -------
            polysys: NormalizedPolynomials
                Wrapped normalized first kind Chebyshev polynomial system."""

        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        n = np.arange(deg)
        one = np.ones_like(n)
        zero = np.zeros_like(n)
        r = np.column_stack((zero, 2*one - (n==0), one))
        return r

    @staticmethod
    def sqnorm(self, n):
        nrm2 = 0.5 * ((n==0) + 1)
        return nrm2

    @staticmethod
    def weighting_dist():

        dist = SemiCircleDistribution(1)
        return dist
