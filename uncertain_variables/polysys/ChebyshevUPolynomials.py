import numpy as np
from .PolynomialSystem import PolynomialSystem
from ..distributions import SemiCircleDistribution

class ChebyshevUPolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        """ Return normalized version of second kind Chebyshev polynomial system.

            Returns
            -------
            polysys: NormalizedPolynomials
                Wrapped normalized second kind Chebyshev polynomial system."""
        
        return self

    @staticmethod
    def recur_coeff(deg):

        n = np.arange(deg)
        one = np.ones_like(n)
        zero = np.zeros_like(n)
        r = np.column_stack((zero, 2*one, one))

        return r

    @staticmethod
    def sqnorm(self, n):

        nrm2 = np.ones_like(n)
        return nrm2

    @staticmethod
    def weighting_dist():

        dist = SemiCircleDistribution(1)
        return dist
