import numpy as np
from .PolynomialSystem import PolynomialSystem
from .NormalizedPolynomials import NormalizedPolynomials
from ..distributions import SemiCircleDistribution

class ChebyshevTPolynomials(PolynomialSystem):
    """ Implementation of Chebyshev polynomials of the first kind.

        Methods
        -------
        normalized()
            Return normalized version of first kind Chebyshev polynomial system.

        recur_coeff(deg)
            Return the recurrence coefficients for first kind Chebyshev polynomials.

        sqnorm(n)
            Return the squared norm of first kind Chebyshev polynomials of degree `n`.
        
        weighting_dist()
            Return the weighting distribution for first kind Chebyshev polynomials."""
    
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
