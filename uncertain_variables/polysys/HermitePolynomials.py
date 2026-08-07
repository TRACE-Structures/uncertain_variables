import numpy as np
from .PolynomialSystem import PolynomialSystem
from .NormalizedPolynomials import NormalizedPolynomials
from ..distributions import NormalDistribution
import scipy.special as sp

class HermitePolynomials(PolynomialSystem):
    """ Implementation of Hermite polynomial system."""
    
    @classmethod
    def normalized(self):
        """ Return normalized version of Hermite polynomial system.

            Returns
            -------
            polysys: NormalizedPolynomials
                Wrapped normalized Hermite polynomial system."""

        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        """ Return the recurrence coefficients for Hermite polynomials.

            Parameters
            ----------
            deg :  int
                Degree up to which recurrence coefficients are computed.

            Returns
            -------
            r : numpy.ndarray
                Recurrence coefficients for Hermite polynomials."""
        
        n = np.arange(deg)
        one = np.ones_like(n)
        zero = np.zeros_like(n)
        r = np.column_stack((zero, one, n))

        return r

    @staticmethod
    def sqnorm(n):
        """ Return the squared norm of Hermite polynomials of degree `n`.

            Parameters
            ----------
            n : array_like
                Degrees for which the squared norms are computed.

            Returns
            -------
            nrm2 : numpy.ndarray
                Array of squared norms corresponding to degrees in `n`."""
        
        nrm2 = sp.factorial(n)
        return nrm2

    @staticmethod
    def weighting_dist():
        """ Return the weighting distribution for Hermite polynomials.

            Returns
            -------
            dist : NormalDistribution
                Weighting distribution for Hermite polynomials."""
        
        dist = NormalDistribution(0,1)
        return dist
