import numpy as np
from .PolynomialSystem import PolynomialSystem
from .NormalizedPolynomials import NormalizedPolynomials
from ..distributions import UniformDistribution

class LegendrePolynomials(PolynomialSystem):
    """ Implementation of Legendre polynomial system."""

    @classmethod
    def normalized(self):
        """ Return normalized version of Legendre polynomial system.

            Returns
            -------
            polysys: NormalizedPolynomials
                Wrapped normalized Legendre polynomial system."""
        
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        """ Return the recurrence coefficients for Legendre polynomials.

            Parameters
            ----------
            deg : int
                Degree up to which recurrence coefficients are computed.

            Returns
            -------
            r : numpy.ndarray
                Recurrence coefficients for Legendre polynomials."""
        
        n = np.arange(deg)
        zer = np.zeros_like(n)
        r = np.column_stack((zer, (2*n+1)/(n+1), n/(n+1)))

        return r

    @staticmethod
    def sqnorm(n):
        """ Return the squared norm of Legendre polynomials of degree `n`.

            Parameters
            ----------
            n : array_like
                Degrees for which the squared norms are computed.

            Returns
            -------
            nrm2 : numpy.ndarray
                Array of squared norms corresponding to degrees in `n`."""
            
        nrm2 = 1/(2*n + 1)
        return nrm2

    @staticmethod
    def weighting_dist():
        """ Return the weighting distribution for Legendre polynomials.

            Returns
            -------
            dist : UniformDistribution
                Weighting distribution for Legendre polynomials."""
        
        dist = UniformDistribution(-1,1)
        return dist
