import numpy as np

from .PolynomialSystem import PolynomialSystem, _probability_sqnorm
from ..distributions import UniformDistribution

from scipy.special import legendre


class LegendrePolynomials(PolynomialSystem):
    """ 
    Implementation of Legendre polynomial system.
    """
    def __init__(self, orthonormal=False):
        """
        Initialize the Legendre polynomial system.

        Parameters
        ----------
        orthonormal : bool, optional
            If True, use orthonormal polynomials. Default is False.
        """
        self.generator = legendre
        self.orthonormal = orthonormal

    def gpc_distribution(self):
        """
        Return the generalized polynomial chaos distribution associated with Legendre polynomials.

        Returns
        -------
        gpc_dist : UniformDistribution
            Uniform distribution over [-1, 1].
        """
        gpc_dist = UniformDistribution(-1, 1)
        return gpc_dist

    def _evaluate(self, deg, x, derivative=False):
        """
        Evaluate the Legendre polynomial system up to degree `deg` at points `x`.

        Parameters
        ----------
        deg : int
            Highest polynomial degree to evaluate.
        x : array_like
            Points at which the polynomials are evaluated.
        derivative : bool, optional
            If True, evaluate the derivative of the polynomials. Default is False.

        Returns
        -------
        y : numpy.ndarray
            Matrix of shape (len(x), deg+1) containing evaluated Legendre polynomials or their derivatives.
        """
        x = np.asarray(x)
        if derivative:
            y = np.column_stack([np.polyder(self.generator(d))(x) for d in range(deg+1)])
        else:
            y = np.column_stack([self.generator(d)(x) for d in range(deg+1)])
        if self.orthonormal:
            y /= np.sqrt(_probability_sqnorm(self.generator, deg, 2))
        return y

    def _sqnorm(self, deg):
        """
        Return squared norms of Legendre polynomials through degree `deg`.

        Parameters
        ----------
        deg : int
            Highest degree.

        Returns
        -------
        nrm2 : numpy.ndarray
            Squared norms in degree order.
        """
        if self.orthonormal:
            y = np.ones(deg+1)
        else:
            y = _probability_sqnorm(self.generator, deg, 2)
        return y
    
# import numpy as np
# from .PolynomialSystem import PolynomialSystem
# from .NormalizedPolynomials import NormalizedPolynomials
# from ..distributions import UniformDistribution

# class LegendrePolynomials(PolynomialSystem):
#     """ Implementation of Legendre polynomial system."""

#     @classmethod
#     def normalized(self):
#         """ Return normalized version of Legendre polynomial system.

#             Returns
#             -------
#             polysys: NormalizedPolynomials
#                 Wrapped normalized Legendre polynomial system."""
        
#         polysys = NormalizedPolynomials(self)
#         return polysys

#     @staticmethod
#     def recur_coeff(deg):
#         """ Return the recurrence coefficients for Legendre polynomials.

#             Parameters
#             ----------
#             deg : int
#                 Degree up to which recurrence coefficients are computed.

#             Returns
#             -------
#             r : numpy.ndarray
#                 Recurrence coefficients for Legendre polynomials."""
        
#         n = np.arange(deg)
#         zer = np.zeros_like(n)
#         r = np.column_stack((zer, (2*n+1)/(n+1), n/(n+1)))

#         return r

#     @staticmethod
#     def sqnorm(n):
#         """ Return the squared norm of Legendre polynomials of degree `n`.

#             Parameters
#             ----------
#             n : array_like
#                 Degrees for which the squared norms are computed.

#             Returns
#             -------
#             nrm2 : numpy.ndarray
#                 Array of squared norms corresponding to degrees in `n`."""
            
#         nrm2 = 1/(2*n + 1)
#         return nrm2

#     @staticmethod
#     def weighting_dist():
#         """ Return the weighting distribution for Legendre polynomials.

#             Returns
#             -------
#             dist : UniformDistribution
#                 Weighting distribution for Legendre polynomials."""
        
#         dist = UniformDistribution(-1,1)
#         return dist
