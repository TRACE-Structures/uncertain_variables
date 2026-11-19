import numpy as np

from .PolynomialSystem import PolynomialSystem, _probability_sqnorm
from ..distributions import ArcsineDistribution
from scipy.special import chebyt


class ChebyshevTPolynomials(PolynomialSystem):
    """ 
    Implementation of Chebyshev polynomials of the first kind.
    """
    def __init__(self, orthonormal=False):
        """
        Initialize Chebyshev polynomials of the first kind.

        Parameters
        ----------
        orthonormal : bool, optional
            If True, use orthonormal polynomials. Default is False.
        """
        self.generator = chebyt
        self.orthonormal = orthonormal

    def gpc_distribution(self):
        """
        Return the generalized polynomial chaos distribution associated with Chebyshev polynomials of the first kind.

        Returns
        -------
        gpc_dist : ArcsineDistribution
            Arc-sine distribution.
        """
        gpc_dist = ArcsineDistribution(scale=2, loc=-1)
        return gpc_dist

    def _evaluate(self, deg, x, derivative=False):
        """
        Evaluate the first kind Chebyshev polynomials up to degree `deg` at points `x`.

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
            Matrix of shape (len(x), deg+1) containing evaluated first kind Chebyshev polynomials or their derivatives.
        """
        x = np.asarray(x)
        if derivative:
            y = np.column_stack([np.polyder(self.generator(d))(x) for d in range(deg+1)])
        else:
            y = np.column_stack([self.generator(d)(x) for d in range(deg+1)])

        if self.orthonormal:
            y /= np.sqrt(_probability_sqnorm(self.generator, deg, np.pi))
        return y

    def _sqnorm(self, deg):
        """
        Return the squared norm of the first kind Chebyshev polynomial of degree `deg`.

        Parameters
        ----------
        deg : int
            Degree of the polynomial.

        Returns
        -------
        y : numpy.ndarray
            Probability-weighted squared norms for degrees 0 through `deg`.
        """
        if self.orthonormal:
            y = np.ones(deg+1)
        else:
            y = _probability_sqnorm(self.generator, deg, np.pi)
        return y


# import numpy as np
# from .PolynomialSystem import PolynomialSystem
# from .NormalizedPolynomials import NormalizedPolynomials
# from ..distributions import SemiCircleDistribution

# class ChebyshevTPolynomials(PolynomialSystem):
#     """ Implementation of Chebyshev polynomials of the first kind."""
    
#     @classmethod
#     def normalized(self):
#         """ Return normalized version of the first kind Chebyshev polynomial system.

#             Returns
#             -------
#             polysys: NormalizedPolynomials
#                 Wrapped normalized first kind Chebyshev polynomial system."""

#         polysys = NormalizedPolynomials(self)
#         return polysys

#     @staticmethod
#     def recur_coeff(deg):
#         n = np.arange(deg)
#         one = np.ones_like(n)
#         zero = np.zeros_like(n)
#         r = np.column_stack((zero, 2*one - (n==0), one))
#         return r

#     @staticmethod
#     def sqnorm(self, n):
#         nrm2 = 0.5 * ((n==0) + 1)
#         return nrm2

#     @staticmethod
#     def weighting_dist():

#         dist = SemiCircleDistribution(1)
#         return dist
