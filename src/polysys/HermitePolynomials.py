import numpy as np

from .PolynomialSystem import PolynomialSystem, _probability_sqnorm
from ..distributions import NormalDistribution

from scipy.special import hermitenorm as hermite


class HermitePolynomials(PolynomialSystem):
    """ 
    Implementation of Hermite polynomial system.
    """
    def __init__(self, orthonormal=False):
        """ 
        Initialize Hermite polynomial system.

        Parameters
        ----------
        orthonormal : bool, optional
            If True, use orthonormal polynomials. Default is False.
        """
        self.generator = hermite
        self.orthonormal = orthonormal

    def gpc_distribution(self):
        """
        Return the generalized polynomial chaos distribution associated with Hermite polynomials.

        Returns
        -------
        gpc_dist : NormalDistribution
            Standard normal distribution.
        """
        gpc_dist = NormalDistribution(0, 1)
        return gpc_dist

    def _evaluate(self, deg, x, derivative=False):
        """
        Evaluate the Hermite polynomial system up to degree `deg` at points `x`.

        Parameters
        ----------
        deg : int
            Highest polynomial degree to evaluate.
        x : array_like
            Points at which the polynomials are evaluated.
        derivative : bool | int, default=False
            If True, evaluate the derivative of the polynomials.
            If an integer n, evaluate the n-th derivative of the polynomials. Default is False.

        Returns
        -------
        y : numpy.ndarray
            Matrix of shape (n_points, deg+1) containing evaluated Hermite polynomials or their derivatives.
        """
        x = np.asarray(x)
        if derivative:
            if isinstance(derivative, bool):
                n = 1 if derivative else 0
            else:
                n = derivative
            y = np.column_stack([np.polyder(self.generator(d), m=n)(x) for d in range(deg+1)])
        else:
            y = np.column_stack([self.generator(d)(x) for d in range(deg+1)])
        if self.orthonormal:
            y /= np.sqrt(_probability_sqnorm(self.generator, deg, np.sqrt(2 * np.pi)))
        return y

    def _sqnorm(self, deg):
        """
        Return the squared norm of the Hermite polynomial of degree `deg`.

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
            y = _probability_sqnorm(self.generator, deg, np.sqrt(2 * np.pi))
        return y

    def _orthonormalize(self):
        """
        Return an orthonormal version of the Hermite polynomial system.
        """
        return HermitePolynomials(orthonormal=True)
    
# import numpy as np
# from .PolynomialSystem import PolynomialSystem
# from .NormalizedPolynomials import NormalizedPolynomials
# from ..distributions import NormalDistribution
# import scipy.special as sp

# class HermitePolynomials(PolynomialSystem):
#     """ Implementation of Hermite polynomial system."""
    
#     @classmethod
#     def normalized(self):
#         """ Return normalized version of Hermite polynomial system.

#             Returns
#             -------
#             polysys: NormalizedPolynomials
#                 Wrapped normalized Hermite polynomial system."""

#         polysys = NormalizedPolynomials(self)
#         return polysys

#     @staticmethod
#     def recur_coeff(deg):
#         """ Return the recurrence coefficients for Hermite polynomials.

#             Parameters
#             ----------
#             deg :  int
#                 Degree up to which recurrence coefficients are computed.

#             Returns
#             -------
#             r : numpy.ndarray
#                 Recurrence coefficients for Hermite polynomials."""
        
#         n = np.arange(deg)
#         one = np.ones_like(n)
#         zero = np.zeros_like(n)
#         r = np.column_stack((zero, one, n))

#         return r

#     @staticmethod
#     def sqnorm(n):
#         """ Return the squared norm of Hermite polynomials of degree `n`.

#             Parameters
#             ----------
#             n : array_like
#                 Degrees for which the squared norms are computed.

#             Returns
#             -------
#             nrm2 : numpy.ndarray
#                 Array of squared norms corresponding to degrees in `n`."""
        
#         nrm2 = sp.factorial(n)
#         return nrm2

#     @staticmethod
#     def weighting_dist():
#         """ Return the weighting distribution for Hermite polynomials.

#             Returns
#             -------
#             dist : NormalDistribution
#                 Weighting distribution for Hermite polynomials."""
        
#         dist = NormalDistribution(0,1)
#         return dist
