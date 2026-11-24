import numpy as np

from .PolynomialSystem import PolynomialSystem, _probability_sqnorm
from ..distributions import ExponentialDistribution

from scipy.special import laguerre


class LaguerrePolynomials(PolynomialSystem):
    """ 
    Implementation of Laguerre polynomial system.
    """
    def __init__(self, orthonormal=False):
        """
        Initialize the Laguerre polynomial system.

        Parameters
        ----------
        orthonormal : bool, optional
            If True, use orthonormal polynomials. Default is False.
        """
        self.generator = laguerre
        self.orthonormal = orthonormal

    def gpc_distribution(self):
        """
        Return the generalized polynomial chaos distribution associated with Laguerre polynomials.

        Returns
        -------
        gpc_dist : ExponentialDistribution
            Exponential distribution with rate parameter 1.
        """
        gpc_dist = ExponentialDistribution(1)
        return gpc_dist

    def _evaluate(self, deg, x, derivative=False):
        """
        Evaluate the Laguerre polynomial system up to degree `deg` at points `x`.

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
            Matrix of shape (n_points, deg+1) containing evaluated Laguerre polynomials or their derivatives.
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
            y /= np.sqrt(_probability_sqnorm(self.generator, deg, 1))
        return y

    def _sqnorm(self, deg):
        """
        Return the squared norm of the Laguerre polynomial of degree `deg`.

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
            y = _probability_sqnorm(self.generator, deg, 1)
        return y    

    def _orthonormalize(self):
        """
        Return an orthonormal version of the Laguerre polynomial system.
        """
        return LaguerrePolynomials(orthonormal=True)


# import numpy as np
# from .PolynomialSystem import PolynomialSystem
# from ..distributions import ExponentialDistribution

# class LaguerrePolynomials(PolynomialSystem):
#     @classmethod
#     def normalized(self):
#         """ Return normalized version of Laguerre polynomial system.
        
#             Returns
#             -------
#             polysys: NormalizedPolynomials
#                 Wrapped normalized Laguerre polynomial system."""

#         return self

#     @staticmethod
#     def recur_coeff(deg):

#         n = np.arange(deg)
#         r = np.column_stack(((2*n + 1)/(n+1), -1/(n + 1), n/(n + 1)))

#         return r

#     @staticmethod
#     def sqnorm(self, n):

#         nrm2 = np.ones_like(n)
#         return nrm2

#     @staticmethod
#     def weighting_dist():

#         dist = ExponentialDistribution(1)
#         return dist
