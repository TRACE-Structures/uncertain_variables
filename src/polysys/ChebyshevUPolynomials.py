import numpy as np

from .PolynomialSystem import PolynomialSystem, _probability_sqnorm
from ..distributions import SemiCircleDistribution

from scipy.special import chebyu 


class ChebyshevUPolynomials(PolynomialSystem):
    """ 
    Implementation of second kind Chebyshev polynomial system.
    """
    def __init__(self, orthonormal=False):
        """ 
        Initialize second kind Chebyshev polynomial system.

        Parameters
        ----------
        orthonormal : bool, optional
            If True, use orthonormal polynomials. Default is False.
        """
        self.generator = chebyu
        self.orthonormal = orthonormal

    def gpc_distribution(self):
        """
        Return the generalized polynomial chaos distribution associated with second kind Chebyshev polynomials.

        Returns
        -------
        gpc_dist : SemiCircleDistribution
            Semi-circle distribution.
        """
        gpc_dist = SemiCircleDistribution(1)
        return gpc_dist

    def _evaluate(self, deg, x, derivative=False):
        """
        Evaluate the second kind Chebyshev polynomial system up to degree `deg` at points `x`.

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
            Matrix of shape (n_points, deg+1) containing evaluated second kind Chebyshev polynomials or their derivatives.
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
            y /= np.sqrt(_probability_sqnorm(self.generator, deg, np.pi / 2))
        return y

    def _sqnorm(self, deg):
        """
        Return the squared norm of the second kind Chebyshev polynomial of degree `deg`.

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
            y = _probability_sqnorm(self.generator, deg, np.pi / 2)
        return y

    def _orthonormalize(self):
        """
        Return an orthonormal version of the second kind Chebyshev polynomial system.
        """
        return ChebyshevUPolynomials(orthonormal=True)
# import numpy as np
# from .PolynomialSystem import PolynomialSystem
# from ..distributions import SemiCircleDistribution

# class ChebyshevUPolynomials(PolynomialSystem):
#     @classmethod
#     def normalized(cls):
#         """ Return normalized version of second kind Chebyshev polynomial system.

#             Returns
#             -------
#             polysys: NormalizedPolynomials
#                 Wrapped normalized second kind Chebyshev polynomial system."""
        
#         return cls

#     @staticmethod
#     def recur_coeff(deg):

#         n = np.arange(deg)
#         one = np.ones_like(n)
#         zero = np.zeros_like(n)
#         r = np.column_stack((zero, 2*one, one))

#         return r

#     @staticmethod
#     def sqnorm(self, n):

#         nrm2 = np.ones_like(n)
#         return nrm2

#     @staticmethod
#     def weighting_dist():

#         dist = SemiCircleDistribution(1)
#         return dist
