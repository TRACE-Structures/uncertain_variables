import numpy as np

from .PolynomialSystem import PolynomialSystem, _probability_sqnorm
from ..distributions import BetaDistribution

from scipy.special import beta as beta_function, jacobi

class JacobiPolynomials(PolynomialSystem):
    """ 
    Implementation of Jacobi polynomial system.

    Attributes
    ----------
    alpha : float
        First parameter of Jacobi polynomials.
    beta : float
        Second parameter of Jacobi polynomials.
    orthonormal : bool
        If True, use orthonormal polynomials. Default is False.
    """

    def __init__(self, alpha: float, beta: float, orthonormal: bool = False):
        """ 

        Parameters
        ----------
        alpha : float
            First parameter of Jacobi polynomials.
        beta : float
            Second parameter of Jacobi polynomials.
        orthonormal : bool, optional
            If True, use orthonormal polynomials. Default is False.
        """

        self.generator = jacobi
        self.alpha = alpha
        self.beta = beta
        self.orthonormal = orthonormal

    def gpc_distribution(self) -> BetaDistribution:
        """
        Return the generalized polynomial chaos distribution associated with Jacobi polynomials.

        Returns
        -------
        gpc_dist : BetaDistribution
            Beta distribution with parameters (beta+1, alpha+1) on [-1, 1].
        """
        gpc_dist = BetaDistribution(self.beta+1, self.alpha+1, scale=2, loc=-1)
        return gpc_dist

    def _evaluate(self, deg, x, derivative=False):
        """
        Evaluate the Jacobi polynomial system up to degree `deg` at points `x`.

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
            Matrix of shape (len(x), deg+1) containing evaluated Jacobi polynomials or their derivatives.
        """
        x = np.asarray(x)
        if derivative:
            y = np.column_stack([np.polyder(self.generator(d, self.alpha, self.beta))(x) for d in range(deg+1)])
        else:
            y = np.column_stack([self.generator(d, self.alpha, self.beta)(x) for d in range(deg+1)])
        if self.orthonormal:
            mass = 2 ** (self.alpha + self.beta + 1) * beta_function(self.alpha + 1, self.beta + 1)
            y /= np.sqrt(_probability_sqnorm(self.generator, deg, mass, self.alpha, self.beta))
        return y

    def _sqnorm(self, deg):
        """
        Return the squared norm of the Jacobi polynomial of degree `deg`.

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
            mass = 2 ** (self.alpha + self.beta + 1) * beta_function(self.alpha + 1, self.beta + 1)
            y = _probability_sqnorm(self.generator, deg, mass, self.alpha, self.beta)
        return y

# import numpy as np
# from .PolynomialSystem import PolynomialSystem
# from .NormalizedPolynomials import NormalizedPolynomials
# from ..distributions import BetaDistribution

# class JacobiPolynomials(PolynomialSystem):
#     """ Implementation of Jacobi polynomial system.

#         Attributes
#         ----------
#         alpha : float
#             First parameter of Jacobi polynomials.

#         beta : float
#             Second parameter of Jacobi polynomials."""
    
#     def __init__(self, alpha, beta):
#         """ 

#         Parameters
#         ----------
#         alpha : float
#             First parameter of Jacobi polynomials.

#         beta : float
#             Second parameter of Jacobi polynomials."""
        
#         self.alpha = alpha
#         self.beta = beta
        
#     def normalized(self):
#         """ Return normalized version of Jacobi polynomial system.

#             Returns
#             -------
#             polysys: NormalizedPolynomials
#                 Wrapped normalized Jacobi polynomial system."""

#         polysys = NormalizedPolynomials(self)
#         return polysys

#     def recur_coeff(self, deg):
#         """ Return the recurrence coefficients for Jacobi polynomials.

#             Parameters
#             ----------
#             deg : int
#                 Degree up to which recurrence coefficients are computed.

#             Returns
#             -------
#             r : numpy.ndarray
#                 Recurrence coefficients for Jacobi polynomials."""
        
#         n = np.array(range(deg)).reshape(-1,1)
#         a = self.alpha
#         b = self.beta
        
#         b_n = (2*n+a+b+1)*(2*n+a+b+2)/( 2*(n+1)*(n+a+b+1) )
#         a_n = (a*a-b*b)*(2*n+a+b+1)/( 2*(n+1)*(n+a+b+1)*(2*n+a+b) )
#         c_n = (n+a)*(n+b)*(2*n+a+b+2)/( (n+1)*(n+a+b+1)*(2*n+a+b) )
        
#         if a+b==0 or a+b==-1:
#             b_n[0]=0.5*(a+b)+1
#             a_n[0]=0.5*(a-b)
#             c_n[0]=0
            
#         r = np.concatenate((a_n, b_n, c_n), axis=1)
            
#         return r

#     def sqnorm(self, n):
#         """ Return the squared norm of Jacobi polynomials of degree `n`.

#             Parameters
#             ----------
#             n : array_like
#                 Degrees for which the squared norms are computed.

#             Returns
#             -------
#             nrm2 : numpy.ndarray
#                 Array of squared norms corresponding to degrees in `n`."""
        
#         nrm2 = PolynomialSystem.sqnorm(self, n)
#         return nrm2

#     def weighting_dist(self):
#         """ Return the weighting distribution for Jacobi polynomials.

#             Returns
#             -------
#             dist : BetaDistribution
#                 Weighting distribution for Jacobi polynomials."""
        
#         dist = BetaDistribution(self.beta+1, self.alpha+1)
#         return dist