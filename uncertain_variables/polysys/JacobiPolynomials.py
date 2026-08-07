import numpy as np
from .PolynomialSystem import PolynomialSystem
from .NormalizedPolynomials import NormalizedPolynomials
from ..distributions import BetaDistribution

class JacobiPolynomials(PolynomialSystem):
    """ Implementation of Jacobi polynomial system.

        Attributes
        ----------
        alpha : float
            First parameter of Jacobi polynomials.

        beta : float
            Second parameter of Jacobi polynomials."""
    
    def __init__(self, alpha, beta):
        """ 

        Parameters
        ----------
        alpha : float
            First parameter of Jacobi polynomials.

        beta : float
            Second parameter of Jacobi polynomials."""
        
        self.alpha = alpha
        self.beta = beta
        
    def normalized(self):
        """ Return normalized version of Jacobi polynomial system.

            Returns
            -------
            polysys: NormalizedPolynomials
                Wrapped normalized Jacobi polynomial system."""

        polysys = NormalizedPolynomials(self)
        return polysys

    def recur_coeff(self, deg):
        """ Return the recurrence coefficients for Jacobi polynomials.

            Parameters
            ----------
            deg : int
                Degree up to which recurrence coefficients are computed.

            Returns
            -------
            r : numpy.ndarray
                Recurrence coefficients for Jacobi polynomials."""
        
        n = np.array(range(deg)).reshape(-1,1)
        a = self.alpha
        b = self.beta
        
        b_n = (2*n+a+b+1)*(2*n+a+b+2)/( 2*(n+1)*(n+a+b+1) )
        a_n = (a*a-b*b)*(2*n+a+b+1)/( 2*(n+1)*(n+a+b+1)*(2*n+a+b) )
        c_n = (n+a)*(n+b)*(2*n+a+b+2)/( (n+1)*(n+a+b+1)*(2*n+a+b) )
        
        if a+b==0 or a+b==-1:
            b_n[0]=0.5*(a+b)+1
            a_n[0]=0.5*(a-b)
            c_n[0]=0
            
        r = np.concatenate((a_n, b_n, c_n), axis=1)
            
        return r

    def sqnorm(self, n):
        """ Return the squared norm of Jacobi polynomials of degree `n`.

            Parameters
            ----------
            n : array_like
                Degrees for which the squared norms are computed.

            Returns
            -------
            nrm2 : numpy.ndarray
                Array of squared norms corresponding to degrees in `n`."""
        
        nrm2 = PolynomialSystem.sqnorm(self, n)
        return nrm2

    def weighting_dist(self):
        """ Return the weighting distribution for Jacobi polynomials.

            Returns
            -------
            dist : BetaDistribution
                Weighting distribution for Jacobi polynomials."""
        
        dist = BetaDistribution(self.beta+1, self.alpha+1)
        return dist