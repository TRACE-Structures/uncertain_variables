import numpy as np
from abc import ABC, abstractmethod

class PolynomialSystem(ABC):
    ''' Abstract base class for polynomial systems.
    
        Methods
        -------
        evaluate(deg, xi)
            Evaluate the polynomial system up to degree deg at points xi.

        sqnorm(n)
            Return the squared norm of the polynomials of degree n.

        sqnorm_by_rc(rc)
            Return the squared norm using recurrence coefficients rc.

        normalized()
            Return the normalized version of the polynomial system.'''
    
    def evaluate(self, deg, xi):
        '''Evaluate polynomial system up to degree `deg` at points `xi`.

            Parameters
            ----------
            deg: int
                Highest polynomial degree to evaluate.
            xi: array_like
                Points at which the polynomials are evaluated.

            Returns
            -------
            y_alpha_j: numpy.ndarray
                Matrix of shape (len(xi), deg+1) containing evaluated polynomials.'''

        k = np.size(xi)
        p = np.zeros([k, deg+2])
        p[:,0] = 0
        p[:,1] = 1
        r = self.recur_coeff(deg+1)
        for d in range(deg):
           p[:, d+2] = (r[d,0] + xi * r[d,1]) * p[:, d+1] - r[d,2] * p[:,d]
        y_alpha_j = p[:,1:]
        return y_alpha_j

    def sqnorm(self, n):
        """ Return the squared norm of the polynomials of degree `n`.

            Parameters
            ----------
            n : array_like
                Degrees for which the squared norms are computed.

            Returns
            -------
            nrm2 : numpy.ndarray
                Array of squared norms corresponding to degrees in `n`."""
        
        deg = max(n.flatten()) + 1
        r = self.recur_coeff(deg)
        nrm2 = self.sqnorm_by_rc(r)
        nrm2 = np.reshape([nrm2[n+1], len(n)])
        return nrm2

    def sqnorm_by_rc(self, rc):
        """ Return the squared norm using recurrence coefficients `rc`.

            Parameters
            ----------
            rc : array_like
                Recurrence coefficients of shape (deg, 3).

            Returns
            -------
            nrm2 : numpy.ndarray
                Array of squared norms for polynomials up to degree `deg`."""
        
        b = rc[:, 1]
        h = b[0] / b[1:]
        c = rc[1:, 2]
        nrm2 = np.concatenate(np.ones([1]),  h.flatten * np.cumprod(c.flatten()))
        return nrm2

    def normalized(self):
        '''Return normalized version of this polynomial system.

        Returns
        -------
        polysys: NormalizedPolynomials
            Wrapped normalized polynomial system.'''

        from .NormalizedPolynomials import NormalizedPolynomials
        polysys = NormalizedPolynomials(self)
        return polysys

    @abstractmethod
    def weighting_dist(self):
        ''' Abstract method to return the weighting distribution of the polynomial system.
            Subclasses must implement this method.'''
        
        pass

    @abstractmethod
    def recur_coeff(self, deg):
        ''' Abstract method to return the recurrence coefficients of the polynomial system.
            Subclasses must implement this method.'''
        
        pass


# class NormalizedPolynomials(PolynomialSystem):
#     """ Wrapper class for normalized polynomial systems.

#         Attributes
#         ----------
#         base_polysys : PolynomialSystem
#             The base polynomial system to be normalized.

#         Methods
#         -------
#         __init__(base_polysys)
#             Initialize the normalized polynomial system with a base polynomial system.

#         recur_coeff(deg)
#             Return the recurrence coefficients for the normalized polynomial system.

#         weighting_dist()
#             Return the weighting distribution of the base polynomial system."""
    
#     def __init__(self, base_polysys):
#         """ Initialize the normalized polynomial system with a base polynomial system.

#             Parameters
#             ----------
#             base_polysys : PolynomialSystem
#                 The base polynomial system to be normalized."""
        
#         self.base_polysys = base_polysys

#     def recur_coeff(self, deg):
#         """_summary_

#             Parameters
#             ----------
#             deg : int
#                 Degree up to which recurrence coefficients are computed.

#             Returns
#             -------
#             r : numpy.ndarray
#                 Recurrence coefficients for the normalized polynomial system."""
        
#         r = self.base_polysys.recur_coeff(deg)
#         n = np.array(range(deg))
#         z = np.concatenate((np.zeros([1]), np.sqrt(self.base_polysys.sqnorm(np.arange(0,deg+1)))), axis=0)
#         r = np.array([r[:, 0]*z[n + 1] / z[n + 2],
#             r[:, 1] * z[n + 1] / z[n + 2],
#             r[:, 2] * z[n] / z[n + 2]])
#         r = r.transpose()
#         return r

#     def weighting_dist(self):
#         """ Return the weighting distribution of the base polynomial system.

#             Returns
#             -------
#             dist : Distribution
#                 Weighting distribution of the base polynomial system."""
        
#         dist = self.base_polysys.weighting_dist()
#         return dist