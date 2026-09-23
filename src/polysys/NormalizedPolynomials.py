import numpy as np
from .PolynomialSystem import PolynomialSystem

class NormalizedPolynomials(PolynomialSystem):
    """ Wrapper class for normalized polynomial systems.

        Attributes
        ----------
        base_polysys : PolynomialSystem
            The base polynomial system to be normalized."""
    
    def __init__(self, base_polysys):
        """ Initialize the normalized polynomial system with a base polynomial system.

            Parameters
            ----------
            base_polysys : PolynomialSystem
                The base polynomial system to be normalized."""
        
        self.base_polysys = base_polysys

    def recur_coeff(self, deg):
        """_summary_

            Parameters
            ----------
            deg : int
                Degree up to which recurrence coefficients are computed.

            Returns
            -------
            r : numpy.ndarray
                Recurrence coefficients for the normalized polynomial system."""
        
        r = self.base_polysys.recur_coeff(deg)
        n = np.array(range(deg))
        z = np.concatenate((np.zeros([1]), np.sqrt(self.base_polysys.sqnorm(np.arange(0,deg+1)))), axis=0)
        r = np.array([r[:, 0]*z[n + 1] / z[n + 2],
            r[:, 1] * z[n + 1] / z[n + 2],
            r[:, 2] * z[n] / z[n + 2]])
        r = r.transpose()
        return r

    def weighting_dist(self):
        """ Return the weighting distribution of the base polynomial system.

            Returns
            -------
            dist : Distribution
                Weighting distribution of the base polynomial system."""
        
        dist = self.base_polysys.weighting_dist()
        return dist