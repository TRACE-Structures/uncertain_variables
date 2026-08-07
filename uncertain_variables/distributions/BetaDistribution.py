from .Distribution import Distribution, unwrap_if_scalar
import numpy as np
import scipy.special as sc
from .TranslatedDistribution import TranslatedDistribution

class BetaDistribution(Distribution):
    ''' Class for beta distribution.
    
        Attributes
        ----------
        a : float
            First shape parameter of the beta distribution.
            
        b : float
            Second shape parameter of the beta distribution.'''
        
    def __init__(self, a, b):
        ''' Initialize the beta distribution with shape parameters a and b.
        
            Parameters
            ----------
            a : float
                First shape parameter of the beta distribution.
                
            b : float
                Second shape parameter of the beta distribution.'''
        
        self.a = a
        self.b = b

    def __repr__(self):
        ''' Returns the string representation of the BetaDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the BetaDistribution object.'''
        
        repr_string = "Beta({}, {})".format(self.a, self.b)
        return repr_string
    
    def __eq__(self, other):
        ''' Check if two BetaDistribution objects are equal.
        
            Parameters
            ----------
            other : BetaDistribution
                Another BetaDistribution object to compare with.
                
            Returns
            -------
            is_equal : bool
                True if the two BetaDistribution objects are equal, False otherwise.'''
        
        if not isinstance(other, BetaDistribution):
            return False
        is_equal = (self.a == other.a) and (self.b == other.b)
        return is_equal
    
    def get_dist_type(self):
        ''' Return the type of the distribution.
        
            Returns
            -------
            dist_type : str
                Type of the distribution.'''
        
        dist_type = "beta"
        return dist_type
    
    def get_dist_params(self):
        ''' Return the parameters of the distribution.
        
            Returns
            -------
            params : tuple
                Parameters of the distribution (a, b).'''
        
        params = (self.a, self.b)
        return params

    def pdf(self, x):
        ''' Return the probability density function of the beta distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.
                
            Returns
            -------
            y : array_like
                Probability density function values at x.'''
        
        x = TranslatedDistribution.translate_points_backwards(x, -1, 2, 0)
        x = np.array(x)
        y = np.zeros(x.shape)
        ind = (x >= 0) & (x <= 1)
        y[ind] = (
            x[ind] ** (self.a - 1)
            * (1 - x[ind]) ** (self.b - 1)
            / sc.beta(self.a, self.b)
        )
        y = unwrap_if_scalar(y)
        return y

    def cdf(self, x):
        ''' Return the cumulative distribution function of the beta distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x.'''
        
        x = TranslatedDistribution.translate_points_backwards(x, -1, 2, 0)
        x = np.array(x)
        y = np.zeros(x.shape)
        ind = (x >= 0) & (x <= 1)
        y[ind] = sc.betainc(self.a, self.b, x[ind])
        y[x > 1] = 1
        y = unwrap_if_scalar(y)
        return y

    def invcdf(self, y):
        ''' Return the inverse cumulative distribution function of the beta distribution, evaluated at y.
            
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.
                
            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y.'''
        
        # TODO implementing the Matlab code
        y = np.array(y)
        x = np.full(y.shape, np.nan)
        ind = (y >= 0) & (y <= 1)
        x[ind] = sc.betaincinv(self.a, self.b, y[ind])
        x = TranslatedDistribution.translate_points_forward(x, -1, 2, 0)
        x = unwrap_if_scalar(x)
        return x

    def moments(self):
        ''' Return the first four moments of the beta distribution.
        
            Returns
            -------
            moments : list
                List containing the first four moments [mean, variance, skewness, kurtosis] of the beta distribution.'''
        
        mean = self.mean()
        var = self.var()
        skew = self.skew()
        kurt = self.kurt()

        moments = [mean, var, skew, kurt]
        moments = TranslatedDistribution.translate_moments(moments, -1, 2, 0)
        return moments

    def mean(self):
        """ Return the mean of the beta distribution.

            Returns
            -------
            mean : float
                Mean of the beta distribution."""
        
        mean = self.a / (self.a + self.b)
        return mean

    def var(self):
        """ Return the variance of the beta distribution.

            Returns
            -------
            var : float
                Variance of the beta distribution."""
        
        var = self.a * self.b / (((self.a + self.b) ** 2) * (self.a + self.b + 1))
        return var

    def skew(self):
        """ Return the skewness of the beta distribution.

            Returns
            -------
            skew : float
                Skewness of the beta distribution."""
        
        skew = (
            2
            * (self.b - self.a)
            * np.sqrt(self.a + self.b + 1)
            / ((self.a + self.b + 2) * np.sqrt(self.a * self.b))
        )
        return skew

    def kurt(self):
        """ Return the kurtosis of the beta distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the beta distribution."""
        
        kurt = (
            6
            * (
                self.a**3
                - (self.a**2) * (2 * self.b - 1)
                + (self.b**2) * (self.b + 1)
                - 2 * self.a * self.b * (self.b + 2)
            )
            / (self.a * self.b * (self.a + self.b + 2) * (self.a + self.b + 3))
        )
        return kurt

    def get_base_dist(self):
        """ Return the GPC base distribution.

            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution."""
        
        dist_germ = self
        return dist_germ

    def base2dist(self, y):
        """ Convert from base (germ) space to beta distribution space.

            Parameters
            ----------
            y : array_like
                Points in base (germ) space.

            Returns
            -------
            x : array_like
                Points in distribution space."""
        
        x = y
        return x

    def dist2base(self, x):
        """ Convert from beta distribution space to base (germ) space.

            Parameters
            ----------
            x : array_like
                Points in distribution space.

            Returns
            -------
            y : array_like
                Points in base (germ) space."""
        
        y = x
        return y

    def orth_polysys(self):
        """ Return the GPC polynomial system for the beta distribution.

            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the beta distribution."""
        
        from polysys import JacobiPolynomials

        polysys = JacobiPolynomials(self.b - 1, self.a - 1)
        return polysys

    def orth_polysys_syschar(self, normalized):
        """ Return the GPC polynomial system characteristic string for the beta distribution.

            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.

            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the beta distribution."""
        
        if self.a == -1 and self.b == 1:
            if normalized:
                polysys_char = "J"
            else:
                polysys_char = "j"
        else:
            polysys_char = []
        return polysys_char
