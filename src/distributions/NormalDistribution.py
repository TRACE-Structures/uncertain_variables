from .Distribution import Distribution
import numpy as np
import scipy.special as sc

class NormalDistribution(Distribution):
    ''' Class for normal (Gaussian) distribution.
    
        Attributes
        ----------
        mu : float
            Mean of the normal distribution.
            
        sigma : float
            Standard deviation of the normal distribution.'''
    
    def __init__(self, mu=0, sigma=1):
        ''' Initialize the normal distribution with mean mu and standard deviation sigma.
        
            Parameters
            ----------
            mu : float, default = 0
                Mean of the normal distribution.

            sigma : float, default = 1
                Standard deviation of the normal distribution.'''
        
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        ''' Returns the string representation of the NormalDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the NormalDistribution object.'''
        
        repr_string = "N({}, {:.2f})".format(self.mu, self.sigma**2)
        return repr_string
    
    def __eq__(self, other):
        ''' Check if two NormalDistribution objects are equal.
        
            Parameters
            ----------
            other : NormalDistribution
                Another NormalDistribution object to compare with.
                
            Returns
            -------
            is_equal : bool
                True if the two NormalDistribution objects are equal, False otherwise.'''
        
        if not isinstance(other, NormalDistribution):
            return False
        is_equal = (self.mu == other.mu) and (self.sigma == other.sigma)
        return is_equal

    def get_dist_type(self):
        """ Return the type identifier of this distribution.

            Returns
            -------
            str
                Always returns ``'translated'``."""

        return "norm"
    
    def get_dist_params(self):
        return self.mu, self.sigma

    def pdf(self, x):
        ''' Return the probability density function of the normal distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.
                
            Returns
            -------
            y : array_like
                Probability density function values at x.'''
        
        mu = self.mu
        sigma = self.sigma
        root = (x - mu) / sigma
        y_exp = root**2
        y_exp = -1 / 2 * y_exp
        y = np.exp(y_exp) / (sigma * np.sqrt(2 * np.pi))
        return y

    def logpdf(self, x):
        ''' Return the log of the probability density function of the normal distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the logpdf.
                
            Returns
            -------
            y : array_like
                Log probability density function values at x.'''
        
        mu = self.mu
        sigma = self.sigma
        root = (x - mu) / sigma
        y = -1 / 2 * (root**2) - np.log(sigma * np.sqrt(2 * np.pi))
        return y

    def cdf(self, x):
        ''' Return the cumulative distribution function of the normal distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x.'''
        
        mu = self.mu
        sigma = self.sigma
        y = 1 / 2 * (1 + sc.erf((x - mu) / (sigma * np.sqrt(2))))
        return y

    def invcdf(self, y):
        ''' Return the inverse cumulative distribution function of the normal distribution, evaluated at y.
        
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.
                
            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y.'''
        
        mu = self.mu
        sigma = self.sigma
        y = np.array(y)
        x = np.full(y.shape, np.nan)  # original
        ind = (y >= 0) & (y <= 1)
        x[ind] = mu + sigma * np.sqrt(2) * sc.erfinv(2 * y[ind] - 1)
        x = x / 1
        return x

    def sample(self, n, method="MC", **params): # ??? TODO
        ''' Return n samples from the normal distribution.

            Parameters
            ----------
            n : int
                Number of samples to generate.
                
            method : str, optional
                Sampling method to use (default is 'MC' for Monte Carlo).
            
            **params : dict
                Additional parameters for the sampling method.
                
            Returns
            -------
            samples : array_like
                Generated samples from the normal distribution.'''
        
        from .UniformDistribution import UniformDistribution
        if method == "MC":
            xi = np.random.randn(n)
        else:
            xi = UniformDistribution().sample(n, method, **params)
        samples = (xi * self.sigma) + self.mu
        return samples

    def mean(self):
        """ Return the mean of the normal distribution.

            Returns
            -------
            mean : float
                Mean of the normal distribution."""
        
        mean = self.mu
        return mean

    def var(self):
        """ Return the variance of the normal distribution.

            Returns
            -------
            var : float
                Variance of the normal distribution."""

        var = self.sigma * self.sigma
        return var

    def skew(self):
        """ Return the skewness of the normal distribution.

            Returns
            -------
            skew : float
                Skewness of the normal distribution."""
        
        skew = 0
        return skew

    def kurt(self):
        """ Return the kurtosis of the normal distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the normal distribution."""

        kurt = 0
        return kurt

    def get_base_dist(self):
        ''' Return the GPC base distribution.

            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution.'''
        
        dist_germ = NormalDistribution(0, 1)
        return dist_germ

    def translate(self, shift, scale):
        ''' Return a translated and scaled version of the normal distribution.
        
            Parameters
            ----------
            shift : float
                Shift to apply to the distribution.

            scale : float
                Scale to apply to the distribution.
            
            Returns
            -------
            new_dist : NormalDistribution
                Translated and scaled normal distribution.'''
        
        new_dist = NormalDistribution(self.mu + shift, self.sigma * scale)
        return new_dist

    def base2dist(self, y):
        """ Convert from base (germ) space to normal distribution space.

            Parameters
            ----------
            y : array_like
                Points in base (germ) space.

            Returns
            -------
            x : array_like
                Points in normal distribution space."""

        x = self.mu + y * self.sigma
        return x
    
    def dist2base(self, x):
        """ Return from normal distribution space to base (germ) space.

            Parameters
            ----------
            x : array_like
                Points in normal distribution space.

            Returns
            -------
            y : array_like
                Points in base (germ) space."""

        y = (x - self.mu) / self.sigma
        return y

    def orth_polysys(self):
        ''' Return the GPC polynomial system for the normal distribution.
        
            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the normal distribution.'''
        
        from polysys import HermitePolynomials

        if self.mu == 0 and self.sigma == 1:
            polysys = HermitePolynomials()
        else:
            polysys = Distribution.orth_polysys(self)
        return polysys

    def orth_polysys_syschar(self, normalized):
        ''' Return the GPC polynomial system characteristic string for the normal distribution.
        
            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.
                
            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the normal distribution.'''
        
        if self.mu == 0 and self.sigma == 1:
            if normalized:
                polysys_char = "h"
            else:
                polysys_char = "H"
        else:
            polysys_char = []
        return polysys_char
