from .Distribution import Distribution, unwrap_if_scalar
import numpy as np
from scipy.stats import norm as sc_norm
from .NormalDistribution import NormalDistribution

class LogNormalDistribution(Distribution):
    ''' Class for log-normal distribution.
    
        Attributes
        ----------
        mu : float
            Mean of the underlying normal distribution.
            
        sigma : float
            Standard deviation of the underlying normal distribution.'''
    
    def __init__(self, mu=0, sigma=1):
        ''' Initialize the log-normal distribution with parameters mu and sigma.
        
            Parameters
            ----------
            mu : float, default = 0
                Mean of the underlying normal distribution.
                
            sigma : float, default = 1
                Standard deviation of the underlying normal distribution.'''
        
        if not (isinstance(mu, (int, float, np.number))):
            raise ValueError("Mean mu must be a number.")  
        if not (isinstance(sigma, (int, float, np.number)) and sigma > 0):
            raise ValueError("Standard deviation sigma must be a positive number.")
        if not np.isfinite(mu):
            raise ValueError("Mean mu must be a finite number.")
        if not np.isfinite(sigma):
            raise ValueError("Standard deviation sigma must be a finite number.")
        
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        ''' Returns the string representation of the LogNormalDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the LogNormalDistribution object.'''
        
        repr_string = "lnN({}, {:.2f})".format(self.mu, self.sigma**2)
        return repr_string
    
    def __eq__(self, other):
        ''' Check if two LogNormalDistribution objects are equal.
        
            Parameters
            ----------
            other : LogNormalDistribution
                Another LogNormalDistribution object to compare with.
                
            Returns
            -------
            is_equal : bool
                True if the two LogNormalDistribution objects are equal, False otherwise.'''
        
        if not isinstance(other, LogNormalDistribution):
            return False
        
        is_equal = (self.mu == other.mu) and (self.sigma == other.sigma)
        return is_equal
    
    def get_dist_type(self):
        """ Return the type of the distribution.

            Returns
            -------
            dist_type : str
                Type of the distribution."""

        dist_type = "lognorm"
        return dist_type
    
    def get_dist_params(self):
        """ Return the parameters of the distribution.

            Returns
            -------
            params : tuple
                Parameters of the distribution (mu, sigma)."""
        
        params = (self.mu, self.sigma)
        return params

    def pdf(self, x):
        ''' Return the probability density function of the log-normal distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.
                
            Returns
            -------
            y : array_like
                Probability density function values at x.'''
        
        x = np.asarray(x)
        y = np.zeros(x.shape)
        mu = self.mu
        sigma = self.sigma
        ind = x > 0
        root = (np.log(x[ind]) - mu) / sigma
        y_exp = root**2
        y_exp = -1 / 2 * y_exp
        y[ind] = np.exp(y_exp) / (x[ind] * sigma * np.sqrt(2 * np.pi))
        y = unwrap_if_scalar(y)
        return y

    # def logpdf(self, x):
    #     ''' Return the log of the probability density function of the log-normal distribution, evaluated at x.
        
    #         Parameters
    #         ----------
    #         x : array_like
    #             Points at which to evaluate the logpdf.
                
    #         Returns
    #         -------
    #         y : array_like
    #             Log probability density function values at x.'''
        
    #     y = np.zeros(x.shape)
    #     ind = x > 0
    #     mu = self.mu
    #     sigma = self.sigma
    #     root = (np.log(x[ind]) - mu) / sigma
    #     y = -1 / 2 * (root**2) - x[ind] * sigma * np.sqrt(2 * np.pi)
    #     return y

    def cdf(self, x):
        ''' Return the cumulative distribution function of the log-normal distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x.'''
        
        x = np.asarray(x)
        y = np.zeros(x.shape)
        ind = x > 0
        mu = self.mu
        sigma = self.sigma
        # y[ind] = 1 / 2 * (1 + sc.erf((np.log(x[ind]) - mu) / (sigma * np.sqrt(2))))
        y[ind] = sc_norm.cdf((np.log(x[ind]) - mu) / sigma)
        y = unwrap_if_scalar(y)
        return y

    def invcdf(self, y):
        ''' Return the inverse cumulative distribution function of the log-normal distribution, evaluated at y.
        
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.
                
            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y.'''
        
        y = np.asarray(y)
        x = np.full(y.shape, np.nan)
        ind = (y >= 0) & (y <= 1)
        mu = self.mu
        sigma = self.sigma
        # x[ind] = np.exp(mu + sigma * np.sqrt(2) * sc.erfinv(2 * y[ind] - 1))
        x[ind] = np.exp(mu + sigma * sc_norm.ppf(y[ind]))
        x = unwrap_if_scalar(x)
        return x

    def sample(self, n, method="MC", seed=None, **params):
        ''' Return n samples from the log-normal distribution.
        
            Parameters
            ----------
            n : int
                Number of samples to generate.
            method : str, optional
                Sampling method to use. Options are "MC" (Monte Carlo), "QMC_Halton" (Quasi-Monte Carlo using Halton sequence),
                "QMC_LHS" (Quasi-Monte Carlo using Latin Hypercube Sampling), and "QMC_Sobol" (Quasi-Monte Carlo using Sobol sequence).
                Default is "MC".
            seed : int, optional
                Seed for the random number generator. Default is None.
            **params : dict
                Additional parameters for the sampling method.
                
            Returns
            -------
            samples : array_like
                Generated samples from the log-normal distribution.'''
        
        from .UniformDistribution import UniformDistribution
        xi = UniformDistribution(0, 1).sample(n, method, seed=seed, **params)
        samples = self.invcdf(xi)
        return samples

    def mean(self):
        ''' Return the mean of the log-normal distribution.
        
            Returns
            -------
            mean : float
                Mean of the log-normal distribution.'''
        
        mean = np.exp(self.mu + (self.sigma**2) / 2)
        return mean

    def var(self):
        ''' Return the variance of the log-normal distribution.
        
            Returns
            -------
            var : float
                Variance of the log-normal distribution.'''
        
        var = (np.exp(self.sigma**2) - 1) * (np.exp(2 * self.mu + self.sigma**2))
        return var

    def skew(self):
        ''' Return the skewness of the log-normal distribution.
        
            Returns
            -------
            skew : float
                Skewness of the log-normal distribution.'''
        
        skew = (np.exp(self.sigma**2) + 2) * (np.sqrt(np.exp(self.sigma**2) - 1))
        return skew

    def kurt(self):
        ''' Return the kurtosis of the log-normal distribution.
        
            Returns
            -------
            kurt : float
                Kurtosis of the log-normal distribution.'''
        
        kurt = (
            np.exp(4 * self.sigma**2)
            + 2 * np.exp(3 * self.sigma**2)
            + 3 * np.exp(2 * self.sigma**2)
            - 6
        )
        return kurt

    def get_base_dist(self):
        ''' Return the GPC base distribution.
            
            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution.'''

        base = NormalDistribution(0, 1)
        return base

    def base2dist(self, y):
        ''' Convert from base (germ) space to log-normal distribution space.
        
            Parameters
            ----------
            y : array_like
                Points in base (germ) space.
                
            Returns
            -------
            x : array_like
                Points in log-normal distribution space.'''

        y = np.asarray(y)
        x = np.exp(y * self.sigma + self.mu)
        return x

    def dist2base(self, x):
        ''' Convert from log-normal distribution space to base (germ) space.
        
            Parameters
            ----------
            x : array_like
                Points in log-normal distribution space.
                
            Returns
            -------
            y : array_like
                Points in base (germ) space.'''

        x = np.asarray(x)
        # ignore RuntimeWarning in case x == 0
        with np.errstate(divide="ignore", invalid="ignore"): 
            y = (np.log(x) - self.mu) / self.sigma
        return y

    def stdnor2base(self, y):  # same as base2dist??
        ''' Convert from standard normal space to log-normal distribution space.
        
            Parameters
            ----------
            y : array_like
                Points in standard normal space.
                
            Returns
            -------
            x : array_like
                Points in log-normal distribution space.'''
        
        y = np.asarray(y)
        x = np.exp(y * self.sigma + self.mu)
        return x

    def base2stdnor(self, x):  # same as dist2base?
        ''' Convert from log-normal distribution space to standard normal space.
        
            Parameters
            ----------
            x : array_like
                Points in log-normal distribution space.
                
            Returns
            -------
            y : array_like
                Points in standard normal space.'''

        x = np.asarray(x)
        with np.errstate(divide="ignore", invalid="ignore"): 
            y = (np.log(x) - self.mu) / self.sigma
        return y

    def orth_polysys(self):
        ''' Return the GPC polynomial system for the log-normal distribution.
        
            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the log-normal distribution.'''
        
        # from polysys import HermitePolynomials

        # if self.mu == 0 and self.sigma == 1:
        #     polysys = HermitePolynomials()
        # else:
        #     polysys = Distribution.orth_polysys(self)
        # return polysys

        from ..polysys import HermitePolynomials

        if self.mu == 0 and self.sigma == 1:
            return HermitePolynomials()
        else:
            raise Exception(f"No polynomial system for this distribution ({self})")

    def orth_polysys_syschar(self, normalized):
        ''' Return the GPC polynomial system characteristic string for the log-normal distribution.
            
            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.
                
            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the log-normal distribution.'''
        
        # if self.mu == 0 and self.sigma == 1:
        #     if normalized:
        #         polysys_char = "h"
        #     else:
        #         polysys_char = "H"
        # else:
        #     polysys_char = []
        # return polysys_char

        if not self.mu == 0 or not self.sigma == 1:
            raise Exception(f"No polynomial system characteristic for this distribution ({self})")
            # OR return []
            
        if normalized:
            return "h"
        else:
            return "H"