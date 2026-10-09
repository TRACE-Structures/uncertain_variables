import numpy as np

from uncertain_variables.distributions.Distribution import Distribution
from uncertain_variables.distributions.UniformDistribution import UniformDistribution
from scipy.stats import lognorm


class LogNormalDistribution(Distribution):
    """ Class for log-normal distribution.
    
        Attributes
        ----------
        mu : float
            Mean of the underlying normal distribution.
            
        sigma : float
            Standard deviation of the underlying normal distribution.
            
        s : float
            Shape parameter of the log-normal distribution, equivalent to 
            sigma of the underlying normal distribution.

        scale : float
            Scale parameter of the log-normal distribution, equivalent to 
            exp(mu) of the underlying normal distribution.

        loc : float
            Location parameter of the log-normal distribution, typically set to 0."""
    
    def __init__(self, mu=0, sigma=1, loc=0):
        """ Initialize the log-normal distribution with parameters mu and sigma.
        
            Parameters
            ----------
            mu : float, default = 0
                Mean of the underlying normal distribution.
                
            sigma : float, default = 1
                Standard deviation of the underlying normal distribution.
                
            loc : float, default = 0
                Location parameter of the log-normal distribution, typically set to 0."""

        # assert sigma > 0
        
        if not (isinstance(mu, (int, float, np.number))):
            raise ValueError("Mean mu must be a number.")  
        if not (isinstance(sigma, (int, float, np.number)) and sigma > 0):
            raise ValueError("Standard deviation sigma must be a positive number.")
        if not (isinstance(loc, (int, float, np.number))):
            raise ValueError("Location loc must be a number.")  
        if not np.isfinite(loc):
            raise ValueError("Location loc must be a finite number.")
        if not np.isfinite(mu):
            raise ValueError("Mean mu must be a finite number.")
        if not np.isfinite(sigma):
            raise ValueError("Standard deviation sigma must be a finite number.")
        
        self.mu = mu
        self.sigma = sigma
        self.loc = loc

        self.s = sigma
        self.scale = np.exp(mu)

    def __repr__(self):
        """ Returns the string representation of the LogNormalDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the LogNormalDistribution object."""   
        
        repr_string = "lnN({:.2f}, {:.2f}), loc={:.2f}".format(self.mu, self.sigma**2, self.loc)
        return repr_string
    
    def __eq__(self, other):
        """ Check if two LogNormalDistribution objects are equal.
        
            Parameters
            ----------
            other : LogNormalDistribution
                Another LogNormalDistribution object to compare with.
                
            Returns
            -------
            is_equal : bool
                True if the two LogNormalDistribution objects are equal, False otherwise."""
        
        if not isinstance(other, LogNormalDistribution):
            return False
        
        is_equal = (self.mu == other.mu) and (self.sigma == other.sigma) and (self.loc == other.loc)
        return is_equal
    
    def get_dist_type(self):
        """ Return the type of the log-normal distribution.

            Returns
            -------
            dist_type : str
                Type of the distribution."""

        dist_type = "lognorm"
        return dist_type
    
    def get_dist_params(self):
        """ Return the parameters of the log-normal distribution.

            Returns
            -------
            params : array_like of shape (3,)
                Distribution parameters [mu, sigma, loc], where mu is 
                the mean of the underlying normal distribution, 
                sigma is the standard deviation of the underlying 
                normal distribution, and loc is the location parameter."""
        
        params = np.array([self.mu, self.sigma, self.loc])
        return params

    def pdf(self, x):
        """ Return the probability density function of the log-normal distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.
                
            Returns
            -------
            y : array_like
                Probability density function values at x."""
        
        # x = np.asarray(x)
        # y = np.zeros(x.shape)
        # mu = self.mu
        # sigma = self.sigma
        # ind = x > 0
        # root = (np.log(x[ind]) - mu) / sigma
        # y_exp = root**2
        # y_exp = -1 / 2 * y_exp
        # y[ind] = np.exp(y_exp) / (x[ind] * sigma * np.sqrt(2 * np.pi))
        # return y

        y = lognorm.pdf(x, s=self.sigma, scale=self.scale, loc=self.loc)
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
        """ Return the cumulative distribution function of the log-normal distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x."""
        
        # x = np.array(x)
        # y = np.zeros(x.shape)
        # ind = x > 0
        # mu = self.mu
        # sigma = self.sigma
        # y[ind] = 1 / 2 * (1 + sc.erf((np.log(x[ind]) - mu) / (sigma * np.sqrt(2))))
        # y = unwrap_if_scalar(y)
        # return y

        y = lognorm.cdf(x, s=self.sigma, scale=self.scale, loc=self.loc)
        return y

    def invcdf(self, y):
        """ Return the inverse cumulative distribution function of the log-normal distribution, evaluated at y.
        
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.
                
            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y."""
        
        # y = np.array(y)
        # x = np.full(y.shape, np.nan)
        # ind = (y >= 0) & (y <= 1)
        # mu = self.mu
        # sigma = self.sigma
        # x[ind] = np.exp(mu + sigma * np.sqrt(2) * sc.erfinv(2 * y[ind] - 1))
        # return x

        x = lognorm.ppf(y, s=self.sigma, scale=self.scale, loc=self.loc)
        return x

    def mean(self):
        """ Return the mean of the log-normal distribution.
        
            Returns
            -------
            mean : float
                Mean of the log-normal distribution."""
        
        mean = lognorm.mean(s=self.sigma, scale=self.scale, loc=self.loc)
        return mean

    def var(self):
        """ Return the variance of the log-normal distribution.
        
            Returns
            -------
            var : float
                Variance of the log-normal distribution."""
        
        var = lognorm.var(s=self.sigma, scale=self.scale, loc=self.loc)
        return var

    def skew(self):
        """ Return the skewness of the log-normal distribution.
        
            Returns
            -------
            skew : float
                Skewness of the log-normal distribution."""
        
        skew = lognorm.stats(s=self.sigma, scale=self.scale, loc=self.loc, moments='s')
        return skew

    def kurt(self):
        """ Return the kurtosis of the log-normal distribution.
        
            Returns
            -------
            kurt : float
                Kurtosis of the log-normal distribution."""
        
        kurt = lognorm.stats(s=self.sigma, scale=self.scale, loc=self.loc, moments='k')
        return kurt

    def sample(self, n, method="MC", seed=None, **params):
        """ Return n samples from the log-normal distribution.
        
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
                Generated samples from the log-normal distribution."""
        
        # from .UniformDistribution import UniformDistribution
        # if method == "MC":
        #     xi = np.random.randn(n)
        # else:
        #     xi = UniformDistribution().sample(n, method, **params)
        # samples = np.exp((xi * self.sigma) + self.mu)
        # return samples

        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Number of samples must be a positive integer.")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer or None.")
        if not np.isfinite(n):
            raise ValueError("Number of samples must be a finite value.")

        y = UniformDistribution(0, 1).sample(n, method, seed=seed, **params)
        samples = self.invcdf(y)
        return samples

    def translate(self, shift, scale):
        """ Return a translated and scaled log-normal distribution.

            The transformation is defined as

            Y = scale * X + shift,

            where X is the original random variable.
        
            Parameters
            ----------
            shift : float
                Shift to apply to the distribution.

            scale : float
                Scale to apply to the distribution.
            
            Returns
            -------
            new_dist : LogNormalDistribution
                Translated and scaled log-normal distribution."""

        if not (isinstance(shift, (int, float, np.number))):
            raise ValueError("Shift must be a numeric value.")
        if not (isinstance(scale, (int, float, np.number)) and scale > 0):
            raise ValueError("Scale must be a positive numeric value.")
        if not np.isfinite(shift):
            raise ValueError("Shift must be a finite value.")
        if not np.isfinite(scale):
            raise ValueError("Scale must be a finite value.")

        new_mu = self.mu + np.log(scale)
        new_sigma = self.sigma
        new_loc = self.loc * scale + shift
        return LogNormalDistribution(new_mu, new_sigma, new_loc)

    def get_shift(self):
        """ Return the shift (location) parameter of the log-normal distribution.

            Returns
            -------
            shift : float
                Shift (location) parameter of the log-normal distribution."""

        shift = self.loc
        return shift

    def get_scale(self):
        """ Return the scale (standard deviation) parameter of the log-normal distribution.

            Returns
            -------
            scale : float
                Scale (standard deviation) parameter of the log-normal distribution."""

        scale = self.sigma
        return scale

    def fix_moments(self, mean, var, loc=0):
        """ Fix the log-normal distribution to have specified mean, variance, and location.

            Parameters
            ----------
            mean : float
                Desired mean of the distribution.
            var : float
                Desired variance of the distribution.
            loc : float, default=0
                Location parameter of the distribution.

            Returns
            -------
            new_dist : LogNormalDistribution
                Log-normal distribution with the specified mean, variance, and location."""
            
        if not isinstance(mean, (int, float, np.number)):
            raise ValueError("Mean must be a numeric value.")
        if not isinstance(var, (int, float, np.number)) or var <= 0:
            raise ValueError("Variance must be a positive numeric value.")
        if not isinstance(loc, (int, float, np.number)):
            raise ValueError("Location must be a numeric value.")

        if mean <= loc:
            raise ValueError("Mean must be greater than loc.")

        if not np.isfinite(mean):
            raise ValueError("Mean must be a finite value.")
        if not np.isfinite(var):
            raise ValueError("Variance must be a finite value.")
        if not np.isfinite(loc):
            raise ValueError("Location must be a finite value.")

        m = mean - loc

        if m <= 0:
            raise ValueError("Mean must be greater than loc.")

        new_sigma = np.sqrt(np.log1p(var / m**2))
        new_mu = np.log(m) - new_sigma**2 / 2

        return LogNormalDistribution(new_mu, new_sigma, loc)
    
    def get_base_dist(self):
        """Return the GPC base distribution for the log-normal distribution.

        Raises
        ------
        Exception
            No orthogonal polynomial system exists for the log-normal
            distribution, so it cannot be used as a GPC base distribution."""

        raise Exception(f"No polynomial system for this distribution ({self})")

    # def base2dist(self, y):
    #     """ Convert from base (germ) space to log-normal distribution space.
        
    #         Parameters
    #         ----------
    #         y : array_like
    #             Points in base (germ) space.
                
    #         Returns
    #         -------
    #         x : array_like
    #             Points in log-normal distribution space."""

    #     y = np.asarray(y)
    #     x = np.exp(y * self.sigma + self.mu)
    #     return x

    # def dist2base(self, x):
    #     """ Convert from log-normal distribution space to base (germ) space.
        
    #         Parameters
    #         ----------
    #         x : array_like
    #             Points in log-normal distribution space.
                
    #         Returns
    #         -------
    #         y : array_like
    #             Points in base (germ) space."""

    #     x = np.asarray(x)
    #     # ignore RuntimeWarning in case x == 0
    #     with np.errstate(divide="ignore", invalid="ignore"): 
    #         y = (np.log(x) - self.mu) / self.sigma
    #     return y

    # def stdnor2base(self, y):  # same as base2dist??
    #     """ Convert from standard normal space to log-normal distribution space.
        
    #         Parameters
    #         ----------
    #         y : array_like
    #             Points in standard normal space.
                
    #         Returns
    #         -------
    #         x : array_like
    #             Points in log-normal distribution space."""
        
    #     y = np.asarray(y)
    #     x = np.exp(y * self.sigma + self.mu)
    #     return x

    # def base2stdnor(self, x):  # same as dist2base?
    #     """ Convert from log-normal distribution space to standard normal space.
        
    #         Parameters
    #         ----------
    #         x : array_like
    #             Points in log-normal distribution space.
                
    #         Returns
    #         -------
    #         y : array_like
    #             Points in standard normal space."""

    #     x = np.asarray(x)
    #     with np.errstate(divide="ignore", invalid="ignore"): 
    #         y = (np.log(x) - self.mu) / self.sigma
    #     return y

    def orth_polysys(self):
        """ Return the GPC polynomial system for the log-normal distribution.
        
            Raises
            ------
            Exception
                No orthogonal polynomial system exists for the log-normal
                distribution, so it cannot be used as a GPC base distribution."""

        raise Exception(f"No polynomial system for this distribution ({self})")


    def orth_polysys_syschar(self, normalized):
        """ Return the GPC polynomial system characteristic string for the log-normal distribution.
            
            Raises
            ------
            Exception
                No orthogonal polynomial system exists for the log-normal
                distribution, so it cannot be used as a GPC base distribution."""

        raise Exception(f"No polynomial system for this distribution ({self})")
