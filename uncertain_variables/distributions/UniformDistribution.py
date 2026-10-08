import numpy as np

from Distribution import Distribution
from scipy.stats import uniform

from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import Sobol


class UniformDistribution(Distribution):
    """ Class for uniform distribution.
    
        Attributes
        ----------
        a : float
            Lower bound of the uniform distribution.
            
        b : float
            Upper bound of the uniform distribution.
            
        loc : float
            Location parameter of the uniform distribution (same as a).
        
        scale : float
            Scale parameter of the uniform distribution (same as b - a)."""
    
    def __init__(self, a=0, b=1):
        """ Initialize the uniform distribution with bounds a and b.
        
            Parameters
            ----------
            a : float, default = -1
                Lower bound of the uniform distribution.
                
            b : float, default = 1
                Upper bound of the uniform distribution."""

        # assert b > a

        if not (isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)) and b > a):
            raise ValueError("Uniform bounds must be numeric with b > a.")
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError("Uniform bounds must be finite numeric values.")

        self.a = a
        self.b = b
        
        self.loc = a
        self.scale = b - a

    def __repr__(self):
        """ Returns the string representation of the UniformDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the UniformDistribution object."""
        
        repr_string = "U({:.2f}, {:.2f})".format(self.a, self.b)
        return repr_string
    
    def __eq__(self, other):        
        """ Check if two UniformDistribution objects are equal.
        
            Parameters
            ----------
            other : UniformDistribution
                Another UniformDistribution object to compare with.
                
            Returns
            -------
            is_equal : bool
                True if the two UniformDistribution objects are equal, False otherwise."""
        
        if not isinstance(other, UniformDistribution):
            return False
        
        is_equal = (self.a == other.a) and (self.b == other.b)
        return is_equal

    def get_dist_type(self):
        """ Return the type of the uniform distribution.
        
            Returns
            -------
            dist_type : str
                Type of the distribution."""
        
        dist_type = "unif"
        return dist_type
    
    def get_dist_params(self):
        """ Return the parameters of the uniform distribution.
        
            Returns
            -------
            params : array_like of shape (2,)
                Distribution parameters in the order [a, b], where a is
                the lower bound and b is the upper bound of the distribution."""
        
        params = np.array([self.a, self.b])
        return params

    def pdf(self, x):
        """ Return the probability density function of the uniform distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.
                
            Returns
            -------
            y : array_like
                Probability density function values at x."""    
        
        # a = self.a
        # b = self.b
        # y = 1 / (b - a) * np.ones(np.size(x))
        # y[x < a] = 0
        # y[x > b] = 0
        # y = unwrap_if_scalar(y)
        # return y

        y = uniform.pdf(x, loc=self.loc, scale=self.scale)
        return y

    # def logpdf(self, x):
    #     """ Return the log of the probability density function of the uniform distribution, evaluated at x.
        
    #         Parameters
    #         ----------
    #         x : array_like
    #             Points at which to evaluate the logpdf.
                
    #         Returns
    #         -------
    #         y : array_like
    #             Log probability density function values at x."""    
        
    #     a = self.a
    #     b = self.b
    #     pdf = self.pdf(x)
    #     pdf = np.array(pdf)  # OR
    #     pdf = pdf.reshape(x.shape)
    #     y = np.zeros(x.shape)
    #     for i in range(len(x)):
    #         if pdf[i] == 0:
    #             y[i] = -np.inf
    #         else:
    #             y[i] = np.log(pdf[i])
    #     return y

    def cdf(self, x):
        """ Return the cumulative distribution function of the uniform distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x."""        
        
        # a = self.a
        # b = self.b
        # y = (x - a) / (b - a)
        # y = np.clip(y, 0, 1)
        # return y

        y = uniform.cdf(x, loc=self.loc, scale=self.scale)
        return y

    def invcdf(self, y):
        """ Return the inverse cumulative distribution function of the uniform distribution, evaluated at y.
            
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.
                
            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y."""    

        # a = self.a
        # b = self.b
        # y = np.array(y)
        # x = np.full(np.size(y), np.nan)
        # ind = (y >= 0) & (y <= 1)
        # x[ind] = a + (b - a) * y[ind]
        # x = unwrap_if_scalar(x)
        # return x

        x = uniform.ppf(y, loc=self.loc, scale=self.scale)
        return x

    def mean(self):
        """ Return the mean of the uniform distribution.
        
            Returns
            -------
            mean : float
                Mean of the uniform distribution."""    
        
        mean = uniform.mean(loc=self.loc, scale=self.scale)
        return mean

    def var(self):
        """ Return the variance of the uniform distribution.
            
            Returns
            -------
            var : float
                Variance of the uniform distribution."""    
        
        var = uniform.var(loc=self.loc, scale=self.scale)
        return var

    def skew(self):
        """ Return the skewness of the uniform distribution.

            Returns
            -------
            skew : float
                Skewness of the uniform distribution."""    
        
        skew = 0
        return skew

    def kurt(self):
        """ Return the excess kurtosis of the uniform distribution.

            Returns
            -------
            kurt : float
                Excess kurtosis of the uniform distribution."""    
        
        kurt = -6 / 5
        return kurt

    def sample(self, n, method="MC", seed=None, **params):
        """ Generate random samples from the uniform distribution using the specified sampling method.

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
                Generated random samples from the uniform distribution. """
        
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Number of samples must be a positive integer.")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer or None.")
        if not np.isfinite(n):
            raise ValueError("Number of samples must be a finite value.")

        if method == "MC":
            rng = np.random.default_rng(seed)
            y = rng.random(n)
        elif method == "QMC_Halton":
            sampler = Halton(d=1, seed=seed)
            y = sampler.random(n)
        elif method == "QMC_LHS":
            sampler = LHS(d=1, seed=seed)
            y = sampler.random(n)
        elif method == "QMC_Sobol":
            sampler = Sobol(d=1, seed=seed)
            y = sampler.random(n)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        return self.invcdf(y)

    def translate(self, shift, scale):
        """ Return a translated and scaled version of the uniform distribution.

            Parameters
            ----------
            shift : float
                Shift to apply to the distribution.
            scale : float
                Scale to apply to the distribution. Must be a positive numeric value.

            Returns
            -------
            new_dist : UniformDistribution
                Translated and scaled uniform distribution."""

        if not (isinstance(shift, (int, float, np.number))):
            raise ValueError("Shift must be a numeric value.")
        if not (isinstance(scale, (int, float, np.number)) and scale > 0):
            raise ValueError("Scale must be a positive numeric value.")
        if not np.isfinite(shift):
            raise ValueError("Shift must be a finite value.")
        if not np.isfinite(scale):
            raise ValueError("Scale must be a finite value.")

        center = (self.a + self.b) / 2 + shift
        half_width = scale * (self.b - self.a) / 2
        return UniformDistribution(center - half_width, center + half_width)

    def get_shift(self):
        """ Return the shift of the uniform distribution from the (-1, 1) base form.
            
            Returns
            -------
            shift : float
                Shift of the uniform distribution."""

        shift = self.loc + self.scale / 2
        return shift

    def get_scale(self):
        """ Return the scale of the uniform distribution, i.e., the width of the distribution.

            Returns
            -------
            scale : float
                Scale of the uniform distribution."""

        scale = self.scale
        return scale

    def fix_moments(self, mean, var):
        """ Fix the uniform distribution to have specified mean and variance.
        
            Parameters
            ----------
            mean : float
                Desired mean of the distribution.
                
            var : float
                Desired variance of the distribution.
                
            Returns
            -------
            new_dist : UniformDistribution
                Translated and scaled uniform distribution with specified moments."""

        if not (isinstance(mean, (int, float, np.number))):
            raise ValueError("Mean must be a numeric value.")
        if not (isinstance(var, (int, float, np.number)) and var > 0):
            raise ValueError("Variance must be a positive numeric value.")
        if not np.isfinite(mean):
            raise ValueError("Mean must be a finite value.")
        if not np.isfinite(var):
            raise ValueError("Variance must be a finite value.")

        a = mean - np.sqrt(3 * var)
        b = mean + np.sqrt(3 * var)
        new_dist = UniformDistribution(a, b)
        return new_dist

    def get_base_dist(self):
        """ Return the GPC base distribution for the uniform distribution.

            Returns
            -------
            dist_germ : UniformDistribution
                GPC base distribution."""
        
        dist_germ = UniformDistribution(-1, 1)
        return dist_germ

    # def base2dist(self, y):
    #     """ Convert from base (germ) space to uniform distribution space.

    #         Parameters
    #         ----------
    #         y : array_like
    #             Points in base (germ) space.

    #         Returns
    #         -------
    #         x : array_like
    #             Points in distribution space."""

        
    #     y = np.asarray(y)
    #     x = self.mean() + y * (self.b - self.a) / 2
    #     return x

    # def dist2base(self, x):
    #     """ Convert from uniform distribution space to base (germ) space.

    #         Parameters
    #         ----------
    #         x : array_like
    #             Points in distribution space.

    #         Returns
    #         -------
    #         y : array_like
    #             Points in base (germ) space."""
        
    #     x = np.asarray(x)
    #     y = (x - self.mean()) * 2 / (self.b - self.a)
    #     return y

    def orth_polysys(self):
        """ Return the GPC polynomial system for the uniform distribution.

            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the uniform distribution."""
        
        # from polysys import LegendrePolynomials

        # if self.a == -1 and self.b == 1:
        #     polysys = LegendrePolynomials()
        # else:
        #     polysys = Distribution.orth_polysys(self)
        # return polysys

        from ..polysys import LegendrePolynomials

        if self.a == -1 and self.b == 1:
            return LegendrePolynomials()
        else: 
            raise Exception(f"No polynomial system for this distribution ({self})") 

    def orth_polysys_syschar(self, normalized):
        """ Return the GPC polynomial system characteristic string for the uniform distribution.

            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.

            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the uniform distribution."""
        
        # if self.a == -1 and self.b == 1:
        #     if normalized:
        #         polysys_char = "p"
        #     else:
        #         polysys_char = "P"
        # else:
        #     polysys_char = []
        # return polysys_char

        if not (self.a == -1 and self.b == 1):
            raise Exception(f"No polynomial system for this distribution ({self})")
            # OR return []

        if normalized == True:
            return "p"
        else:
            return "P"

    # def get_bounds(self, delta=0):
    #     """ Return the bounds of the uniform distribution.

    #         Parameters
    #         ----------
    #         delta : int, optional
    #             Expansion factor for the bounds (default is 0).

    #         Returns
    #         -------
    #         bounds : numpy.ndarray
    #             Array containing the lower and upper bounds of the uniform distribution."""
        
    #     if not isinstance(delta, (int, float, np.number)):
    #         raise ValueError("delta must be a numeric value")
        
    #     a = self.a
    #     b = self.b
    #     ab = b - a

    #     # The modified function contracts the bounds by delta instead of expanding them
    #     # bounds = np.array([a - ab * delta, b + ab * delta])
        
    #     bounds = np.array([a + ab * delta, b - ab * delta])
    #     return bounds
