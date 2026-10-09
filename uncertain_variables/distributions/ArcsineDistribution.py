import numpy as np

from uncertain_variables.distributions.Distribution import Distribution
from uncertain_variables.distributions.UniformDistribution import UniformDistribution
from scipy.stats import arcsine

class ArcsineDistribution(Distribution):
    """Class for arcsine distribution.
    
        Attributes
        ----------
        scale : float
            Scale parameter of the arcsine distribution. The length of the interval, default is 1.

        loc : float
            Location parameter of the arcsine distribution. The starting point of the interval, default is 0."""

    def __init__(self, scale=1, loc=0):
        """
        Initialize the arcsine distribution with scale and location parameters.
        
            Parameters
            ----------
            scale : float, default = 1
                Scale parameter of the arcsine distribution. The length of the interval, default is 1.

            loc : float, default = 0
                Location parameter of the arcsine distribution. The starting point of the interval, default is 0."""
        
        if not (isinstance(scale, (int, float, np.number)) and scale > 0):
            raise ValueError("Parameter 'scale' must be a positive number.")
        if not (isinstance(loc, (int, float, np.number))):
            raise ValueError("Parameter 'loc' must be a number.")
        if not np.isfinite(scale):
            raise ValueError("Parameter 'scale' must be a finite number.")
        if not np.isfinite(loc):
            raise ValueError("Parameter 'loc' must be a finite number.")

        self.scale = scale
        self.loc = loc
    
    def __repr__(self):
        """ Returns the string representation of the ArcsineDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the ArcsineDistribution object."""

        repr_string = "Arcsine() scale={:.2f}, loc={:.2f})".format(self.scale, self.loc)
        return repr_string

    def __eq__(self, other):
        """Check if two ArcsineDistribution objects are equal.
        
            Parameters
            ----------
            other : ArcsineDistribution
                Another ArcsineDistribution object to compare with.
                
            Returns
            -------
            is_equal : bool
                True if the two ArcsineDistribution objects are equal, False otherwise."""

        if not isinstance(other, ArcsineDistribution):
            return False
        
        is_equal = (self.scale == other.scale) and (self.loc == other.loc)
        return is_equal
    
    def get_dist_type(self):
        """ Return the type of the distribution.
        
            Returns
            -------
            dist_type : str
                Type of the distribution."""

        dist_type = "arcsine"
        return dist_type

    def get_dist_params(self):
        """Return the parameters of the arcsine distribution.

            Returns
            -------
            params : array_like of shape (2,)
                Distribution parameters [scale, loc], where scale is the length of the interval, 
                and loc is the starting point of the interval."""
        
        params = np.array([self.scale, self.loc])
        return params
    
    def pdf(self, x):
        """Return the probability density function of the arcsine distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.
                
            Returns
            -------
            y : array_like
                Probability density function values at x."""
        
        y = arcsine.pdf(x, scale=self.scale, loc=self.loc)
        return y

    def cdf(self, x):
        """ Return the cumulative distribution function of the arcsine distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x."""

        y = arcsine.cdf(x, scale=self.scale, loc=self.loc)
        return y

    def invcdf(self, y):
        """ Return the inverse cumulative distribution function of the arcsine distribution, evaluated at y.
        
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.

            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y."""

        x = arcsine.ppf(y, scale=self.scale, loc=self.loc)
        return x

    def mean(self):
        """ Return the mean of the arcsine distribution.

            Returns
            -------
            mean : float
                Mean of the arcsine distribution."""
        
        mean = arcsine.mean(scale=self.scale, loc=self.loc)
        return mean

    def var(self):
        """ Return the variance of the arcsine distribution.

            Returns
            -------
            var : float
                Variance of the arcsine distribution."""
        
        var = arcsine.var(scale=self.scale, loc=self.loc)
        return var

    def skew(self):
        """ Return the skewness of the arcsine distribution.

            Returns
            -------
            skew : float
                Skewness of the arcsine distribution."""
        
        skew = arcsine.stats(scale=self.scale, loc=self.loc, moments='s')
        return skew

    def kurt(self):
        """ Return the kurtosis of the arcsine distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the arcsine distribution."""
        
        kurt = arcsine.stats(scale=self.scale, loc=self.loc, moments='k')
        return kurt

    def sample(self, n, method="MC", seed=None, **params):
        """Return n samples from the arcsine distribution.
        
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
                Generated samples from the arcsine distribution."""

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
        """ Return a translated and scaled beta distribution.

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
            new_dist : BetaDistribution
                Translated and scaled beta distribution."""

        new_loc = scale * self.loc + shift
        new_scale = self.scale * scale

        new_dist = ArcsineDistribution(scale=new_scale, loc=new_loc)
        return new_dist

    def get_shift(self):
        """ Return the shift of the arcsine distribution from the (0, 1) base form.

            Returns
            -------
            shift : float
                Shift of the arcsine distribution."""

        shift = self.loc
        return shift

    def get_scale(self):
        """ Return the scale of the arcsine distribution, i.e., the width of the distribution.

            Returns
            -------
            scale : float
                Scale of the arcsine distribution."""

        scale = self.scale
        return scale

    def fix_moments(self, mean, var):
        """ Fix the arcsine distribution to have specified mean and variance.

            Parameters
            ----------
            mean : float
                Desired mean of the distribution.
                
            var : float
                Desired variance of the distribution.
                
            Returns
            -------
            new_dist : ArcsineDistribution
                Translated and scaled arcsine distribution with specified moments."""

        if not (isinstance(mean, (int, float, np.number))):
            raise ValueError("Mean must be a numeric value.")
        if not (isinstance(var, (int, float, np.number)) and var > 0):
            raise ValueError("Variance must be a positive numeric value.")
        if not np.isfinite(mean):
            raise ValueError("Mean must be a finite value.")
        if not np.isfinite(var):
            raise ValueError("Variance must be a finite value.")

        scale = 2 * np.sqrt(2 * var)
        loc = mean - scale / 2
        return ArcsineDistribution(scale=scale, loc=loc)

    def get_base_dist(self):
        """ Return the GPC base distribution for the arcsine distribution.
        
            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution."""
                    
        return ArcsineDistribution(scale=2, loc=-1)
    
    def orth_polysys(self):
        """ Return the GPC polynomial system for the arcsine distribution.
        
            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the arcsine distribution."""

        from ..polysys import ChebyshevTPolynomials
    
        if self.scale == 2 and self.loc == -1:
            return ChebyshevTPolynomials()
        else:
            raise Exception(f"No polynomial system for this distribution ({self})") 

    def orth_polysys_syschar(self, normalized):
        """ Return the GPC polynomial system characteristic string for the arcsine distribution.

            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.

            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the arcsine distribution."""
        
        if not (self.scale == 2 and self.loc == -1):
            raise Exception(f"No polynomial system for this distribution ({self})")

        if normalized == True:
            return "j"
        else:
            return "J"