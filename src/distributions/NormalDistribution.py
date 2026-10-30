import pint
import numpy as np

from src.distributions.Distribution import Distribution
from src.distributions.UniformDistribution import UniformDistribution
from scipy.stats import norm


class NormalDistribution(Distribution):
    """ Normal (Gaussian) probability distribution.
    
        Attributes
        ----------
        mu : float
            Mean of the normal distribution (default: 0).
            
        sigma : float
            Positive standard deviation of the normal distribution (default: 1).
            
        loc : float
            Location parameter (equal to ``mu``).

        scale : float
            Scale parameter (equal to ``sigma``)."""
    
    def __init__(self, mu=0, sigma=1):
        """ Initialize a normal distribution with mean ``mu`` and standard deviation ``sigma``.
        
            Parameters
            ----------
            mu : float, default: 0
                Mean of the normal distribution.

            sigma : float, default: 1
                Standard deviation of the normal distribution."""

        # assert sigma > 0
        
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

        self.loc = self.mu
        self.scale = self.sigma

    def __repr__(self):
        """ Returns the string representation of the NormalDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the NormalDistribution object."""
        
        repr_string = "N({:.2f}, {:.2f})".format(self.mu, self.sigma)
        return repr_string
    
    def __eq__(self, other):
        """ Check if two NormalDistribution objects are equal.
        
            Parameters
            ----------
            other : NormalDistribution
                Another NormalDistribution object to compare with.
                
            Returns
            -------
            is_equal : bool
                True if the two NormalDistribution objects are equal, False otherwise."""
        
        if not isinstance(other, NormalDistribution):
            return False
        
        is_equal = (self.mu == other.mu) and (self.sigma == other.sigma)
        return is_equal

    def get_dist_type(self):
        """ Return the type of the distribution.
        
            Returns
            -------
            dist_type : str
                Type of the distribution."""

        dist_type = "norm"
        return dist_type
    
    def get_dist_params(self):
        """ Return the parameters of the normal distribution.

            Returns
            -------
            params : array_like of shape (2,)
                Distribution parameters in the order [mu, sigma], where mu is
                the mean and sigma is the standard deviation of the distribution."""
        
        params = np.array([self.mu, self.sigma])
        return params

    def pdf(self, x):
        """ Return the probability density function of the normal distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.
                
            Returns
            -------
            y : array_like
                Probability density function values at x."""
        
        # mu = self.mu
        # sigma = self.sigma
        # root = (x - mu) / sigma
        # y_exp = root**2
        # y_exp = -1 / 2 * y_exp
        # y = np.exp(y_exp) / (sigma * np.sqrt(2 * np.pi))
        # return y

        y = norm.pdf(x, loc=self.loc, scale=self.scale)
        return y

    # def logpdf(self, x):
    #     ''' Return the log of the probability density function of the normal distribution, evaluated at x.
        
    #         Parameters
    #         ----------
    #         x : array_like
    #             Points at which to evaluate the logpdf.
                
    #         Returns
    #         -------
    #         y : array_like
    #             Log probability density function values at x.'''

    #     x = np.asarray(x)
    #     mu = self.mu
    #     sigma = self.sigma
    #     root = (x - mu) / sigma
    #     y = -1 / 2 * (root**2) - np.log(sigma * np.sqrt(2 * np.pi))
    #     y = unwrap_if_scalar(y)
    #     return y

    def cdf(self, x):
        """ Return the cumulative distribution function of the normal distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x."""

        # mu = self.mu
        # sigma = self.sigma
        # y = 1 / 2 * (1 + sc.erf((x - mu) / (sigma * np.sqrt(2))))
        # return y

        y = norm.cdf(x, loc=self.loc, scale=self.scale)
        return y

    def ppf(self, y):
        """ Return the percentage point function (inverse of cdf) of the normal distribution, evaluated at y.
        
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the PPF.
                
            Returns
            -------
            x : array_like
                PPF values evaluated at y."""
        
        # mu = self.mu
        # sigma = self.sigma
        # y = np.array(y)
        # x = np.full(y.shape, np.nan)  # original
        # ind = (y >= 0) & (y <= 1)
        # x[ind] = mu + sigma * np.sqrt(2) * sc.erfinv(2 * y[ind] - 1)
        # x = x / 1
        # return x

        x = norm.ppf(y, loc=self.loc, scale=self.scale)
        return x

    def mean(self):
        """ Return the mean of the normal distribution.

            Returns
            -------
            mean : float
                Mean of the normal distribution."""
        
        mean = norm.mean(loc=self.loc, scale=self.scale)
        return mean

    def var(self):
        """ Return the variance of the normal distribution.

            Returns
            -------
            var : float
                Variance of the normal distribution."""

        var = norm.var(loc=self.loc, scale=self.scale)
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
        """ Return the excess kurtosis of the normal distribution.

            Returns
            -------
            kurt : float
                Excess kurtosis of the normal distribution."""

        kurt = 0
        return kurt

    def sample(self, n, method="MC", seed=None): 
        """ Generate random samples from the normal distribution using the specified sampling method.

            Parameters
            ----------
            n : int
                Number of samples to generate.

            method : str, default: "MC"
                Sampling method to use. Options are "MC" (Monte Carlo), "QMC_Halton" (Quasi-Monte Carlo using Halton sequence),
                "QMC_LHS" (Quasi-Monte Carlo using Latin Hypercube Sampling), and "QMC_Sobol" (Quasi-Monte Carlo using Sobol sequence).

            seed : int or None, default: None
                Seed for the random number generator.

            Returns
            -------
            samples : array_like
                Generated random samples from the normal distribution."""

        
        # from .UniformDistribution import UniformDistribution
        # if method == "MC":
        #     xi = np.random.randn(n)
        # else:
        #     xi = UniformDistribution().sample(n, method, **params)
        # samples = (xi * self.sigma) + self.mu
        # return samples

        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Number of samples must be a positive integer.")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer or None.")
        if not np.isfinite(n):
            raise ValueError("Number of samples must be a finite value.")

        y = UniformDistribution(0, 1).sample(n, method, seed=seed)
        samples = self.ppf(y)
        return samples

    def transform(self, shift=0, scale=1):
        """ Return a translated and scaled normal distribution.

            The transformation is defined as

            Y = scale * X + shift,

            where X is the original random variable.
        
        
            Parameters
            ----------
            shift : float, default: 0
                Shift to apply to the distribution.

            scale : float, default: 1
                Positive scale factor to apply to the distribution.
            
            Returns
            -------
            new_dist : NormalDistribution
                Translated and scaled normal distribution."""

        # new_dist = NormalDistribution(self.mu + shift, self.sigma * scale)
        # return new_dist
    
        if not (isinstance(shift, (int, float, np.number))):
            raise ValueError("Shift must be a numeric value.")
        if not (isinstance(scale, (int, float, np.number)) and scale > 0):
            raise ValueError("Scale must be a positive numeric value.")
        if not np.isfinite(shift):
            raise ValueError("Shift must be a finite value.")
        if not np.isfinite(scale):
            raise ValueError("Scale must be a finite value.")
        
        new_mu = scale * self.mu + shift
        new_sigma = self.scale * scale
        return NormalDistribution(new_mu, new_sigma)

    def get_shift(self):
        """ Return the shift (location) parameter of the normal distribution.

            Returns
            -------
            shift : float
                Shift (location) parameter of the normal distribution."""

        shift = self.loc
        return shift

    def get_scale(self):
        """ Return the scale (standard deviation) parameter of the normal distribution.

            Returns
            -------
            scale : float
                Scale (standard deviation) parameter of the normal distribution."""

        scale = self.scale
        return scale

    def fix_moments(self, mean=0, var=1):
        """ Fix the normal distribution to have specified mean and variance.
                
            Parameters
            ----------
            mean : float, default: 0
                Desired mean of the distribution.
                
            var : float, default: 1
                Desired positive variance of the distribution.
                
            Returns
            -------
            new_dist : NormalDistribution
                Translated and scaled normal distribution with specified moments."""

        if not (isinstance(mean, (int, float, np.number))):
            raise ValueError("Mean must be a numeric value.")
        if not (isinstance(var, (int, float, np.number)) and var > 0):
            raise ValueError("Variance must be a positive numeric value.")
        if not np.isfinite(mean):
            raise ValueError("Mean must be a finite value.")
        if not np.isfinite(var):
            raise ValueError("Variance must be a finite value.")

        new_loc = mean
        new_scale = np.sqrt(var)
        return NormalDistribution(new_loc, new_scale)
    
    def gpc_dist(self):
        """ Return the GPC germ distribution for the normal distribution.

            Returns
            -------
            germ_dist : NormalDistribution
                GPC germ distribution."""
        
        germ_dist = NormalDistribution(0, 1)
        return germ_dist

    # def base2dist(self, y):
    #     """ Convert from base (germ) space to normal distribution space.

    #         Parameters
    #         ----------
    #         y : array_like
    #             Points in base (germ) space.

    #         Returns
    #         -------
    #         x : array_like
    #             Points in normal distribution space."""

    #     y = np.asarray(y)
    #     x = self.mu + y * self.sigma
    #     return x
    
    # def dist2base(self, x):
    #     """ Convert from normal distribution space to base (germ) space.

    #         Parameters
    #         ----------
    #         x : array_like
    #             Points in normal distribution space.

    #         Returns
    #         -------
    #         y : array_like
    #             Points in base (germ) space."""

    #     x = np.asarray(x)
    #     y = (x - self.mu) / self.sigma
    #     return y

    def gpc_polysys(self):
        """ Return the GPC polynomial system for the normal distribution.
        
            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the normal distribution."""
        
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

    def gpc_syschar(self, normalized):
        """ Return the GPC polynomial system characteristic string for the normal distribution.
        
            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.
                
            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the normal distribution."""
        
        # if self.mu == 0 and self.sigma == 1:
        #     if normalized:
        #         polysys_char = "h"
        #     else:
        #         polysys_char = "H"
        # else:
        #     polysys_char = []
        # return polysys_char

        if self.mu == 0 and self.sigma == 1:
            return "h" if normalized else "H"

        raise Exception(f"No polynomial system for this distribution ({self})")

    def convert_to(self, from_unit, to_unit):
        """ Convert the distribution parameters to a different unit.

            Parameters
            ----------
            from_unit : str
                Source unit.

            to_unit : str
                Target unit.

            Returns
            -------
            report : str
                Description of the conversion.

            Raises
            ------
            pint.DimensionalityError
                If the source and target units are incompatible."""

        ureg = pint.UnitRegistry()
        self.mu = ureg.Quantity(self.mu, from_unit).to(to_unit).magnitude
        self.sigma = ureg.Quantity(self.sigma, from_unit).to(to_unit).magnitude
        self.loc = ureg.Quantity(self.loc, from_unit).to(to_unit).magnitude

        self.s = self.sigma
        self.scale = self.sigma

        report = f"Converted from {from_unit} to {to_unit}."
        return report