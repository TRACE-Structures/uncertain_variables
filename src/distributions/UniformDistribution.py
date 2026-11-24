import pint
import numpy as np

from .Distribution import Distribution
from ..polysys import LegendrePolynomials

from scipy.stats import uniform

from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import Sobol


class UniformDistribution(Distribution):
    """ Continuous uniform distribution on a finite interval.
    
        Attributes
        ----------
        a : float
            Lower endpoint of the interval (default: 0).
            
        b : float
            Upper endpoint of the interval (default: 1).
            
        loc : float
            Lower endpoint of the interval (equal to ``a``).
        
        scale : float
            Width of the interval (equal to ``b - a``)."""
    
    def __init__(self, a=0, b=1):
        """ Initialize a uniform distribution on the interval ``[a, b]``.
        
            Parameters
            ----------
            a : float, default: 0
                Lower endpoint of the interval.
                
            b : float, default: 1
                Upper endpoint of the interval."""

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
        """ Return the type of the distribution.
        
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

        y = uniform.pdf(x, loc=self.loc, scale=self.scale)
        return y

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
        
        y = uniform.cdf(x, loc=self.loc, scale=self.scale)
        return y

    def ppf(self, y):
        """ Return the percentage point function (inverse of cdf) of the uniform distribution, evaluated at y.
            
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the PPF.
                
            Returns
            -------
            x : array_like
                PPF values evaluated at y."""    

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

    def sample(self, n, method="MC", seed=None):
        """ Generate random samples from the uniform distribution using the specified sampling method.

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
            y = sampler.random(n).ravel()
        elif method == "QMC_LHS":
            sampler = LHS(d=1, seed=seed)
            y = sampler.random(n).ravel()
        elif method == "QMC_Sobol":
            sampler = Sobol(d=1, seed=seed)
            y = sampler.random(n).ravel()
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        return self.ppf(y)

    def transform(self, shift=0, scale=1):
        """ Return a translated and scaled uniform distribution.

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

        new_loc = scale * self.loc + shift
        new_scale = self.scale * scale
        new_a = new_loc
        new_b = new_loc + new_scale

        new_dist = UniformDistribution(new_a, new_b)
        return new_dist

    def get_shift(self):
        """ Return the shift of the uniform distribution from the (0, 1) base form.
            
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

    def fix_moments(self, mean=0.5, var=1/12):
        """ Fix the uniform distribution to have specified mean and variance.
        
            Parameters
            ----------
            mean : float, default: 0.5
                Desired mean of the distribution.
                
            var : float, default: 1/12
                Desired positive variance of the distribution.
                
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

    def gpc_dist(self):
        """ Return the GPC germ distribution for the uniform distribution.

            Returns
            -------
            germ_dist : UniformDistribution
                GPC germ distribution."""
        
        germ_dist = UniformDistribution(-1, 1)
        return germ_dist

    def gpc_polysys(self, orthonormal=True):
        """ Return the GPC polynomial system for the uniform distribution.

            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the uniform distribution."""
        if self.a == -1 and self.b == 1:
            return LegendrePolynomials(orthonormal=orthonormal)
        else: 
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
        self.a = ureg.Quantity(self.a, from_unit).to(to_unit).magnitude
        self.b = ureg.Quantity(self.b, from_unit).to(to_unit).magnitude

        self.loc = self.a
        self.scale = self.b - self.a
        
        report = f"Converted from {from_unit} to {to_unit}."
        return report

    def copy(self):
        """ Create a deep copy of the UniformDistribution.

            Returns
            -------
            UniformDistribution
                A new UniformDistribution instance with the same attributes as the current UniformDistribution."""

        return UniformDistribution(a=self.a, b=self.b)