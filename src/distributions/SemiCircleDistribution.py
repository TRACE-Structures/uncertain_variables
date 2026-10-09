import numpy as np

from uncertain_variables.distributions.Distribution import Distribution
from uncertain_variables.distributions.UniformDistribution import UniformDistribution
from scipy.stats import semicircular

class SemiCircleDistribution(Distribution):
    """ Class for Wigner semicircle distribution.

        Attributes
        ----------
        radius : float
            Radius of the semicircle."
            
        scale : float
            Scale of the semicircle distribution, equivalent to the radius of the semicircle.
            
        loc : float
            Location parameter of the semicircle distribution, the center of the semicircle."""

    def __init__(self, radius, loc=0):
        """ Initialize the Wigner semicircle distribution with radius.

            Parameters
            ----------
            radius : float
                Radius of the semicircle.
            
            loc : float, optional
                Location parameter of the semicircle distribution, the center of the semicircle. Default is 0."""
        
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Radius must be a positive number.")
        if not isinstance(loc, (int, float)):
            raise ValueError("Location must be a numeric value.")

        self.radius = radius
        self.loc = loc
        
        self.scale = radius

    def __repr__(self):
        """ Returns the string representation of the SemiCircleDistribution object.

            Returns
            -------
            repr_string : str
                String representation of the SemiCircleDistribution object."""
        
        repr_string = "W({:.2f}), loc({:.2f})".format(self.radius, self.loc)
        return repr_string
    
    def __eq__(self, other):
        """ Check if two SemiCircleDistribution distributions are equal.

            Parameters
            ----------
            other : SemiCircleDistribution
                Another SemiCircleDistribution distribution to compare with.

            Returns
            -------
            is_equal : bool
                True if the two distributions are equal, False otherwise."""
        
        if not isinstance(other, SemiCircleDistribution):
            return False
        
        is_equal = (self.radius == other.radius) and (self.loc == other.loc)
        return is_equal

    def get_dist_type(self):
        """ Return the type of the Wigner semicircle distribution.

            Returns
            -------
            dist_type : str
                Type of the distribution."""
        
        dist_type = "semicircle"
        return dist_type
    
    def get_dist_params(self):
        """ Return the parameters of the Wigner semicircle distribution.

            Returns
            -------
            params : array_like of shape (2,)
                Distribution parameters [radius, loc], where radius is 
                the radius of the semicircle distribution and loc is the location parameter."""
        
        params = np.array([self.radius, self.loc])
        return params
    
    def pdf(self, x):
        """ Return the probability density function of the Wigner semicircle distribution, evaluated at x.

            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.

            Returns
            -------
            y : array_like
                Probability density function values at x. """
        
        # x = np.asarray(x)
        # y = np.zeros(x.shape)
        # ind = (x >= -self.radius) & (x <= self.radius)
        # y[ind] = (2 / (np.pi * self.radius**2)) * np.sqrt(self.radius**2 - x[ind] ** 2)
        # y = unwrap_if_scalar(y)
        # return y

        y = semicircular.pdf(x, scale=self.scale, loc=self.loc)
        return y

    def cdf(self, x):
        """ Return the cumulative distribution function of the Wigner semicircle distribution, evaluated at x.

            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.

            Returns
            -------
            y : array_like
                Cumulative distribution function values at x."""
        
        # x = np.asarray(x)
        # y = np.zeros(x.shape)
        # ind1 = x < -self.radius
        # ind2 = (x >= -self.radius) & (x <= self.radius)
        # ind3 = x > self.radius
        # y[ind1] = 0
        # y[ind2] = 1 / 2 + (x[ind2] * np.sqrt(self.radius**2 - x[ind2]**2)) / (np.pi * self.radius**2) + (
        #     np.arcsin(x[ind2] / self.radius)
        # ) / np.pi
        # y[ind3] = 1
        # y = unwrap_if_scalar(y)
        # return y

        y = semicircular.cdf(x, scale=self.scale, loc=self.loc)
        return y

    def invcdf(self, y):
        """ Return the inverse cumulative distribution function of the Wigner semicircle distribution, evaluated at y.

            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.

            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y. """
        
        # y = np.asarray(y)
        # x = np.full(y.shape, np.nan)
        # ind = (y >= 0) & (y <= 1)
        # x[ind] = self.radius * np.sin(np.pi * (y[ind] - 1 / 2))
        # return x

        x = semicircular.ppf(y, scale=self.scale, loc=self.loc)
        return x
    
    def mean(self):
        """ Return the mean of the Wigner semicircle distribution.

            Returns
            -------
            mean : float
                Mean of the Wigner semicircle distribution. """
        
        mean = semicircular.mean(scale=self.scale, loc=self.loc)
        return mean
    
    def var(self):
        """ Return the variance of the Wigner semicircle distribution.

            Returns
            -------
            var : float
                Variance of the Wigner semicircle distribution. """
        
        var = semicircular.var(scale=self.scale, loc=self.loc)
        return var
    
    def skew(self):
        """ Return the skewness of the Wigner semicircle distribution.

            Returns
            -------
            skew : float
                Skewness of the Wigner semicircle distribution. """
        
        skew = semicircular.stats(scale=self.scale, loc=self.loc, moments='s')
        return skew
    
    def kurt(self):
        """ Return the kurtosis of the Wigner semicircle distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the Wigner semicircle distribution. """
        
        kurt = semicircular.stats(scale=self.scale, loc=self.loc, moments='k')
        return kurt

    def sample(self, n, method="MC", seed=None, **params): 
        """Return n samples from the Wigner semicircle distribution.

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
                Generated samples from the Wigner semicircle distribution."""

        # from .UniformDistribution import UniformDistribution
        # xi = UniformDistribution(0, 1).sample(n, method, seed=seed, **params)
        # samples = self.invcdf(xi)
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
        """ Return a translated and scaled Wigner semicircle distribution.

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
            new_dist : SemiCircleDistribution
                Translated and scaled Wigner semicircle distribution."""

        if not (isinstance(shift, (int, float, np.number))):
            raise ValueError("Shift must be a numeric value.")
        if not (isinstance(scale, (int, float, np.number)) and scale > 0):
            raise ValueError("Scale must be a positive numeric value.")
        if not np.isfinite(shift):
            raise ValueError("Shift must be a finite value.")
        if not np.isfinite(scale):
            raise ValueError("Scale must be a finite value.")

        new_radius = self.radius * scale
        new_loc = self.loc * scale + shift
        return SemiCircleDistribution(new_radius, new_loc)

    def get_shift(self):
        """ Return the shift (location) parameter of the Wigner semicircle distribution.

            Returns
            -------
            shift : float
                Shift (location) parameter of the Wigner semicircle distribution."""

        shift = self.loc
        return shift

    def get_scale(self):
        """ Return the scale (radius) parameter of the Wigner semicircle distribution.

            Returns
            -------
            scale : float
                Scale (radius) parameter of the Wigner semicircle distribution."""

        scale = self.radius
        return scale

    def fix_moments(self, mean, var):
        """ Fix the Wigner semicircle distribution to have specified mean and variance.

            Parameters
            ----------
            mean : float
                Desired mean of the distribution.
            var : float
                Desired variance of the distribution.

            Returns
            -------
            new_dist : SemiCircleDistribution
                Wigner semicircle distribution with the specified mean and variance."""

        new_radius = np.sqrt(4 * var)
        new_loc = mean
        return SemiCircleDistribution(new_radius, new_loc)

    def get_base_dist(self):
        """ Return the GPC base distribution for the Wigner semicircle distribution.

            Returns
            -------
            dist_germ : SemiCircleDistribution object
                GPC base distribution."""
        
        dist_germ = SemiCircleDistribution(radius=1)
        return dist_germ

    # def base2dist(self, y):
    #     """ Convert from base (germ) space to Wigner semicircle distribution space.
        
    #         Parameters
    #         ----------
    #         y : array_like
    #             Points in base (germ) space.

    #         Returns
    #         -------
    #         x : array_like
    #             Points in Wigner semicircle distribution space."""
        
    #     y = np.asarray(y)
    #     x = y / self.radius
    #     return x

    # def dist2base(self, x):
    #     """ Convert from Wigner semicircle distribution space to base (germ) space.
  
    #           Parameters
    #           ----------
    #           x : array_like
    #               Points in Wigner semicircle distribution space.
              
    #           Returns
    #           -------
    #           y : array_like
    #               Points in base (germ) space."""
        
    #     x = np.asarray(x)
    #     y = x * self.radius
    #     return y

    def orth_polysys(self):
        """ Return the GPC polynomial system for the Wigner semicircle distribution.
                
            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the Wigner semicircle distribution."""

        from ..polysys.ChebyshevUPolynomials import ChebyshevUPolynomials

        if self.radius == 1 and self.loc == 0:
            return ChebyshevUPolynomials()
        else:  
            raise Exception(f"No polynomial system for this distribution ({self})")

    def orth_polysys_syschar(self, normalized):
        """ Return the GPC polynomial system characteristic string for the Wigner semicircle distribution.
        
            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.
                
            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the Wigner semicircle distribution."""

        if not (self.radius == 1 and self.loc == 0):
            raise Exception(f"No polynomial system for this distribution ({self})")

        if normalized == True:
            return "u"
        else:
            return "U"