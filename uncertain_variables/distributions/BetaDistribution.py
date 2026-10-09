import numpy as np
from sklearn.preprocessing import scale

from uncertain_variables.distributions.Distribution import Distribution
from uncertain_variables.distributions.UniformDistribution import UniformDistribution
from scipy.stats import beta

class BetaDistribution(Distribution):
    """ Class for beta distribution.
    
        Attributes
        ----------
        a : float
            First shape parameter of the beta distribution.
            
        b : float
            Second shape parameter of the beta distribution.
        
        scale : float
            Scale parameter of the beta distribution. The length of the interval, default is 1.

        loc : float
            Location parameter of the beta distribution. The starting point of the interval, default is 0."""
        
    def __init__(self, a, b, scale=1, loc=0):
        """ Initialize the beta distribution with shape parameters a and b 
            and optional scale and loc parameters.
        
            Parameters
            ----------
            a : float
                First shape parameter of the beta distribution.
                
            b : float
                Second shape parameter of the beta distribution.
                
            scale : float, default = 1
                Scale parameter of the beta distribution. The length of the interval, default is 1.

            loc : float, default = 0
                Location parameter of the beta distribution. The start of the interval, default is 0."""

        if not (isinstance(a, (int, float, np.number)) and a > 0):
            raise ValueError("Parameter 'a' must be a positive number.")
        if not (isinstance(b, (int, float, np.number)) and b > 0):
            raise ValueError("Parameter 'b' must be a positive number.")
        if not (isinstance(scale, (int, float, np.number)) and scale > 0):
            raise ValueError("Parameter 'scale' must be a positive number.")
        if not (isinstance(loc, (int, float, np.number))):
            raise ValueError("Parameter 'loc' must be a number.")
        if not np.isfinite(a):
            raise ValueError("Parameter 'a' must be a finite number.")
        if not np.isfinite(b):
            raise ValueError("Parameter 'b' must be a finite number.")
        if not np.isfinite(scale):
            raise ValueError("Parameter 'scale' must be a finite number.")
        if not np.isfinite(loc):
            raise ValueError("Parameter 'loc' must be a finite number.")

        self.a = a
        self.b = b
        self.scale = scale
        self.loc = loc

    def __repr__(self):
        """ Returns the string representation of the BetaDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the BetaDistribution object."""
        
        repr_string = "Beta({:.2f}, {:.2f}), scale={:.2f}, loc={:.2f}".format(self.a, self.b, self.scale, self.loc)
        return repr_string
    
    def __eq__(self, other):
        """ Check if two BetaDistribution objects are equal.
        
            Parameters
            ----------
            other : BetaDistribution
                Another BetaDistribution object to compare with.
                
            Returns
            -------
            is_equal : bool
                True if the two BetaDistribution objects are equal, False otherwise."""
        
        if not isinstance(other, BetaDistribution):
            return False
        
        is_equal = (self.a == other.a) and (self.b == other.b) and (self.scale == other.scale) and (self.loc == other.loc)
        return is_equal
    
    def get_dist_type(self):
        """ Return the type of the beta distribution.
        
            Returns
            -------
            dist_type : str
                Type of the distribution."""
        
        dist_type = "beta"
        return dist_type
    
    def get_dist_params(self):
        """ Return the parameters of the beta distribution.

            Returns
            -------
            params : array_like of shape (4,)
                Distribution parameters [a, b, scale, loc], where a and b are 
                the shape parameters of the beta distribution, loc is the start of the interval, 
                and scale is the length of the interval."""
        
        params = np.array([self.a, self.b, self.scale, self.loc])
        return params

    def pdf(self, x):
        """ Return the probability density function of the beta distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.
                
            Returns
            -------
            y : array_like
                Probability density function values at x."""
        
        # x = TranslatedDistribution.translate_points_backwards(x, -1, 2, 0)
        # x = np.array(x)
        # y = np.zeros(x.shape)
        # ind = (x >= 0) & (x <= 1)
        # y[ind] = (
        #     x[ind] ** (self.a - 1)
        #     * (1 - x[ind]) ** (self.b - 1)
        #     / sc.beta(self.a, self.b)
        # )
        # y = unwrap_if_scalar(y)
        # return y

        y = beta.pdf(x, self.a, self.b, scale=self.scale, loc=self.loc)
        return y

    def cdf(self, x):
        """ Return the cumulative distribution function of the beta distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x."""
        
        # x = TranslatedDistribution.translate_points_backwards(x, -1, 2, 0)
        # x = np.array(x)
        # y = np.zeros(x.shape)
        # ind = (x >= 0) & (x <= 1)
        # y[ind] = sc.betainc(self.a, self.b, x[ind])
        # y[x > 1] = 1
        # y = unwrap_if_scalar(y)
        # return y

        y = beta.cdf(x, self.a, self.b, scale=self.scale, loc=self.loc)
        return y

    def invcdf(self, y):
        """ Return the inverse cumulative distribution function of the beta distribution, evaluated at y.
            
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.
                
            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y."""
        
        # TODO implementing the Matlab code
        # y = np.array(y)
        # x = np.full(y.shape, np.nan)
        # ind = (y >= 0) & (y <= 1)
        # x[ind] = sc.betaincinv(self.a, self.b, y[ind])
        # x = TranslatedDistribution.translate_points_forward(x, -1, 2, 0)
        # x = unwrap_if_scalar(x)
        # return x

        x = beta.ppf(y, self.a, self.b, scale=self.scale, loc=self.loc)
        return x

    # def moments(self):
    #     """ Return the first four moments of the beta distribution.
        
    #         Returns
    #         -------
    #         moments : list
    #             List containing the first four moments [mean, variance, skewness, kurtosis] of the beta distribution."""
        
    #     mean = self.mean()
    #     var = self.var()
    #     skew = self.skew()
    #     kurt = self.kurt()

    #     # moments = [mean, var, skew, kurt]
    #     # moments = TranslatedDistribution.translate_moments(moments, -1, 2, 0)

    #     moments = np.array([mean, var, skew, kurt])
    #     return moments

    def mean(self):
        """ Return the mean of the beta distribution.

            Returns
            -------
            mean : float
                Mean of the beta distribution."""
        
        mean = beta.mean(self.a, self.b, scale=self.scale, loc=self.loc)
        return mean

    def var(self):
        """ Return the variance of the beta distribution.

            Returns
            -------
            var : float
                Variance of the beta distribution."""
        
        var = beta.var(self.a, self.b, scale=self.scale, loc=self.loc)
        return var

    def skew(self):
        """ Return the skewness of the beta distribution.

            Returns
            -------
            skew : float
                Skewness of the beta distribution."""
        
        # skew = (
        #     2
        #     * (self.b - self.a)
        #     * np.sqrt(self.a + self.b + 1)
        #     / ((self.a + self.b + 2) * np.sqrt(self.a * self.b))
        # )

        skew = beta.stats(self.a, self.b, scale=self.scale, loc=self.loc, moments='s')
        return skew

    def kurt(self):
        """ Return the kurtosis of the beta distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the beta distribution."""
        
        # kurt = (
        #     6
        #     * (
        #         self.a**3
        #         - (self.a**2) * (2 * self.b - 1)
        #         + (self.b**2) * (self.b + 1)
        #         - 2 * self.a * self.b * (self.b + 2)
        #     )
        #     / (self.a * self.b * (self.a + self.b + 2) * (self.a + self.b + 3))
        # )

        kurt = beta.stats(self.a, self.b, scale=self.scale, loc=self.loc, moments='k')
        return kurt

    def sample(self, n, method="MC", seed=None, **params): 
        """Return n samples from the beta distribution.

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
                Generated samples from the beta distribution."""

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
        
        if not (isinstance(shift, (int, float, np.number))):
            raise ValueError("Shift must be a numeric value.")
        if not (isinstance(scale, (int, float, np.number)) and scale > 0):
            raise ValueError("Scale must be a positive numeric value.")
        if not np.isfinite(shift):
            raise ValueError("Shift must be a finite value.")
        if not np.isfinite(scale):
            raise ValueError("Scale must be a finite value.")

        new_a = self.a
        new_b = self.b
        new_loc = self.loc * scale + shift
        new_scale = self.scale * scale

        new_dist = BetaDistribution(new_a, new_b, loc=new_loc, scale=new_scale)
        return new_dist

    def get_shift(self):
        """ Return the shift of the beta distribution from the (0, 1) base form.

            Returns
            -------
            shift : float
                Shift parameter of the beta distribution."""

        shift = self.loc
        return shift

    def get_scale(self):
        """ Return the scale of the beta distribution, i.e., the width of the distribution.

            Returns
            -------
            scale : float
                Scale parameter of the beta distribution."""

        scale = self.scale
        return scale

    def fix_moments(self, mean, var, scale=1, loc=0):
        """Return a beta distribution with the specified moments and support.

        Parameters
        ----------
        mean : float
            Desired mean of the distribution.
        var : float
            Desired variance of the distribution.
        scale : float, default=1
            The length of the interval, default is 1.
        loc : float, default=0
             The start of the interval, default is 0.

        Returns
        -------
        new_dist : BetaDistribution
            Beta distribution with the specified mean, variance, location,
            and scale."""
        
        if not isinstance(mean, (int, float, np.number)):
            raise ValueError("Mean must be a numeric value.")
        if not isinstance(var, (int, float, np.number)) or var <= 0:
            raise ValueError("Variance must be a positive numeric value.")
        if not isinstance(scale, (int, float, np.number)) or scale <= 0:
            raise ValueError("Scale must be a positive numeric value.")
        if not isinstance(loc, (int, float, np.number)):
            raise ValueError("Location must be a numeric value.")

        m = (mean - loc) / scale
        v = var / scale**2

        if not 0 < m < 1:
            raise ValueError("Mean must lie strictly inside the support.")
        if not 0 < v < m * (1 - m):
            raise ValueError("Variance is incompatible with the beta distribution.")

        temp = m * (1 - m) / v - 1
        a = m * temp
        b = (1 - m) * temp

        return BetaDistribution(a, b, scale=scale, loc=loc)

    def get_base_dist(self, a, b):
        """ Return the GPC base distribution for the beta distribution.

            Returns
            -------
            dist_germ : BetaDistribution
                GPC base distribution."""
        
        dist_germ = BetaDistribution(a, b, scale=2, loc=-1)
        return dist_germ

    # def base2dist(self, y):
    #     """ Convert from base (germ) space to beta distribution space.

    #         Parameters
    #         ----------
    #         y : array_like
    #             Points in base (germ) space.

    #         Returns
    #         -------
    #         x : array_like
    #             Points in distribution space."""
        
    #     x = y
    #     return x

    # def dist2base(self, x):
    #     """ Convert from beta distribution space to base (germ) space.

    #         Parameters
    #         ----------
    #         x : array_like
    #             Points in distribution space.

    #         Returns
    #         -------
    #         y : array_like
    #             Points in base (germ) space."""
        
    #     y = x
    #     return y

    def orth_polysys(self):
        """ Return the GPC polynomial system for the beta distribution.

            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the beta distribution."""
        
        # from polysys import JacobiPolynomials

        # polysys = JacobiPolynomials(self.b - 1, self.a - 1)
        # return polysys

        from ..polysys import JacobiPolynomials

        if self.scale == 2 and self.loc == -1:
            return JacobiPolynomials(self.b - 1, self.a - 1)
        else:
            raise Exception(f"No polynomial system for this distribution ({self})") 

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
        
        # if self.a == -1 and self.b == 1:
        #     if normalized:
        #         polysys_char = "J"
        #     else:
        #         polysys_char = "j"
        # else:
        #     polysys_char = []
        # return polysys_char

        if not (self.scale == 2 and self.loc == -1):
            raise Exception(f"No polynomial system for this distribution ({self})")

        if normalized == True:
            return "j"
        else:
            return "J"
