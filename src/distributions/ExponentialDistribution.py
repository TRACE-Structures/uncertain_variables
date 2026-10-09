import numpy as np

from uncertain_variables.distributions.Distribution import Distribution
from uncertain_variables.distributions.UniformDistribution import UniformDistribution
from scipy.stats import expon

class ExponentialDistribution(Distribution):
    """ Class for exponential distribution.

        Attributes
        ----------
        lambda_ : float
            Rate parameter of the exponential distribution.
            
        scale : float
            Scale parameter of the exponential distribution (1/lambda_).
            
        loc : float
            Location parameter of the exponential distribution, typically set to 0."""
    
    def __init__(self, lambda_=1, loc=0):
        """ Initialize the exponential distribution with rate parameter lambda_.

            Parameters
            ----------
            lambda_ : float
                Rate parameter of the exponential distribution."""

        if not (isinstance(lambda_, (int, float, np.number)) and lambda_ > 0):
            raise ValueError("lambda_ must be a positive number.")

        self.lambda_ = lambda_

        self.scale = 1 / lambda_
        self.loc = loc

    def __repr__(self):
        """ Returns the string representation of the ExponentialDistribution object.

            Returns
            -------
            repr_string : str
                String representation of the ExponentialDistribution object."""
        
        repr_string = "Exp({:.2f}), loc={:.2f}".format(self.lambda_, self.loc)
        return repr_string
    
    def __eq__(self, other):
        """ Check if two ExponentialDistribution objects are equal.

            Parameters
            ----------
            other : ExponentialDistribution
                Another ExponentialDistribution object to compare with.

            Returns
            -------
            is_equal : bool
                True if the two ExponentialDistribution objects are equal, False otherwise."""
        
        if not isinstance(other, ExponentialDistribution):
            return False
        
        is_equal = (self.lambda_ == other.lambda_) and (self.loc == other.loc)
        return is_equal
    
    def get_dist_type(self):
        """ Return the type of the exponential distribution.

            Returns
            -------
            type_string : str
                Type of the distribution."""
        
        type_string = "exp"
        return type_string
    
    def get_dist_params(self):
        """ Return the parameters of the exponential distribution.

            Returns
            -------
            params : array_like of shape (2,)
                Distribution parameters [lambda_, loc], where lambda_ is 
                the rate parameter and loc is the location parameter."""
        
        params = np.array([self.lambda_, self.loc])
        return params

    def pdf(self, x):
        """ Return the probability density function of the exponential distribution, evaluated at x.

            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.

            Returns
            -------
            y : array_like
                Probability density function values at x."""

        # x = np.asarray(x)
        # lambda_ = self.lambda_
        # y = np.zeros(x.shape)
        # ind = x >= 0
        # y[ind] = lambda_ * np.exp(-lambda_ * x[ind])
        # y = unwrap_if_scalar(y)
        # return y
        y = expon.pdf(x, scale=self.scale, loc=self.loc)
        return y

    def cdf(self, x):
        """ Return the cumulative distribution function of the exponential distribution, evaluated at x.

            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.

            Returns
            -------
            y : array_like
                Cumulative distribution function values at x."""
        
        # x = np.asarray(x)
        # lambda_ = self.lambda_
        # y = np.zeros(x.shape)
        # ind = x >= 0
        # y[ind] = 1 - np.exp(-lambda_ * x[ind])
        # y = unwrap_if_scalar(y)
        # return y

        y = expon.cdf(x, scale=self.scale, loc=self.loc)
        return y

    def invcdf(self, y):
        """ Return the inverse cumulative distribution function of the exponential distribution, evaluated at y.

            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.

            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y."""
        
        # y = np.asarray(y)
        # lambda_ = self.lambda_
        # x = np.full(np.size(y), np.nan)
        # ind = (y >= 0) & (y <= 1)
        # # ignore RuntimeWarning in case x == 0
        # with np.errstate(divide="ignore", invalid="ignore"):
        #     x[ind] = -np.log(1 - y[ind]) / lambda_
        # x = unwrap_if_scalar(x)
        # return x

        x = expon.ppf(y, scale=self.scale, loc=self.loc)
        return x

    def mean(self):
        """ Return the mean of the exponential distribution.

            Returns
            -------
            mean : float
                Mean of the exponential distribution."""
        
        mean = expon.mean(scale=self.scale, loc=self.loc)
        return mean

    def var(self):
        """ Return the variance of the exponential distribution.

            Returns
            -------
            var : float
                Variance of the exponential distribution."""
            
        var = expon.var(scale=self.scale, loc=self.loc)
        return var

    def skew(self):
        """ Return the skewness of the exponential distribution.

            Returns
            -------
            skew : float
                Skewness of the exponential distribution."""
        
        return 2

    def kurt(self):
        """ Return the kurtosis of the exponential distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the exponential distribution."""
        
        return 6

    def sample(self, n, method="MC", seed=None, **params):
        """Return n samples from the exponential distribution.
        
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
                        Generated samples from the exponential distribution."""
        
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
        """ Return a translated and scaled exponential distribution.

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
            new_dist : ExponentialDistribution
                Translated and scaled exponential distribution."""

        if not (isinstance(shift, (int, float, np.number))):
            raise ValueError("Shift must be a numeric value.")
        if not (isinstance(scale, (int, float, np.number)) and scale > 0):
            raise ValueError("Scale must be a positive numeric value.")
        if not np.isfinite(shift):
            raise ValueError("Shift must be a finite value.")
        if not np.isfinite(scale):
            raise ValueError("Scale must be a finite value.")

        new_loc = scale * self.loc + shift
        new_lambda = self.lambda_ / scale
        return ExponentialDistribution(lambda_=new_lambda, loc=new_loc)

    def get_shift(self):
        """ Return the shift (location) parameter of the exponential distribution.

            Returns
            -------
            shift : float
                Shift (location) parameter of the exponential distribution."""

        shift = self.loc
        return shift

    def get_scale(self):
        """ Return the scale 1/lambda_ parameter of the exponential distribution.

            Returns
            -------
            scale : float
                Scale (1/lambda_) parameter of the exponential distribution."""

        scale = self.scale
        return scale

    def fix_moments(self, mean, var, loc=0):
        """ Fix the exponential distribution to have specified mean, variance, and location.

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
            new_dist : ExponentialDistribution
                Exponential distribution with the specified mean, variance, and location."""
        
        if not isinstance(mean, (int, float, np.number)):
            raise ValueError("Mean must be a numeric value.")
        if not isinstance(var, (int, float, np.number)) and var is not None:
            raise ValueError("Variance must be a numeric value or None.")
        if not isinstance(loc, (int, float, np.number)):
            raise ValueError("Location must be a numeric value.")

        if not np.isfinite(mean):
            raise ValueError("Mean must be a finite value.")
        if var is not None and not np.isfinite(var):
            raise ValueError("Variance must be a finite value.")
        if not np.isfinite(loc):
            raise ValueError("Location must be a finite value.")

        if mean <= loc:
            raise ValueError("Mean must be greater than loc.")

        if var is None:
            var = (mean - loc)**2
        elif not isinstance(var, (int, float, np.number)) or not np.isfinite(var) or var <= 0:
            raise ValueError("Variance must be a positive finite number.")
        elif not np.isclose(var, (mean - loc)**2):
            raise ValueError("Mean and variance are incompatible with the specified location.")

        new_lambda = 1 / (mean - loc)
        return ExponentialDistribution(new_lambda, loc)

    def get_base_dist(self):
        """ Return the GPC base distribution.

            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution."""
            
        base = ExponentialDistribution(1)
        return base

    def orth_polysys(self):
        """ Return the GPC polynomial system for the exponential distribution.

            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the exponential distribution."""
            
        # if self.lambda_:
        #     from polysys import LaguerrePolynomials

        #     polysys = LaguerrePolynomials()
        # else:
        #     Distribution.orth_polysys()
        # return polysys

        from ..polysys import LaguerrePolynomials

        if self.lambda_ == 1:
            return LaguerrePolynomials()
        else:
            raise Exception(f"No polynomial system for this distribution ({self})")

    def orth_polysys_syschar(self, normalized):
        """ Return the GPC polynomial system characteristic string for the exponential distribution.

            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.

            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the exponential distribution."""

        # if self.a == -1 and self.b == 1:
        #     if normalized:
        #         polysys_char = 'L'
        #     else:
        #         polysys_char = 'l'
        # else:
        #     polysys_char = []
        # return polysys_char

        if not self.lambda_ == 1:
            raise Exception(f"No polynomial system for this distribution ({self})")
            # OR return []

        if normalized:
            polysys_char = 'l'
        else:
            polysys_char = 'L'
        return polysys_char

    # def base2dist(self, y):
    #     """ Convert from base (germ) space to exponential distribution space.

    #         Parameters
    #         ----------
    #         y : array_like
    #             Points in base (germ) space.

    #         Returns
    #         -------
    #         x : array_like
    #             Points in exponential distribution space."""

    #     y = np.asarray(y)
    #     x = y / self.lambda_
    #     return x

    # def dist2base(self, x):
    #     """ Convert from exponential distribution space to base (germ) space.

    #         Parameters
    #         ----------
    #         x : array_like
    #             Points in exponential distribution space.
            
    #         Returns
    #         -------
    #         y : array_like
    #             Points in base (germ) space."""
        
    #     y = x * self.lambda_
    #     return y
