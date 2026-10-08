from .Distribution import Distribution, unwrap_if_scalar
import numpy as np

class ExponentialDistribution(Distribution):
    """ Class for exponential distribution.

        Attributes
        ----------
        lambda_ : float
            Rate parameter of the exponential distribution."""
    
    def __init__(self, lambda_=1):
        """ Initialize the exponential distribution with rate parameter lambda_.

            Parameters
            ----------
            lambda_ : float
                Rate parameter of the exponential distribution."""

        if not (isinstance(lambda_, (int, float, np.number)) and lambda_ > 0):
            raise ValueError("lambda_ must be a positive number.")

        self.lambda_ = lambda_

    def __repr__(self):
        """ Returns the string representation of the ExponentialDistribution object.

            Returns
            -------
            repr_string : str
                String representation of the ExponentialDistribution object."""
        
        repr_string = "Exp({})".format(self.lambda_)
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
        
        is_equal = self.lambda_ == other.lambda_
        return is_equal
    
    def get_dist_type(self):
        """ Return the type of the distribution.

            Returns
            -------
            type_string : str
                Type of the distribution."""
        
        type_string = "exp"
        return type_string
    
    def get_dist_params(self):
        """ Return the parameters of the distribution.

            Returns
            -------
            params : float
                Parameters of the distribution (lambda_)."""
        
        params = self.lambda_
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

        x = np.asarray(x)
        lambda_ = self.lambda_
        y = np.zeros(x.shape)
        ind = x >= 0
        y[ind] = lambda_ * np.exp(-lambda_ * x[ind])
        y = unwrap_if_scalar(y)
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
        
        x = np.asarray(x)
        lambda_ = self.lambda_
        y = np.zeros(x.shape)
        ind = x >= 0
        y[ind] = 1 - np.exp(-lambda_ * x[ind])
        y = unwrap_if_scalar(y)
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
        
        y = np.asarray(y)
        lambda_ = self.lambda_
        x = np.full(np.size(y), np.nan)
        ind = (y >= 0) & (y <= 1)
        # ignore RuntimeWarning in case x == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            x[ind] = -np.log(1 - y[ind]) / lambda_
        x = unwrap_if_scalar(x)
        return x

    def mean(self):
        """ Return the mean of the exponential distribution.

            Returns
            -------
            mean : float
                Mean of the exponential distribution."""
        
        mean = 1 / self.lambda_
        return mean

    def var(self):
        """ Return the variance of the exponential distribution.

            Returns
            -------
            var : float
                Variance of the exponential distribution."""
            
        var = 1 / self.lambda_**2
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
        
        from .UniformDistribution import UniformDistribution
        xi = UniformDistribution().sample(n, method, seed=seed, **params)
        samples = self.invcdf(xi)
        return samples

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

    def get_base_dist(self):
        """ Return the GPC base distribution.

            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution."""
            
        base = ExponentialDistribution(1)
        return base

    def base2dist(self, y):
        """ Convert from base (germ) space to exponential distribution space.

            Parameters
            ----------
            y : array_like
                Points in base (germ) space.

            Returns
            -------
            x : array_like
                Points in exponential distribution space."""

        y = np.asany    
        x = y / self.lambda_
        return x

    def dist2base(self, x):
        """ Convert from exponential distribution space to base (germ) space.

            Parameters
            ----------
            x : array_like
                Points in exponential distribution space.
            
            Returns
            -------
            y : array_like
                Points in base (germ) space."""
        
        y = x * self.lambda_
        return y
