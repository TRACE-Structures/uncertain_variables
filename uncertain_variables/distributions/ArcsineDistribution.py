import numpy as np

from uncertain_variables.distributions import NormalDistribution

from .Distribution import Distribution, unwrap_if_scalar


class ArcsineDistribution(Distribution):
    """Class for arcsine distribution on the interval [-1, 1].
    
        Attributes
        ----------
        None
            The arcsine distribution on the interval [-1, 1] has no parameters."""

    def __init__(self):
        """
        Initialize the arcsine distribution on the interval [-1, 1].
        """
        pass
    
    def __repr__(self):
        """ Returns the string representation of the ArcsineDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the ArcsineDistribution object."""

        repr_string = "Arcsine()"
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

        return isinstance(other, ArcsineDistribution)
    
    def get_dist_type(self):
        """ Return the type of the distribution.
        
            Returns
            -------
            dist_type : str
                Type of the distribution."""

        dist_type = "arcsin"
        return dist_type

    def get_dist_params(self):
        """Return the parameters of the arcsine distribution.

            Returns
            -------
            params : tuple
                Parameters of the distribution (none)."""
        
        params = ()
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
        
        x = np.asarray(x)
        ind = (x >= -1) & (x <= 1)
        y = np.zeros(x.shape)
        y[ind] = 1 / (np.pi * np.sqrt(1 - x[ind]**2))
        y = unwrap_if_scalar(y)
        return y

    def cdf(self, x):
        """Return the cumulative distribution function of the arcsine distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x."""
        
        x = np.asarray(x)
        ind = (x >= -1) & (x <= 1)
        y = np.zeros(x.shape)
        y[ind] = 0.5 + np.arcsin(x[ind]) / np.pi
        y = unwrap_if_scalar(y)
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

        y = np.asarray(y)
        ind = (y >= 0) & (y <= 1)
        x = np.full(y.shape, np.nan)
        x[ind] = np.sin(np.pi * (y[ind] - 0.5))
        x = unwrap_if_scalar(x)
        return x

    def mean(self):
        """ Return the mean of the arcsine distribution.

            Returns
            -------
            mean : float
                Mean of the arcsine distribution."""
        
        return 0

    def var(self):
        """ Return the variance of the arcsine distribution.

            Returns
            -------
            var : float
                Variance of the arcsine distribution."""
        
        return 0.5

    def skew(self):
        """ Return the skewness of the arcsine distribution.

            Returns
            -------
            skew : float
                Skewness of the arcsine distribution."""
        
        return 0

    def kurt(self):
        """ Return the kurtosis of the arcsine distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the arcsine distribution."""
        
        return -1.5

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

        from .UniformDistribution import UniformDistribution
        xi = UniformDistribution().sample(n, method, seed=seed, **params)
        samples = self.invcdf(xi)
        return samples

    def get_base_dist(self):
        """ Return the GPC base distribution.
        
            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution."""
                    
        return self

    def base2dist(self, y):
        """ Convert from base (germ) space to arcsine distribution space.
        
            Parameters
            ----------
            y : array_like
                Points in base (germ) space.

            Returns
            -------
            x : array_like
                Points in arcsine distribution space."""

        return np.asarray(y)

    def dist2base(self, x):
        """ Convert from arcsine distribution space to base (germ) space.
        
            Parameters
            ----------
            x : array_like
                Points in arcsine distribution space.
            
            Returns
            -------
            y : array_like
                Points in base (germ) space."""
        
        return np.asarray(x)

    def orth_polysys(self):
        """ Return the GPC polynomial system for the arcsine distribution.
        
            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the arcsine distribution."""

        from ..polysys import ChebyshevTPolynomials

        return ChebyshevTPolynomials()

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

        if normalized:
            polysys_char = 't'
        else:
            polysys_char = 'T'
        return polysys_char