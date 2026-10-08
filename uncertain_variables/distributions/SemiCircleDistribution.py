from .Distribution import Distribution, unwrap_if_scalar
import numpy as np

class SemiCircleDistribution(Distribution):
    """ Class for Wigner semicircle distribution.

        Attributes
        ----------
        radius : float
            Radius of the semicircle."""

    def __init__(self, radius):
        """ Initialize the Wigner semicircle distribution with radius.

            Parameters
            ----------
            radius : float
                Radius of the semicircle."""
        
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Radius must be a positive number.")
        
        self.radius = radius

    def __repr__(self):
        """ Returns the string representation of the WignerSemicirlceDistribution object.

            Returns
            -------
            repr_string : str
                String representation of the WignerSemicircleDistribution object."""
        
        repr_string = "W({})".format(self.radius)
        return repr_string
    
    def __eq__(self, other):
        """ Check if two Wigner semicircle distributions are equal.

            Parameters
            ----------
            other : SemiCircleDistribution
                Another Wigner semicircle distribution to compare with.

            Returns
            -------
            is_equal : bool
                True if the two distributions are equal, False otherwise."""
        
        if not isinstance(other, SemiCircleDistribution):
            return False
        
        is_equal = self.radius == other.radius
        return is_equal

    def get_dist_type(self):
        """ Return the type of the distribution.

            Returns
            -------
            dist_type : str
                Type of the distribution."""
        
        dist_type = "semicircle"
        return dist_type
    
    def get_dist_params(self):
        """ Return the parameters of the distribution.

            Returns
            -------
            params : float
                Parameters of the distribution (radius)."""
        
        params = self.radius
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
        
        x = np.asarray(x)
        y = np.zeros(x.shape)
        ind = (x >= -self.radius) & (x <= self.radius)
        y[ind] = (2 / (np.pi * self.radius**2)) * np.sqrt(self.radius**2 - x[ind] ** 2)
        y = unwrap_if_scalar(y)
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
        
        x = np.asarray(x)
        y = np.zeros(x.shape)
        ind1 = x < -self.radius
        ind2 = (x >= -self.radius) & (x <= self.radius)
        ind3 = x > self.radius
        y[ind1] = 0
        y[ind2] = 1 / 2 + (x[ind2] * np.sqrt(self.radius**2 - x[ind2]**2)) / (np.pi * self.radius**2) + (
            np.arcsin(x[ind2] / self.radius)
        ) / np.pi
        y[ind3] = 1
        y = unwrap_if_scalar(y)
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
        
        y = np.asarray(y)
        x = np.full(y.shape, np.nan)
        ind = (y >= 0) & (y <= 1)
        x[ind] = self.radius * np.sin(np.pi * (y[ind] - 1 / 2))
        return x
    
    def mean(self):
        """ Return the mean of the Wigner semicircle distribution.

            Returns
            -------
            mean : float
                Mean of the Wigner semicircle distribution. """
        
        mean = 0
        return mean
    
    def var(self):
        """ Return the variance of the Wigner semicircle distribution.

            Returns
            -------
            var : float
                Variance of the Wigner semicircle distribution. """
        
        var = self.radius**2 / 4
        return var
    
    def skew(self):
        """ Return the skewness of the Wigner semicircle distribution.

            Returns
            -------
            skew : float
                Skewness of the Wigner semicircle distribution. """
        
        skew = 0
        return skew
    
    def kurt(self):
        """ Return the kurtosis of the Wigner semicircle distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the Wigner semicircle distribution. """
        
        kurt = -1
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

        from .UniformDistribution import UniformDistribution
        xi = UniformDistribution(0, 1).sample(n, method, seed=seed, **params)
        samples = self.invcdf(xi)
        return samples

    def get_base_dist(self):
        ''' Return the GPC base distribution.

            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution.'''
        
        dist_germ = SemiCircleDistribution(1)
        return dist_germ

    def base2dist(self, y):
        """ Convert from base (germ) space to Wigner semicircle distribution space.
        
            Parameters
            ----------
            y : array_like
                Points in base (germ) space.

            Returns
            -------
            x : array_like
                Points in Wigner semicircle distribution space."""
        
        y = np.asarray(y)
        x = y / self.radius
        return x

    def dist2base(self, x):
        """ Convert from Wigner semicircle distribution space to base (germ) space.
  
              Parameters
              ----------
              x : array_like
                  Points in Wigner semicircle distribution space.
              
              Returns
              -------
              y : array_like
                  Points in base (germ) space."""
        
        x = np.asarray(x)
        y = x * self.radius
        return y

    def orth_polysys(self):
        ''' Return the GPC polynomial system for the Wigner semicircle distribution.
                
                    Returns
                    -------
                    polysys : PolynomialSystem object
                        GPC polynomial system for the Wigner semicircle distribution.'''

        from ..polysys.ChebyshevUPolynomials import ChebyshevUPolynomials

        if self.radius == 1:
            return ChebyshevUPolynomials()
        else:  
            raise Exception(f"No polynomial system for this distribution ({self})")

    def orth_polysys_syschar(self, normalized):
        ''' Return the GPC polynomial system characteristic string for the Wigner semicircle distribution.
        
            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.
                
            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the Wigner semicircle distribution.'''

        if not self.radius == 1:
            raise Exception(f"No polynomial system for this distribution ({self})")

        if normalized:
            polysys_char = "u"
        else:
            polysys_char = "U"
        return polysys_char