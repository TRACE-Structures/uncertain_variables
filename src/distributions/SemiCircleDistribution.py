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
        
        assert radius > 0
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
            type_string : str
                Type of the distribution."""
        
        type_string = "wigner"
        return type_string
    
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
        
        x = np.array(x)
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
        
        x = np.array(x)
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
        
        y = np.array(y)
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
    
    def get_base_dist(self):
        ''' Return the GPC base distribution.

            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution.'''
        
        dist_germ = SemiCircleDistribution(1)
        return dist_germ

