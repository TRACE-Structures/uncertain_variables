from .Distribution import Distribution, unwrap_if_scalar
import numpy as np

class UniformDistribution(Distribution):
    ''' Class for uniform distribution.
    
        Attributes
        ----------
        a : float
            Lower bound of the uniform distribution.
            
        b : float
            Upper bound of the uniform distribution.'''
    
    def __init__(self, a=0, b=1):
        ''' Initialize the uniform distribution with bounds a and b.
        
            Parameters
            ----------
            a : float, default = 0
                Lower bound of the uniform distribution.
                
            b : float, default = 1
                Upper bound of the uniform distribution.'''

        if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > a):
            raise ValueError("Uniform bounds must be numeric with b > a.")
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError("Uniform bounds must be finite numeric values.")

        self.a = a
        self.b = b

    def __repr__(self):
        ''' Returns the string representation of the UniformDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the UniformDistribution object.'''
        
        repr_string = "U({}, {})".format(self.a, self.b)
        return repr_string
    
    def __eq__(self, other):        
        ''' Check if two UniformDistribution objects are equal.
        
            Parameters
            ----------
            other : UniformDistribution
                Another UniformDistribution object to compare with.
                
            Returns
            -------
            is_equal : bool
                True if the two UniformDistribution objects are equal, False otherwise.'''
        
        if not isinstance(other, UniformDistribution):
            return False
        
        is_equal = (self.a == other.a) and (self.b == other.b)
        return is_equal

    def get_dist_type(self):
        ''' Return the type of the distribution.
        
            Returns
            -------
            dist_type : str
                Type of the distribution.'''
        
        dist_type = "unif"
        return dist_type
    
    def get_dist_params(self):
        ''' Return the parameters of the distribution.
        
            Returns
            -------
            params : tuple
                Parameters of the distribution (a, b).'''
        
        params = [self.a, self.b]
        return params

    def pdf(self, x):
        ''' Return the probability density function of the uniform distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the pdf.
                
            Returns
            -------
            y : array_like
                Probability density function values at x.'''
        
        x = np.asarray(x)
        a = self.a
        b = self.b
        y = 1 / (b - a) * np.ones(np.size(x))
        y[x < a] = 0
        y[x > b] = 0
        y = unwrap_if_scalar(y)
        return y

    def logpdf(self, x):
        ''' Return the log of the probability density function of the uniform distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the logpdf.
                
            Returns
            -------
            y : array_like
                Log probability density function values at x.'''
        
        # a = self.a
        # b = self.b
        # pdf = self.pdf(x)
        # pdf = np.array(pdf)  # OR
        # pdf = pdf.reshape(x.shape)
        # y = np.zeros(x.shape)
        # for i in range(len(x)):
        #     if pdf[i] == 0:
        #         y[i] = -np.inf
        #     else:
        #         y[i] = np.log(pdf[i])
        # return y

        x = np.asarray(x)
        pdf = self.pdf(x)
        if np.isscalar(pdf):
            pdf = np.array([pdf])
        y = np.zeros(pdf.shape)
        for i in range(len(pdf)):
            if pdf[i] == 0:
                y[i] = -np.inf
            else:
                y[i] = np.log(pdf[i])
        y = unwrap_if_scalar(y)
        return y

    def cdf(self, x):
        ''' Return the cumulative distribution function of the uniform distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the cdf.
                
            Returns
            -------
            y : array_like
                Cumulative distribution function values at x.'''
        
        # a = self.a
        # b = self.b
        # y = (x - a) / (b - a)
        # y = np.clip(y, 0, 1)
        # return y

        a = self.a
        b = self.b
        x = np.asarray(x)   
        y = (x - a) / (b - a)
        y = np.clip(y, 0, 1)
        y = unwrap_if_scalar(y)
        return y

    def invcdf(self, y):
        ''' Return the inverse cumulative distribution function of the uniform distribution, evaluated at y.
            
            Parameters
            ----------
            y : array_like
                Points at which to evaluate the invcdf.
                
            Returns
            -------
            x : array_like
                Inverse cumulative distribution function values at y.'''
        
        a = self.a
        b = self.b
        y = np.asarray(y)
        x = np.full(np.size(y), np.nan)
        ind = (y >= 0) & (y <= 1)
        x[ind] = a + (b - a) * y[ind]
        x = unwrap_if_scalar(x)
        return x

    def mean(self):
        ''' Return the mean of the uniform distribution.
        
            Returns
            -------
            mean : float
                Mean of the uniform distribution.'''
        
        mean = 0.5 * (self.a + self.b)
        return mean

    def var(self):
        ''' Return the variance of the uniform distribution.
            
            Returns
            -------
            var : float
                Variance of the uniform distribution.'''
        
        var = (self.b - self.a) ** 2 / 12
        return var

    def skew(self):
        ''' Return the skewness of the uniform distribution.

            Returns
            -------
            skew : float
                Skewness of the uniform distribution.'''
        skew = 0
        return skew

    def kurt(self):
        ''' Return the kurtosis of the uniform distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the uniform distribution.'''
        
        kurt = -6 / 5
        return kurt

    def translate(self, shift, scale):
        """ Return a translated and scaled version of the uniform distribution.

            Parameters
            ----------
            shift : float
                Shift to apply to the distribution.
            scale : float
                Scale to apply to the distribution. Must be a positive numeric value.

            Returns
            -------
            new_dist : UniformDistribution
                Translated and scaled uniform distribution."""
        
        # m = (self.a + self.b) / 2
        # v = scale * (self.b - self.a) / 2

        # a = m + shift - v
        # b = m + shift + v
        # new_dist = UniformDistribution(a, b)
        # return new_dist
    
        if not (isinstance(shift, (int, float))):
            raise ValueError("Shift must be a numeric value.")
        if not (isinstance(scale, (int, float)) and scale > 0):
            raise ValueError("Scale must be a positive numeric value.")

        m = (self.a + self.b) / 2
        v = scale * (self.b - self.a) / 2

        a = m + shift - v
        b = m + shift + v
        new_dist = UniformDistribution(a, b)
        return new_dist

    def get_base_dist(self):
        """ Return the GPC base distribution.

            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution."""
        
        dist_germ = UniformDistribution(-1, 1)
        return dist_germ

    def base2dist(self, y):
        """ Convert from base (germ) space to uniform distribution space.

            Parameters
            ----------
            y : array_like
                Points in base (germ) space.

            Returns
            -------
            x : array_like
                Points in distribution space."""

        
        y = np.asarray(y)
        x = self.mean() + y * (self.b - self.a) / 2
        return x

    def dist2base(self, x):
        """ Convert from uniform distribution space to base (germ) space.

            Parameters
            ----------
            x : array_like
                Points in distribution space.

            Returns
            -------
            y : array_like
                Points in base (germ) space."""
        
        x = np.asarray(x)
        y = (x - self.mean()) * 2 / (self.b - self.a)
        return y

    def orth_polysys(self):
        """ Return the GPC polynomial system for the uniform distribution.

            Returns
            -------
            polysys : PolynomialSystem object
                GPC polynomial system for the uniform distribution."""
        
        # from polysys import LegendrePolynomials

        # if self.a == -1 and self.b == 1:
        #     polysys = LegendrePolynomials()
        # else:
        #     polysys = Distribution.orth_polysys(self)
        # return polysys

        from ..polysys import LegendrePolynomials

        if self.a == -1 and self.b == 1:
            return LegendrePolynomials()
        else: 
            raise Exception(f"No polynomial system for this distribution ({self})") 

    def orth_polysys_syschar(self, normalized):
        """ Return the GPC polynomial system characteristic string for the uniform distribution.

            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.

            Returns
            -------
            polysys_char : str
                GPC polynomial system characteristic string for the uniform distribution."""
        
        # if self.a == -1 and self.b == 1:
        #     if normalized:
        #         polysys_char = "p"
        #     else:
        #         polysys_char = "P"
        # else:
        #     polysys_char = []
        # return polysys_char

        if not (self.a == -1 and self.b == 1):
            raise Exception(f"No polynomial system for this distribution ({self})")

        if normalized == True:
            return "p"
        else:
            return "P"

    def get_bounds(self, delta=0):
        """ Return the bounds of the uniform distribution.

            Parameters
            ----------
            delta : int, optional
                Expansion factor for the bounds (default is 0).

            Returns
            -------
            bounds : numpy.ndarray
                Array containing the lower and upper bounds of the uniform distribution."""
        
        if not isinstance(delta, (int, float)):
            raise ValueError("delta must be a numeric value")
        
        a = self.a
        b = self.b

        ab = b - a

        # The modified function contracts the bounds by delta instead of expanding them
        # bounds = np.array([a - ab * delta, b + ab * delta])
        
        bounds = np.array([a + ab * delta, b - ab * delta])
        return bounds
