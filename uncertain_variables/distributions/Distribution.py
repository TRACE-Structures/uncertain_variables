import numpy as np
from abc import ABC, abstractmethod
from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import Sobol

np.random.seed(seed=1234)

def unwrap_if_scalar(arr):
    """ Return the item if the array has only one element, else return the array.

        Parameters
        ----------
        arr : array_like
            Input array.

        Returns
        -------
        output : scalar or array_like
            Unwrapped item if arr has one element, else arr itself."""
    
    output = arr.item() if arr.size == 1 else arr
    return output


class Distribution(ABC):
    ''' Abstract class for probability distributions.'''
            
    @abstractmethod
    def pdf(self, x):
        ''' Abstract method to return the probability density function of the distribution, evaluated at x.
            Subclasses must implement this method.'''
        
        pass

    @abstractmethod
    def cdf(self, x):
        ''' Abstract method to return the cumulative distribution function of the distribution, evaluated at x.
            Subclasses must implement this method.'''
        
        pass

    @abstractmethod
    def invcdf(self, y):
        ''' Abstract method to return the inverse cumulative distribution function of the distribution, evaluated at y.
            Subclasses must implement this method.'''
        
        pass

    @abstractmethod
    def mean(self):
        ''' Abstract method to return the mean of the distribution.
            Subclasses must implement this method.'''
        
        pass

    @abstractmethod
    def var(self):
        ''' Abstract method to return the variance of the distribution.
            Subclasses must implement this method.'''
        
        pass

    @abstractmethod
    def skew(self):
        ''' Abstract method to return the skewness of the distribution.
            Subclasses must implement this method.'''
        
        pass

    @abstractmethod
    def kurt(self):
        ''' Abstract method to return the kurtosis of the distribution.
            Subclasses must implement this method.'''
        
        pass

    def moments(self):
        ''' Return the moments of the distribution.
        
            Returns
            -------
            moments : list
                List containing the mean, variance, skewness, and kurtosis of the distribution.'''
        
        moments = [self.mean(), self.var(), self.skew(), self.kurt()]
        return moments

    def logpdf(self, x):
        ''' Return the log of the probability density function of the distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the logpdf.
                
            Returns
            -------
            y : array_like
                Log probability density function values at x.'''
        
        pdf = self.pdf(x)
        y = np.log(pdf)
        return y

    def sample(self, n, method="MC", **params):
        if method == "MC":
            yi = np.random.rand(n)
        elif method == "QMC_Halton":
            sampler = Halton(d=1)
            yi = sampler.random(n)
        elif method == "QMC_LHS":
            sampler = LHS(d=1)
            yi = sampler.random(n)
        elif method == "QMC_Sobol":
            sampler = Sobol(d=1)
            yi = sampler.random(n)
        # elif method == 'Sobol_saltelli': # It's not


    def translate(self, shift, scale):
        ''' Return a translated and scaled version of the distribution.
            
            Parameters
            ----------
            shift : float
                Shift to apply to the distribution.

            scale : float
                Scale to apply to the distribution.
                
            Returns
            -------
            tdist : TranslatedDistribution
                Translated and scaled distribution.'''
        from .TranslatedDistribution import TranslatedDistribution
        tdist = TranslatedDistribution(self, shift, scale)
        return tdist

    def get_shift(self):
        """ Return the shift of the distribution.

            Returns
            -------
            shift : float
                Shift of the distribution."""
        
        shift = self-shift
        return self.shift

    def get_scale(self):
        """ Return the scale of the distribution.

            Returns
            -------
            scale : float
                Scale of the distribution."""
        
        scale = self.scale
        return scale

    def fix_moments(self, mean, var):
        ''' Fix the distribution to have specified mean and variance.
        
            Parameters
            ----------
            mean : float
                Desired mean of the distribution.
                
            var : float
                Desired variance of the distribution.
                
            Returns
            -------
            new_dist : TranslatedDistribution
                Translated and scaled distribution with specified moments.'''
        
        old_mean, old_var = self.mean(), self.var()
        self.shift = mean - old_mean
        self.scale = np.sqrt(var / old_var)
        new_dist = self.translate(self.shift, self.scale)
        return new_dist

    def fix_bounds(self, min, max, q0=0, q1=1):
        ''' Fix the distribution to have specified bounds.
        
            Parameters
            ----------
            min : float
                Desired minimum of the distribution.
                
            max : float
                Desired maximum of the distribution.
                
            q0 : float, optional
                Quantile corresponding to the minimum (default is 0).
                
            q1 : float, optional
                Quantile corresponding to the maximum (default is 1).
                
            Returns
            -------
            new_dist : TranslatedDistribution
                Translated and scaled distribution with specified bounds.'''
        
        if not (0 <= q0 <= 1):
            raise ValueError(f"q0 must be between 0 and 1, got {q0}")
        if not (q0 <= q1 <= 1):
            raise ValueError(f"q1 must be between q0 and 1, got {q1}")

        old_min = self.invcdf(q0)
        old_max = self.invcdf(q1)

        if not np.isfinite(old_min):
            # raise ValueError(f"Lower quantile (q0) gives infinity (unbounded distribution?)")
            print(
                f"Lower quantile (q0) gives infinity (unbounded distribution?). Using new q0=0.02"
            )
            old_min, _ = self.get_bounds()
        if not np.isfinite(old_max):
            # raise ValueError(f"Upper quantile (q1) gives infinity (unbounded distribution?)")
            print(
                f"Upper quantile (q1) gives infinity (unbounded distribution?). Using new q1=0.98"
            )
            _, old_max = self.get_bounds()

        center = self.mean()
        self.scale = (max - min) / (old_max - old_min)
        self.shift = min - ((old_min - center) * self.scale + center)
        new_dist = self.translate(self.shift, self.scale)
        return new_dist

    def stdnor2base(self, x):
        ''' Convert from standard normal space to distribution space.
        
            Parameters
            ----------
            x : array_like
                Points in standard normal space.
                
            Returns
            -------
            y : array_like
                Points in distribution space.'''
        
        from .NormalDistribution import NormalDistribution
        y = self.invcdf(NormalDistribution().cdf(x))
        return y

    def base2stdnor(self, y):
        ''' Convert from distribution space to standard normal space.
        
            Parameters
            ----------
            y : array_like
                Points in distribution space.
                
            Returns
            -------
            x : array_like
                Points in standard normal space.'''
        
        from .NormalDistribution import NormalDistribution
        x = NormalDistribution().invcdf(self.cdf(y))
        return x

    def get_base_dist(self):
        ''' Return the GPC base distribution.
            
            Returns
            -------
            dist_germ : Distribution object
                GPC base distribution.'''
        
        from .NormalDistribution import NormalDistribution
        dist_germ = NormalDistribution(0, 1)
        return dist_germ

    def base2dist(self, y):
        ''' Convert from base (germ) space to distribution space.
            
            Parameters
            ----------
            y : array_like
                Points in base (germ) space.
                
            Returns
            -------
            x : array_like
                Points in distribution space.'''
        
        x = self.invcdf(self.get_base_dist().cdf(y))
        return x

    def dist2base(self, x):
        ''' Convert from distribution space to base (germ) space.
            
            Parameters
            ----------
            x : array_like
                Points in distribution space.
                
            Returns
            -------
            y : array_like
                Points in base (germ) space.'''
        
        y = self.get_base_dist().invcdf(self.cdf(x))
        return y

    def orth_polysys(self):
        """ Return the GPC polynomial system.

            Raises
            ------
            Exception
                If no polynomial system is defined for this distribution."""
        
        raise Exception(f"No polynomial system for this distribution ({self})")

    def get_bounds(self, delta=0.02):
        ''' Return the bounds of the distribution.
            
            Parameters
            ----------
            delta : float, optional
                Small probability to define the bounds (default is 0.02).
            
            Returns
            -------
            bounds : array_like
                Bounds of the distribution as [lower_bound, upper_bound].'''
        
        bounds = self.invcdf(np.array([delta, 1 - delta]))
        return bounds
    
    