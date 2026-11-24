import numpy as np
from abc import ABC, abstractmethod

class Distribution(ABC):
    """ Abstract class for probability distributions."""
    
    @abstractmethod
    def get_dist_type(self):
        """ Abstract method to return the type of the distribution.
            Subclasses must implement this method."""
        
        pass

    @abstractmethod
    def get_dist_params(self):
        """ Abstract method to return the parameters of the distribution.
            Subclasses must implement this method."""

        pass
    
    @abstractmethod
    def pdf(self, x):
        """ Abstract method to return the probability density function of the distribution, evaluated at x.
            Subclasses must implement this method."""
        
        pass

    @abstractmethod
    def cdf(self, x):
        """ Abstract method to return the cumulative distribution function of the distribution, evaluated at x.
            Subclasses must implement this method."""
        
        pass

    @abstractmethod
    def ppf(self, y):
        """ Abstract method to return the percentage point function (inverse of cdf) of the distribution, evaluated at y.
            Subclasses must implement this method."""
        
        pass

    @abstractmethod
    def mean(self):
        """ Abstract method to return the mean of the distribution.
            Subclasses must implement this method."""
        
        pass

    @abstractmethod
    def var(self):
        """ Abstract method to return the variance of the distribution.
            Subclasses must implement this method."""
        
        pass

    @abstractmethod
    def skew(self):
        """ Abstract method to return the skewness of the distribution.
            Subclasses must implement this method."""
        
        pass

    @abstractmethod
    def kurt(self):
        """ Abstract method to return the kurtosis of the distribution.
            Subclasses must implement this method."""

        pass

    def moments(self):
        """ Return the moments of the distribution.
        
            Returns
            -------
            moments : array_like
                Array containing the mean, variance, skewness, and kurtosis of the distribution."""

        moments = [self.mean(), self.var(), self.skew(), self.kurt()]
        return moments

    def logpdf(self, x):
        """ Return the log of the probability density function of the distribution, evaluated at x.
        
            Parameters
            ----------
            x : array_like
                Points at which to evaluate the logpdf.
                
            Returns
            -------
            y : array_like
                Log probability density function values at x."""

        # pdf = self.pdf(x)
        # y = np.log(pdf)
        # return y

        with np.errstate(divide="ignore", invalid="ignore"):
            pdf = self.pdf(x)
            y = np.log(pdf)
        return y

    # def sample(self, n, method="MC", **params):
    #     if method == "MC":
    #         yi = np.random.rand(n)
    #     elif method == "QMC_Halton":
    #         sampler = Halton(d=1)
    #         yi = sampler.random(n)
    #     elif method == "QMC_LHS":
    #         sampler = LHS(d=1)
    #         yi = sampler.random(n)
    #     elif method == "QMC_Sobol":
    #         sampler = Sobol(d=1)
    #         yi = sampler.random(n)
    #     # elif method == 'Sobol_saltelli': # It's not

    @abstractmethod
    def sample(self, n, method, seed):
        """ Abstract method to generate random samples from the distribution.
            Subclasses must implement this method."""

        pass

    # def translate(self, shift, scale):
    #     """ Return a translated and scaled version of the distribution.
            
    #         Parameters
    #         ----------
    #         shift : float
    #             Shift to apply to the distribution.

    #         scale : float
    #             Scale to apply to the distribution.  Must be a positive numeric value.
                
    #         Returns
    #         -------
    #         tdist : TranslatedDistribution
    #             Translated and scaled distribution."""

    #     if not (isinstance(shift, (int, float))):
    #         raise ValueError("Shift must be a numeric value.")
    #     if not (isinstance(scale, (int, float)) and scale > 0):
    #         raise ValueError("Scale must be a positive numeric value.")
        
    #     from .TranslatedDistribution import TranslatedDistribution
    #     tdist = TranslatedDistribution(self, shift, scale)
    #     return tdist

    @abstractmethod
    def transform(self, shift, scale):
        """ Abstract method to translate the distribution by a shift and scale.
            Subclasses must implement this method."""
        
        pass

    @abstractmethod
    def get_shift(self):
        """ Return the shift of the distribution.
            Subclasses must implement this method."""

        pass

    @abstractmethod
    def get_scale(self):
        """ Return the scale of the distribution.
            Subclasses must implement this method."""

        pass

    def get_bounds(self, delta=0.02):
        """ Return the bounds of the distribution.
            
            Parameters
            ----------
            delta : float, optional, default: 0.02
                Small probability to define the bounds.
            
            Returns
            -------
            bounds : array_like
                Bounds of the distribution as [lower_bound, upper_bound]."""

        if not (isinstance(delta, (int, float, np.number)) and 0 <= delta <= 0.5):
            raise ValueError("Delta must be a float between 0 and 0.5.")
        
        bounds = np.array([self.ppf(delta), self.ppf(1 - delta)])
        return bounds
    
    @abstractmethod
    def fix_moments(self, mean, var):
        """ Fix the distribution to have specified mean and variance.
            Subclasses must implement this method."""
        
        pass

    def fix_bounds(self, new_min, new_max, q0=0, q1=1):
        """ Fix the distribution to have specified bounds.
        
            Parameters
            ----------
            new_min : float
                Desired minimum of the distribution.
                
            new_max : float
                Desired maximum of the distribution.
                
            q0 : float, optional
                Quantile corresponding to the minimum (default is 0).
                
            q1 : float, optional
                Quantile corresponding to the maximum (default is 1).
                
            Returns
            -------
            new_dist : Distribution
                Translated and scaled distribution with specified bounds."""

        if not (isinstance(new_min, (int, float))):
            raise ValueError("Minimum must be a numeric value.")
        if not (isinstance(new_max, (int, float))):
            raise ValueError("Maximum must be a numeric value.")
        if new_min >= new_max:
            raise ValueError("Minimum must be less than maximum.")
        if not (0 <= q0 <= 1):
            raise ValueError(f"q0 must be between 0 and 1, got {q0}")
        if not (q0 <= q1 <= 1):
            raise ValueError(f"q1 must be between q0 and 1, got {q1}")

        old_min = self.ppf(q0)
        old_max = self.ppf(q1)

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

        new_scale = (new_max - new_min) / (old_max - old_min)
        new_shift = new_min - old_min * new_scale
        new_dist = self.transform(shift=new_shift, scale=new_scale)
        return new_dist

    def stdnor_to_germ(self, x):
        """ Convert from standard normal space to germ space.
        
            Parameters
            ----------
            x : array_like
                Points in standard normal space.
                
            Returns
            -------
            y : array_like
                Points in germ space.""" 

        # Imported lazily to avoid a circular import with NormalDistribution.
        from src.distributions.NormalDistribution import NormalDistribution
        y = self.ppf(NormalDistribution().gpc_dist().cdf(x))
        return y

    def germ_to_stdnor(self, y):
        """ Convert from germ space to standard normal space.
        
            Parameters
            ----------
            y : array_like
                Points in germ space.
                
            Returns
            -------
            x : array_like
                Points in standard normal space."""

        # Imported lazily to avoid a circular import with NormalDistribution.
        from src.distributions.NormalDistribution import NormalDistribution
        x = NormalDistribution().ppf(self.cdf(y))
        return x

    @abstractmethod
    def gpc_dist(self):
        """ Return the GPC germ distribution.
            Subclasses must implement this method."""

        pass

    def germ_to_dist(self, y):
        """ Convert from germ space to distribution space.
            
            Parameters
            ----------
            y : array_like
                Points in germ space.
                
            Returns
            -------
            x : array_like
                Points in distribution space."""

        x = self.ppf(self.gpc_dist().cdf(y))
        return x

    def dist_to_germ(self, x):
        """ Convert from distribution space to germ space.
            
            Parameters
            ----------
            x : array_like
                Points in distribution space.
                
            Returns
            -------
            y : array_like
                Points in germ space."""
        
        y = self.gpc_dist().ppf(self.cdf(x))
        return y

    @abstractmethod
    def gpc_polysys(self, orthonormal):
        """ Return the GPC polynomial system.
            Subclasses must implement this method."""

        pass

    @abstractmethod
    def convert_to(self, from_unit, to_unit):
        """ Convert the distribution parameters to a different unit.
            Subclasses must implement this method."""

        pass

    @abstractmethod
    def copy(self):
        """ Create a deep copy of the Distribution.

            Returns
            -------
            Distribution
                A new Distribution instance with the same attributes as the current Distribution."""

        pass