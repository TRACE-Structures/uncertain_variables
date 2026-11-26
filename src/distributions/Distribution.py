from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..polysys import PolynomialSystem


class Distribution(ABC):
    """ 
    Abstract class for probability distributions.
    """
    
    @abstractmethod
    def get_dist_type(self) -> str:
        """ 
        Abstract method to return the type of the distribution.
        """
        ...

    @abstractmethod
    def get_dist_params(self) -> dict[str, np.float64]:
        """ 
        Abstract method to return the parameters of the distribution.
        """
        ...
    
    @abstractmethod
    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """ 
        Abstract method to return the Probability Density Function of the distribution, evaluated at x.
        """
        ...

    @abstractmethod
    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """ 
        Abstract method to return the Cumulative Distribution Function of the distribution, evaluated at x.
        """
        ...

    @abstractmethod
    def ppf(self, y: ArrayLike) -> NDArray[np.float64]:
        """ 
        Abstract method to return the Percentage Point Function (inverse of CDF) of the distribution, evaluated at y.
        """
        ...

    @abstractmethod
    def mean(self) -> np.float64:
        """ 
        Abstract method to return the mean of the distribution.
        """
        ...

    @abstractmethod
    def var(self) -> np.float64:
        """ 
        Abstract method to return the variance of the distribution.
        """
        ...

    @abstractmethod
    def skew(self) -> np.float64:
        """ 
        Abstract method to return the skewness of the distribution.
        """
        ...

    @abstractmethod
    def kurt(self) -> np.float64:
        """ 
        Abstract method to return the kurtosis of the distribution.
        """
        ...

    def moments(self) -> dict[str, np.float64]:
        """ 
        Return the moments of the distribution.

        Returns
        -------
        moments : dict[str, np.float64]
            Dictionary containing the mean, variance, skewness, and kurtosis of the distribution.
        """
        return {
            "mean": self.mean(),
            "variance": self.var(),
            "skewness": self.skew(),
            "kurtosis": self.kurt()
        }

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """ 
        Return the log of the Probability Density Function of the distribution, evaluated at x.

        Parameters
        ----------
        x : ArrayLike
            Points at which to evaluate the log of the Probability Density Function.
            
        Returns
        -------
        y : NDArray[np.float64]
            Log of the Probability Density Function values at x.
        """
        if np.isscalar(x):
            x = [x]
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError("Input points x must be one-dimensional.")
        if x.dtype.kind not in "iuf":
            raise TypeError("Input points x must be real numbers.")
        x = x.astype(np.float64, copy=False)

        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(self.pdf(x))

        return y

    @abstractmethod
    def sample(
        self,
        n: int,
        method: Literal["MC", "QMC_Halton", "QMC_LHS", "QMC_Sobol"] = "MC",
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """ 
        Abstract method to generate random samples from the distribution.
        """
        ...

    @abstractmethod
    def transform(
        self,
        shift: float,
        scale: float,
    ) -> "Distribution":
        """ 
        Transform the distribution by a shift and scale.
        """
        ...

    @abstractmethod
    def get_shift(self) -> np.float64:
        """ 
        Return the shift of the distribution.
        """
        ...

    @abstractmethod
    def get_scale(self) -> np.float64:
        """ 
        Return the scale of the distribution.
        """
        ...

    def get_bounds(self, delta: float = 0.02) -> tuple[np.float64, np.float64]:
        """ 
        Return the bounds of the distribution.
            
        Parameters
        ----------
        delta : float, default is 0.02
            Small probability to define the bounds.
        
        Returns
        -------
        bounds : tuple[np.float64, np.float64]
            Bounds of the distribution as [lower_bound, upper_bound].
        """
        if not (isinstance(delta, (int, float, np.integer, np.floating))):
            raise TypeError("Delta must be a real number.")
        if not (0 <= delta <= 0.5):
            raise ValueError("Delta must be between 0 and 0.5.")
        delta = np.float64(delta)
        
        return (self.ppf(delta), self.ppf(1 - delta))
    
    @abstractmethod
    def fix_moments(
        self,
        mean: float,
        var: float,
    ) -> "Distribution":
        """ 
        Fix the distribution to have specified mean and variance.
        """
        ...  

    def stdnor_to_germ(self, x: ArrayLike) -> NDArray[np.float64]:
        """ 
        Convert from standard normal space to germ space.
        
        Parameters
        ----------
        x : ArrayLike
            Points in standard normal space.
            
        Returns
        -------
        y : NDArray[np.float64]
            Points in germ space.
        """ 
        if np.isscalar(x):
            x = [x]
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError("Input points x must be one-dimensional.")
        if x.dtype.kind not in "iuf":
            raise TypeError("Input points x must be real numbers.")
        x = x.astype(np.float64, copy=False)

        from .NormalDistribution import NormalDistribution

        return self.ppf(NormalDistribution().gpc_dist().cdf(x))

    def germ_to_stdnor(self, y: ArrayLike) -> NDArray[np.float64]:
        """ 
        Convert from germ space to standard normal space.
        
        Parameters
        ----------
        y : ArrayLike
            Points in germ space.
            
        Returns
        -------
        x : NDArray[np.float64]
            Points in standard normal space.
        """
        if np.isscalar(y):
            y = [y]
        y = np.array(y)

        if y.ndim != 1:
            raise ValueError("Input points y must be one-dimensional.")
        if y.dtype.kind not in "iuf":
            raise TypeError("Input points y must be real numbers.")
        y = y.astype(np.float64, copy=False)

        from .NormalDistribution import NormalDistribution

        return NormalDistribution().ppf(self.cdf(y))

    @abstractmethod
    def gpc_dist(self) -> "Distribution":
        """
        Return the probability distribution associated with the GPCE germ.
        """
        ...

    def germ_to_dist(self, y: ArrayLike) -> NDArray[np.float64]:
        """ 
        Convert from germ space to distribution space.
            
        Parameters
        ----------
        y : ArrayLike
            Points in germ space.
                
        Returns
        -------
        x : NDArray[np.float64]
            Points in distribution space.
        """
        if np.isscalar(y):
            y = [y]
        y = np.array(y)

        if y.ndim != 1:
            raise ValueError("Input points y must be one-dimensional.")
        if y.dtype.kind not in "iuf":
            raise TypeError("Input points y must be real numbers.")
        y = y.astype(np.float64, copy=False)

        return self.ppf(self.gpc_dist().cdf(y))

    def dist_to_germ(self, x: ArrayLike) -> NDArray[np.float64]:
        """ 
        Convert from distribution space to germ space.
        
        Parameters
        ----------
        x : ArrayLike
            Points in distribution space.
            
        Returns
        -------
        y : NDArray[np.float64]
            Points in germ space.
        """
        if np.isscalar(x):
            x = [x]
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError("Input points x must be one-dimensional.")
        if x.dtype.kind not in "iuf":
            raise TypeError("Input points x must be real numbers.")
        x = x.astype(np.float64, copy=False)

        return self.gpc_dist().ppf(self.cdf(x))

    @abstractmethod
    def gpc_polysys(self, orthonormal: bool = False) -> "PolynomialSystem":
        """ 
        Return the polynomial system associated with the distribution.
        """
        ...

    @abstractmethod
    def convert_to(
        self,
        from_unit: str,
        to_unit: str,
    ) -> None:
        """ 
        Convert the distribution parameters to a different unit.
        """
        ...

    @abstractmethod
    def copy(self) -> "Distribution":
        """ 
        Create a deep copy of the Distribution.
        """
        ...
