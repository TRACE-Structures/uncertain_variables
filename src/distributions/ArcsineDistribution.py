import pint
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import arcsine

from .Distribution import Distribution
from .UniformDistribution import UniformDistribution

class ArcsineDistribution(Distribution):
    """
    Arcsine probability distribution on a finite interval.
    
    Attributes
    ----------
    scale : float, default is 1
        Length of the interval.
    loc : float, default is 0
        Starting point of the interval.
    """

    def __init__(
        self,
        scale: float = 1,
        loc: float = 0,
    ) -> None:
        """
        Initialize an arcsine distribution on the interval '[loc, loc + scale]'.
        """
        if not (isinstance(scale, (int, float, np.integer, np.floating))):
            raise TypeError("Parameter 'scale' must be a real number.")
        scale = np.float64(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Parameter 'scale' must be a positive finite number.")

        if not (isinstance(loc, (int, float, np.integer, np.floating))):
            raise TypeError("Parameter 'loc' must be a real number.")
        loc = np.float64(loc)
        if not np.isfinite(loc):
            raise ValueError("Parameter 'loc' must be a finite number.")

        self.scale = scale
        self.loc = loc
    
    def __repr__(self) -> str:
        """ 
        Returns the string representation of the ArcsineDistribution object.
        
        Returns
        -------
        repr_string : str
            String representation of the ArcsineDistribution object.
        """
        return "Arcsine(scale={:.2f}, loc={:.2f})".format(self.scale, self.loc)

    def __eq__(self, other: "ArcsineDistribution") -> bool:
        """ 
        Check if two ArcsineDistribution objects are equal.
        
        Parameters
        ----------
        other : ArcsineDistribution
            Another ArcsineDistribution object to compare with.
            
        Returns
        -------
        is_equal : bool
            True if the two ArcsineDistribution objects are equal, False otherwise.
        """
        if not isinstance(other, ArcsineDistribution):
            return False
        
        return (self.scale == other.scale) and (self.loc == other.loc)
    
    def get_dist_type(self) -> str:
        """ 
        Return the type of the distribution.
        
        Returns
        -------
        dist_type : str
            Type of the distribution.
        """
        return "arcsine"

    def get_dist_params(self) -> dict[str, np.float64]:
        """ 
        Return the parameters of the arcsine distribution.

        Returns
        -------
        params : dict
            Distribution parameters {"scale": scale, "loc": loc}, where scale is the length of the interval, 
            and loc is the starting point of the interval.
        """
        return {
            "scale": self.scale,
            "loc": self.loc
        }
        
    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """ 
        Return the probability density function of the arcsine distribution, evaluated at x.
        
        Parameters
        ----------
        x : ArrayLike
            Points at which to evaluate the pdf.
            
        Returns
        -------
        y : NDArray[np.float64]
            Probability density function values at x.
        """
        if np.isscalar(x):
            x = [x]
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError("Input points 'x' must be one-dimensional.")
        if x.dtype.kind not in "iuf":
            raise TypeError("Input points 'x' must be real numbers.")
        x = x.astype(np.float64, copy=False)
        
        return arcsine.pdf(x, scale=self.scale, loc=self.loc)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """ 
        Return the cumulative distribution function of the arcsine distribution, evaluated at x.
        
        Parameters
        ----------
        x : ArrayLike
            Points at which to evaluate the cdf.
            
        Returns
        -------
        y : NDArray[np.float64]
            Cumulative distribution function values at x.
        """
        if np.isscalar(x):
            x = [x]
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError("Input points 'x' must be one-dimensional.")
        if x.dtype.kind not in "iuf":
            raise TypeError("Input points 'x' must be real numbers.")
        x = x.astype(np.float64, copy=False)

        return arcsine.cdf(x, scale=self.scale, loc=self.loc)

    def ppf(self, y: ArrayLike) -> NDArray[np.float64]:
        """ 
        Return the percentage point function (inverse of cdf) of the arcsine distribution, evaluated at y.
        
        Parameters
        ----------
        y : ArrayLike
            Points at which to evaluate the PPF.

        Returns
        -------
        x : NDArray[np.float64]
            PPF values evaluated at y.
        """
        if np.isscalar(y):
            y = [y]
        y = np.array(y)

        if y.ndim != 1:
            raise ValueError("Input points 'y' must be one-dimensional.")
        if y.dtype.kind not in "iuf":
            raise TypeError("Input points 'y' must be real numbers.")
        y = y.astype(np.float64, copy=False)

        return arcsine.ppf(y, scale=self.scale, loc=self.loc)

    def mean(self) -> np.float64:
        """
        Return the mean of the arcsine distribution.

        Returns
        -------
        mean : np.float64
            Mean of the arcsine distribution.
        """
        return arcsine.mean(scale=self.scale, loc=self.loc)

    def var(self) -> np.float64:
        """ 
        Return the variance of the arcsine distribution.

        Returns
        -------
        var : np.float64
            Variance of the arcsine distribution.
        """
        return arcsine.var(scale=self.scale, loc=self.loc)

    def skew(self) -> np.float64:
        """
        Return the skewness of the arcsine distribution.

        Returns
        -------
        skew : np.float64
            Skewness of the arcsine distribution.
        """
        return arcsine.stats(scale=self.scale, loc=self.loc, moments='s')

    def kurt(self) -> np.float64:
        """
        Return the kurtosis of the arcsine distribution.

        Returns
        -------
        kurt : np.float64
            Kurtosis of the arcsine distribution.
        """
        return arcsine.stats(scale=self.scale, loc=self.loc, moments='k')

    def sample(
        self,
        n: int,
        method: Literal["MC", "QMC_Halton", "QMC_LHS", "QMC_Sobol"] = "MC", 
        seed: int | None = None
    ) -> np.NDArray[np.float64]:
        """
        Return n samples from the arcsine distribution.
        
        Parameters
        ----------
        n : int
            Number of samples to generate.
        method : Literal["MC", "QMC_Halton", "QMC_LHS", "QMC_Sobol"], default is "MC"
            Sampling methods are "MC" (Monte Carlo), "QMC_Halton" (Halton sequence),
            "QMC_LHS" (Latin hypercube sampling), and "QMC_Sobol" (Sobol sequence).
        seed : int or None, default is None
            Seed for the random number generator.

        Returns
        -------
        samples : NDArray[np.float64]
            Generated samples from the arcsine distribution.
        """
        if not isinstance(n, (int, np.integer)):
            raise TypeError("Parameter 'n' (number of samples) must be an integer.")
        if n <= 0 or not np.isfinite(n):
            raise ValueError("Parameter 'n' (number of samples) must be a positive finite integer.")

        if method not in ["MC", "QMC_Halton", "QMC_LHS", "QMC_Sobol"]:
            raise ValueError("Invalid sampling method. Method must be one of 'MC', 'QMC_Halton', 'QMC_LHS', 'QMC_Sobol'.")

        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError("Parameter 'seed' must be an integer or None.")

        y = UniformDistribution(0, 1).sample(n, method, seed=seed)
        return self.ppf(y)

    def transform(
        self, 
        shift: float = 0,
        scale: float = 1
    ) -> "ArcsineDistribution":
        """ 
        Return a transformed arcsine distribution.

        The transformation is defined as

        Y = scale * X + shift,

        where X is the original random variable.
    
        Parameters
        ----------
        shift : float, default is 0
            Shift to apply to the distribution.
        scale : float, default is 1
            Positive scale factor to apply to the distribution.
        
        Returns
        -------
        new_dist : ArcsineDistribution
            Transformed arcsine distribution.
        """
        if not isinstance(shift, (int, float, np.integer, np.floating)):
            raise TypeError("Parameter 'shift' must be a numeric value.")
        shift = np.float64(shift)
        if not np.isfinite(shift):
            raise ValueError("Parameter 'shift' must be a finite value.")

        if not isinstance(scale, (int, float, np.integer, np.floating)):
            raise TypeError("Parameter 'scale' must be a numeric value.")
        scale = np.float64(scale)
        if not np.isfinite(scale):
            raise ValueError("Parameter 'scale' must be a finite value.")
        if scale <= 0:
            raise ValueError("Parameter 'scale' must be a positive value.")

        new_loc = scale * self.loc + shift
        new_scale = self.scale * scale

        return ArcsineDistribution(scale=new_scale, loc=new_loc)

    def get_shift(self) -> np.float64:
        """ 
        Return the shift of the arcsine distribution, i.e., the location parameter.

        Returns
        -------
        shift : np.float64
            Shift of the arcsine distribution.
        """
        return self.loc

    def get_scale(self) -> np.float64:
        """ 
        Return the scale of the arcsine distribution, i.e., length of the interval.

        Returns
        -------
        scale : np.float64
            Scale of the arcsine distribution.
        """
        return self.scale

    def fix_moments(
        self,
        mean: float = 0.5,
        var: float = 0.125
    ) -> "ArcsineDistribution":
        """
        Fix the arcsine distribution to have specified mean and variance.

        Parameters
        ----------
        mean : float, default: 0.5
            Desired mean of the distribution.
        var : float, default: 0.125
            Desired positive variance of the distribution.
            
        Returns
        -------
        new_dist : ArcsineDistribution
            Translated and scaled arcsine distribution with specified moments.
        """
        if not isinstance(mean, (int, float, np.integer, np.floating)):
            raise TypeError("Parameter 'mean' must be a numeric value.")
        mean = np.float64(mean)
        if not np.isfinite(mean):
            raise ValueError("Parameter 'mean' must be a finite value.")

        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError("Parameter 'var' must be a numeric value.")
        var = np.float64(var)
        if not np.isfinite(var):
            raise ValueError("Parameter 'var' must be a finite value.")
        if var <= 0:
            raise ValueError("Parameter 'var' must be a positive value.")

        scale = 2 * np.sqrt(2 * var)
        loc = mean - scale / 2
        
        return ArcsineDistribution(scale=scale, loc=loc)

    def gpc_dist(self) -> "ArcsineDistribution":
        """ 
        Return the GPC germ distribution for the arcsine distribution.
        
        Returns
        -------
        germ_dist : ArcsineDistribution
            GPC germ distribution.
        """          
        return ArcsineDistribution(scale=2, loc=-1)
    
    def gpc_polysys(self, orthonormal: bool = False) -> "ChebyshevTPolynomials":
        """ 
        Return the GPC polynomial system for the arcsine distribution.

        Parameters
        ----------
        orthonormal : bool, default is False
            Whether to return an orthonormal polynomial system.
        
        Returns
        -------
        polysys : ChebyshevTPolynomials
            GPC polynomial system for the arcsine distribution.

        Raises
        ------
        ValueError
            If no polynomial system exists for the current distribution.
        """    
        if not isinstance(orthonormal, (bool, np.bool_)):
            raise TypeError("Parameter 'orthonormal' must be a boolean value.")
        
        if self.scale == 2 and self.loc == -1:
            from ..polysys import ChebyshevTPolynomials

            return ChebyshevTPolynomials(orthonormal=orthonormal)
        else:
            raise ValueError(f"No polynomial system for this distribution ({self})") 

    def convert_to(
        self,
        from_unit: str,
        to_unit: str
    ) -> None:
        """ 
        Convert the distribution parameters from one unit to another in place.

        Parameters
        ----------
        from_unit : str
            Current unit of the distribution parameters. (e.g., 'm', 'cm', 'inch')
        to_unit : str
            Target unit of the distribution parameters.

        Raises
        ------
        pint.DimensionalityError
            If the source and target units are incompatible.
        """
        ureg = pint.UnitRegistry()
        self.scale = ureg.Quantity(self.scale, from_unit).to(to_unit).magnitude
        self.loc = ureg.Quantity(self.loc, from_unit).to(to_unit).magnitude
        
    def copy(self) -> "ArcsineDistribution":
        """ 
        Create a deep copy of the ArcsineDistribution.

        Returns
        -------
        ArcsineDistribution
            A new ArcsineDistribution instance with the same attributes as the current ArcsineDistribution.
        """
        return ArcsineDistribution(scale=self.scale, loc=self.loc)
    