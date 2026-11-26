import pint
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta

from .Distribution import Distribution
from .UniformDistribution import UniformDistribution


class BetaDistribution(Distribution):
    """ 
    Beta probability distribution on a finite interval.
    
    Attributes
    ----------
    alpha : float
        First shape parameter of the beta distribution.
    beta : float
        Second shape parameter of the beta distribution.
    scale : float, default is 1
        Length of the interval.
    loc : float, default is 0
        Starting point of the interval.
    """
        
    def __init__(
        self,
        alpha: float,
        beta: float,
        scale: float = 1,
        loc: float = 0,
    ) -> None:
        """
        Initialize a beta distribution with shape parameters 'a' and 'b'
        on the interval '[loc, loc + scale]'.
        """
        if not (isinstance(alpha, (int, float, np.integer, np.floating))):
            raise TypeError("Parameter 'alpha' must be a numeric value.")
        alpha = np.float64(alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Parameter 'alpha' must be a positive finite number.")

        if not (isinstance(beta, (int, float, np.integer, np.floating))):
            raise TypeError("Parameter 'beta' must be a numeric value.")
        beta = np.float64(beta)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("Parameter 'beta' must be a positive finite number.")

        if not (isinstance(scale, (int, float, np.integer, np.floating))):
            raise TypeError("Parameter 'scale' must be a numeric value.")
        scale = np.float64(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Parameter 'scale' must be a positive finite number.")

        if not (isinstance(loc, (int, float, np.integer, np.floating))):
            raise TypeError("Parameter 'loc' must be a numeric value.")
        loc = np.float64(loc)
        if not np.isfinite(loc):
            raise ValueError("Parameter 'loc' must be a finite number.")

        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.loc = loc

    def __repr__(self) -> str:
        """ 
        Returns the string representation of the BetaDistribution object.
        
        Returns
        -------
        repr_string : str
            String representation of the BetaDistribution object.
        """
        return "Beta({:.2f}, {:.2f}), scale={:.2f}, loc={:.2f}".format(self.alpha, self.beta, self.scale, self.loc)
    
    def __eq__(self, other: "BetaDistribution") -> bool:
        """ 
        Check if two BetaDistribution objects are equal.
        
        Parameters
        ----------
        other : BetaDistribution
            Another BetaDistribution object to compare with.
            
        Returns
        -------
        is_equal : bool
            True if the two BetaDistribution objects are equal, False otherwise.
        """
        if not isinstance(other, BetaDistribution):
            return False
        
        return(self.alpha == other.alpha) and (self.beta == other.beta) and (self.scale == other.scale) and (self.loc == other.loc)
    
    def get_dist_type(self) -> str:
        """ 
        Return the type of the distribution.
        
        Returns
        -------
        dist_type : str
            Type of the distribution.
        """
        return "beta"
    
    def get_dist_params(self) -> dict[str, np.float64]:
        """ 
        Return the parameters of the beta distribution.

        Returns
        -------
        params : dict
            Distribution parameters {"a": a, "b": b, "scale": scale, "loc": loc}, where a and b are 
            the shape parameters of the beta distribution, loc is the start of the interval, 
            and scale is the length of the interval.
        """
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "scale": self.scale,
            "loc": self.loc
        }

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """ 
        Return the probability density function of the beta distribution, evaluated at x.
        
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

        return beta.pdf(x, self.alpha, self.beta, scale=self.scale, loc=self.loc)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """ 
        Return the cumulative distribution function of the beta distribution, evaluated at x.
        
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

        return beta.cdf(x, self.alpha, self.beta, scale=self.scale, loc=self.loc)

    def ppf(self, y: ArrayLike) -> NDArray[np.float64]:
        """ 
        Return the percentage point function (inverse of cdf) of the beta distribution, evaluated at y.
        
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

        return beta.ppf(y, self.alpha, self.beta, scale=self.scale, loc=self.loc)

    def mean(self) -> np.float64:
        """ 
        Return the mean of the beta distribution.

        Returns
        -------
        mean : np.float64
            Mean of the beta distribution.
        """
        return beta.mean(self.alpha, self.beta, scale=self.scale, loc=self.loc)

    def var(self) -> np.float64:
        """ 
        Return the variance of the beta distribution.

        Returns
        -------
        var : np.float64
            Variance of the beta distribution.
        """
        return beta.var(self.alpha, self.beta, scale=self.scale, loc=self.loc)

    def skew(self) -> np.float64:
        """ 
        Return the skewness of the beta distribution.

        Returns
        -------
        skew : np.float64
            Skewness of the beta distribution.
        """
        return beta.stats(self.alpha, self.beta, scale=self.scale, loc=self.loc, moments='s')

    def kurt(self) -> np.float64:
        """ 
        Return the kurtosis of the beta distribution.

        Returns
        -------
        kurt : np.float64
            Kurtosis of the beta distribution.
        """
        return beta.stats(self.alpha, self.beta, scale=self.scale, loc=self.loc, moments='k')

    def sample(
        self,
        n: int,
        method: Literal["MC", "QMC_Halton", "QMC_LHS", "QMC_Sobol"] = "MC", 
        seed: int | None = None
    ) -> np.NDArray[np.float64]:
        """
        Return n samples from the beta distribution.
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
            Generated samples from the beta distribution.
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
    ) -> "BetaDistribution":
        """ 
        Return a translated and scaled beta distribution.

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
        new_dist : BetaDistribution
            Translated and scaled beta distribution.
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

        new_loc = self.loc * scale + shift
        new_scale = self.scale * scale

        new_dist = BetaDistribution(alpha=self.alpha, beta=self.beta, loc=new_loc, scale=new_scale)
        return new_dist

    def get_shift(self) -> np.float64:
        """ 
        Return the shift of the beta distribution, i.e., the location parameter.

        Returns
        -------
        shift : np.float64
            Shift parameter of the beta distribution.
        """
        return self.loc

    def get_scale(self) -> np.float64:
        """ 
        Return the scale of the beta distribution, i.e., the length of the interval.

        Returns
        -------
        scale : np.float64
            Scale parameter of the beta distribution.
        """
        return self.scale

    def fix_moments(
        self,
        mean: float,
        var: float,
        scale: float = 1,
        loc: float = 0
    ) -> "BetaDistribution":
        """
        Fix the beta distribution to have the specified mean, variance, location, and scale.

        Parameters
        ----------
        mean : float
            Desired mean of the distribution.
        var : float
            Desired positive variance of the distribution.
        scale : float, default is 1
            Length of the interval. 
        loc : float, default is 0
            Starting point of the interval.

        Returns
        -------
        new_dist : BetaDistribution
            Beta distribution with the specified mean, variance, location, and scale.
        """
        if not isinstance(mean, (int, float, np.integer, np.floating)):
            raise ValueError("Parameter 'mean' must be a numeric value.")
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

        if not isinstance(scale, (int, float, np.integer, np.floating)):
            raise TypeError("Parameter 'scale' must be a numeric value.")
        scale = np.float64(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Parameter 'scale' must be a finite positive value.")

        if not isinstance(loc, (int, float, np.integer, np.floating)):
            raise TypeError("Parameter 'loc' must be a numeric value.")
        loc = np.float64(loc)
        if not np.isfinite(loc):
            raise ValueError("Parameter 'loc' must be a finite value.")

        m = (mean - loc) / scale
        v = var / scale**2

        if not 0 < m < 1:
            raise ValueError("Parameter 'mean' must lie strictly inside the support.")
        if not 0 < v < m * (1 - m):
            raise ValueError("Parameter 'var' is incompatible with the beta distribution.")

        temp = m * (1 - m) / v - 1
        alpha = m * temp
        beta = (1 - m) * temp

        return BetaDistribution(alpha=alpha, beta=beta, scale=scale, loc=loc)

    def gpc_dist(
        self,
        alpha: float,
        beta: float
    ) -> "BetaDistribution":
        """ 
        Return the GPC germ distribution for the beta distribution.

        Parameters
        ----------
        alpha : float
            First positive shape parameter of the germ beta distribution.
        beta : float
            Second positive shape parameter of the germ beta distribution.

        Returns
        -------
        germ_dist : BetaDistribution
            Beta germ distribution with shape parameters 'alpha' and 'beta' on the standard interval '[-1, 1]'.
        """
        return BetaDistribution(alpha=alpha, beta=beta, scale=2, loc=-1)

    def gpc_polysys(self, orthonormal: bool = True) -> "JacobiPolynomials":
        """ 
        Return the GPC polynomial system for the beta distribution.

        Parameters
        ----------
        orthonormal : bool, default is True
            Whether to return an orthonormal polynomial system.

        Returns
        -------
        polysys : JacobiPolynomials
            GPC polynomial system for the beta distribution.

        Raises
        ------
        ValueError
            If no polynomial system exists for the current distribution.
        """
        if not isinstance(orthonormal, (bool, np.bool_)):
            raise TypeError("Parameter 'orthonormal' must be a boolean value.")
        
        if self.scale == 2 and self.loc == -1:
            from ..polysys import JacobiPolynomials

            return JacobiPolynomials(self.beta - 1, self.alpha - 1, orthonormal=orthonormal)
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

    def copy(self) -> "BetaDistribution":
        """ 
        Create a deep copy of the BetaDistribution.

        Returns
        -------
        BetaDistribution
            A new BetaDistribution instance with the same attributes as the current BetaDistribution.
        """
        return BetaDistribution(alpha=self.alpha, beta=self.beta, scale=self.scale, loc=self.loc)
    