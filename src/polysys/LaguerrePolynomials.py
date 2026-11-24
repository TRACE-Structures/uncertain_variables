import numpy as np
from numpy.typing import NDArray

from scipy.special import laguerre

from ..distributions import ExponentialDistribution
from .PolynomialSystem import PolynomialSystem, _probability_sqnorm


class LaguerrePolynomials(PolynomialSystem):
    """ 
    Implementation of Laguerre polynomial system.

    Parameters
    ----------
    orthonormal : bool, default is False
        If True, use orthonormal polynomials. Default is False.
    """

    def __init__(self, orthonormal: bool = False) -> None:
        """
        Initialize the Laguerre polynomial system.
        """
        super().__init__(orthonormal=orthonormal)
        self.generator = laguerre

    def gpc_distribution(self) -> ExponentialDistribution:
        """
        Return the generalized polynomial chaos distribution associated with Laguerre polynomials.

        Returns
        -------
        gpc_dist : ExponentialDistribution
            Exponential distribution with rate parameter 1.
        """
        gpc_dist = ExponentialDistribution(1)
        return gpc_dist
    
    def _raw_sqnorm(self, deg: int | np.integer) -> NDArray[np.float64]:
        """
        Return the raw squared norms of the Laguerre polynomials up to degree `deg`.

        Parameters
        ----------
        deg : int | np.integer
            Maximum polynomial degree to evaluate.

        Returns
        -------
        y : NDArray[np.float64]
            Probability-weighted squared norms for degrees 0 through `deg`.
        """
        return _probability_sqnorm(self.generator, deg, 1)
    
    def _orthonormalize(self) -> "LaguerrePolynomials":
        """
        Return an orthonormal version of the Laguerre polynomial system.
        """
        return LaguerrePolynomials(orthonormal=True)
