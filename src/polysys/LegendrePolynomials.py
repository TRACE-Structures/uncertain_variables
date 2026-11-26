import numpy as np
from numpy.typing import NDArray

from scipy.special import legendre

from ..distributions import UniformDistribution
from .PolynomialSystem import PolynomialSystem, _probability_sqnorm


class LegendrePolynomials(PolynomialSystem):
    """ 
    Implementation of Legendre polynomial system.

    Parameters
    ----------
    orthonormal : bool, default is False
        If True, use orthonormal polynomials. Default is False.
    """

    def __init__(self, orthonormal: bool = False) -> None:
        """
        Initialize the Legendre polynomial system.
        """
        if not isinstance(orthonormal, (bool, np.bool_)):
            raise TypeError("Parameter 'orthonormal' must be a boolean.")
        
        super().__init__(orthonormal=bool(orthonormal))
        self.generator = legendre

    def gpc_distribution(self) -> UniformDistribution:
        """
        Return the generalized polynomial chaos distribution associated with Legendre polynomials.

        Returns
        -------
        gpc_dist : UniformDistribution
            Uniform distribution over [-1, 1].
        """
        return UniformDistribution(-1, 1)
    
    def _raw_sqnorm(self, deg: int) -> NDArray[np.float64]:
        """
        Return the raw squared norms of the Legendre polynomials up to degree `deg`.

        Parameters
        ----------
        deg : int
            Maximum polynomial degree to evaluate.

        Returns
        -------
        y : NDArray[np.float64]
            Probability-weighted squared norms for degrees 0 through `deg`.
        """
        return _probability_sqnorm(self.generator, deg, 2)
    
    def orthonormalize(self) -> "LegendrePolynomials":
        """
        Return an orthonormal version of the Legendre polynomial system.

        Returns
        -------
        orthonormal_polysys : LegendrePolynomials
            A new instance of the Legendre polynomial system with orthonormal polynomials.
        """
        return LegendrePolynomials(orthonormal=True)
    