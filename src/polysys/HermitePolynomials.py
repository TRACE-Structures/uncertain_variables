import numpy as np
from numpy.typing import NDArray

from scipy.special import hermitenorm as hermite

from ..distributions import NormalDistribution
from .PolynomialSystem import PolynomialSystem, _probability_sqnorm


class HermitePolynomials(PolynomialSystem):
    """ 
    Implementation of Hermite polynomial system.

    Parameters
    ----------
    orthonormal : bool, default is False
        If True, use orthonormal polynomials. Default is False.
    """

    def __init__(self, orthonormal: bool = False) -> None:
        """ 
        Initialize Hermite polynomial system.
        """
        if not isinstance(orthonormal, (bool, np.bool_)):
            raise TypeError("Parameter 'orthonormal' must be a boolean.")
        
        super().__init__(orthonormal=bool(orthonormal))
        self.generator = hermite

    def gpc_distribution(self) -> NormalDistribution:
        """
        Return the generalized polynomial chaos distribution associated with Hermite polynomials.

        Returns
        -------
        gpc_dist : NormalDistribution
            Standard normal distribution.
        """
        return NormalDistribution(0, 1)

    def _raw_sqnorm(self, deg: int) -> NDArray[np.float64]:
        """
        Return the raw squared norms of the Hermite polynomials up to degree `deg`.

        Parameters
        ----------
        deg : int
            Maximum polynomial degree to evaluate.

        Returns
        -------
        y : NDArray[np.float64]
            Probability-weighted squared norms for degrees 0 through `deg`.
        """
        return _probability_sqnorm(self.generator, deg, np.sqrt(2 * np.pi))

    def orthonormalize(self) -> "HermitePolynomials":
        """
        Return an orthonormal version of the Hermite polynomial system.

        Returns
        -------
        orthonormal_polysys : HermitePolynomials
            A new instance of the Hermite polynomial system with orthonormal polynomials.
        """
        return HermitePolynomials(orthonormal=True)
    