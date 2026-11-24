import numpy as np
from numpy.typing import NDArray

from scipy.special import chebyt

from ..distributions import ArcsineDistribution
from .PolynomialSystem import PolynomialSystem, _probability_sqnorm


class ChebyshevTPolynomials(PolynomialSystem):
    """ 
    Implementation of Chebyshev polynomials of the first kind.

    Parameters
    ----------
    orthonormal : bool, default is False
        If True, use orthonormal polynomials. Default is False.
    """
    
    def __init__(self, orthonormal: bool = False) -> None:
        """
        Initialize Chebyshev polynomials of the first kind.
        """
        super().__init__(orthonormal=orthonormal)
        self.generator = chebyt

    def gpc_distribution(self) -> ArcsineDistribution:
        """
        Return the generalized polynomial chaos distribution associated with Chebyshev polynomials of the first kind.

        Returns
        -------
        gpc_dist : ArcsineDistribution
            Arc-sine distribution.
        """
        return ArcsineDistribution(scale=2, loc=-1)

    def _raw_sqnorm(self, deg: int | np.integer) -> NDArray[np.float64]:
        """
        Return the raw squared norms of the Chebyshev polynomials of the first kind up to degree `deg`.

        Parameters
        ----------
        deg : int | np.integer
            Maximum polynomial degree to evaluate.

        Returns
        -------
        y : NDArray[np.float64]
            Probability-weighted squared norms for degrees 0 through `deg`.
        """
        return _probability_sqnorm(self.generator, deg, np.pi)

    def _orthonormalize(self) -> "ChebyshevTPolynomials":
        """
        Return an orthonormal version of the first kind Chebyshev polynomial system.
        """
        return ChebyshevTPolynomials(orthonormal=True)
