import numpy as np
from numpy.typing import NDArray

from scipy.special import chebyu

from ..distributions.SemiCircleDistribution import SemiCircleDistribution
from .PolynomialSystem import PolynomialSystem, _probability_sqnorm


class ChebyshevUPolynomials(PolynomialSystem):
    """ 
    Implementation of Chebyshev polynomials of the second kind.

    Parameters
    ----------
    orthonormal : bool, default is False
        If True, use orthonormal polynomials. Default is False.
    """

    def __init__(self, orthonormal: bool = False) -> None:
        """ 
        Initialize Chebyshev polynomials of the second kind.
        """
        if not isinstance(orthonormal, (bool, np.bool_)):
            raise TypeError("Parameter 'orthonormal' must be a boolean.")
        
        super().__init__(orthonormal=bool(orthonormal))
        self.generator = chebyu

    def gpc_distribution(self) -> SemiCircleDistribution:
        """
        Return the generalized polynomial chaos distribution associated with second kind Chebyshev polynomials.

        Returns
        -------
        gpc_dist : SemiCircleDistribution
            Semi-circle distribution.
        """
        return SemiCircleDistribution(1)

    def _raw_sqnorm(self, deg: int) -> NDArray[np.float64]:
        """
        Return the raw squared norms of the second kind Chebyshev polynomials up to degree `deg`.

        Parameters
        ----------
        deg : int
            Maximum polynomial degree to evaluate.

        Returns
        -------
        y : NDArray[np.float64]
            Probability-weighted squared norms for degrees 0 through `deg`.
        """
        return _probability_sqnorm(self.generator, deg, np.pi / 2)

    def orthonormalize(self) -> "ChebyshevUPolynomials":
        """
        Return an orthonormal version of the second kind Chebyshev polynomial system.

        Returns
        -------
        orthonormal_polysys : ChebyshevUPolynomials
            A new instance of the second kind Chebyshev polynomial system with orthonormal polynomials.
        """
        return ChebyshevUPolynomials(orthonormal=True)
