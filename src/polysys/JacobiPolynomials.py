import numpy as np
from numpy.typing import NDArray

from scipy.special import jacobi
from scipy.special import beta as beta_function 

from ..distributions import BetaDistribution
from .PolynomialSystem import PolynomialSystem, _probability_sqnorm


class JacobiPolynomials(PolynomialSystem):
    """ 
    Implementation of Jacobi polynomial system.

    Parameters
    ----------
    alpha : float
        First parameter of Jacobi polynomials.
    beta : float
        Second parameter of Jacobi polynomials.
    orthonormal : bool, default is False
        If True, use orthonormal polynomials. Default is False.
    """

    def __init__(
        self, 
        alpha: float,
        beta: float,
        orthonormal: bool = False,
    ) -> None:
        """ 
        Initialize Jacobi polynomial system.
        """
        if not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise TypeError("Alpha must be a real number.")
        alpha = np.float64(alpha)

        if not isinstance(beta, (int, float, np.integer, np.floating)):
            raise TypeError("Beta must be a real number.")
        beta = np.float64(beta)

        if alpha <= -1 or beta <= -1:
            raise ValueError("Alpha and beta must be greater than -1.")
        
        if not isinstance(orthonormal, (bool, np.bool_)):
            raise TypeError("Parameter 'orthonormal' must be a boolean.")
        
        super().__init__(orthonormal=bool(orthonormal))
        self.alpha = alpha
        self.beta = beta
        self.generator = jacobi

    def gpc_distribution(self) -> BetaDistribution:
        """
        Return the generalized polynomial chaos distribution associated with Jacobi polynomials.

        Returns
        -------
        gpc_dist : BetaDistribution
            Beta distribution with parameters (beta+1, alpha+1) on [-1, 1].
        """
        return BetaDistribution(self.beta+1, self.alpha+1, scale=2, loc=-1)

    def _raw_sqnorm(self, deg: int) -> NDArray[np.float64]:
        """
        Return the raw squared norms of the Jacobi polynomials up to degree `deg`.

        Parameters
        ----------
        deg : int
            Maximum polynomial degree to evaluate.

        Returns
        -------
        y : NDArray[np.float64]
            Probability-weighted squared norms for degrees 0 through `deg`.
        """
        mass = (
            2 ** (self.alpha + self.beta + 1) 
            * beta_function(self.alpha + 1, self.beta + 1)
        )
        return _probability_sqnorm(self.generator, deg, mass, self.alpha, self.beta)

    def orthonormalize(self) -> "JacobiPolynomials":
        """
        Return an orthonormal version of the Jacobi polynomial system.

        Returns
        -------
        orthonormal_polysys : JacobiPolynomials
            A new instance of the Jacobi polynomial system with orthonormal polynomials.
        """
        return JacobiPolynomials(self.alpha, self.beta, orthonormal=True)
    