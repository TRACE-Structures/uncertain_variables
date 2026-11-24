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
    alpha : float | np.floating
        First parameter of Jacobi polynomials.
    beta : float | np.floating
        Second parameter of Jacobi polynomials.
    orthonormal : bool, default is False
        If True, use orthonormal polynomials. Default is False.
    """

    def __init__(
        self, 
        alpha: float | np.floating,
        beta: float | np.floating, 
        orthonormal: bool = False
    ) -> None:
        """ 
        Initialize Jacobi polynomial system.
        """
        if not np.isfinite(alpha) or not np.isfinite(beta):
            raise ValueError("Alpha and beta must be finite numbers.")
        if alpha <= -1 or beta <= -1:
            raise ValueError("Alpha and beta must be greater than -1.")
        
        super().__init__(orthonormal=orthonormal)
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

    def _raw_sqnorm(self, deg: int | np.integer) -> NDArray[np.float64]:
        """
        Return the raw squared norms of the Jacobi polynomials up to degree `deg`.

        Parameters
        ----------
        deg : int | np.integer
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

    def _orthonormalize(self) -> "JacobiPolynomials":
        """
        Return an orthonormal version of the Jacobi polynomial system.
        """
        return JacobiPolynomials(self.alpha, self.beta, orthonormal=True)
    