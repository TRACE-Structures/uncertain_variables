from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Any

from ..distributions.Distribution import Distribution


def _probability_sqnorm(
    generator: Callable,
    deg: int,
    weight_mass: float,
    *args: Any,
) -> NDArray[np.floating]:
    """
    Convert SciPy's weight-integral norms to probability-weighted norms.

    Parameters
    ----------
    generator : Callable
        Function that generates the polynomial of a given degree.
    deg : int
        Maximum polynomial degree to evaluate.
    weight_mass : float
        Integral of the weight function over the domain.
    *args : Any
        Additional arguments to pass to the generator function.

    Returns
    -------
    norms : NDArray[np.floating]
        Array of probability-weighted squared norms for degrees up to `deg`.
    """
    if not callable(generator):
        raise TypeError("Generator must be callable.")
    if not isinstance(deg, (int, np.integer)):
        raise TypeError("Degree must be an integer.")
    if deg < 0:
        raise ValueError("Degree must be non-negative.")
    if weight_mass <= 0:
        raise ValueError("Weight mass must be positive.")

    norms = np.array([generator(n, *args).normcoef for n in range(1, deg + 1)])
    return np.concatenate(([1.0], norms**2 / weight_mass))


class PolynomialSystem(ABC):
    """
    Abstract base class for univariate polynomial systems.
    """
    def __init__(self, orthonormal: bool = False):
        """
        Initialize the polynomial system.

        Parameters
        ----------
        orthonormal : bool, optional
            If True, use orthonormal polynomials. Default is False.
        """
        self.orthonormal = orthonormal

    @abstractmethod
    def gpc_distribution(self) -> Distribution:
        """
        Return the probability distribution associated with
        the polynomial system.
        """
        ...

    def evaluate(
        self, 
        deg: int, 
        x: ArrayLike, 
        derivative: bool | int = False,
    ) -> NDArray[np.floating]:
        """
        Evaluate the polynomial system up to degree `deg` at points `x`.

        Parameters
        ----------
        deg : int
            Maximum polynomial degree to evaluate.
        x : ArrayLike
            One-dimensional array of real-valued evaluation points.
        derivative : bool | int, default=False
            If True, evaluate the derivative of the polynomials.
            If an integer n, evaluate the n-th derivative of the polynomials. Default is False.

        Returns
        -------
        y : NDArray[np.floating]
            Matrix of shape (n_points, deg+1) containing evaluated polynomials
            or first derivatives. Column k corresponds to degree k.
        """
        if not isinstance(deg, (int, np.integer)):
            raise TypeError("Degree must be an integer.")
        if deg < 0:
            raise ValueError("Degree must be non-negative.")
        if not isinstance(derivative, (bool, int)):
            raise TypeError("Derivative flag must be a boolean or an integer.")
        
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("Input points x must be one-dimensional.")
        if x.dtype.kind not in "iuf":
            raise TypeError("Input points x must be real numbers.")
        return self._evaluate(int(deg), x.astype(float, copy=False), derivative)

    @abstractmethod
    def _evaluate(
        self,
        deg: int,
        x: NDArray[np.floating],
        derivative: bool | int = False,
    ) -> NDArray[np.floating]:
        """
        Abstract method to evaluate the polynomial system up to degree `deg` at points `x`.
        Subclasses must implement this method.
        """
        ...

    def sqnorm(self, deg: int) -> NDArray[np.float64]:
        """
        Return the squared norm of the polynomial system up to degree `deg`.

        Parameters
        ----------
        deg : int
            Highest polynomial degree to evaluate.

        Returns
        -------
        y : NDArray[np.floating64]
            Squared norms under the probability distribution returned by
            `gpc_distribution`, in degree order, with shape (deg+1,).
        """
        if not isinstance(deg, (int, np.integer)):
            raise TypeError("Degree must be an integer.")
        if deg < 0:
            raise ValueError("Degree must be non-negative.")

        return self._sqnorm(deg)

    @abstractmethod
    def _sqnorm(self, deg: int) -> NDArray[np.float64]:
        """
        Abstract method to return the squared norm of the polynomial systemup to degree `deg`.
        Subclasses must implement this method.
        """
        ...

    def orthonormalize(self) -> "PolynomialSystem":
        """
        Return the orthonormal version of the polynomial system.

        Returns
        -------
        normalized_system : PolynomialSystem
            A new instance of the polynomial system with orthonormal polynomials.
        """
        return self._orthonormalize()

    @abstractmethod
    def _orthonormalize(self) -> "PolynomialSystem":
        """
        Abstract method to return an orthonormal version of the polynomial system.
        Subclasses must implement this method.
        """
        ...