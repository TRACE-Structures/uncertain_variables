from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..distributions.Distribution import Distribution


def _probability_sqnorm(
    generator: Callable[..., Any],
    deg: int,
    weight_mass: float,
    *args: Any,
) -> NDArray[np.float64]:
    """
    Convert SciPy's weight-integral norms to probability-weighted norms.

    Parameters
    ----------
    generator : Callable[..., Any]
        Function that generates the polynomial of a given degree.
    deg : int
        Maximum polynomial degree to evaluate.
    weight_mass : float
        Integral of the weight function over the domain.
    *args : Any
        Additional arguments to pass to the generator function.

    Returns
    -------
    norms : NDArray[np.float64]
        Array of probability-weighted squared norms for degrees up to `deg`.
    """
    norms = np.array([generator(n, *args).normcoef for n in range(1, deg + 1)])
    return np.concatenate(([1.0], norms**2 / weight_mass))


class PolynomialSystem(ABC):
    """
    Abstract base class for univariate polynomial systems.
    """
    generator: Callable[..., Any]
    
    def __init__(self, orthonormal: bool = False) -> None:
        """
        Initialize the polynomial system.

        Parameters
        ----------
        orthonormal : bool, default is False
            If True, use orthonormal polynomials. Default is False.
        """
        if not isinstance(orthonormal, (bool, np.bool_)):
            raise TypeError("Orthonormal must be a boolean.")
        
        self.orthonormal = bool(orthonormal)

    def evaluate(
        self, 
        deg: int, 
        x: ArrayLike, 
    ) -> NDArray[np.float64]:
        """
        Evaluate the polynomial system up to degree `deg` at points `x`.

        Parameters
        ----------
        deg : int
            Maximum polynomial degree to evaluate.
        x : ArrayLike
            One-dimensional array of real-valued evaluation points.

        Returns
        -------
        y : NDArray[np.float64]
            Matrix of shape (n_points, deg+1) containing evaluated polynomials.
            Column k corresponds to degree k.
        """
        if not isinstance(deg, (int, np.integer)):
            raise TypeError("Degree must be an integer.")
        if deg < 0:
            raise ValueError("Degree must be non-negative.")

        if np.isscalar(x):
            x = [x]
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError("Input points x must be one-dimensional.")
        if x.dtype.kind not in "iuf":
            raise TypeError("Input points x must be real numbers.")
        x = x.astype(np.float64, copy=False)

        y = np.column_stack([self.generator(d)(x) for d in range(deg+1)])
        if self.orthonormal:
            y /= np.sqrt(self._raw_sqnorm(deg))

        return y

    def evaluate_derivative(
        self,
        deg: int,
        order: int,
        x: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the `order`-th derivative of the polynomial system up to degree `deg` at points `x`.

        Parameters
        ----------
        deg : int
            Maximum polynomial degree to evaluate.
        order : int
            Order of the derivative to evaluate.
        x : ArrayLike
            One-dimensional array of real-valued evaluation points.

        Returns
        -------
        y : NDArray[np.float64]
            Matrix of shape (n_points, deg+1) containing evaluated derivatives of the polynomials.
            Column k corresponds to degree k.
        """
        if not isinstance(deg, (int, np.integer)):
            raise TypeError("Degree must be an integer.")
        if deg < 0:
            raise ValueError("Degree must be non-negative.")
        
        if not isinstance(order, (int, np.integer)):
            raise TypeError("Order must be an integer.")
        if order < 0:
            raise ValueError("Order must be non-negative.")

        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError("Input points x must be one-dimensional.")
        if x.dtype.kind not in "iuf":
            raise TypeError("Input points x must be real numbers.")
        x = x.astype(np.float64, copy=False)

        y = np.column_stack([self.generator(d).deriv(m=order)(x) for d in range(deg+1)])
        if self.orthonormal:
            y /= np.sqrt(self._raw_sqnorm(deg))

        return y

    def sqnorm(self, deg: int) -> NDArray[np.float64]:
        """
        Return the squared norm of the polynomial system up to degree `deg`.

        If the polynomial system is orthonormal, this will return an array of ones.

        Parameters
        ----------
        deg : int
            Highest polynomial degree to evaluate.

        Returns
        -------
        y : NDArray[np.float64]
            Squared norms under the probability distribution returned by
            `gpc_distribution`, in degree order, with shape (deg+1,).
        """
        if not isinstance(deg, (int, np.integer)):
            raise TypeError("Degree must be an integer.")
        if deg < 0:
            raise ValueError("Degree must be non-negative.")

        if self.orthonormal:
            return np.ones(deg + 1)
        return self._raw_sqnorm(deg)

    @abstractmethod
    def gpc_distribution(self) -> Distribution:
        """
        Return the probability distribution associated with
        the polynomial system.
        """
        ...

    @abstractmethod
    def _raw_sqnorm(self, deg: int) -> NDArray[np.float64]:
        """
        Return squared norms of the non-normalized polynomial system.
        """
        ...

    @abstractmethod
    def orthonormalize(self) -> "PolynomialSystem":
        """
        Abstract method to return an orthonormal version of the polynomial system.
        Subclasses must implement this method.
        """
        ...
