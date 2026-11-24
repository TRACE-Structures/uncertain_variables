"""
Polynomial systems for polynomial chaos expansions (PCE).

PolynomialSystem(ABC):
    Abstract base class for univariate polynomial systems.

ChebyshevTPolynomials(PolynomialSystem)
    Implementation of Chebyshev polynomials of the first kind.

ChebyshevUPolynomials(PolynomialSystem)
    Implementation of Chebyshev polynomials of the second kind.

HermitePolynomials(PolynomialSystem)
    Implementation of Hermite polynomial system.

JacobiPolynomials(PolynomialSystem)
    Implementation of Jacobi polynomial system.

LaguerrePolynomials(PolynomialSystem)
    Implementation of Laguerre polynomial system.

LegendrePolynomials(PolynomialSystem)
    Implementation of Legendre polynomial system.
"""

from .PolynomialSystem import PolynomialSystem
from .ChebyshevTPolynomials import ChebyshevTPolynomials
from .ChebyshevUPolynomials import ChebyshevUPolynomials
from .HermitePolynomials import HermitePolynomials
from .JacobiPolynomials import JacobiPolynomials
from .LaguerrePolynomials import LaguerrePolynomials
from .LegendrePolynomials import LegendrePolynomials
