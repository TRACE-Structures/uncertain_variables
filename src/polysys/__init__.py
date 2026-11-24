''' This module implements various polynomial systems used in polynomial chaos expansions.'''

from .PolynomialSystem import PolynomialSystem
from .ChebyshevTPolynomials import ChebyshevTPolynomials
from .ChebyshevUPolynomials import ChebyshevUPolynomials
from .HermitePolynomials import HermitePolynomials
from .JacobiPolynomials import JacobiPolynomials
from .LaguerrePolynomials import LaguerrePolynomials
from .LegendrePolynomials import LegendrePolynomials
from .Monomials import MonomialPolynomials

def syschar_to_polysys(syschar):
    """ Convert a system characteristic string to the corresponding polynomial system.

        Parameters
        ----------
        syschar : str
            Characteristic string representing the polynomial system.

        Returns
        -------
        polysys : type or PolynomialSystem
            Uppercase codes and ``m`` return classes; lowercase codes return
            orthonormal instances. Jacobi's ``J`` class requires parameters."""
    
    poly_dict = {'H': HermitePolynomials,
                'h': HermitePolynomials(orthonormal=True),
                'P': LegendrePolynomials,
                'p': LegendrePolynomials(orthonormal=True),
                'T': ChebyshevTPolynomials,
                't': ChebyshevTPolynomials(orthonormal=True),
                'U': ChebyshevUPolynomials,
                'u': ChebyshevUPolynomials(orthonormal=True),
                'L': LaguerrePolynomials,
                'l': LaguerrePolynomials(orthonormal=True),
                'J': JacobiPolynomials,
                'm': MonomialPolynomials
                }
    
    polysys = poly_dict[syschar]
    return polysys
