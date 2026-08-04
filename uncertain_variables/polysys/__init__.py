''' This module implements various polynomial systems used in polynomial chaos expansions.'''

from .PolynomialSystem import *
from .ChebyshevTPolynomials import *
from .ChebyshevUPolynomials import *
from .HermitePolynomials import *
from .JacobiPolynomials import *
from .LaguerrePolynomials import *
from .LegendrePolynomials import *
from .Monomials import *

def syschar_to_polysys(syschar):
    """ Convert a system characteristic string to the corresponding polynomial system.

        Parameters
        ----------
        syschar : str
            Characteristic string representing the polynomial system.

        Returns
        -------
        polysys : PolynomialSystem
            Corresponding polynomial system."""
    
    poly_dict = {'H': HermitePolynomials,
                'h': HermitePolynomials.normalized(),
                'P': LegendrePolynomials,
                'p': LegendrePolynomials.normalized(),
                'T': ChebyshevTPolynomials,
                't': ChebyshevTPolynomials.normalized(),
                'U': ChebyshevUPolynomials,
                'u': ChebyshevUPolynomials.normalized(),
                'L': LaguerrePolynomials,
                'l': LaguerrePolynomials.normalized(),
                'J': JacobiPolynomials,
                #'j': JacobiPolynomials.normalized(),
                'm': Monomials
                }
    
    polysys = poly_dict[syschar]
    return polysys
