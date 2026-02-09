''' This module implements various polynomial systems used in polynomial chaos expansions.

    Classes
    -------
    PolynomialSystem
        Abstract base class for polynomial systems.

    NormalizedPolynomials
        Wrapper class for normalized polynomial systems.
        
    LegendrePolynomials
        Implementation of Legendre polynomial system.
        
    HermitePolynomials
        Implementation of Hermite polynomial system.
        
    JacobiPolynomials
        Implementation of Jacobi polynomial system.
        
    ChebyshevTPolynomials
        Implementation of Chebyshev polynomials of the first kind.
        
    ChebyshevUPolynomials
        Implementation of Chebyshev polynomials of the second kind.
        
    LaguerrePolynomials
        Implementation of Laguerre polynomial system.
                
    Functions
    ---------
    syschar_to_polysys(syschar)
        Convert a system characteristic string to the corresponding polynomial system.'''

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
                'j': JacobiPolynomials.normalized(),
                'm': Monomials
                }
    
    polysys = poly_dict[syschar]
    return polysys
