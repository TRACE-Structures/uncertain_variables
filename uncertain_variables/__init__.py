__version__ = "0.1.7"

from .polysys import *
from .distributions import *
from .variable_set import *
from .variable import *


def generate_stndrn_variable_set(sigmas):
    Q = VariableSet()
    for i in range(len(sigmas)):
        s = Variable('pn_' + str(i+1), NormalDistribution(0, sigmas[i]))
        Q.add(s)
    return Q