"""Tests for the SemiCircleDistribution class."""

import numpy as np
import pytest

from uncertain_variables.polysys.ChebyshevUPolynomials import ChebyshevUPolynomials
from .TestDistribution import TestDistribution as DistributionTestClass

from uncertain_variables.distributions import SemiCircleDistribution


RADIUS = 2


class TestSemiCircleDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return SemiCircleDistribution(1)

    @pytest.fixture
    def support(self):
        return -RADIUS, RADIUS
    
	# =====================================================================
	# 1. API / basic behaviour
	# =====================================================================
    def test_init_stores_radius(self, dist):
        assert dist.radius == 1

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "semicircle"

def test_get_dist_params(self, dist):
    assert dist.get_dist_params() == RADIUS

def test_eq(self):
    d1 = SemiCircleDistribution(RADIUS)
    d2 = SemiCircleDistribution(RADIUS)
    d3 = SemiCircleDistribution(1)
    assert d1 == d2
    assert d1 != d3

def test_repr(self):
    assert repr(SemiCircleDistribution(RADIUS)) == "W({})".format(RADIUS)

def test_base_distribution_has_unit_radius(self, dist):
    germ = dist.get_base_dist()
    assert isinstance(germ, SemiCircleDistribution)
    assert germ.radius == 1

# =====================================================================
# 2. Mathematical identities (semicircle-specific parts)
# =====================================================================
def test_pdf_is_zero_outside_support(self, dist):
    x = np.array([-RADIUS - 1, RADIUS + 1])
    assert np.allclose(dist.pdf(x), 0)

def test_pdf_at_center(self, dist):
    assert np.isclose(dist.pdf(0), 2 / (np.pi * RADIUS))

def test_cdf_limits_and_symmetry(self, dist):
    assert dist.cdf(-RADIUS) == 0
    assert dist.cdf(RADIUS) == 1
    assert np.isclose(dist.cdf(0), 0.5)
    x = np.array([0.25, 1.0])
    assert np.allclose(dist.cdf(-x), 1 - dist.cdf(x))

def test_invcdf_inverts_cdf(self, dist):
    q = np.linspace(0.01, 0.99, 50)
    assert np.allclose(dist.cdf(dist.invcdf(q)), q)

def test_base_round_trip(self, dist):
    q = np.linspace(0.01, 0.99, 50)
    assert np.allclose(dist.invcdf(dist.cdf(dist.invcdf(q))), dist.invcdf(q))

# =====================================================================
# 3. Reference values
# =====================================================================
def test_moments_reference_values(self, dist):
    assert np.allclose(dist.moments(), [0, RADIUS ** 2 / 4, 0, -1])

# =====================================================================
# 4. Edge cases / invalid parameters
# =====================================================================
@pytest.mark.parametrize("radius", [0, -1])
def test_invalid_parameters_raise_assertion_error(self, radius):
    with pytest.raises(Exception):
        SemiCircleDistribution(radius)
