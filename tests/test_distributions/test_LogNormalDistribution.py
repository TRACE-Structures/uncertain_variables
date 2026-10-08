"""Tests for the LogNormalDistribution class."""

import numpy as np
import pytest

from .TestDistribution import TestDistribution as DistributionTestClass
from uncertain_variables.distributions import NormalDistribution
from uncertain_variables.distributions import LogNormalDistribution


MU, SIGMA = 2, 5


class TestLogNormalDistribution(DistributionTestClass):

	KURT_IS_EXCESS = True

	# ---- fixtures -------------------------------------------------------
	@pytest.fixture
	def dist(self):
		return LogNormalDistribution(0, 1)

	@pytest.fixture
	def support(self):
		return 0, np.inf

	# =====================================================================
	# 1. API / basic behaviour
	# =====================================================================
	def test_init_stores_parameters(self):
		d = LogNormalDistribution(MU, SIGMA)
		assert np.allclose((d.mu, d.sigma), (MU, SIGMA), atol=self.ATOL, rtol=self.RTOL)

	def test_init_defaults_to_standard_lognormal(self):
		d = LogNormalDistribution()
		assert np.allclose((d.mu, d.sigma), (0, 1), atol=self.ATOL, rtol=self.RTOL)

	def test_get_dist_type(self, dist):
		assert dist.get_dist_type() == "lognorm"

	def test_get_dist_params(self, dist):
		assert np.allclose(dist.get_dist_params(), (0, 1), atol=self.ATOL, rtol=self.RTOL)

	def test_eq(self):
		d1 = LogNormalDistribution(MU, SIGMA)
		d2 = LogNormalDistribution(MU, SIGMA)
		d3 = LogNormalDistribution(0, 1)
		d4 = NormalDistribution(0, 1)
		assert d1 == d2
		assert d1 != d3
		assert d3 != d4

	def test_repr(self):
		d = LogNormalDistribution(MU, SIGMA)
		assert repr(d) == "lnN({}, {:.2f})".format(MU, SIGMA ** 2)

	# =====================================================================
	# 2. Mathematical identities  (lognormal-specific parts)
	# =====================================================================
	def test_pdf_is_zero_at_and_below_zero(self, dist):
		x = np.array([-1.0, 0.0])
		assert np.allclose(dist.pdf(x), 0, atol=self.ATOL, rtol=self.RTOL)

	def test_cdf_is_zero_at_and_below_zero(self, dist):
		x = np.array([-1.0, 0.0])
		assert np.allclose(dist.cdf(x), 0, atol=self.ATOL, rtol=self.RTOL)

	def test_mean_reference_value(self):
		d = LogNormalDistribution(MU, SIGMA)
		expected = np.exp(MU + SIGMA ** 2 / 2)
		assert np.isclose(d.mean(), expected, atol=self.ATOL, rtol=self.RTOL)

	def test_variance_reference_value(self):
		d = LogNormalDistribution(MU, SIGMA)
		expected = (np.exp(SIGMA ** 2) - 1) * np.exp(2 * MU + SIGMA ** 2)
		assert np.isclose(d.var(), expected, atol=self.ATOL, rtol=self.RTOL)

	def test_base_distribution_is_standard_normal(self, dist):
		germ = dist.get_base_dist()
		assert isinstance(germ, NormalDistribution)
		assert np.allclose((germ.mu, germ.sigma), (0, 1), atol=self.ATOL, rtol=self.RTOL)

	def test_base2dist(self):
		d = LogNormalDistribution(MU, SIGMA)
		y = np.linspace(-3, 3, 50)
		x = np.exp(MU + SIGMA * y)
		assert np.allclose(d.base2dist(y), x, atol=self.ATOL, rtol=self.RTOL)

	def test_dist2base(self):
		d = LogNormalDistribution(MU, SIGMA)
		y = np.linspace(-3, 3, 50)
		x = d.base2dist(y)
		assert np.allclose(d.dist2base(x), y, atol=self.ATOL, rtol=self.RTOL)

	# =====================================================================
	# 3. Reference values
	# =====================================================================
	def test_moments_reference_values(self):
		d = LogNormalDistribution(MU, SIGMA)
		mean = np.exp(MU + SIGMA ** 2 / 2)
		var = (np.exp(SIGMA ** 2) - 1) * np.exp(2 * MU + SIGMA ** 2)
		skew = (np.exp(SIGMA ** 2) + 2) * np.sqrt(np.exp(SIGMA ** 2) - 1)
		kurt = (
			np.exp(4 * SIGMA ** 2)
			+ 2 * np.exp(3 * SIGMA ** 2)
			+ 3 * np.exp(2 * SIGMA ** 2)
			- 6
		)
		assert np.allclose(d.moments(), [mean, var, skew, kurt], atol=self.ATOL, rtol=self.RTOL)

	def test_pdf_matches_with_scipy(self):
		from scipy.stats import lognorm
		d = LogNormalDistribution(MU, SIGMA)
		x = np.linspace(0.01, 100, 50)
		expected_pdf = lognorm.pdf(x, s=SIGMA, scale=np.exp(MU))
		assert np.allclose(d.pdf(x), expected_pdf, atol=self.ATOL, rtol=self.RTOL)

	def test_cdf_matches_with_scipy(self):
		from scipy.stats import lognorm
		d = LogNormalDistribution(MU, SIGMA)
		x = np.linspace(0.01, 100, 50)
		expected_cdf = lognorm.cdf(x, s=SIGMA, scale=np.exp(MU))
		assert np.allclose(d.cdf(x), expected_cdf, atol=self.ATOL, rtol=self.RTOL)

	def test_invcdf_matches_with_scipy(self):
		from scipy.stats import lognorm
		d = LogNormalDistribution(MU, SIGMA)
		q = np.linspace(0.01, 0.99, 50)
		expected_invcdf = lognorm.ppf(q, s=SIGMA, scale=np.exp(MU))
		assert np.allclose(d.invcdf(q), expected_invcdf, atol=self.ATOL, rtol=self.RTOL)

	# =====================================================================
	# 4. Edge cases / invalid parameters
	# =====================================================================
	@pytest.mark.parametrize("mu,sigma", [(0, 0), (0, -1)])
	def test_invalid_parameters_raise_assertion_error(self, mu, sigma):
		with pytest.raises(Exception):
			LogNormalDistribution(mu, sigma)
