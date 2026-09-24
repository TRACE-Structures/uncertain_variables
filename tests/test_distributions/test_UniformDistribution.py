"""Tests for the UniformDistribution class."""

import numpy as np
import pytest

from uncertain_variables.polysys.LegendrePolynomials import LegendrePolynomials

from .TestDistribution import TestDistribution as DistributionTestClass
from uncertain_variables.distributions import NormalDistribution

from uncertain_variables.distributions import UniformDistribution


A, B = 2, 5


class TestUniformDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return UniformDistribution(0, 1)

    @pytest.fixture
    def support(self):
        return 0, 1

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    def test_init_stores_parameters(self):
        d = UniformDistribution(A, B)
        assert np.allclose((d.a, d.b), (A, B), atol=self.ATOL, rtol=self.RTOL)

    def test_init_defaults_to_unit_interval(self):
        d = UniformDistribution()
        assert np.allclose((d.a, d.b), (0, 1), atol=self.ATOL, rtol=self.RTOL)

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "unif"

    def test_get_dist_params(self, dist):
        assert np.allclose(dist.get_dist_params(), (0, 1), atol=self.ATOL, rtol=self.RTOL)

    def test_eq(self):
        d1 = UniformDistribution(A, B)
        d2 = UniformDistribution(A, B)
        d3 = UniformDistribution(0, 1)
        d4 = NormalDistribution(0, 1)
        assert d1 == d2
        assert d1 != d3
        assert d3 != d4

    def test_repr(self):
        d = UniformDistribution(A, B)
        assert repr(d) == f"U({A}, {B})"

    def test_translate_returns_uniform(self, dist):
        translated = dist.translate(1, 2)
        assert isinstance(translated, UniformDistribution)
        assert np.allclose((translated.a, translated.b), (0.5, 2.5), atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 2. Mathematical identities  (uniform-specific parts)
    # =====================================================================
    def test_pdf_is_constant_on_the_support(self, dist):
        x = np.linspace(dist.a, dist.b, 100)
        pdf_values = dist.pdf(x)
        assert np.allclose(pdf_values, pdf_values[0], atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_is_zero_outside_the_support(self, dist):
        x = np.array([dist.a - 1, dist.b + 1])
        pdf_values = dist.pdf(x)
        assert np.allclose(pdf_values, 0, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_is_zero_below_support(self, dist):
        x = dist.a - 1
        cdf_value = dist.cdf(x)
        assert np.allclose(cdf_value, 0, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_is_one_above_support(self, dist):
        x = dist.b + 1
        cdf_value = dist.cdf(x)
        assert np.allclose(cdf_value, 1, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_is_linear_on_the_support(self, dist):
        x = np.linspace(dist.a, dist.b, 100)
        cdf_values = dist.cdf(x)
        expected_values = (x - dist.a) / (dist.b - dist.a)
        assert np.allclose(cdf_values, expected_values, atol=self.ATOL, rtol=self.RTOL)

    def test_mean_is_midpoint_of_support(self, dist):
        mean_value = dist.mean()
        expected_value = (dist.a + dist.b) / 2
        assert np.isclose(mean_value, expected_value, atol=self.ATOL, rtol=self.RTOL)

    def test_variance_is_square_of_support_length_divided_by_12(self, dist):
        variance_value = dist.var()
        expected_value = ((dist.b - dist.a) ** 2) / 12
        assert np.isclose(variance_value, expected_value, atol=self.ATOL, rtol=self.RTOL)

    def test_base_distribution_is_uniform_on_minus_one_to_one(self, dist):
        germ = dist.get_base_dist()
        assert isinstance(germ, UniformDistribution)
        assert np.allclose((germ.a, germ.b), (-1, 1), atol=self.ATOL, rtol=self.RTOL)

    def test_base2dist(self, dist):
        germ = dist.get_base_dist()
        y = np.linspace(germ.a, germ.b, 50)
        x = np.linspace(dist.a, dist.b, 50)
        assert np.allclose(dist.base2dist(y), x, atol=self.ATOL, rtol=self.RTOL)

    def test_dist2base(self, dist):
        germ = dist.get_base_dist()
        x = np.linspace(dist.a, dist.b, 50)
        y = dist.dist2base(x)
        expected_y = np.linspace(germ.a, germ.b, 50)
        assert np.allclose(y, expected_y, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 3. Reference values
    # =====================================================================
    @pytest.mark.parametrize(
        "x, expected",
        [(1.0, 0.0), (2.0, 1 / 3), (3.5, 1 / 3), (5.0, 1 / 3), (6.0, 0.0)],
    )
    def test_pdf_reference_values(self, x, expected):
        dist = UniformDistribution(A, B)
        assert np.allclose(dist.pdf(x), expected, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "x, expected",
        [(1.0, 0.0), (2.0, 0.0), (2.75, 0.25), (3.5, 0.5), (5.0, 1.0), (7.0, 1.0)],
    )
    def test_cdf_reference_values(self, x, expected):
        dist = UniformDistribution(A, B)
        assert np.allclose(dist.cdf(x), expected, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "x, expected",
        [(0.0, 2.0), (0.25, 2.75), (0.5, 3.5), (1.0, 5.0)],
    )
    def test_invcdf_reference_values(self, x, expected):
        dist = UniformDistribution(A, B)
        assert np.allclose(dist.invcdf(x), expected, atol=self.ATOL, rtol=self.RTOL)

    def test_moments_reference_values(self):
        dist = UniformDistribution(A, B)
        assert np.allclose(dist.moments(), [7 / 2, 3 ** 2 / 12, 0, -6 / 5], atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_matches_with_scipy(self):
        from scipy.stats import uniform
        dist = UniformDistribution(A, B)
        x = np.linspace(dist.a, dist.b, 50)
        expected_pdf = uniform.pdf(x, loc=dist.a, scale=dist.b - dist.a)
        assert np.allclose(dist.pdf(x), expected_pdf, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_matches_with_scipy(self):
        from scipy.stats import uniform
        dist = UniformDistribution(A, B)
        x = np.linspace(dist.a, dist.b, 50)
        expected_cdf = uniform.cdf(x, loc=dist.a, scale=dist.b - dist.a)
        assert np.allclose(dist.cdf(x), expected_cdf, atol=self.ATOL, rtol=self.RTOL)

    def test_invcdf_matches_with_scipy(self):
        from scipy.stats import uniform
        dist = UniformDistribution(A, B)
        x = np.linspace(0, 1, 50)
        expected_invcdf = uniform.ppf(x, loc=dist.a, scale=dist.b - dist.a)
        assert np.allclose(dist.invcdf(x), expected_invcdf, atol=self.ATOL, rtol=self.RTOL)

    def test_orth_polysys_is_legendre(self):
        d1 = UniformDistribution(-1, 1)
        d2 = UniformDistribution(A, B)
        assert isinstance(d1.orth_polysys(), LegendrePolynomials)
        with pytest.raises(Exception):
            d2.orth_polysys()

    def test_orth_polysys_syschar(self, dist):
        d1 = UniformDistribution(-1, 1)
        d2 = UniformDistribution(A, B)
        assert d1.orth_polysys_syschar(False) == "P"
        assert d1.orth_polysys_syschar(True) == "p"
        with pytest.raises(Exception):
            d2.orth_polysys_syschar(False)   
                 
    def test_get_bounds_are_exact(self):
        dist = UniformDistribution(A, B)
        bounds = dist.get_bounds()
        assert np.allclose(bounds, [A, B], atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 4. Edge cases / invalid parameters
    # =====================================================================
    @pytest.mark.parametrize(
        "lower, upper",
        [
            (B, A),
            (A, A),
            (np.inf, B),
            (A, np.inf),
        ],
    )
    def test_invalid_parameters_raise_value_error(self, lower, upper):
        with pytest.raises(ValueError):
            UniformDistribution(lower, upper)

