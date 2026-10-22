"""Tests for the UniformDistribution class."""

import numpy as np
import pytest

from src.polysys.LegendrePolynomials import LegendrePolynomials

from .TestDistribution import TestDistribution as DistributionTestClass
from src.distributions import NormalDistribution

from src.distributions import UniformDistribution


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

    def test_eq_same_params(self):
        assert UniformDistribution(A, B) == UniformDistribution(A, B)

    @pytest.mark.parametrize("other", [(0, 1), (A, B + 1), (A - 1, B), (B, B + 3)])
    def test_eq_same_type_different_params(self, other):
        assert UniformDistribution(A, B) != UniformDistribution(*other)

    def test_eq_different_distribution_type(self):
        assert UniformDistribution(0, 1) != NormalDistribution(0, 1)

    def test_eq_non_distribution_object(self):
        assert UniformDistribution(A, B) != (A, B)
        assert UniformDistribution(A, B) != "U(2.00, 5.00)"

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (A, B, "U(2.00, 5.00)"),
            (0, 1, "U(0.00, 1.00)"),
            (-1, 1, "U(-1.00, 1.00)"),
            (0.125, 2.5, "U(0.12, 2.50)"),
        ],
    )
    def test_repr(self, a, b, expected):
        assert repr(UniformDistribution(a, b)) == expected

    def test_translate_returns_uniform(self, dist):
        translated = dist.translate(1, 2)
        assert isinstance(translated, UniformDistribution)
        assert np.allclose((translated.a, translated.b), (1, 3), atol=self.ATOL, rtol=self.RTOL)

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

    @pytest.mark.parametrize("lower, upper", [("a", 1), (0, None), (-np.inf, 1), (0, np.nan)])
    def test_non_numeric_or_nonfinite_parameters_raise_value_error(self, lower, upper):
        with pytest.raises(ValueError):
            UniformDistribution(lower, upper)

    # =====================================================================
    # 5. Moments in specific cases
    # =====================================================================
    MOMENT_CASES = [
        (0, 1, 0.5, 1 / 12),
        (-1, 1, 0.0, 1 / 3),
        (A, B, 3.5, 0.75),
        (-10, -4, -7.0, 3.0),
        (0.5, 0.75, 0.625, 0.25 ** 2 / 12),
    ]

    @pytest.mark.parametrize("a, b, mean, var", MOMENT_CASES)
    def test_mean_specific_cases(self, a, b, mean, var):
        assert np.isclose(UniformDistribution(a, b).mean(), mean, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("a, b, mean, var", MOMENT_CASES)
    def test_var_specific_cases(self, a, b, mean, var):
        assert np.isclose(UniformDistribution(a, b).var(), var, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("a, b, mean, var", MOMENT_CASES)
    def test_skew_specific_cases(self, a, b, mean, var):
        assert np.isclose(UniformDistribution(a, b).skew(), 0.0, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("a, b, mean, var", MOMENT_CASES)
    def test_kurt_specific_cases(self, a, b, mean, var):
        assert np.isclose(UniformDistribution(a, b).kurt(), -6 / 5, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 6. Sampling
    # =====================================================================
    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    @pytest.mark.parametrize("a, b", [(0, 1), (A, B), (-4, -1)])
    def test_sample_within_support_all_methods(self, method, a, b):
        d = UniformDistribution(a, b)
        s = np.asarray(d.sample(64, method=method, seed=1))
        assert s.shape == (64,)
        assert np.all(s >= a) and np.all(s <= b)

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_statistics_all_methods(self, method):
        d = UniformDistribution(A, B)
        s = np.asarray(d.sample(1024, method=method, seed=7))
        assert abs(np.mean(s) - d.mean()) < 0.1
        assert abs(np.var(s) - d.var()) < 0.1

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_is_reproducible_with_seed(self, method):
        d = UniformDistribution(A, B)
        s1 = d.sample(32, method=method, seed=123)
        s2 = d.sample(32, method=method, seed=123)
        assert np.array_equal(s1, s2)

    def test_sample_mc_differs_for_different_seeds(self):
        d = UniformDistribution(A, B)
        assert not np.array_equal(d.sample(32, "MC", seed=1), d.sample(32, "MC", seed=2))

    def test_sample_default_method_is_mc(self):
        d = UniformDistribution(A, B)
        assert np.array_equal(d.sample(16, seed=5), d.sample(16, "MC", seed=5))

    @pytest.mark.parametrize("n", [0, -3, 2.5, "10", None])
    def test_sample_invalid_n_raises(self, n):
        with pytest.raises(ValueError):
            UniformDistribution(A, B).sample(n)

    @pytest.mark.parametrize("seed", [1.5, "abc"])
    def test_sample_invalid_seed_raises(self, seed):
        with pytest.raises(ValueError):
            UniformDistribution(A, B).sample(8, seed=seed)

    def test_sample_unknown_method_raises(self):
        with pytest.raises(ValueError):
            UniformDistribution(A, B).sample(8, method="unknown")

    # =====================================================================
    # 7. translate / get_shift / get_scale
    # =====================================================================
    @pytest.mark.parametrize(
        "shift, scale",
        [(0, 1), (1, 2), (-3, 0.5), (2.5, 4), (0, 10), (-1, 1)],
    )
    def test_translate_maps_bounds_affinely(self, shift, scale):
        d = UniformDistribution(A, B)
        t = d.translate(shift, scale)
        assert isinstance(t, UniformDistribution)
        assert np.allclose((t.a, t.b), (scale * A + shift, scale * B + shift), atol=self.ATOL, rtol=self.RTOL)

    def test_translate_default_is_identity(self):
        d = UniformDistribution(A, B)
        assert d.translate() == d

    def test_translate_does_not_modify_original(self):
        d = UniformDistribution(A, B)
        d.translate(3, 2)
        assert (d.a, d.b) == (A, B)

    def test_translate_composition(self):
        d = UniformDistribution(A, B)
        t = d.translate(1, 2).translate(-3, 0.5)
        expected = d.translate(0.5 * 1 - 3, 0.5 * 2)
        assert np.allclose(t.get_dist_params(), expected.get_dist_params(), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("shift, scale", [(0, 0), (0, -1), (np.inf, 1), (0, np.inf), ("a", 1), (0, "b"), (None, 1)])
    def test_translate_invalid_raises(self, shift, scale):
        with pytest.raises(ValueError):
            UniformDistribution(A, B).translate(shift, scale)

    @pytest.mark.parametrize("a, b", [(0, 1), (A, B), (-1, 1), (-7.5, -2.5)])
    def test_get_shift_and_scale(self, a, b):
        d = UniformDistribution(a, b)
        assert np.isclose(d.get_shift(), (a + b) / 2, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(d.get_scale(), b - a, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("shift, scale", [(1, 2), (-3, 0.5), (2.5, 4)])
    def test_get_scale_after_translate(self, shift, scale):
        d = UniformDistribution(A, B)
        t = d.translate(shift, scale)
        assert np.isclose(t.get_scale(), scale * d.get_scale(), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.get_shift(), scale * d.get_shift() + shift, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 8. fix_moments (verified by quadrature)
    # =====================================================================
    @pytest.mark.parametrize(
        "mean, var",
        [(0, 1), (0.5, 1 / 12), (3.0, 2.0), (-5.0, 0.01), (100.0, 25.0), (-0.3, 7.5)],
    )
    @pytest.mark.parametrize("start", [(0, 1), (A, B), (-1, 1)])
    def test_fix_moments_matches_by_quadrature(self, start, mean, var):
        new = UniformDistribution(*start).fix_moments(mean, var)
        assert isinstance(new, UniformDistribution)
        lo, hi = new.a, new.b
        q_mean = self._quad(lambda t: t * new.pdf(t), lo, hi)
        q_var = self._central_moment(new, 2, lo, hi)
        assert np.isclose(q_mean, mean, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(q_var, var, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        # analytic moments agree too
        assert np.isclose(new.mean(), mean, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(new.var(), var, atol=self.ATOL, rtol=self.RTOL)

    def test_fix_moments_pdf_integrates_to_one(self):
        new = UniformDistribution(A, B).fix_moments(1.0, 4.0)
        assert np.isclose(self._quad(new.pdf, new.a, new.b), 1.0, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)

    def test_fix_moments_defaults(self):
        new = UniformDistribution(A, B).fix_moments()
        assert np.allclose((new.a, new.b), (0, 1), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mean, var", [(0, 0), (0, -1), ("a", 1), (0, "b"), (np.inf, 1), (0, np.inf), (np.nan, 1)])
    def test_fix_moments_invalid_raises(self, mean, var):
        with pytest.raises(ValueError):
            UniformDistribution(A, B).fix_moments(mean, var)

    # =====================================================================
    # 9. fix_bounds
    # =====================================================================
    @pytest.mark.parametrize("new_min, new_max", [(0, 1), (-1, 1), (10, 20), (-5.5, -0.5)])
    def test_fix_bounds_default_quantiles(self, new_min, new_max):
        new = UniformDistribution(A, B).fix_bounds(new_min, new_max)
        assert np.allclose((new.a, new.b), (new_min, new_max), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("q0, q1", [(0.1, 0.9), (0.25, 0.75), (0, 0.5), (0.5, 1)])
    def test_fix_bounds_with_quantiles(self, q0, q1):
        new = UniformDistribution(A, B).fix_bounds(10, 20, q0, q1)
        assert np.isclose(new.invcdf(q0), 10, atol=1e-8)
        assert np.isclose(new.invcdf(q1), 20, atol=1e-8)

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((1, 1), {}),
            ((2, 1), {}),
            (("a", 1), {}),
            ((0, None), {}),
            ((0, 1), {"q0": -0.1}),
            ((0, 1), {"q0": 1.1}),
            ((0, 1), {"q0": 0.6, "q1": 0.4}),
            ((0, 1), {"q1": 1.5}),
        ],
    )
    def test_fix_bounds_invalid_raises(self, args, kwargs):
        with pytest.raises(ValueError):
            UniformDistribution(A, B).fix_bounds(*args, **kwargs)

    def test_get_bounds_are_exact(self):
        dist = UniformDistribution(A, B)
        bounds = dist.get_bounds(delta=0)
        assert np.allclose(bounds, [A, B], atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 10. GPC related functions
    # =====================================================================    
    @pytest.mark.parametrize("a, b", [(0, 1), (A, B), (-1, 1), (-9, -3)])
    def test_get_base_dist_is_always_standard_uniform(self, a, b):
        germ = UniformDistribution(a, b).get_base_dist()
        assert germ == UniformDistribution(-1, 1)

    def test_orth_polysys_is_legendre(self):
        d1 = UniformDistribution(-1, 1)
        d2 = UniformDistribution(A, B)
        assert isinstance(d1.orth_polysys(), LegendrePolynomials)
        with pytest.raises(Exception):
            d2.orth_polysys()

    def test_orth_polysys_syschar(self):
        d1 = UniformDistribution(-1, 1)
        d2 = UniformDistribution(A, B)
        assert d1.orth_polysys_syschar(False) == "P"
        assert d1.orth_polysys_syschar(True) == "p"
        with pytest.raises(Exception):
            d2.orth_polysys_syschar(False)
        with pytest.raises(Exception):
            d2.orth_polysys_syschar(True)