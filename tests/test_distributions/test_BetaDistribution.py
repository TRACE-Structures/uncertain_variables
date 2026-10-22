"""Tests for the BetaDistribution class."""

import numpy as np
import pytest
from scipy.stats import beta as scipy_beta

from src.polysys.JacobiPolynomials import JacobiPolynomials

from .TestDistribution import TestDistribution as DistributionTestClass
from src.distributions import NormalDistribution

from src.distributions import UniformDistribution

from src.distributions import BetaDistribution


A, B, SCALE, LOC = 2, 3, 4, 1


def _beta_moments(a, b, scale, loc):
    s = a + b
    mean = loc + scale * a / s
    var = scale ** 2 * a * b / (s ** 2 * (s + 1))
    skew = 2 * (b - a) * np.sqrt(s + 1) / ((s + 2) * np.sqrt(a * b))
    kurt = 6 * ((a - b) ** 2 * (s + 1) - a * b * (s + 2)) / (a * b * (s + 2) * (s + 3))
    return mean, var, skew, kurt


class TestBetaDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return BetaDistribution(A, B, scale=SCALE, loc=LOC)

    @pytest.fixture
    def support(self):
        return LOC, LOC + SCALE

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    def test_init_stores_parameters(self):
        d = BetaDistribution(A, B, scale=SCALE, loc=LOC)
        assert np.allclose((d.a, d.b, d.scale, d.loc), (A, B, SCALE, LOC), atol=self.ATOL, rtol=self.RTOL)

    def test_init_defaults_to_unit_interval(self):
        d = BetaDistribution(A, B)
        assert (d.scale, d.loc) == (1, 0)

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "beta"

    @pytest.mark.parametrize("a, b, scale, loc", [(2, 3, 1, 0), (A, B, SCALE, LOC), (0.5, 0.5, 2, -1)])
    def test_get_dist_params(self, a, b, scale, loc):
        params = BetaDistribution(a, b, scale=scale, loc=loc).get_dist_params()
        assert np.asarray(params).shape == (4,)
        assert np.allclose(params, [a, b, scale, loc], atol=self.ATOL, rtol=self.RTOL)

    def test_eq_same_params(self):
        assert BetaDistribution(A, B, SCALE, LOC) == BetaDistribution(A, B, SCALE, LOC)

    @pytest.mark.parametrize(
        "other",
        [
            (A + 1, B, SCALE, LOC),
            (A, B + 1, SCALE, LOC),
            (A, B, SCALE + 1, LOC),
            (A, B, SCALE, LOC + 1),
            (B, A, SCALE, LOC),
        ],
    )
    def test_eq_same_type_different_params(self, other):
        assert BetaDistribution(A, B, SCALE, LOC) != BetaDistribution(*other)

    def test_eq_different_distribution_type(self):
        assert BetaDistribution(1, 1, 1, 0) != UniformDistribution(0, 1)
        assert BetaDistribution(A, B) != NormalDistribution(0, 1)

    def test_eq_non_distribution_object(self):
        assert BetaDistribution(A, B) != (A, B, 1, 0)
        assert BetaDistribution(A, B) != "Beta"

    @pytest.mark.parametrize(
        "args, expected",
        [
            ((A, B, SCALE, LOC), "Beta(2.00, 3.00), scale=4.00, loc=1.00"),
            ((A, B), "Beta(2.00, 3.00), scale=1.00, loc=0.00"),
            ((0.5, 1.25, 2, -1), "Beta(0.50, 1.25), scale=2.00, loc=-1.00"),
        ],
    )
    def test_repr(self, args, expected):
        assert repr(BetaDistribution(*args)) == expected

    # =====================================================================
    # 2. Mathematical identities (beta-specific parts)
    # =====================================================================
    def test_pdf_is_zero_outside_the_support(self, dist, support):
        lo, hi = support
        assert np.allclose(dist.pdf(np.array([lo - 1, hi + 1])), 0, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_is_zero_below_and_one_above_support(self, dist, support):
        lo, hi = support
        assert np.isclose(dist.cdf(lo - 1), 0, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(dist.cdf(hi + 1), 1, atol=self.ATOL, rtol=self.RTOL)

    def test_symmetric_beta_has_symmetric_pdf(self):
        d = BetaDistribution(3, 3, scale=2, loc=-1)
        x = np.linspace(0.05, 1, 20)
        assert np.allclose(d.pdf(x), d.pdf(-x), atol=self.ATOL, rtol=self.RTOL)

    def test_beta_1_1_is_uniform(self):
        d = BetaDistribution(1, 1, scale=3, loc=2)
        u = UniformDistribution(2, 5)
        x = np.linspace(1, 6, 50)
        assert np.allclose(d.pdf(x), u.pdf(x), atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.cdf(x), u.cdf(x), atol=self.ATOL, rtol=self.RTOL)

    def test_get_bounds_default_delta(self, dist):
        assert np.allclose(dist.get_bounds(), dist.invcdf([0.02, 0.98]), atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 3. Reference values
    # =====================================================================
    @pytest.mark.parametrize("a, b, scale, loc", [(A, B, SCALE, LOC), (0.5, 0.5, 1, 0), (5, 1.5, 2, -1), (1, 1, 1, 0)])
    def test_pdf_cdf_invcdf_match_scipy(self, a, b, scale, loc):
        d = BetaDistribution(a, b, scale=scale, loc=loc)
        x = np.linspace(loc + 0.01 * scale, loc + 0.99 * scale, 40)
        q = np.linspace(0.01, 0.99, 40)
        assert np.allclose(d.pdf(x), scipy_beta.pdf(x, a, b, loc=loc, scale=scale), atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.cdf(x), scipy_beta.cdf(x, a, b, loc=loc, scale=scale), atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.invcdf(q), scipy_beta.ppf(q, a, b, loc=loc, scale=scale), atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_reference_values(self):
        # Beta(2, 3) on [0, 1]: pdf = 12 x (1 - x)^2
        d = BetaDistribution(2, 3)
        for x in (0.0, 0.25, 0.5, 1.0):
            assert np.isclose(d.pdf(x), 12 * x * (1 - x) ** 2, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_reference_values(self):
        # Beta(2, 2) on [0, 1]: cdf = 3x^2 - 2x^3
        d = BetaDistribution(2, 2)
        for x in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert np.isclose(d.cdf(x), 3 * x ** 2 - 2 * x ** 3, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 4. Edge cases / invalid parameters
    # =====================================================================
    @pytest.mark.parametrize(
        "a, b, scale, loc",
        [
            (0, 1, 1, 0),
            (-1, 1, 1, 0),
            (1, 0, 1, 0),
            (1, -2, 1, 0),
            (1, 1, 0, 0),
            (1, 1, -1, 0),
            (np.inf, 1, 1, 0),
            (1, np.inf, 1, 0),
            (1, 1, np.inf, 0),
            (1, 1, 1, np.inf),
            (1, 1, 1, np.nan),
            ("a", 1, 1, 0),
            (1, None, 1, 0),
            (1, 1, "s", 0),
            (1, 1, 1, "l"),
        ],
    )
    def test_invalid_parameters_raise_value_error(self, a, b, scale, loc):
        with pytest.raises(ValueError):
            BetaDistribution(a, b, scale=scale, loc=loc)

    # =====================================================================
    # 5. Moments in specific cases
    # =====================================================================
    MOMENT_CASES = [
        (1, 1, 1, 0),
        (2, 2, 1, 0),
        (2, 5, 1, 0),
        (A, B, SCALE, LOC),
        (5, 1.5, 2, -1),
        (0.5, 0.5, 10, -5),
        (3, 3, 2, -1),
    ]

    @pytest.mark.parametrize("a, b, scale, loc", MOMENT_CASES)
    def test_mean_specific_cases(self, a, b, scale, loc):
        mean = _beta_moments(a, b, scale, loc)[0]
        assert np.isclose(BetaDistribution(a, b, scale, loc).mean(), mean, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("a, b, scale, loc", MOMENT_CASES)
    def test_var_specific_cases(self, a, b, scale, loc):
        var = _beta_moments(a, b, scale, loc)[1]
        assert np.isclose(BetaDistribution(a, b, scale, loc).var(), var, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("a, b, scale, loc", MOMENT_CASES)
    def test_skew_specific_cases(self, a, b, scale, loc):
        skew = _beta_moments(a, b, scale, loc)[2]
        assert np.isclose(BetaDistribution(a, b, scale, loc).skew(), skew, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("a, b, scale, loc", MOMENT_CASES)
    def test_kurt_specific_cases(self, a, b, scale, loc):
        kurt = _beta_moments(a, b, scale, loc)[3]
        assert np.isclose(BetaDistribution(a, b, scale, loc).kurt(), kurt, atol=self.ATOL, rtol=self.RTOL)

    def test_symmetric_beta_has_zero_skew(self):
        assert np.isclose(BetaDistribution(4, 4).skew(), 0, atol=self.ATOL)

    def test_beta_1_1_has_uniform_excess_kurtosis(self):
        assert np.isclose(BetaDistribution(1, 1).kurt(), -6 / 5, atol=self.ATOL, rtol=self.RTOL)

    def test_skew_sign_follows_shape_parameters(self):
        assert BetaDistribution(2, 5).skew() > 0
        assert BetaDistribution(5, 2).skew() < 0

    # =====================================================================
    # 6. Sampling
    # =====================================================================
    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    @pytest.mark.parametrize("a, b, scale, loc", [(A, B, SCALE, LOC), (0.5, 0.5, 1, 0), (5, 1.5, 2, -1)])
    def test_sample_within_support_all_methods(self, method, a, b, scale, loc):
        s = np.asarray(BetaDistribution(a, b, scale, loc).sample(64, method=method, seed=1))
        assert s.shape == (64,)
        assert np.all(s >= loc) and np.all(s <= loc + scale)

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_statistics_all_methods(self, dist, method):
        s = np.asarray(dist.sample(2048, method=method, seed=7))
        assert abs(np.mean(s) - dist.mean()) < 0.1
        assert abs(np.var(s) - dist.var()) < 0.1

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_is_reproducible_with_seed(self, dist, method):
        assert np.array_equal(dist.sample(32, method=method, seed=123), dist.sample(32, method=method, seed=123))

    def test_sample_mc_differs_for_different_seeds(self, dist):
        assert not np.array_equal(dist.sample(32, "MC", seed=1), dist.sample(32, "MC", seed=2))

    def test_sample_default_method_is_mc(self, dist):
        assert np.array_equal(dist.sample(16, seed=5), dist.sample(16, "MC", seed=5))

    @pytest.mark.parametrize("n", [0, -3, 2.5, "10", None])
    def test_sample_invalid_n_raises(self, dist, n):
        with pytest.raises(ValueError):
            dist.sample(n)

    @pytest.mark.parametrize("seed", [1.5, "abc"])
    def test_sample_invalid_seed_raises(self, dist, seed):
        with pytest.raises(ValueError):
            dist.sample(8, seed=seed)

    def test_sample_unknown_method_raises(self, dist):
        with pytest.raises(ValueError):
            dist.sample(8, method="unknown")

    # =====================================================================
    # 7. translate / get_shift / get_scale
    # =====================================================================
    @pytest.mark.parametrize("shift, scale", [(0, 1), (1, 2), (-3, 0.5), (2.5, 4), (0, 10), (-1, 1)])
    def test_translate_maps_parameters_affinely(self, dist, shift, scale):
        t = dist.translate(shift, scale)
        assert isinstance(t, BetaDistribution)
        assert (t.a, t.b) == (A, B)
        assert np.isclose(t.loc, LOC * scale + shift, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.scale, SCALE * scale, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("shift, scale", [(1, 2), (-3, 0.5), (2.5, 4)])
    def test_translate_transforms_moments(self, dist, shift, scale):
        t = dist.translate(shift, scale)
        assert np.isclose(t.mean(), scale * dist.mean() + shift, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.var(), scale ** 2 * dist.var(), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.skew(), dist.skew(), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.kurt(), dist.kurt(), atol=self.ATOL, rtol=self.RTOL)

    def test_translate_default_is_identity(self, dist):
        assert dist.translate() == dist

    def test_translate_does_not_modify_original(self, dist):
        dist.translate(3, 2)
        assert (dist.a, dist.b, dist.scale, dist.loc) == (A, B, SCALE, LOC)

    def test_translate_composition(self, dist):
        t = dist.translate(1, 2).translate(-3, 0.5)
        expected = dist.translate(0.5 * 1 - 3, 0.5 * 2)
        assert np.allclose(t.get_dist_params(), expected.get_dist_params(), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "shift, scale",
        [(0, 0), (0, -1), (np.inf, 1), (0, np.inf), ("a", 1), (0, "b"), (None, 1), (np.nan, 1)],
    )
    def test_translate_invalid_raises(self, dist, shift, scale):
        with pytest.raises(ValueError):
            dist.translate(shift, scale)

    @pytest.mark.parametrize("a, b, scale, loc", [(A, B, 1, 0), (A, B, SCALE, LOC), (A, B, 2, -1), (0.5, 3, 7, -2.5)])
    def test_get_shift_and_scale(self, a, b, scale, loc):
        d = BetaDistribution(a, b, scale, loc)
        assert np.isclose(d.get_shift(), loc, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(d.get_scale(), scale, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("shift, scale", [(1, 2), (-3, 0.5), (2.5, 4)])
    def test_get_shift_scale_after_translate(self, dist, shift, scale):
        t = dist.translate(shift, scale)
        assert np.isclose(t.get_scale(), scale * dist.get_scale(), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.get_shift(), scale * dist.get_shift() + shift, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 8. fix_moments (verified by quadrature)
    # =====================================================================
    # The generic mean=10, variance=4 check is infeasible on this support.
    def test_fix_moments(self, dist):
        new_dist = dist.fix_moments(3, 0.5, scale=SCALE, loc=LOC)
        assert np.isclose(new_dist.mean(), 3, rtol=self.RTOL, atol=self.ATOL)
        assert np.isclose(new_dist.var(), 0.5, rtol=self.RTOL, atol=self.ATOL)

    def test_fix_moments_does_not_modify_original(self, dist):
        original_params = dist.get_dist_params().copy()
        dist.fix_moments(3, 0.5, scale=SCALE, loc=LOC)
        assert np.array_equal(dist.get_dist_params(), original_params)

    # (mean, var, scale, loc); chosen so both shape parameters stay >= 1,
    # i.e. the pdf is bounded and quadrature is reliable.
    FIX_MOMENT_CASES = [
        (0.5, 0.05, 1, 0),
        (0.3, 0.02, 1, 0),
        (0.7, 0.01, 1, 0),
        (10.0, 4.0, 20, 0),
        (-0.2, 0.1, 2, -1),
        (5.0, 0.25, 4, 3),
        (100.0, 9.0, 30, 85),
    ]

    @pytest.mark.parametrize("mean, var, scale, loc", FIX_MOMENT_CASES)
    def test_fix_moments_matches_by_quadrature(self, mean, var, scale, loc):
        new = BetaDistribution(2, 3).fix_moments(mean, var, scale=scale, loc=loc)
        assert isinstance(new, BetaDistribution)
        assert (new.scale, new.loc) == (scale, loc)
        lo, hi = loc, loc + scale
        q_mean = self._quad(lambda t: t * new.pdf(t), lo, hi)
        q_var = self._central_moment(new, 2, lo, hi)
        assert np.isclose(q_mean, mean, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(q_var, var, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(new.mean(), mean, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(new.var(), var, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mean, var, scale, loc", FIX_MOMENT_CASES)
    def test_fix_moments_pdf_integrates_to_one(self, mean, var, scale, loc):
        new = BetaDistribution(2, 3).fix_moments(mean, var, scale=scale, loc=loc)
        assert np.isclose(self._quad(new.pdf, loc, loc + scale), 1.0, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)

    def test_fix_moments_shape_parameters(self):
        # mean 0.5, var 0.05 -> a = b = 2
        new = BetaDistribution(7, 9).fix_moments(0.5, 0.05)
        assert np.allclose((new.a, new.b), (2, 2), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "mean, var, scale, loc",
        [
            (0.5, 0, 1, 0),
            (0.5, -1, 1, 0),
            (0.5, 0.05, 0, 0),
            (0.5, 0.05, -1, 0),
            ("a", 0.05, 1, 0),
            (0.5, "b", 1, 0),
            (0.5, 0.05, "c", 0),
            (0.5, 0.05, 1, "d"),
            (0.0, 0.05, 1, 0),       # mean on the boundary
            (1.0, 0.05, 1, 0),
            (1.5, 0.05, 1, 0),       # mean outside the support
            (0.5, 0.25, 1, 0),       # var == m(1-m)
            (0.5, 0.3, 1, 0),        # var > m(1-m)
            (10, 4, 1, 0),           # incompatible with the default support
        ],
    )
    def test_fix_moments_invalid_raises(self, mean, var, scale, loc):
        with pytest.raises(ValueError):
            BetaDistribution(A, B).fix_moments(mean, var, scale=scale, loc=loc)

    # =====================================================================
    # 9. fix_bounds
    # =====================================================================
    @pytest.mark.parametrize("new_min, new_max", [(0, 1), (-1, 1), (10, 20), (-5.5, -0.5)])
    def test_fix_bounds_default_quantiles(self, dist, new_min, new_max):
        new = dist.fix_bounds(new_min, new_max)
        assert isinstance(new, BetaDistribution)
        assert (new.a, new.b) == (A, B)
        assert np.allclose((new.loc, new.loc + new.scale), (new_min, new_max), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("q0, q1", [(0.1, 0.9), (0.25, 0.75), (0, 0.5), (0.5, 1)])
    def test_fix_bounds_with_quantiles(self, dist, q0, q1):
        new = dist.fix_bounds(10, 20, q0, q1)
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
    def test_fix_bounds_invalid_raises(self, dist, args, kwargs):
        with pytest.raises(ValueError):
            dist.fix_bounds(*args, **kwargs)

    # =====================================================================
    # 10. GPC related functions
    # =====================================================================
    @pytest.mark.parametrize("a, b", [(1, 1), (A, B), (0.5, 4), (7, 2)])
    def test_get_base_dist_is_beta_on_minus_one_to_one(self, dist, a, b):
        germ = dist.get_base_dist(a, b)
        assert isinstance(germ, BetaDistribution)
        assert germ == BetaDistribution(a, b, scale=2, loc=-1)

    def test_orth_polysys_is_jacobi(self):
        ps = BetaDistribution(A, B, scale=2, loc=-1).orth_polysys()
        assert isinstance(ps, JacobiPolynomials)
        assert np.isclose(ps.alpha, B - 1)
        assert np.isclose(ps.beta, A - 1)
        with pytest.raises(Exception):
            BetaDistribution(A, B, scale=1, loc=0).orth_polysys()

    def test_orth_polysys_syschar(self):
        d = BetaDistribution(A, B, scale=2, loc=-1)
        assert d.orth_polysys_syschar(False) == "J"
        assert d.orth_polysys_syschar(True) == "j"
        with pytest.raises(Exception):
            BetaDistribution(A, B, scale=1, loc=0).orth_polysys_syschar(False)
        with pytest.raises(Exception):
            BetaDistribution(A, B, scale=1, loc=0).orth_polysys_syschar(True)