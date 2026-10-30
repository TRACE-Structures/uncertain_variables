"""Tests for the ExponentialDistribution class."""

import numpy as np
import pytest
from scipy.stats import expon

from src.polysys.LaguerrePolynomials import LaguerrePolynomials

from .TestDistribution import TestDistribution as DistributionTestClass
from src.distributions import NormalDistribution

from src.distributions import ExponentialDistribution


LAMBDA, LOC = 2, 1


class TestExponentialDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return ExponentialDistribution(LAMBDA, LOC)

    @pytest.fixture
    def support(self):
        return LOC, np.inf

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    def test_init_stores_parameters(self, dist):
        assert np.isclose(dist.lambda_, LAMBDA, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(dist.loc, LOC, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(dist.scale, 1 / LAMBDA, atol=self.ATOL, rtol=self.RTOL)

    def test_init_defaults(self):
        d = ExponentialDistribution()
        assert (d.lambda_, d.loc) == (1, 0)

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "exp"

    @pytest.mark.parametrize("lambda_, loc", [(1, 0), (LAMBDA, LOC), (0.25, -3.5)])
    def test_get_dist_params(self, lambda_, loc):
        params = ExponentialDistribution(lambda_, loc).get_dist_params()
        assert np.asarray(params).shape == (2,)
        assert np.allclose(params, [lambda_, loc], atol=self.ATOL, rtol=self.RTOL)

    def test_eq_same_params(self):
        assert ExponentialDistribution(LAMBDA, LOC) == ExponentialDistribution(LAMBDA, LOC)

    @pytest.mark.parametrize("other", [(1, LOC), (LAMBDA, 0), (3, 2), (LOC, LAMBDA)])
    def test_eq_same_type_different_params(self, other):
        assert ExponentialDistribution(LAMBDA, LOC) != ExponentialDistribution(*other)

    def test_eq_different_distribution_type(self):
        assert ExponentialDistribution(1) != NormalDistribution(0, 1)

    def test_eq_non_distribution_object(self):
        assert ExponentialDistribution(LAMBDA, LOC) != (LAMBDA, LOC)
        assert ExponentialDistribution(LAMBDA, LOC) != "Exp"

    @pytest.mark.parametrize(
        "args, expected",
        [
            ((LAMBDA, LOC), "Exp(2.00), loc=1.00"),
            ((1,), "Exp(1.00), loc=0.00"),
            ((0.125, -2.5), "Exp(0.12), loc=-2.50"),
        ],
    )
    def test_repr(self, args, expected):
        assert repr(ExponentialDistribution(*args)) == expected

    # =====================================================================
    # 2. Mathematical identities (exponential-specific parts)
    # =====================================================================
    def test_pdf_is_zero_below_support(self, dist):
        assert np.allclose(dist.pdf(np.array([LOC - 1, LOC - 5])), 0, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_is_zero_below_support(self, dist):
        assert np.allclose(dist.cdf(np.array([LOC - 1, LOC - 5])), 0, atol=self.ATOL, rtol=self.RTOL)

    def test_cdf_at_loc_is_zero(self, dist):
        assert np.isclose(dist.cdf(LOC), 0, atol=self.ATOL, rtol=self.RTOL)

    def test_pdf_at_loc_equals_lambda(self, dist):
        assert np.isclose(dist.pdf(LOC), LAMBDA, atol=self.ATOL, rtol=self.RTOL)

    def test_ppf_at_zero_is_loc_and_at_one_is_inf(self, dist):
        assert np.isclose(dist.ppf(0), LOC, atol=self.ATOL, rtol=self.RTOL)
        assert dist.ppf(1) == np.inf

    def test_memoryless_property(self, dist):
        s, t = 0.3, 0.8
        surv = lambda x: 1 - dist.cdf(x)
        assert np.isclose(surv(LOC + s + t) / surv(LOC + s), surv(LOC + t), atol=self.ATOL, rtol=self.RTOL)

    def test_median(self, dist):
        assert np.isclose(dist.ppf(0.5), LOC + np.log(2) / LAMBDA, atol=self.ATOL, rtol=self.RTOL)

    def test_get_bounds_default_delta(self, dist):
        assert np.allclose(dist.get_bounds(), dist.ppf([0.02, 0.98]), atol=self.ATOL, rtol=self.RTOL)

    def test_germ_to_dist(self):
        d = ExponentialDistribution(LAMBDA)
        y = np.linspace(0, 10, 50)
        assert np.allclose(d.germ_to_dist(y), y / LAMBDA, atol=self.ATOL, rtol=self.RTOL)

    def test_dist_to_germ(self):
        d = ExponentialDistribution(LAMBDA)
        x = np.linspace(0, 10, 50)
        assert np.allclose(d.dist_to_germ(x), x * LAMBDA, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 3. Reference values
    # =====================================================================
    @pytest.mark.parametrize("lambda_, loc", [(LAMBDA, LOC), (1, 0), (0.5, -2), (5, 3)])
    def test_matches_scipy(self, lambda_, loc):
        d = ExponentialDistribution(lambda_, loc)
        x = np.linspace(loc, loc + 10 / lambda_, 50)
        q = np.linspace(0.01, 0.99, 50)
        assert np.allclose(d.pdf(x), expon.pdf(x, loc=loc, scale=1 / lambda_), atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.cdf(x), expon.cdf(x, loc=loc, scale=1 / lambda_), atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.ppf(q), expon.ppf(q, loc=loc, scale=1 / lambda_), atol=self.ATOL, rtol=self.RTOL)

    def test_reference_values(self):
        d = ExponentialDistribution(2)
        x = np.array([0.0, 0.5, 1.0, 2.0])
        assert np.allclose(d.pdf(x), 2 * np.exp(-2 * x), atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.cdf(x), 1 - np.exp(-2 * x), atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.ppf([0.0, 0.5]), [0.0, np.log(2) / 2], atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 4. Edge cases / invalid parameters
    # =====================================================================
    @pytest.mark.parametrize("lambda_", [0, -1, -0.5, np.nan, "a", None])
    def test_invalid_parameters_raise_value_error(self, lambda_):
        with pytest.raises(ValueError):
            ExponentialDistribution(lambda_)

    # =====================================================================
    # 5. Moments in specific cases
    # =====================================================================
    MOMENT_CASES = [
        (1, 0, 1.0, 1.0),
        (LAMBDA, 0, 0.5, 0.25),
        (LAMBDA, LOC, 1.5, 0.25),
        (0.25, -3, 1.0, 16.0),
        (10, 5, 5.1, 0.01),
    ]

    @pytest.mark.parametrize("lambda_, loc, mean, var", MOMENT_CASES)
    def test_mean_specific_cases(self, lambda_, loc, mean, var):
        assert np.isclose(ExponentialDistribution(lambda_, loc).mean(), mean, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("lambda_, loc, mean, var", MOMENT_CASES)
    def test_var_specific_cases(self, lambda_, loc, mean, var):
        assert np.isclose(ExponentialDistribution(lambda_, loc).var(), var, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("lambda_, loc, mean, var", MOMENT_CASES)
    def test_skew_specific_cases(self, lambda_, loc, mean, var):
        assert np.isclose(ExponentialDistribution(lambda_, loc).skew(), 2.0, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("lambda_, loc, mean, var", MOMENT_CASES)
    def test_kurt_specific_cases(self, lambda_, loc, mean, var):
        assert np.isclose(ExponentialDistribution(lambda_, loc).kurt(), 6.0, atol=self.ATOL, rtol=self.RTOL)

    def test_moments_reference_values(self):
        d = ExponentialDistribution(LAMBDA)
        assert np.allclose(d.moments(), [1 / LAMBDA, 1 / LAMBDA ** 2, 2, 6], atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 6. Sampling
    # =====================================================================
    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    @pytest.mark.parametrize("lambda_, loc", [(LAMBDA, LOC), (1, 0), (0.5, -3)])
    def test_sample_within_support_all_methods(self, method, lambda_, loc):
        s = np.asarray(ExponentialDistribution(lambda_, loc).sample(64, method=method, seed=1))
        assert s.shape == (64,)
        assert np.all(s >= loc)

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_statistics_all_methods(self, dist, method):
        s = np.asarray(dist.sample(4096, method=method, seed=7))
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
    # 7. transform / get_shift / get_scale
    # =====================================================================
    @pytest.mark.parametrize("shift, scale", [(0, 1), (1, 2), (-3, 0.5), (2.5, 4), (0, 10), (-1, 1)])
    def test_transform_maps_parameters_affinely(self, dist, shift, scale):
        t = dist.transform(shift, scale)
        assert isinstance(t, ExponentialDistribution)
        assert np.isclose(t.lambda_, LAMBDA / scale, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.loc, LOC * scale + shift, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("shift, scale", [(1, 2), (-3, 0.5), (2.5, 4)])
    def test_transform_transforms_moments(self, dist, shift, scale):
        t = dist.transform(shift, scale)
        assert np.isclose(t.mean(), scale * dist.mean() + shift, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.var(), scale ** 2 * dist.var(), atol=self.ATOL, rtol=self.RTOL)

    def test_transform_default_is_identity(self, dist):
        assert dist.transform() == dist

    def test_transform_does_not_modify_original(self, dist):
        dist.transform(3, 2)
        assert (dist.lambda_, dist.loc) == (LAMBDA, LOC)

    def test_transform_composition(self, dist):
        t = dist.transform(1, 2).transform(-3, 0.5)
        expected = dist.transform(0.5 * 1 - 3, 0.5 * 2)
        assert np.allclose(t.get_dist_params(), expected.get_dist_params(), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "shift, scale",
        [(0, 0), (0, -1), (np.inf, 1), (0, np.inf), ("a", 1), (0, "b"), (None, 1), (np.nan, 1)],
    )
    def test_transform_invalid_raises(self, dist, shift, scale):
        with pytest.raises(ValueError):
            dist.transform(shift, scale)

    @pytest.mark.parametrize("lambda_, loc", [(1, 0), (LAMBDA, LOC), (0.25, -3.5)])
    def test_get_shift_and_scale(self, lambda_, loc):
        d = ExponentialDistribution(lambda_, loc)
        assert np.isclose(d.get_shift(), loc, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(d.get_scale(), 1 / lambda_, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("shift, scale", [(1, 2), (-3, 0.5), (2.5, 4)])
    def test_get_shift_scale_after_transform(self, dist, shift, scale):
        t = dist.transform(shift, scale)
        assert np.isclose(t.get_scale(), scale * dist.get_scale(), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(t.get_shift(), scale * dist.get_shift() + shift, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 8. fix_moments (verified by quadrature)
    # =====================================================================
    # Only var == (mean - loc) ** 2 is attainable for an exponential.
    def test_fix_moments(self, dist):
        new_dist = dist.fix_moments(10, 81, loc=1)
        assert np.isclose(new_dist.mean(), 10, rtol=self.RTOL, atol=self.ATOL)
        assert np.isclose(new_dist.var(), 81, rtol=self.RTOL, atol=self.ATOL)

    def test_fix_moments_does_not_modify_original(self, dist):
        original_params = dist.get_dist_params().copy()
        dist.fix_moments(10, 81, loc=1)
        assert np.array_equal(dist.get_dist_params(), original_params)

    # (mean, loc): only var == (mean - loc)^2 is attainable
    FIX_MOMENT_CASES = [(1.0, 0.0), (3.0, 1.0), (-2.0, -5.0), (10.0, 0.0), (0.5, 0.25), (100.0, 90.0)]

    @pytest.mark.parametrize("mean, loc", FIX_MOMENT_CASES)
    @pytest.mark.parametrize("start", [(1, 0), (LAMBDA, LOC)])
    def test_fix_moments_matches_by_quadrature(self, start, mean, loc):
        var = (mean - loc) ** 2
        new = ExponentialDistribution(*start).fix_moments(mean, var, loc=loc)
        assert isinstance(new, ExponentialDistribution)
        assert np.isclose(new.loc, loc, atol=self.ATOL, rtol=self.RTOL)
        lo, hi = loc, loc + 60 * (mean - loc)
        q_mean = self._quad(lambda t: t * new.pdf(t), lo, hi)
        q_var = self._central_moment(new, 2, lo, hi)
        assert np.isclose(q_mean, mean, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(q_var, var, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(new.mean(), mean, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(new.var(), var, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mean, loc", FIX_MOMENT_CASES)
    def test_fix_moments_infers_variance_when_none(self, mean, loc):
        new = ExponentialDistribution(1).fix_moments(mean, None, loc=loc)
        assert np.isclose(new.mean(), mean, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(new.var(), (mean - loc) ** 2, atol=self.ATOL, rtol=self.RTOL)

    def test_fix_moments_defaults(self):
        new = ExponentialDistribution(LAMBDA, LOC).fix_moments()
        assert np.allclose((new.lambda_, new.loc), (1, 0), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "mean, var, loc",
        [
            (1, 2, 0),            # incompatible variance
            (2, 1, 0),
            (1, 0, 0),
            (1, -1, 0),
            (0, 0.5, 0),          # mean == loc
            (-1, 1, 0),           # mean < loc
            ("a", 1, 0),
            (1, "b", 0),
            (1, 1, "c"),
            (np.inf, 1, 0),
            (1, np.inf, 0),
            (1, 1, np.inf),
            (np.nan, 1, 0),
        ],
    )
    def test_fix_moments_invalid_raises(self, mean, var, loc):
        with pytest.raises(ValueError):
            ExponentialDistribution(LAMBDA).fix_moments(mean, var, loc=loc)

    # =====================================================================
    # 9. fix_bounds 
    # =====================================================================
    @pytest.mark.parametrize("new_min, new_max", [(0, 1), (-1, 1), (10, 20), (-5.5, -0.5)])
    def test_fix_bounds_default_quantiles(self, dist, new_min, new_max):
        new = dist.fix_bounds(new_min, new_max)
        assert isinstance(new, ExponentialDistribution)
        assert np.isclose(new.ppf(0), new_min, atol=1e-8)
        assert np.isclose(new.ppf(0.98), new_max, atol=1e-8)

    @pytest.mark.parametrize("q0, q1", [(0.1, 0.9), (0.25, 0.75), (0, 0.5), (0.5, 0.99)])
    def test_fix_bounds_with_quantiles(self, dist, q0, q1):
        new = dist.fix_bounds(10, 20, q0, q1)
        assert np.isclose(new.ppf(q0), 10, atol=1e-8)
        assert np.isclose(new.ppf(q1), 20, atol=1e-8)

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
    @pytest.mark.parametrize("lambda_, loc", [(1, 0), (LAMBDA, LOC), (0.25, -3)])
    def test_gpc_dist_is_unit_rate_exponential(self, lambda_, loc):
        germ = ExponentialDistribution(lambda_, loc).gpc_dist()
        assert isinstance(germ, ExponentialDistribution)
        assert germ == ExponentialDistribution(1, 0)

    def test_gpc_polysys_is_laguerre(self):
        d1 = ExponentialDistribution(1)
        d2 = ExponentialDistribution(LAMBDA)
        assert isinstance(d1.gpc_polysys(), LaguerrePolynomials)
        with pytest.raises(Exception):
            d2.gpc_polysys()

    def test_gpc_syschar_is_l(self):
        d1 = ExponentialDistribution(1)
        d2 = ExponentialDistribution(LAMBDA)
        assert d1.gpc_syschar(False) == "L"
        assert d1.gpc_syschar(True) == "l"
        with pytest.raises(Exception):
            d2.gpc_syschar(False)
        with pytest.raises(Exception):
            d2.gpc_syschar(True)

    # =====================================================================
    # 11. Unit conversions
    # =====================================================================
    def test_convert_to_round_trip(self):
        dist = ExponentialDistribution(1, 0)
        old_params = dist.get_dist_params()
        dist.convert_to("m", "cm")
        dist.convert_to("cm", "m")
        new_params = dist.get_dist_params()
        assert np.allclose(old_params, new_params)

    def test_convert_to_exception(self):
        d = ExponentialDistribution(1, 0)
        d.convert_to("m", "cm")
        expected_params = [100, 0]
        new_params = d.get_dist_params()
        assert np.allclose(new_params, expected_params)