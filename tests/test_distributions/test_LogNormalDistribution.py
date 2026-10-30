"""Tests for the LogNormalDistribution class."""

import numpy as np
import pytest
from scipy.stats import lognorm

from .TestDistribution import TestDistribution as DistributionTestClass
from src.distributions import NormalDistribution

from src.distributions import UniformDistribution

from src.distributions import LogNormalDistribution


MU, SIGMA, LOC = 2, 0.5, 1


class TestLogNormalDistribution(DistributionTestClass):

    KURT_IS_EXCESS = True

    # ---- fixtures -------------------------------------------------------
    @pytest.fixture
    def dist(self):
        return LogNormalDistribution(0.1, 0.35)

    @pytest.fixture
    def support(self):
        return 0, np.inf

    # =====================================================================
    # 1. API / basic behaviour
    # =====================================================================
    def test_init_stores_parameters(self):
        d = LogNormalDistribution(MU, SIGMA, LOC)
        assert np.allclose((d.mu, d.sigma, d.loc), (MU, SIGMA, LOC), atol=self.ATOL, rtol=self.RTOL)

    def test_init_defaults_to_standard_lognormal(self):
        d = LogNormalDistribution()
        assert np.allclose((d.mu, d.sigma, d.loc), (0, 1, 0), atol=self.ATOL, rtol=self.RTOL)

    def test_get_dist_type(self, dist):
        assert dist.get_dist_type() == "lognorm"

    @pytest.mark.parametrize("mu, sigma, loc", [(0, 1, 0), (MU, SIGMA, LOC), (-3.5, 0.25, -2)])
    def test_get_dist_params(self, mu, sigma, loc):
        params = LogNormalDistribution(mu, sigma, loc).get_dist_params()
        assert np.asarray(params).shape == (3,)
        assert np.allclose(params, [mu, sigma, loc], atol=self.ATOL, rtol=self.RTOL)

    def test_eq_same_params(self):
        assert LogNormalDistribution(MU, SIGMA, LOC) == LogNormalDistribution(MU, SIGMA, LOC)

    @pytest.mark.parametrize(
        "other",
        [(0, SIGMA, LOC), (MU, 1, LOC), (MU, SIGMA, 0), (MU + 1, SIGMA + 1, LOC + 1)],
    )
    def test_eq_same_type_different_params(self, other):
        assert LogNormalDistribution(MU, SIGMA, LOC) != LogNormalDistribution(*other)

    def test_eq_different_distribution_type(self):
        assert LogNormalDistribution(0, 1) != NormalDistribution(0, 1)
        assert LogNormalDistribution(0, 1) != UniformDistribution(0, 1)

    def test_eq_non_distribution_object(self):
        assert LogNormalDistribution(MU, SIGMA, LOC) != (MU, SIGMA, LOC)
        assert LogNormalDistribution() != "lnN(0.00, 1.00), loc=0.00"

    @pytest.mark.parametrize(
        "mu, sigma, loc, expected",
        [
            (MU, SIGMA, LOC, "lnN(2.00, 0.25), loc=1.00"),
            (0, 1, 0, "lnN(0.00, 1.00), loc=0.00"),
            (-1.235, 0.5, 2.125, "lnN(-1.24, 0.25), loc=2.12"),
        ],
    )
    def test_repr(self, mu, sigma, loc, expected):
        assert repr(LogNormalDistribution(mu, sigma, loc)) == expected

    # =====================================================================
    # 2. Mathematical identities (lognormal-specific parts)
    # =====================================================================
    def test_pdf_and_cdf_are_zero_at_and_below_location(self):
        d = LogNormalDistribution(0.4, 0.8, 2.5)
        x = np.array([-1.0, 2.0, 2.5])
        assert np.allclose(d.pdf(x), 0, atol=self.ATOL, rtol=self.RTOL)
        assert np.allclose(d.cdf(x), 0, atol=self.ATOL, rtol=self.RTOL)

    def test_median_is_loc_plus_exp_mu(self):
        d = LogNormalDistribution(MU, SIGMA, LOC)
        assert np.isclose(d.ppf(0.5), LOC + np.exp(MU), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(d.cdf(LOC + np.exp(MU)), 0.5, atol=self.ATOL, rtol=self.RTOL)

    def test_ppf_endpoints(self, dist):
        assert np.isclose(dist.ppf(0), 0, atol=self.ATOL, rtol=self.RTOL)
        assert dist.ppf(1) == np.inf

    def test_ppf_outside_unit_interval_is_nan(self, dist):
        assert np.isnan(dist.ppf(-0.1))
        assert np.isnan(dist.ppf(1.1))

    def test_get_bounds_default_delta(self, dist):
        assert np.allclose(dist.get_bounds(), dist.ppf([0.02, 0.98]), atol=self.ATOL, rtol=self.RTOL)

    def test_get_bounds_at_zero_and_one(self):
        bounds = LogNormalDistribution(0.4, 0.5, LOC).get_bounds(delta=0)
        assert np.isclose(bounds[0], LOC, atol=self.ATOL, rtol=self.RTOL)
        assert bounds[1] == np.inf

    # =====================================================================
    # 3. Reference values
    # =====================================================================
    @pytest.mark.parametrize(
        "mu, sigma, loc",
        [(0, 1, 0), (MU, SIGMA, LOC), (-1, 0.3, -2), (2, 0.8, 0)],
    )
    def test_pdf_cdf_and_ppf_match_scipy(self, mu, sigma, loc):
        d = LogNormalDistribution(mu, sigma, loc)
        x = loc + np.exp(np.linspace(mu - 1.5 * sigma, mu + 1.5 * sigma, 40))
        q = np.linspace(0.01, 0.99, 40)
        assert np.allclose(
            d.pdf(x), lognorm.pdf(x, s=sigma, scale=np.exp(mu), loc=loc),
            atol=self.ATOL, rtol=self.RTOL,
        )
        assert np.allclose(
            d.cdf(x), lognorm.cdf(x, s=sigma, scale=np.exp(mu), loc=loc),
            atol=self.ATOL, rtol=self.RTOL,
        )
        assert np.allclose(
            d.ppf(q), lognorm.ppf(q, s=sigma, scale=np.exp(mu), loc=loc),
            atol=self.ATOL, rtol=self.RTOL,
        )

    def test_moments_reference_values(self):
        d = LogNormalDistribution(MU, SIGMA, LOC)
        s2 = SIGMA ** 2
        mean = LOC + np.exp(MU + s2 / 2)
        var = (np.exp(s2) - 1) * np.exp(2 * MU + s2)
        skew = (np.exp(s2) + 2) * np.sqrt(np.exp(s2) - 1)
        kurt = np.exp(4 * s2) + 2 * np.exp(3 * s2) + 3 * np.exp(2 * s2) - 6
        assert np.allclose(d.moments(), [mean, var, skew, kurt], atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 4. Edge cases / invalid parameters
    # =====================================================================
    @pytest.mark.parametrize(
        "mu, sigma, loc",
        [
            ("a", 1, 0), (None, 1, 0), (np.inf, 1, 0), (-np.inf, 1, 0), (np.nan, 1, 0),
            (0, "a", 0), (0, None, 0), (0, 0, 0), (0, -1, 0), (0, np.inf, 0), (0, np.nan, 0),
            (0, 1, "a"), (0, 1, None), (0, 1, np.inf), (0, 1, -np.inf), (0, 1, np.nan),
        ],
    )
    def test_invalid_parameters_raise_value_error(self, mu, sigma, loc):
        with pytest.raises(ValueError):
            LogNormalDistribution(mu, sigma, loc)

    # =====================================================================
    # 5. Moments in specific cases
    # =====================================================================
    MOMENT_CASES = [
        (0, 1, 0),
        (0, 0.5, 0),
        (MU, SIGMA, LOC),
        (-1, 0.25, -3),
        (2, 0.1, 10),
    ]

    @pytest.mark.parametrize("mu, sigma, loc", MOMENT_CASES)
    def test_mean_specific_cases(self, mu, sigma, loc):
        expected = loc + np.exp(mu + sigma ** 2 / 2)
        assert np.isclose(LogNormalDistribution(mu, sigma, loc).mean(), expected, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mu, sigma, loc", MOMENT_CASES)
    def test_var_specific_cases(self, mu, sigma, loc):
        expected = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        assert np.isclose(LogNormalDistribution(mu, sigma, loc).var(), expected, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mu, sigma, loc", MOMENT_CASES)
    def test_skew_specific_cases(self, mu, sigma, loc):
        expected = (np.exp(sigma ** 2) + 2) * np.sqrt(np.exp(sigma ** 2) - 1)
        assert np.isclose(LogNormalDistribution(mu, sigma, loc).skew(), expected, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("mu, sigma, loc", MOMENT_CASES)
    def test_kurt_specific_cases(self, mu, sigma, loc):
        s2 = sigma ** 2
        expected = np.exp(4 * s2) + 2 * np.exp(3 * s2) + 3 * np.exp(2 * s2) - 6
        assert np.isclose(LogNormalDistribution(mu, sigma, loc).kurt(), expected, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 6. Sampling
    # =====================================================================
    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    @pytest.mark.parametrize("mu, sigma, loc", [(0, 0.4, 0), (-0.2, 0.7, 1.5)])
    def test_sample_within_support_all_methods(self, method, mu, sigma, loc):
        samples = np.asarray(LogNormalDistribution(mu, sigma, loc).sample(64, method=method, seed=1))
        assert samples.shape == (64,)
        assert np.all(np.isfinite(samples))
        assert np.all(samples >= loc)

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_statistics_all_methods(self, method):
        d = LogNormalDistribution(0, 0.35)
        samples = np.asarray(d.sample(4096, method=method, seed=7))
        assert abs(np.mean(samples) - d.mean()) < 0.1
        assert abs(np.var(samples) - d.var()) < 0.1

    @pytest.mark.parametrize("method", DistributionTestClass.SAMPLE_METHODS)
    def test_sample_is_reproducible_with_seed(self, method):
        d = LogNormalDistribution(0.2, 0.35, 1)
        assert np.array_equal(d.sample(32, method=method, seed=123), d.sample(32, method=method, seed=123))

    def test_sample_mc_differs_for_different_seeds(self):
        d = LogNormalDistribution(0, 0.35)
        assert not np.array_equal(d.sample(32, "MC", seed=1), d.sample(32, "MC", seed=2))

    def test_sample_default_method_is_mc(self):
        d = LogNormalDistribution(0.2, 0.35)
        assert np.array_equal(d.sample(16, seed=5), d.sample(16, "MC", seed=5))

    @pytest.mark.parametrize("n", [0, -3, 2.5, "10", None])
    def test_sample_invalid_n_raises(self, n):
        with pytest.raises(ValueError):
            LogNormalDistribution().sample(n)

    @pytest.mark.parametrize("seed", [1.5, "abc"])
    def test_sample_invalid_seed_raises(self, seed):
        with pytest.raises(ValueError):
            LogNormalDistribution().sample(8, seed=seed)

    def test_sample_unknown_method_raises(self):
        with pytest.raises(ValueError):
            LogNormalDistribution().sample(8, method="unknown")

    # =====================================================================
    # 7. transform / get_shift / get_scale
    # =====================================================================
    @pytest.mark.parametrize("shift, scale", [(0, 1), (1, 2), (-3, 0.5), (2.5, 4), (0, 10), (-1, 1)])
    def test_transform_maps_parameters_affinely(self, shift, scale):
        d = LogNormalDistribution(MU, SIGMA, LOC)
        translated = d.transform(shift, scale)
        assert isinstance(translated, LogNormalDistribution)
        assert np.isclose(translated.mu, MU + np.log(scale), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(translated.sigma, SIGMA, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(translated.loc, LOC * scale + shift, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize("shift, scale", [(1, 2), (-3, 0.5), (2.5, 4)])
    def test_transform_transforms_moments(self, shift, scale):
        d = LogNormalDistribution(MU, SIGMA, LOC)
        translated = d.transform(shift, scale)
        assert np.isclose(translated.mean(), scale * d.mean() + shift, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(translated.var(), scale ** 2 * d.var(), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(translated.skew(), d.skew(), atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(translated.kurt(), d.kurt(), atol=self.ATOL, rtol=self.RTOL)

    def test_transform_default_is_identity(self):
        d = LogNormalDistribution(MU, SIGMA, LOC)
        assert d.transform() == d

    def test_transform_does_not_modify_original(self):
        d = LogNormalDistribution(MU, SIGMA, LOC)
        d.transform(3, 2)
        assert (d.mu, d.sigma, d.loc) == (MU, SIGMA, LOC)

    def test_transform_composition(self):
        d = LogNormalDistribution(MU, SIGMA, LOC)
        translated = d.transform(1, 2).transform(-3, 0.5)
        expected = d.transform(0.5 * 1 - 3, 0.5 * 2)
        assert np.allclose(translated.get_dist_params(), expected.get_dist_params(), atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "shift, scale",
        [(0, 0), (0, -1), (np.inf, 1), (0, np.inf), ("a", 1), (0, "b"), (None, 1), (np.nan, 1)],
    )
    def test_transform_invalid_raises(self, shift, scale):
        with pytest.raises(ValueError):
            LogNormalDistribution(MU, SIGMA, LOC).transform(shift, scale)

    @pytest.mark.parametrize("mu, sigma, loc", [(0, 1, 0), (MU, SIGMA, LOC), (-2, 0.25, -3.5)])
    def test_get_shift_and_scale(self, mu, sigma, loc):
        d = LogNormalDistribution(mu, sigma, loc)
        assert np.isclose(d.get_shift(), loc, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(d.get_scale(), sigma, atol=self.ATOL, rtol=self.RTOL)

    # =====================================================================
    # 8. fix_moments (verified by quadrature)
    # =====================================================================
    FIX_MOMENT_CASES = [
        (1.0, 0.1, 0),
        (2.0, 0.5, 0),
        (5.0, 3.0, 1),
        (10.0, 20.0, -2),
        (0.8, 0.03, 0.2),
    ]

    @pytest.mark.parametrize("mean, var, loc", FIX_MOMENT_CASES)
    def test_fix_moments_matches_by_quadrature(self, mean, var, loc):
        new = LogNormalDistribution(0.3, 0.8, -1).fix_moments(mean, var, loc)
        assert isinstance(new, LogNormalDistribution)
        assert np.isclose(new.loc, loc, atol=self.ATOL, rtol=self.RTOL)
        lo, hi = new.loc, np.inf
        q_mean = self._quad(lambda t: t * new.pdf(t), lo, hi)
        q_var = self._central_moment(new, 2, lo, hi)
        assert np.isclose(q_mean, mean, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(q_var, var, atol=self.QUAD_ATOL, rtol=self.QUAD_RTOL)
        assert np.isclose(new.mean(), mean, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(new.var(), var, atol=self.ATOL, rtol=self.RTOL)

    def test_fix_moments_does_not_modify_original(self, dist):
        original_params = dist.get_dist_params().copy()
        dist.fix_moments(10, 4)
        assert np.array_equal(dist.get_dist_params(), original_params)

    def test_fix_moments_defaults(self):
        new = LogNormalDistribution(2, 0.25, 4).fix_moments()
        expected_var = np.exp(1) * (np.exp(1) - 1)
        assert np.isclose(new.mean(), 1, atol=self.ATOL, rtol=self.RTOL)
        assert np.isclose(new.var(), expected_var, atol=self.ATOL, rtol=self.RTOL)
        assert new.loc == 0

    @pytest.mark.parametrize(
        "mean, var, loc",
        [
            (0, 1, 0), (-1, 1, 0), (1, 1, 1), (1, 1, 2),
            (1, 0, 0), (1, -1, 0), ("a", 1, 0), (1, "b", 0), (1, 1, "c"),
            (np.inf, 1, 0), (np.nan, 1, 0), (1, np.inf, 0), (1, np.nan, 0), (1, 1, np.inf),
        ],
    )
    def test_fix_moments_invalid_raises(self, mean, var, loc):
        with pytest.raises(ValueError):
            LogNormalDistribution().fix_moments(mean, var, loc)

    # =====================================================================
    # 9. fix_bounds
    # =====================================================================
    @pytest.mark.parametrize("new_min, new_max", [(0, 1), (-1, 1), (10, 20), (-5.5, -0.5)])
    def test_fix_bounds_default_quantiles(self, dist, new_min, new_max):
        new = dist.fix_bounds(new_min, new_max)
        assert isinstance(new, LogNormalDistribution)
        assert np.isclose(new.ppf(0), new_min, atol=1e-8)
        assert np.isclose(new.ppf(0.98), new_max, atol=1e-8)

    @pytest.mark.parametrize("q0, q1", [(0.1, 0.9), (0.25, 0.75), (0.01, 0.5), (0.5, 0.99)])
    def test_fix_bounds_with_quantiles(self, dist, q0, q1):
        new = dist.fix_bounds(10, 20, q0, q1)
        assert np.isclose(new.ppf(q0), 10, atol=1e-8)
        assert np.isclose(new.ppf(q1), 20, atol=1e-8)

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((1, 1), {}), ((2, 1), {}), (("a", 1), {}), ((0, None), {}),
            ((0, 1), {"q0": -0.1}), ((0, 1), {"q0": 1.1}),
            ((0, 1), {"q0": 0.6, "q1": 0.4}), ((0, 1), {"q1": 1.5}),
        ],
    )
    def test_fix_bounds_invalid_raises(self, dist, args, kwargs):
        with pytest.raises(ValueError):
            dist.fix_bounds(*args, **kwargs)

    # =====================================================================
    # 10. GPC related functions
    # =====================================================================
    def test_gpc_dist_is_not_available(self, dist):
        with pytest.raises(Exception):
            dist.gpc_dist()

    def test_germ_conversions_require_a_gpc_distribution(self, dist):
        with pytest.raises(Exception):
            dist.germ_to_dist(0.5)
        with pytest.raises(Exception):
            dist.dist_to_germ(1.0)

    @pytest.mark.parametrize("mu, sigma, loc", [(0, 1, 0), (MU, SIGMA, LOC), (-1, 0.25, -2)])
    def test_gpc_polysys_invalid_raises(self, mu, sigma, loc):
        with pytest.raises(Exception):
            LogNormalDistribution(mu, sigma, loc).gpc_polysys()

    @pytest.mark.parametrize("normalized", [False, True])
    def test_gpc_syschar_invalid_raises(self, normalized):
        with pytest.raises(Exception):
            LogNormalDistribution().gpc_syschar(normalized)

    # =====================================================================
    # 11. Unit conversions
    # =====================================================================
    def test_convert_to_round_trip(self):
        dist = LogNormalDistribution(MU, SIGMA, LOC)
        old_params = dist.get_dist_params()
        dist.convert_to("m", "cm")
        dist.convert_to("cm", "m")
        new_params = dist.get_dist_params()
        assert np.allclose(old_params, new_params)

    def test_convert_to_exception(self):
        d = LogNormalDistribution(MU, SIGMA, LOC)
        d.convert_to("m", "cm")
        expected_params = [100*MU, 100*SIGMA, 100*LOC]
        new_params = d.get_dist_params()
        assert np.allclose(new_params, expected_params)