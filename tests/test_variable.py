"""Tests for the Variable class."""

import numpy as np
import pint
import pytest

from src.distributions import ArcsineDistribution, NormalDistribution
from src.polysys import ChebyshevTPolynomials
from src.variable import Variable


class TestVariable:
    @pytest.fixture
    def random_variable(self):
        return Variable("temperature", NormalDistribution(mu=10, sigma=2))

    # =====================================================================
    # 1. Construction and basic behavior
    # =====================================================================
    def test_init_with_distribution(self, random_variable):
        assert random_variable.name == "temperature"
        assert isinstance(random_variable.dist, NormalDistribution)
        assert random_variable.unit is None

    @pytest.mark.parametrize(
        "value",
        [
            "not a distribution",
            object(),
            None,
            2.5,
            True,
            np.bool_(False),
        ],
    )
    def test_init_rejects_non_distribution(self, value):
        with pytest.raises(TypeError, match="dist must be an instance of Distribution"):
            Variable("invalid", value)

    def test_repr_random_variable(self):
        variable = Variable("x", NormalDistribution(mu=1, sigma=2))
        assert repr(variable) == "Variable(x, N(1.00, 2.00), None)"

    def test_equality_for_matching_variables(self):
        assert Variable("x", NormalDistribution(1, 2)) == Variable(
            "x", NormalDistribution(1, 2)
        )

    def test_inequality_for_different_variable_values(self):
        assert Variable("x", NormalDistribution(1, 2)) != Variable(
            "x", NormalDistribution(1, 3)
        )

    def test_inequality_for_different_names_or_units(self):
        distribution = NormalDistribution(1, 2)
        assert Variable("x", distribution, unit="m") != Variable(
            "y", distribution, unit="m"
        )
        assert Variable("x", distribution, unit="m") != Variable(
            "x", distribution, unit="cm"
        )

    def test_equality_with_non_variable_is_false(self, random_variable):
        assert random_variable != object()

    def test_get_dist_type(self):
        variable = Variable("x", NormalDistribution())
        assert variable.get_dist_type() == "norm"

    def test_get_dist_params_for_random_variable(self, random_variable):
        assert np.allclose(random_variable.get_dist_params(), [10, 2])

    # =====================================================================
    # 2. Probability functions
    # =====================================================================
    def test_random_variable_probability_methods_delegate_to_distribution(
        self, random_variable
    ):
        points = np.array([6.0, 10.0, 14.0])
        probabilities = np.array([0.1, 0.5, 0.9])
        assert np.allclose(random_variable.pdf(points), random_variable.dist.pdf(points))
        assert np.allclose(random_variable.cdf(points), random_variable.dist.cdf(points))
        assert np.allclose(
            random_variable.ppf(probabilities),
            random_variable.dist.ppf(probabilities),
        )

    # =====================================================================
    # 3. Moments and sampling
    # =====================================================================
    def test_random_variable_moments_delegate_to_distribution(self, random_variable):
        assert random_variable.mean() == random_variable.dist.mean()
        assert random_variable.var() == random_variable.dist.var()
        assert random_variable.skew() == random_variable.dist.skew()
        assert random_variable.kurt() == random_variable.dist.kurt()
        assert random_variable.moments() == random_variable.dist.moments()

    def test_random_variable_sample_shape_and_seed(self, random_variable):
        first = random_variable.sample(20, method="MC", seed=42)
        second = random_variable.sample(20, method="MC", seed=42)
        assert np.asarray(first).shape == (20,)
        assert np.array_equal(first, second)

    # =====================================================================
    # 4. Polynomial-chaos and germ-space helpers
    # =====================================================================
    @pytest.fixture
    def arcsine_variable(self):
        return Variable("x", ArcsineDistribution(scale=2, loc=-1))

    def test_get_gpc_dist(self, arcsine_variable):
        assert arcsine_variable.get_gpc_dist() == ArcsineDistribution(scale=2, loc=-1)

    def test_get_gpc_syschar(self, arcsine_variable):
        assert arcsine_variable.get_gpc_syschar(normalized=False) == "T"
        assert arcsine_variable.get_gpc_syschar(normalized=True) == "t"

    def test_get_gpc_polysys(self, arcsine_variable):
        assert isinstance(arcsine_variable.get_gpc_polysys(), ChebyshevTPolynomials)

    def test_germ_variable_transformations_are_inverse(self, arcsine_variable):
        germ_points = np.array([-0.8, -0.2, 0.0, 0.4, 0.9])
        variable_points = arcsine_variable.germ_to_dist(germ_points)
        recovered = arcsine_variable.dist_to_germ(variable_points)
        assert np.allclose(recovered, germ_points)

    # =====================================================================
    # 5. Unit conversion
    # =====================================================================
    def test_convert_random_variable_to_another_unit(self):
        variable = Variable("x", ArcsineDistribution(scale=2, loc=-1), unit="m")
        report = variable.convert_to("cm")
        assert variable.dist.scale == pytest.approx(200.0)
        assert variable.dist.loc == pytest.approx(-100.0)
        assert variable.unit == "cm"
        assert report == "Variable x: Converted from m to cm."

    def test_convert_random_variable_round_trip(self):
        variable = Variable("x", ArcsineDistribution(scale=2, loc=-1), unit="m")
        variable.convert_to("cm")
        variable.convert_to("m")
        assert variable.dist == ArcsineDistribution(scale=2, loc=-1)
        assert variable.unit == "m"

    def test_convert_without_source_unit_raises(self):
        variable = Variable("x", NormalDistribution())
        report = variable.convert_to("cm")
        assert variable.unit == "cm"
        assert report == "Converted from None to cm."

    def test_incompatible_random_unit_conversion_does_not_change_variable_unit(self):
        variable = Variable("x", ArcsineDistribution(scale=2, loc=-1), unit="m")
        original_params = variable.dist.get_dist_params().copy()
        with pytest.raises(pint.DimensionalityError):
            variable.convert_to("s")
        assert variable.unit == "m"
        assert np.array_equal(variable.dist.get_dist_params(), original_params)
