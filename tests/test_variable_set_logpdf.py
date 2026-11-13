import numpy as np
import pytest

from src.distributions import UniformDistribution
from src.variable import Variable
from src.variable_set import VariableSet


def test_joint_logpdf_avoids_underflow_from_multiplying_marginal_pdfs():
    variables = VariableSet()
    for i in range(40):
        variables.add(Variable(f"x{i}", UniformDistribution(0, 1e10)))
    points = np.full((40, 2), 5e9)

    logpdf = variables.logpdf(points)

    assert logpdf.shape == (1, 2)
    np.testing.assert_allclose(logpdf, np.full((1, 2), 40 * np.log(1e-10)))


def test_marginal_logpdf_returns_one_row_per_variable():
    variables = VariableSet()
    variables.add(Variable("x", UniformDistribution(0, 10)))
    variables.add(Variable("y", UniformDistribution(0, 100)))
    points = np.array([[5.0, 6.0], [50.0, 60.0]])

    logpdf = variables.logpdf(points, marginal=True)

    assert logpdf.shape == (2, 2)
    np.testing.assert_allclose(
        logpdf,
        np.array([[np.log(0.1), np.log(0.1)], [np.log(0.01), np.log(0.01)]]),
    )


def test_cdf_returns_one_row_per_variable():
    variables = VariableSet()
    variables.add(Variable("x", UniformDistribution(0, 10)))
    variables.add(Variable("y", UniformDistribution(0, 100)))
    points = np.array([[5.0, 6.0], [50.0, 60.0]])

    cdf = variables.cdf(points)

    assert cdf.shape == (2, 2)
    np.testing.assert_allclose(cdf, np.array([[0.5, 0.6], [0.5, 0.6]]))


def test_cdf_rejects_input_with_wrong_dimensions_or_variable_count():
    variables = VariableSet()
    variables.add(Variable("x", UniformDistribution(0, 1)))

    for points in (np.array([0.5]), np.array([[0.5], [0.6]])):
        with pytest.raises(ValueError):
            variables.cdf(points)
