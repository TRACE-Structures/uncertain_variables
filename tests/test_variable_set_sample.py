import numpy as np
import pytest

from src.distributions import UniformDistribution
from src.variable import Variable
from src.variable_set import VariableSet


@pytest.mark.parametrize("method", ["MC", "QMC_Halton", "QMC_LHS", "QMC_Sobol"])
def test_sample_returns_reproducible_independent_rows(method):
    variables = VariableSet()
    variables.add(Variable("x", UniformDistribution()))
    variables.add(Variable("y", UniformDistribution()))

    first = variables.sample(8, method=method, seed=123)
    second = variables.sample(8, method=method, seed=123)

    assert first.shape == (2, 8)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0], first[1])


def test_sobol_saltelli_returns_samples_for_each_variable():
    variables = VariableSet()
    variables.add(Variable("x", UniformDistribution(0, 1)))
    variables.add(Variable("y", UniformDistribution(2, 4)))

    samples = variables.sample(8, method="Sobol_saltelli", seed=123)

    assert samples.shape[1] == 2
    assert np.all((samples[:, 0] >= 0) & (samples[:, 0] <= 1))
    assert np.all((samples[:, 1] >= 2) & (samples[:, 1] <= 4))


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_sample_rejects_invalid_sample_count(n):
    variables = VariableSet()
    variables.add(Variable("x", UniformDistribution()))

    with pytest.raises(ValueError, match="positive integer"):
        variables.sample(n)


def test_sample_rejects_invalid_seed():
    variables = VariableSet()
    variables.add(Variable("x", UniformDistribution()))

    with pytest.raises(TypeError, match="seed must be an integer"):
        variables.sample(2, seed="123")
