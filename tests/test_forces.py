
import numpy as np
from numpy.testing import assert_allclose

from nbody.forces import gravitational_force


def test_gravitational_force_x_axis():
    # Arrange
    m1 = 2.0
    m2 = 3.0
    r1 = np.array([0.0, 0.0])
    r2 = np.array([2.0, 0.0])

    expected = np.array([1.5, 0.0])

    # Act
    result = gravitational_force(m1, r1, m2, r2)

    # Assert
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_action_reaction():
    m1 = 2.0
    m2 = 3.0
    r1 = np.array([0.0, 1.0])
    r2 = np.array([2.0, 0.0])

    F = gravitational_force(m1, r1, m2, r2)
    F_prime = -gravitational_force(m2, r2, m1, r1)

    assert_allclose(F, F_prime, rtol=1e-12, atol=1e-12)


def test_inverse_square():
    m1 = 2.0
    m2 = 3.0
    r1 = np.array([0.0, 0.0])
    r2 = np.array([2.0, 0.0])
    r3 = np.array([4.0, 0.0])



    F = gravitational_force(m1, r1, m2, r2)
    F_2 = gravitational_force(m1, r1, m2, r3)

    assert_allclose(0.25 * F, F_2, rtol=1e-12, atol=1e-12)

def test_inputs_unchanged():
    m1 = 2.0
    m2 = 3.0
    r1 = np.array([1.0, 2.0])
    r2 = np.array([4.0, 6.0])

    inputs = [r1.copy(), r2.copy()]

    gravitational_force(m1, r1, m2, r2)

    post_inputs = [r1, r2]

    assert_allclose(inputs, post_inputs, rtol=1e-12, atol=1e-12)

