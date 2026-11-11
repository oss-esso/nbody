
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nbody.initial_conditions import circular_binary
from nbody.dynamics import accelerations


def test_equal_mass_binary():
    bodies = circular_binary(
        m1=1.0,
        m2=1.0,
        separation=2.0,
        G=1.0,
    )

    assert len(bodies) == 2

    assert_allclose(bodies[0].position, [-1.0, 0.0])
    assert_allclose(bodies[1].position, [1.0, 0.0])

    assert_allclose(bodies[0].velocity, [0.0, -0.5])
    assert_allclose(bodies[1].velocity, [0.0, 0.5])


def test_center_of_mass_and_momentum():
    bodies = circular_binary(
        m1=2.0,
        m2=3.0,
        separation=5.0,
    )

    masses = np.array([b.mass for b in bodies])
    positions = np.array([b.position for b in bodies])
    velocities = np.array([b.velocity for b in bodies])

    center_of_mass = np.sum(
        masses[:, None] * positions, axis=0
    ) / np.sum(masses)

    total_momentum = np.sum(
        masses[:, None] * velocities, axis=0
    )

    assert_allclose(center_of_mass, [0, 0], atol=1e-12)
    assert_allclose(total_momentum, [0, 0], atol=1e-12)

    assert_allclose(
        np.linalg.norm(positions[1] - positions[0]),
        5.0,
    )


def test_centripetal_acceleration():
    m1, m2 = 2.0, 3.0
    d = 4.0
    G = 1.0

    bodies = circular_binary(m1, m2, d, G)

    omega_squared = G * (m1 + m2) / d**3

    actual = accelerations(bodies, G=G)

    expected = np.array([
        -omega_squared * b.position
        for b in bodies
    ])

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "m1,m2,d,G",
    [
        (0.0, 1.0, 2.0, 1.0),
        (1.0, -1.0, 2.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 2.0, -1.0),
        (np.nan, 1.0, 2.0, 1.0),
        (1.0, 1.0, np.inf, 1.0),
    ],
)
def test_invalid_parameters(m1, m2, d, G):
    with pytest.raises(ValueError):
        circular_binary(m1, m2, d, G)
