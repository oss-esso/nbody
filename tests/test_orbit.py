
import numpy as np
from numpy.testing import assert_allclose

from nbody.initial_conditions import circular_binary
from nbody.simulation import simulate


def test_binary_orbit_conserves_com_and_momentum():
    bodies = circular_binary(
        m1=1.0,
        m2=1.0,
        separation=2.0,
        G=1.0,
    )

    period = 4 * np.pi
    steps = 4000

    positions, velocities = simulate(
        bodies,
        dt=period / steps,
        steps=steps,
        G=1.0,
    )

    assert positions.shape == (4001, 2, 2)
    assert velocities.shape == (4001, 2, 2)

    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(velocities))

    # Equal masses: the COM is their average position.
    center_of_mass = (
        positions[:, 0, :] + positions[:, 1, :]
    ) / 2

    # Equal masses: zero momentum means v0 + v1 = 0.
    total_momentum = (
        velocities[:, 0, :] + velocities[:, 1, :]
    )

    assert_allclose(
        center_of_mass,
        np.zeros_like(center_of_mass),
        atol=1e-10,
    )

    assert_allclose(
        total_momentum,
        np.zeros_like(total_momentum),
        atol=1e-10,
    )
