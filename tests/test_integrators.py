
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nbody.models import Body
from nbody.integrators import euler_step


def make_body(mass, position, velocity):
    return Body(
        mass=mass,
        position=np.array(position, dtype=float),
        velocity=np.array(velocity, dtype=float),
    )


def test_empty_system():
    assert euler_step([], dt=0.1) == []


def test_single_body_constant_velocity():
    body = make_body(
        mass=2.0,
        position=[1.0, 2.0],
        velocity=[2.0, -3.0],
    )

    result = euler_step([body], dt=0.25)

    assert len(result) == 1
    assert_allclose(result[0].position, [1.5, 1.25])
    assert_allclose(result[0].velocity, [2.0, -3.0])


def test_two_body_euler_step():
    bodies = [
        make_body(2.0, [0.0, 0.0], [0.1, 0.0]),
        make_body(3.0, [2.0, 0.0], [-0.2, 0.0]),
    ]

    result = euler_step(bodies, dt=0.1)

    # Explicit Euler uses the OLD velocities
    # when updating positions.
    expected_positions = [
        [0.01, 0.0],
        [1.98, 0.0],
    ]

    # Initial gravitational accelerations:
    # a0 = +0.75, a1 = -0.50
    expected_velocities = [
        [0.175, 0.0],
        [-0.25, 0.0],
    ]

    for i in range(2):
        assert_allclose(
            result[i].position,
            expected_positions[i],
            atol=1e-12,
        )
        assert_allclose(
            result[i].velocity,
            expected_velocities[i],
            atol=1e-12,
        )


def test_original_bodies_unchanged():
    bodies = [
        make_body(2.0, [1.0, 2.0], [0.5, 0.0]),
        make_body(3.0, [4.0, 6.0], [0.0, -0.5]),
    ]

    original_positions = [b.position.copy() for b in bodies]
    original_velocities = [b.velocity.copy() for b in bodies]

    result = euler_step(bodies, dt=0.1)

    for i in range(len(bodies)):
        assert result[i] is not bodies[i]
        assert_array_equal(
            bodies[i].position,
            original_positions[i],
        )
        assert_array_equal(
            bodies[i].velocity,
            original_velocities[i],
        )

    # Returned arrays must not share mutable storage.
    for i in range(len(bodies)):
        assert not np.shares_memory(
            result[i].position, bodies[i].position
        )
        assert not np.shares_memory(
            result[i].velocity, bodies[i].velocity
        )


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_invalid_timestep(dt):
    body = make_body(1.0, [0.0, 0.0], [1.0, 0.0])

    with pytest.raises(ValueError):
        euler_step([body], dt=dt)


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
def test_invalid_timestep_empty_system(dt):
    with pytest.raises(ValueError):
        euler_step([], dt=dt)


