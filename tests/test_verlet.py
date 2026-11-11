
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nbody.models import Body
from nbody.integrators import velocity_verlet_step


def make_body(mass, position, velocity):
    return Body(
        mass=mass,
        position=np.array(position, dtype=float),
        velocity=np.array(velocity, dtype=float),
    )


def test_empty_system():
    assert velocity_verlet_step([], dt=0.1) == []


def test_single_body_constant_velocity():
    body = make_body(
        2.0, [1.0, 2.0], [3.0, -2.0]
    )

    result = velocity_verlet_step([body], dt=0.5)

    assert_allclose(result[0].position, [2.5, 1.0])
    assert_allclose(result[0].velocity, [3.0, -2.0])


def test_two_body_position_update():
    bodies = [
        make_body(2.0, [0.0, 0.0], [0.1, 0.0]),
        make_body(3.0, [2.0, 0.0], [-0.2, 0.0]),
    ]

    result = velocity_verlet_step(bodies, dt=0.1)

    # Initial accelerations:
    # a0 = 0.75, a1 = -0.50

    # r_new = r_old + v_old*dt + 0.5*a_old*dt**2
    assert_allclose(
        result[0].position,
        [0.01375, 0.0],
        atol=1e-12,
    )

    assert_allclose(
        result[1].position,
        [1.9775, 0.0],
        atol=1e-12,
    )


def test_velocity_uses_new_acceleration():
    bodies = [
        make_body(2.0, [0.0, 0.0], [0.1, 0.0]),
        make_body(3.0, [2.0, 0.0], [-0.2, 0.0]),
    ]

    result = velocity_verlet_step(bodies, dt=0.1)

    # New separation after the position update
    separation = 1.9775 - 0.01375

    a0_old = 0.75
    a1_old = -0.50

    a0_new = 3.0 / separation**2
    a1_new = -2.0 / separation**2

    expected_v0 = 0.1 + 0.05 * (a0_old + a0_new)
    expected_v1 = -0.2 + 0.05 * (a1_old + a1_new)

    assert_allclose(
        result[0].velocity,
        [expected_v0, 0.0],
        atol=1e-12,
    )

    assert_allclose(
        result[1].velocity,
        [expected_v1, 0.0],
        atol=1e-12,
    )


def test_total_momentum_conserved():
    bodies = [
        make_body(2.0, [1, 2], [0.1, -0.2]),
        make_body(3.0, [4, 6], [0.2, 0.1]),
        make_body(5.0, [-2, 3], [-0.3, 0.2]),
    ]

    old_momentum = sum(
        (b.mass * b.velocity for b in bodies),
        np.zeros(2),
    )

    result = velocity_verlet_step(bodies, dt=0.01)

    new_momentum = sum(
        (b.mass * b.velocity for b in result),
        np.zeros(2),
    )

    assert_allclose(
        new_momentum,
        old_momentum,
        rtol=0,
        atol=1e-12,
    )


def test_original_bodies_unchanged():
    bodies = [
        make_body(2.0, [0, 0], [0.1, 0]),
        make_body(3.0, [2, 0], [-0.2, 0]),
    ]

    positions = [b.position.copy() for b in bodies]
    velocities = [b.velocity.copy() for b in bodies]

    result = velocity_verlet_step(bodies, dt=0.1)

    for i, body in enumerate(bodies):
        assert result[i] is not body
        assert_array_equal(body.position, positions[i])
        assert_array_equal(body.velocity, velocities[i])

        assert not np.shares_memory(
            result[i].position, body.position
        )
        assert not np.shares_memory(
            result[i].velocity, body.velocity
        )


@pytest.mark.parametrize(
    "dt", [0.0, -0.1, np.nan, np.inf]
)
def test_invalid_timestep(dt):
    with pytest.raises(ValueError):
        velocity_verlet_step([], dt=dt)


@pytest.mark.parametrize(
    "G", [0.0, -1.0, np.nan, np.inf]
)
def test_invalid_gravitational_constant(G):
    with pytest.raises(ValueError):
        velocity_verlet_step([], dt=0.1, G=G)


def test_coincident_bodies():
    bodies = [
        make_body(1.0, [0, 0], [0, 0]),
        make_body(2.0, [0, 0], [0, 0]),
    ]

    with pytest.raises(ValueError):
        velocity_verlet_step(bodies, dt=0.1)


def test_collision_at_new_positions():
    bodies = [
        make_body(1.0, [-1, 0], [0.995, 0]),
        make_body(1.0, [1, 0], [-0.995, 0]),
    ]

    # At dt=1, initial accelerations are +/-0.25.
    # Each position update contributes +/-0.125,
    # so the resulting positions are -0.13 and +0.13.
    # This case stays separated; it must remain valid.
    result = velocity_verlet_step(bodies, dt=1.0)

    assert np.all(np.isfinite(
        [b.position for b in result]
    ))
    assert np.all(np.isfinite(
        [b.velocity for b in result]
    ))
