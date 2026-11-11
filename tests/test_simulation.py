
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nbody.models import Body
from nbody.simulation import simulate


def make_body(mass, position, velocity):
    return Body(
        mass=mass,
        position=np.array(position, dtype=float),
        velocity=np.array(velocity, dtype=float),
    )


def test_zero_steps():
    bodies = [
        make_body(2.0, [1.0, 2.0], [3.0, 4.0])
    ]

    positions, velocities = simulate(bodies, dt=0.1, steps=0)

    assert positions.shape == (1, 1, 2)
    assert velocities.shape == (1, 1, 2)

    assert_array_equal(positions[0, 0], [1.0, 2.0])
    assert_array_equal(velocities[0, 0], [3.0, 4.0])


def test_empty_system():
    positions, velocities = simulate(
        [], dt=0.1, steps=5
    )

    assert positions.shape == (6, 0, 2)
    assert velocities.shape == (6, 0, 2)


def test_single_body_linear_motion():
    bodies = [
        make_body(2.0, [1.0, 2.0], [3.0, -1.0])
    ]

    positions, velocities = simulate(
        bodies, dt=0.5, steps=3
    )

    expected_positions = np.array([
        [1.0, 2.0],
        [2.5, 1.5],
        [4.0, 1.0],
        [5.5, 0.5],
    ])

    assert positions.shape == (4, 1, 2)
    assert velocities.shape == (4, 1, 2)

    assert_allclose(positions[:, 0, :], expected_positions)

    for v in velocities[:, 0, :]:
        assert_allclose(v, [3.0, -1.0])


def test_two_body_two_steps():
    bodies = [
        make_body(2.0, [0.0, 0.0], [0.1, 0.0]),
        make_body(3.0, [2.0, 0.0], [-0.2, 0.0]),
    ]

    positions, velocities = simulate(
        bodies, dt=0.1, steps=2
    )

    expected_positions = np.array([
        [[0.0, 0.0], [2.0, 0.0]],
        [[0.01, 0.0], [1.98, 0.0]],
        [[0.0275, 0.0], [1.955, 0.0]],
    ])

    assert_allclose(positions, expected_positions, atol=1e-12)

    # At step 1, separation is 1.97.
    # Euler uses that separation for the step-2 velocity update.
    a0 = 3.0 / (1.97 ** 2)
    a1 = -2.0 / (1.97 ** 2)

    assert_allclose(
        velocities[2, 0],
        [0.175 + 0.1 * a0, 0.0],
        atol=1e-12,
    )
    assert_allclose(
        velocities[2, 1],
        [-0.25 + 0.1 * a1, 0.0],
        atol=1e-12,
    )


def test_original_bodies_unchanged():
    bodies = [
        make_body(2.0, [1.0, 2.0], [0.5, 0.0]),
        make_body(3.0, [4.0, 6.0], [0.0, -0.5]),
    ]

    original_positions = [b.position.copy() for b in bodies]
    original_velocities = [b.velocity.copy() for b in bodies]

    simulate(bodies, dt=0.1, steps=10)

    for i, body in enumerate(bodies):
        assert_array_equal(body.position, original_positions[i])
        assert_array_equal(body.velocity, original_velocities[i])


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_steps(steps):
    bodies = [
        make_body(1.0, [0.0, 0.0], [0.0, 0.0])
    ]

    with pytest.raises(ValueError):
        simulate(bodies, dt=0.1, steps=steps)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_invalid_timestep(dt):
    with pytest.raises(ValueError):
        simulate([], dt=dt, steps=0)



import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nbody.models import Body
from nbody.simulation import simulate
from nbody.integrators import velocity_verlet_step


def make_bodies():
    return [
        Body(
            mass=2.0,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.1, 0.0]),
        ),
        Body(
            mass=3.0,
            position=np.array([2.0, 0.0]),
            velocity=np.array([-0.2, 0.0]),
        ),
    ]


def test_default_remains_euler():
    bodies = make_bodies()

    default = simulate(bodies, dt=0.1, steps=3)
    explicit = simulate(
        bodies, dt=0.1, steps=3, method="euler"
    )

    for actual, expected in zip(default, explicit):
        assert_array_equal(actual, expected)


def test_verlet_single_step():
    bodies = make_bodies()

    positions, velocities = simulate(
        bodies, dt=0.1, steps=1, method="verlet"
    )

    expected = velocity_verlet_step(bodies, dt=0.1)

    for i, body in enumerate(expected):
        assert_allclose(positions[1, i], body.position)
        assert_allclose(velocities[1, i], body.velocity)


def test_verlet_multiple_steps():
    bodies = make_bodies()

    positions, velocities = simulate(
        bodies, dt=0.1, steps=2, method="verlet"
    )

    first = velocity_verlet_step(bodies, dt=0.1)
    second = velocity_verlet_step(first, dt=0.1)

    assert positions.shape == (3, 2, 2)
    assert velocities.shape == (3, 2, 2)

    for i in range(2):
        assert_allclose(positions[0, i], bodies[i].position)
        assert_allclose(positions[1, i], first[i].position)
        assert_allclose(positions[2, i], second[i].position)

        assert_allclose(velocities[2, i], second[i].velocity)


def test_methods_produce_different_results():
    bodies = make_bodies()

    euler_positions, _ = simulate(
        bodies, dt=0.1, steps=1, method="euler"
    )
    verlet_positions, _ = simulate(
        bodies, dt=0.1, steps=1, method="verlet"
    )

    assert not np.allclose(euler_positions, verlet_positions)


@pytest.mark.parametrize("method", ["euler", "verlet"])
def test_empty_system(method):
    positions, velocities = simulate(
        [], dt=0.1, steps=3, method=method
    )

    assert positions.shape == (4, 0, 2)
    assert velocities.shape == (4, 0, 2)


@pytest.mark.parametrize("method", ["euler", "verlet"])
def test_zero_steps(method):
    bodies = make_bodies()

    positions, velocities = simulate(
        bodies, dt=0.1, steps=0, method=method
    )

    assert positions.shape == (1, 2, 2)
    assert velocities.shape == (1, 2, 2)

    for i, body in enumerate(bodies):
        assert_array_equal(positions[0, i], body.position)
        assert_array_equal(velocities[0, i], body.velocity)


@pytest.mark.parametrize("method", ["rk4", "", None])
def test_invalid_method(method):
    with pytest.raises(ValueError):
        simulate([], dt=0.1, steps=0, method=method)


def test_verlet_preserves_original_inputs():
    bodies = make_bodies()

    old_positions = [b.position.copy() for b in bodies]
    old_velocities = [b.velocity.copy() for b in bodies]

    simulate(bodies, dt=0.1, steps=3, method="verlet")

    for i, body in enumerate(bodies):
        assert_array_equal(body.position, old_positions[i])
        assert_array_equal(body.velocity, old_velocities[i])
