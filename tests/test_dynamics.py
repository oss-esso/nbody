
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nbody.models import Body
from nbody.dynamics import accelerations


def make_body(mass, x, y):
    return Body(
        mass=mass,
        position=np.array([x, y], dtype=float),
        velocity=np.zeros(2),
    )


def test_empty_system():
    result = accelerations([])

    assert result.shape == (0, 2)


def test_single_body():
    bodies = [make_body(5.0, 2.0, 3.0)]

    result = accelerations(bodies)

    assert result.shape == (1, 2)
    assert_allclose(result, [[0.0, 0.0]], atol=1e-12)


def test_two_body_accelerations():
    bodies = [
        make_body(2.0, 0.0, 0.0),
        make_body(3.0, 2.0, 0.0),
    ]

    result = accelerations(bodies)

    expected = np.array([
        [0.75, 0.0],
        [-0.50, 0.0],
    ])

    assert result.shape == (2, 2)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_three_body_accelerations():
    bodies = [
        make_body(2.0, 0.0, 0.0),
        make_body(3.0, 3.0, 0.0),
        make_body(5.0, 0.0, 4.0),
    ]

    result = accelerations(bodies)

    # Independently calculated Newtonian accelerations.
    expected = np.array([
        [1/3, 5/16],
        [-2/9 - 3/25, 4/25],
        [9/125, -1/8 - 12/125],
    ])

    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_total_internal_force_is_zero():
    bodies = [
        make_body(2.0, 1.0, 2.0),
        make_body(3.0, 4.0, 6.0),
        make_body(5.0, -2.0, 3.0),
        make_body(7.0, 3.0, -4.0),
    ]

    a = accelerations(bodies)
    masses = np.array([body.mass for body in bodies])

    total_force = np.sum(masses[:, None] * a, axis=0)

    assert_allclose(total_force, [0.0, 0.0], atol=1e-12)


def test_gravitational_constant_scaling():
    bodies = [
        make_body(2.0, 0.0, 0.0),
        make_body(3.0, 2.0, 0.0),
    ]

    a1 = accelerations(bodies, G=1.0)
    a2 = accelerations(bodies, G=2.0)

    assert_allclose(a2, 2 * a1, rtol=1e-12, atol=1e-12)


def test_inputs_unchanged():
    bodies = [
        make_body(2.0, 1.0, 2.0),
        make_body(3.0, 4.0, 6.0),
    ]

    original_positions = [b.position.copy() for b in bodies]
    original_velocities = [b.velocity.copy() for b in bodies]
    original_masses = [b.mass for b in bodies]

    accelerations(bodies)

    for i, body in enumerate(bodies):
        assert_array_equal(body.position, original_positions[i])
        assert_array_equal(body.velocity, original_velocities[i])
        assert body.mass == original_masses[i]


def test_coincident_bodies_rejected():
    bodies = [
        make_body(2.0, 1.0, 2.0),
        make_body(3.0, 1.0, 2.0),
    ]

    with pytest.raises(ValueError):
        accelerations(bodies)
