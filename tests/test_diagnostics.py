
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nbody.models import Body
from nbody.diagnostics import total_energy


def make_body(mass, position, velocity=(0.0, 0.0)):
    return Body(
        mass=mass,
        position=np.array(position, dtype=float),
        velocity=np.array(velocity, dtype=float),
    )


def test_empty_system():
    assert total_energy([]) == 0.0


def test_single_body_kinetic_energy():
    body = make_body(2.0, [0, 0], [3, 4])

    # K = 1/2 * 2 * (3² + 4²) = 25
    assert_allclose(total_energy([body]), 25.0)


def test_two_stationary_bodies():
    bodies = [
        make_body(2.0, [0, 0]),
        make_body(3.0, [3, 4]),
    ]

    # Separation = 5
    # U = -G*m1*m2/r = -6/5
    assert_allclose(total_energy(bodies), -1.2)


def test_kinetic_plus_potential():
    bodies = [
        make_body(2.0, [0, 0], [1, 0]),
        make_body(3.0, [3, 4], [0, 2]),
    ]

    # K1 = 1, K2 = 6, U = -1.2
    assert_allclose(total_energy(bodies), 5.8)


def test_three_body_pair_counting():
    bodies = [
        make_body(1.0, [0, 0]),
        make_body(2.0, [2, 0]),
        make_body(3.0, [5, 0]),
    ]

    # U01 = -2/2 = -1
    # U02 = -3/5 = -0.6
    # U12 = -6/3 = -2
    assert_allclose(total_energy(bodies), -3.6)


def test_gravitational_constant_scaling():
    bodies = [
        make_body(2.0, [0, 0]),
        make_body(3.0, [3, 4]),
    ]

    assert_allclose(total_energy(bodies, G=2.0), -2.4)


@pytest.mark.parametrize("G", [0.0, -1.0, np.nan, np.inf])
def test_invalid_G(G):
    with pytest.raises(ValueError):
        total_energy([], G=G)


def test_coincident_positions():
    bodies = [
        make_body(1.0, [0, 0]),
        make_body(2.0, [0, 0]),
    ]

    with pytest.raises(ValueError):
        total_energy(bodies)


from nbody.diagnostics import total_angular_momentum
from nbody.initial_conditions import circular_binary
from numpy.testing import assert_array_equal


def test_angular_momentum_empty():
    assert total_angular_momentum([]) == 0.0


def test_angular_momentum_single_body():
    body = make_body(
        2.0,
        [3.0, 4.0],
        [-1.0, 2.0],
    )

    # Lz = 2 * (3*2 - 4*(-1)) = 20
    assert_allclose(total_angular_momentum([body]), 20.0)


def test_angular_momentum_two_bodies():
    bodies = [
        make_body(2.0, [1.0, 0.0], [0.0, 3.0]),
        make_body(3.0, [0.0, 2.0], [4.0, 0.0]),
    ]

    # L1 = 2 * (1*3 - 0*0) = 6
    # L2 = 3 * (0*0 - 2*4) = -24
    # Total = -18

    assert_allclose(
        total_angular_momentum(bodies),
        -18.0,
    )


def test_angular_momentum_circular_binary():
    bodies = circular_binary(
        m1=1.0,
        m2=1.0,
        separation=2.0,
        G=1.0,
    )

    # Each body contributes 0.5
    assert_allclose(
        total_angular_momentum(bodies),
        1.0,
        atol=1e-12,
    )


def test_angular_momentum_order_independence():
    bodies = [
        make_body(2.0, [1.0, 2.0], [-1.0, 3.0]),
        make_body(3.0, [4.0, -2.0], [2.0, 1.0]),
        make_body(5.0, [-3.0, 1.0], [0.0, -2.0]),
    ]

    original = total_angular_momentum(bodies)
    reversed_order = total_angular_momentum(bodies[::-1])

    assert_allclose(original, reversed_order, atol=1e-12)


def test_angular_momentum_inputs_unchanged():
    bodies = [
        make_body(2.0, [1.0, 2.0], [3.0, 4.0]),
    ]

    original_position = bodies[0].position.copy()
    original_velocity = bodies[0].velocity.copy()

    total_angular_momentum(bodies)

    assert_array_equal(bodies[0].position, original_position)
    assert_array_equal(bodies[0].velocity, original_velocity)
