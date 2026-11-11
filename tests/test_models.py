
import numpy as np
import pytest

from nbody.models import Body


def test_body_init():
    body = Body(
        mass=2.0,
        position=np.array([1.0, 2.0]),
        velocity=np.array([0.0, 0.0]),
    )

    np.testing.assert_array_equal(body.position, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(body.velocity, np.array([0.0, 0.0]))
    
def test_zero_mass_body():
    with pytest.raises(ValueError):
        Body(
            mass=0.0,
            position=np.array([1.0, 2.0]),
            velocity=np.array([0.0, 0.0]),
        )
def test_invalid_position_body():
    with pytest.raises(ValueError):
        Body(
            mass=2.0,
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.0, 0.0]),
        )
def test_nonfinite_velocity():
    with pytest.raises(ValueError):
        Body(
            mass=2.0,
            position=np.array([1.0, 2.0]),
            velocity=np.array([0.0, np.nan]),
        )




