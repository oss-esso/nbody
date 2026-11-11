import numpy as np
from nbody.integrators import euler_step, velocity_verlet_step
from nbody.models import Body

def simulate(
    bodies: list[Body],
    dt: float,
    steps: int,
    G: float = 1.0,
    method: str = 'euler'
) -> tuple[np.ndarray, np.ndarray]:

    if steps < 0 or not(np.isfinite(steps)) or not type(steps) is int:
        raise ValueError('steps must be non-negative and finite')

    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")

    if method not in ['euler', 'verlet']:
        raise ValueError('method not supported')

    #method selection

    if method == 'euler':
        step = euler_step
    elif method == 'verlet':
        step = velocity_verlet_step
    
    N = len(bodies)
    positions = np.zeros((steps+1, N, 2))
    velocities = np.zeros((steps+1, N, 2))

    if len(bodies) == 0:
        return positions, velocities

    positions[0, :, :] = np.array([body.position for body in bodies])
    velocities[0, :, :] = np.array([body.velocity for body in bodies])

    if steps == 0:
        return (positions, velocities)

    old_bodies = bodies


    for i in range(steps):

        new_bodies = step(bodies=old_bodies, dt=dt, G=G)
        positions[i+1, :, :] = np.array([body.position for body in new_bodies])
        velocities[i+1, :, :] = np.array([body.velocity for body in new_bodies])

        old_bodies = new_bodies

    return positions, velocities


        