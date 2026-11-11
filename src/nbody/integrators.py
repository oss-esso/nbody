from nbody.models import Body
from nbody.dynamics import accelerations
import numpy as np

def euler_step(
        bodies: list[Body],
        dt: float,
        G: float = 1.0
) -> list[Body]:
    
    if dt <= 0 or not np.isfinite(dt):
            raise ValueError('timestep must be finite and positive')
    
    if len(bodies) == 0:
        return []
    

    velocities = np.array([body.velocity for body in bodies])
    acceleration = accelerations(bodies, G)

    #update
    new_positions = np.array([body.position for body in bodies]) + dt * velocities
    new_velocities = np.array([body.velocity for body in bodies]) + dt * acceleration

    new_bodies = [Body(mass=bodies[i].mass, position=new_positions[i], velocity=new_velocities[i]) for i in range(len(bodies))]

    return new_bodies


def velocity_verlet_step(
        bodies: list[Body],
        dt: float,
        G: float = 1.0
) -> list[Body]:
    
    if dt <= 0 or not np.isfinite(dt):
            raise ValueError('timestep must be finite and positive')

    if G <= 0 or not np.isfinite(G):
                raise ValueError('G must be finite and positive')
    
    if len(bodies) == 0:
        return []
    

    velocities = np.array([body.velocity for body in bodies])
    acceleration = accelerations(bodies, G)

    #update
    new_positions = np.array([body.position for body in bodies]) + dt * velocities + 0.5 * dt**2 * acceleration
    new_bodies = [Body(mass=bodies[i].mass, position=new_positions[i], velocity=velocities[i]) for i in range(len(bodies))]
    new_velocities = np.array([body.velocity for body in bodies]) + 0.5 * dt * (acceleration + accelerations(new_bodies, G))

    new_bodies = [Body(mass=bodies[i].mass, position=new_positions[i], velocity=new_velocities[i]) for i in range(len(bodies))]

    return new_bodies