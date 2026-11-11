import numpy as np
from nbody.models import Body
from nbody.forces import gravitational_force

def accelerations(
    bodies: list[Body],
    G: float = 1.0,
) -> np.ndarray:
    
#    for i in range(len(bodies)):
#        for j in range(j+1, n):
#            if np.arrays_equal(bodies[j].position, bodies[i].position):
#                raise ValueError(f'position of body {i} and of body {j} coincide')

    n = len(bodies)
    accel = np.zeros((n, 2))

    if n < 2:
        return accel


    for i in range(n):
        for j in range(i+1, n):
            F = gravitational_force(bodies[i].mass, bodies[i].position, bodies[j].mass, bodies[j].position, G=G)
            accel[i, :] += F / bodies[i].mass
            accel[j, :] -= F / bodies[j].mass
    
    return accel