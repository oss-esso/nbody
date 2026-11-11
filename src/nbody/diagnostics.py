import numpy as np
from nbody.models import Body

def total_energy(
        bodies: list[Body],
        G: float = 1.0,
) -> float:

    if G <= 0 or not np.isfinite(G):
        raise ValueError('G must be positive and finite')


    T = np.sum([0.5 * body.mass * body.velocity**2 for body in bodies])
    def potential(bodies, i):
        pot = 0
        body_i = bodies[i]
        for j in range(i+1, len(bodies)):
            body_j = bodies[j]
            if np.array_equal(body_j.position , body_i.position):
                raise ValueError(f"objects {i} and {j} must sit at different places")
            
            pot += G * body_i.mass * body_j.mass / np.linalg.norm(body_j.position - body_i.position)
        return pot

    U = np.sum([potential(bodies, i) for i in range(len(bodies))])

    return T - U

def total_angular_momentum(
        bodies: list[Body],
) -> float:

    Lz = 0.0
    N = len(bodies)

    if N == 0:
        return Lz

    for body in bodies:
        L_i = body.mass * (body.position[0]*body.velocity[1] - body.position[1]*body.velocity[0])
        Lz += L_i

    return Lz