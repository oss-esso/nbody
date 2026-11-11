import numpy as np
from nbody.models import Body
import random

def circular_binary(
    m1: float,
    m2: float,
    separation: float,
    G: float = 1.0,
) -> list[Body]:

    if not np.isfinite(m1) or m1 <= 0:
        raise ValueError("m1 must be finite and positive")
    if not np.isfinite(m2) or m2 <= 0:
        raise ValueError("m2 must be finite and positive")
    if not np.isfinite(G) or G <= 0:
            raise ValueError("G must be finite and positive")
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("separation must be finite and positive")

    x2 = m1 / (m2 + m1) * separation
    x1 = x2 - separation

    omega = (G * (m1 + m2) / separation ** 3) ** 0.5

    v1 = np.array([0.0, omega*x1])
    v2 = np.array([0.0, omega*x2])

    return [Body(m1, np.array([x1, 0.0]), v1),Body(m2, np.array([x2, 0.0]), v2),]

