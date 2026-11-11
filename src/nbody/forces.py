import numpy as np


def gravitational_force(
        m1: float,
        r1: np.ndarray,
        m2: float,
        r2: np.ndarray,
        G: float = 1.0
) -> np.ndarray:

    if not np.isfinite(m1) or m1 <= 0:
        raise ValueError("m1 must be finite and positive")
    if not np.isfinite(m2) or m2 <= 0:
        raise ValueError("m2 must be finite and positive")
    if not np.isfinite(G) or G <= 0:
            raise ValueError("G must be finite and positive")
    if r1.shape != (2,) or not np.all(np.isfinite(r1)):
        raise ValueError("positions must be finite arrays")
    if r2.shape != (2,) or not np.all(np.isfinite(r2)):
        raise ValueError("positions must be finite arrays")
    if np.array_equal(r1, r2):
         raise ValueError("objects must sit at different places")

    num = G * m1 * m2 * (r2-r1)
    den = np.linalg.norm(r2-r1) ** 3

    F = num / den


    return F