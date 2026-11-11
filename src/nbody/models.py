from dataclasses import dataclass

import numpy as np


@dataclass
class Body:
    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.mass) or self.mass <= 0:
                raise ValueError("mass must be finite and positive")
        if self.position.shape != (2,) or not np.all(np.isfinite(self.position)):
            raise ValueError("position must be finite array")
        if self.velocity.shape != (2,) or not np.all(np.isfinite(self.velocity)):
            raise ValueError("velocity must be finite array")

