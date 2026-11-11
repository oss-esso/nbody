
import numpy as np
import matplotlib.pyplot as plt

from nbody.initial_conditions import circular_binary
from nbody.simulation import simulate

m1 = 1
m2 = 1
d = 2
G = 1

T = 4 * np.pi
step_counts = [400, 1000, 4000, 8000]
for steps in step_counts:
    dt = T / steps

    bodies = circular_binary(m1, m2, separation=d, G=G)
    positions, velocities = simulate(bodies, dt, steps, G)

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(len(bodies)):
        x = positions[:, i, 0]
        y = positions[:, i, 1]

        ax.plot(x, y, label=f"Body {i}")
        ax.scatter(x[0], y[0], marker="o", s=60)
        ax.scatter(x[-1], y[-1], marker="x", s=80)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Two-body orbit — Explicit Euler")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()

    # Separation diagnostics
    initial_separation = np.linalg.norm(
        positions[0, 1] - positions[0, 0]
    )

    final_separation = np.linalg.norm(
        positions[-1, 1] - positions[-1, 0]
    )

    relative_change = (
        final_separation - initial_separation
    ) / initial_separation
    print(f"Step count: {steps:.6f}")
    print(f"Initial separation: {initial_separation:.6f}")
    print(f"Final separation:   {final_separation:.6f}")
    print(f"Relative change:    {relative_change:.4%}")
