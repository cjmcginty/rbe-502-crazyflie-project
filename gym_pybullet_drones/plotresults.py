# plotresults.py

import numpy as np
import matplotlib.pyplot as plt
from trajectory.circle import circle
from trajectory.diamond import diamond

def sample_trajectory(traj_fun, tf, dt=0.01):
    """
    Sample a trajectory function over [0, tf].

    traj_fun: function(t, tf) -> desired_state dict
    tf      : total duration
    dt      : timestep
    """
    t = np.arange(0.0, tf + dt, dt)
    N = len(t)

    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))
    acc = np.zeros((N, 3))

    for i, ti in enumerate(t):
        desired = traj_fun(ti, tf)
        pos[i, :] = desired['pos']
        vel[i, :] = desired['vel']
        acc[i, :] = desired['acc']

    return t, pos, vel, acc

def plot_time_series(t, data, labels, title_prefix):
    """
    Plot 3 components vs time in stacked subplots.

    data: (N,3) array (pos, vel, or acc)
    labels: ('x', 'y', 'z') etc.
    """
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

    for i, ax in enumerate(axs):
        ax.plot(t, data[:, i])
        ax.set_ylabel(f"{labels[i]}")

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(title_prefix)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_3d_trajectory(pos, title_prefix):
    """
    3D plot of the trajectory.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], marker='o', label='start')
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], marker='x', label='end')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(f"{title_prefix} – 3D trajectory")
    ax.legend()
    fig.tight_layout()

def plot_trajectory_set(name, traj_fun, tf, dt=0.01):
    """
    Convenience fn to sample and plot everything for one trajectory.
    """
    t, pos, vel, acc = sample_trajectory(traj_fun, tf=tf, dt=dt)

    # Position
    plot_time_series(t, pos, ('x [m]', 'y [m]', 'z [m]'),
                     f"{name} – Position vs Time")

    # Velocity
    plot_time_series(t, vel, ('vx [m/s]', 'vy [m/s]', 'vz [m/s]'),
                     f"{name} – Velocity vs Time")

    # Acceleration
    plot_time_series(t, acc, ('ax [m/s²]', 'ay [m/s²]', 'az [m/s²]'),
                     f"{name} – Acceleration vs Time")

    # 3D trajectory
    plot_3d_trajectory(pos, name)

def main():
    # Circular trajectory: phases 0–5, 5–10, 10–15 s
    circle_tf = 15.0
    #plot_trajectory_set("Circle trajectory", circle, tf=circle_tf, dt=0.01)

    # Diamond trajectory: total duration tf = 8 s
    diamond_tf = 8.0
    plot_trajectory_set("Diamond trajectory", diamond, tf=diamond_tf, dt=0.01)

    # Show all figures
    plt.show()

if __name__ == "__main__":
    main()
