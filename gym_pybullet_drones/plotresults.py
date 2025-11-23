# plotresults.py

import numpy as np
import matplotlib.pyplot as plt
from trajectory.circle import circle
from trajectory.diamond import diamond


def sample_trajectory(traj_fun, tf, dt=0.01):
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


def plot_time_series(t, data, labels, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, lbl in enumerate(labels):
        ax.plot(t, data[:, i], label=lbl)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()


def plot_3d_trajectory(pos, title_prefix):
    from mpl_toolkits.mplot3d import Axes3D

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
    t, pos, vel, acc = sample_trajectory(traj_fun, tf=tf, dt=dt)

    plot_time_series(
        t, pos,
        labels=('x [m]', 'y [m]', 'z [m]'),
        title=f"{name} – Position vs Time",
        ylabel="Position [m]"
    )

    plot_time_series(
        t, vel,
        labels=('vx [m/s]', 'vy [m/s]', 'vz [m/s]'),
        title=f"{name} – Velocity vs Time",
        ylabel="Velocity [m/s]"
    )

    plot_time_series(
        t, acc,
        labels=('ax [m/s²]', 'ay [m/s²]', 'az [m/s²]'),
        title=f"{name} – Acceleration vs Time",
        ylabel="Acceleration [m/s²]"
    )

    plot_3d_trajectory(pos, name)


def main():
    circle_tf = 15.0
    plot_trajectory_set("Circle trajectory", circle, tf=circle_tf, dt=0.01)

    diamond_tf = 8.0
    plot_trajectory_set("Diamond trajectory", diamond, tf=diamond_tf, dt=0.01)

    plt.show()


if __name__ == "__main__":
    main()
