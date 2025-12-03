import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from gym_pybullet_drones.trajectory.circle import circle as proj_circle
from gym_pybullet_drones.trajectory.diamond import diamond as proj_diamond

g = 9.81
m = 0.03
Ixx = 1.4e-5
Iyy = 1.4e-5
Izz = 2.2e-5

def build_AB(m, Ixx, Iyy, Izz, g=9.81):
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))

    IDX_X, IDX_Y, IDX_Z = 0, 1, 2
    IDX_PHI, IDX_THETA, IDX_PSI = 3, 4, 5
    IDX_VX, IDX_VY, IDX_VZ = 6, 7, 8
    IDX_P, IDX_Q, IDX_R = 9, 10, 11

    A[IDX_X, IDX_VX] = 1.0
    A[IDX_Y, IDX_VY] = 1.0
    A[IDX_Z, IDX_VZ] = 1.0
    A[IDX_PHI, IDX_P] = 1.0
    A[IDX_THETA, IDX_Q] = 1.0
    A[IDX_PSI, IDX_R] = 1.0

    A[IDX_VX, IDX_THETA] = g
    A[IDX_VY, IDX_PHI] = -g

    B[IDX_VZ, 0] = 1.0 / m

    B[IDX_P, 1] = 1.0 / Ixx
    B[IDX_Q, 2] = 1.0 / Iyy
    B[IDX_R, 3] = 1.0 / Izz

    return A, B

def design_lqr(A, B):
    Q = np.diag([2.0,  20.0,  4.0,
                 5.0,  5.0,  1.0,
                 1.0,  1.0,  2.0,
                 2.0,  2.0,  1.0])

    R = np.diag([2.0, 2.0, 2.0, 2.0])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K, Q, R

def circle_traj(T, dt):
    t = np.arange(0.0, T, dt)
    rd = []
    vd = []

    for ti in t:
        state = proj_circle(ti, tf=T)
        rd.append(state["pos"])
        vd.append(state["vel"])

    return t, np.vstack(rd), np.vstack(vd)


def diamond_traj(T, dt):
    t = np.arange(0.0, T, dt)
    rd = []
    vd = []

    for ti in t:
        state = proj_diamond(ti, tfinal=T)
        rd.append(state["pos"])
        vd.append(state["vel"])

    return t, np.vstack(rd), np.vstack(vd)
    

def simulate_lqr(A, B, K, traj_func, T=8.0, dt=0.004):
    t, rd, vd = traj_func(T, dt)
    N = len(t)

    x = np.zeros(12)
    X_hist = np.zeros((N, 12))
    U_hist = np.zeros((N, 4))

    def f(x, u_tilde):
        return A @ x + B @ u_tilde

    for k in range(N):
        pos_d = rd[k]
        vel_d = vd[k]
        psi_d = 0.0

        xd = np.array([pos_d[0], pos_d[1], pos_d[2],
                       0.0, 0.0, psi_d,
                       vel_d[0], vel_d[1], vel_d[2],
                       0.0, 0.0, 0.0])

        e = x - xd

        e = np.clip(e, -1.0, 1.0)

        u_tilde = -K @ e
        u_tilde = np.clip(u_tilde, -1.0, 1.0)

        k1 = f(x, u_tilde)
        k2 = f(x + 0.5 * dt * k1, u_tilde)
        k3 = f(x + 0.5 * dt * k2, u_tilde)
        k4 = f(x + dt * k3, u_tilde)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        X_hist[k, :] = x
        U_hist[k, :] = u_tilde

    pos_sim = X_hist[:, 0:3]
    err = pos_sim - rd
    rmse = np.sqrt(np.mean(err**2, axis=0))

    return t, rd, pos_sim, rmse

def main():
    A, B = build_AB(m, Ixx, Iyy, Izz, g)
    K, Q, R = design_lqr(A, B)

    print("LQR gain K shape:", K.shape)

    t_c, rd_c, pos_c, rmse_c = simulate_lqr(A, B, K, circle_traj, T=8.0, dt=0.002)
    print("Circle RMSE [x, y, z]:", rmse_c)

    t_d, rd_d, pos_d, rmse_d = simulate_lqr(A, B, K, diamond_traj, T=8.0, dt=0.002)
    print("Diamond RMSE [x, y, z]:", rmse_d)

    plt.figure()
    plt.suptitle("LQR tracking - Circle trajectory")

    labels = ["x", "y", "z"]
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(t_c, rd_c[:, i], linestyle="--", label=f"{labels[i]} reference")
        plt.plot(t_c, pos_c[:, i], label=f"{labels[i]} actual")
        plt.ylabel(labels[i])
        if i == 0:
            plt.legend()
    plt.xlabel("Time [s]")

    plt.figure()
    plt.suptitle("LQR tracking - Diamond trajectory")

    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(t_d, rd_d[:, i], linestyle="--", label=f"{labels[i]} reference")
        plt.plot(t_d, pos_d[:, i], label=f"{labels[i]} actual")
        plt.ylabel(labels[i])
        if i == 0:
            plt.legend()
    plt.xlabel("Time [s]")

    plt.show()

if __name__ == "__main__":
    main()







