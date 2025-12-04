import numpy as np
import pybullet as p
import pybullet_data
from scipy.linalg import solve_continuous_are

def get_drone_parameters(urdf_path):
    p.connect(p.DIRECT)
    drone = p.loadURDF(urdf_path)

    mass = p.getDynamicsInfo(drone, -1)[0]
    inertia = p.getDynamicsInfo(drone, -1)[2]  # tuple (Ixx, Iyy, Izz)

    p.disconnect()

    Ixx, Iyy, Izz = inertia
    return mass, Ixx, Iyy, Izz


def design_lqr():

    # ---- Load real parameters from URDF ----
    urdf_path = "gym_pybullet_drones/assets/cf2x.urdf"   # or cf2p.urdf
    m, Ixx, Iyy, Izz = get_drone_parameters(urdf_path)

    g = 9.81

    print("Loaded from URDF:")
    print("Mass =", m)
    print("Ixx =", Ixx)
    print("Iyy =", Iyy)
    print("Izz =", Izz)

    # ---- 1. Build A and B matrices using these values ----
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))

    A[0, 6] = 1.0
    A[1, 7] = 1.0
    A[2, 8] = 1.0

    A[3, 9] = 1.0
    A[4, 10] = 1.0
    A[5, 11] = 1.0

    A[6, 4] =  g
    A[7, 3] = -g

    B[8, 0]  = 1.0 / m
    B[9, 1]  = 1.0 / Ixx
    B[10, 2] = 1.0 / Iyy
    B[11, 3] = 1.0 / Izz

    print("A =", A)
    print("B =", B)

     # ---- 2. Choose Q, R ----
    # Very strong position + velocity weights, modest attitude, low rates
    Q = np.diag([
        2000.0, 2000.0, 3000.0,   # x, y, z (huge -> track the path hard)
        2.0,    2.0,    1.0,      # phi, theta, psi (cheap -> allowed to tilt)
        120.0,  120.0,  80.0,     # vx, vy, vz (react strongly to motion)
        5.0,    5.0,    3.0       # p, q, r (keep rates reasonably small)
    ])

    # Inputs: modest cost on thrust, moderate cost on torques
    R = np.diag([
        1e-2,   # thrust
        0.3,    # tau_x
        0.3,    # tau_y
        0.3     # tau_z
    ])

    # ---- 3. Solve LQR ----
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    print("K =\n", K)
    np.save("lqr_gain.npy", K)
    print("Saved to lqr_gain.npy")


if __name__ == "__main__":
    design_lqr()
