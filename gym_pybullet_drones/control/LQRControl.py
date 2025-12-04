import numpy as np
import math
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


class LQRControl(BaseControl):
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)

        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P]:
            print("[ERROR] LQRControl requires DroneModel.CF2X or CF2P")
            exit()

        # ---- Load precomputed LQR gain ----
        # K has shape (4, 12) for the 12-state model:
        # [x, y, z, phi, theta, psi, xdot, ydot, zdot, p, q, r]
        try:
            self.K = np.load("lqr_gain.npy")
        except FileNotFoundError:
            print("[ERROR] lqr_gain.npy not found. Run your design_lqr.py script first.")
            exit()

        if self.K.shape != (4, 12):
            print(f"[ERROR] lqr_gain.npy has wrong shape {self.K.shape}, expected (4, 12)")
            exit()

        # Same motor model and mixer as DSLPIDControl
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([
                [-0.5, -0.5, -1.0],
                [-0.5,  0.5,  1.0],
                [ 0.5,  0.5, -1.0],
                [ 0.5, -0.5,  1.0]
            ])
        else:  # CF2P
            self.MIXER_MATRIX = np.array([
                [ 0.0, -1.0, -1.0],
                [ 1.0,  0.0,  1.0],
                [ 0.0,  1.0, -1.0],
                [-1.0,  0.0,  1.0]
            ])

        self.reset()

    def reset(self):
        super().reset()
        # no additional LQR state to reset for now

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3),
                       target_acc=np.zeros(3)
                       ):
        """
        LQR control law around hover.

        State ordering for the LQR design:
        x = [x, y, z, phi, theta, psi, xdot, ydot, zdot, p, q, r]^T

        We form the *error* state (current - reference) and then apply:
            u = u_hover - K * x_err
        where u = [total_thrust, tau_x, tau_y, tau_z].
        """

        # Get drone mass from URDF (same as DSLPIDControl)
        mass = self._getURDFParameter('m')

        # --- 1. Build state error vector x_err (12,) ---

        # Current orientation as roll, pitch, yaw
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))  # (phi, theta, psi)

        # Errors: current - desired (matches linearization around zero error)
        pos_error = cur_pos - target_pos          # ex, ey, ez
        rpy_error = cur_rpy - target_rpy          # ephi, etheta, epsi
        vel_error = cur_vel - target_vel          # evx, evy, evz
        ang_vel_error = cur_ang_vel - target_rpy_rates  # ep, eq, er

        # Stack into 12x1 error state
        x_err = np.hstack([
            pos_error,
            rpy_error,
            vel_error,
            ang_vel_error
        ])

        # --- 2. LQR feedback: u = u_hover - K x_err ---

        # Hover thrust in Newtons
        u_hover = np.array([mass * self.GRAVITY, 0.0, 0.0, 0.0])  # [u1, tau_x, tau_y, tau_z]
        u = u_hover - self.K @ x_err  # (4,)

        total_thrust_N = u[0]           # total thrust in Newtons
        target_torques = u[1:4]         # [tau_x, tau_y, tau_z] (units consistent with mixer)

        # Prevent negative thrust (can't pull downward)
        total_thrust_N = max(0.0, total_thrust_N)

        # --- 3. Map thrust (N) & torques -> motor PWM, then to RPM ---

        # Convert desired total thrust [N] to a *baseline* PWM command
        # match exactly the logic from DSLPIDControl:
        # scalar_thrust is the force along body z; here we treat u[0] as that
        scalar_thrust = total_thrust_N

        # self.KF is defined in BaseControl (thrust coefficient)
        thrust_pwm = (math.sqrt(scalar_thrust / (4.0 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        # Add differential contributions from target torques via the mixer
        target_torques = np.clip(target_torques, -3200.0, 3200.0)
        pwm = thrust_pwm + np.dot(self.MIXER_MATRIX, target_torques)

        # Clip PWM to allowed range and convert to RPM
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        # For logging consistency with DSLPIDControl: return position error and yaw error
        des_rpy = target_rpy

        return rpm, pos_error, des_rpy
