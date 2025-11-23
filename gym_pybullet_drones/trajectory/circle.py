import numpy as np

def circle(t, tf=8):
    """
    Generate the desired state of a drone following a circular trajectory.

    The function computes the drone’s position, velocity, and acceleration
    at a given time 't' while following a circular trajectory.

    Parameters:
        t (float): Current time (in seconds).
        tf (float): Total trajectory duration.

    Returns:
        desired_state (dict): 
            - 'pos'   (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'   (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'   (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk'  (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'   (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """

    """
    Write your code here.
    """
    t = float(np.clip(t, 0, tf))

    T1 = tf / 3
    T2 = 2 * tf / 3
    R  = 1
    z_d = 1

    r0 = np.array([0, 0, 0.5])
    r1 = np.array([R, 0, z_d])

    def min_jerk_scalar(t_local, T_seg):
        if T_seg <= 0:
            return 0, 0, 0

        tau = np.clip(t_local / T_seg, 0, 1)
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        s_dot  = (30*tau**2 - 60*tau**3 + 30*tau**4) / T_seg
        s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / (T_seg**2)
        return s, s_dot, s_ddot

    def straight_segment(t_local, T_seg, p_start, p_end):
        d = p_end - p_start
        s, s_dot, s_ddot = min_jerk_scalar(t_local, T_seg)

        pos = p_start + s * d
        vel = s_dot * d
        acc = s_ddot * d
        return pos, vel, acc

    if t <= T1:
        pos, vel, acc = straight_segment(t, T1, r0, r1)

    elif t <= T2:
        Tc = T2 - T1
        t_local = t - T1
        s, s_dot, s_ddot = min_jerk_scalar(t_local, Tc)
        theta   = 2 * np.pi * s
        theta_dot  = 2 * np.pi * s_dot
        theta_ddot = 2 * np.pi * s_ddot

        x = R * np.cos(theta)
        y = R * np.sin(theta)
        z = z_d

        vx = -R * np.sin(theta) * theta_dot
        vy = R * np.cos(theta) * theta_dot
        vz = 0

        ax = -R * np.cos(theta) * (theta_dot**2) - R * np.sin(theta) * theta_ddot
        ay = -R * np.sin(theta) * (theta_dot**2) + R * np.cos(theta) * theta_ddot
        az = 0

        pos = np.array([x,  y,  z])
        vel = np.array([vx, vy, vz])
        acc = np.array([ax, ay, az])

    else:
        T3 = tf - T2
        t_local = t - T2
        pos, vel, acc = straight_segment(t_local, T3, r1, r0)

    desired_state = {
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state
