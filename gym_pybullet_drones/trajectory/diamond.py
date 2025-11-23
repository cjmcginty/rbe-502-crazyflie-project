import numpy as np

def diamond(t, tfinal=8):
    """
    Generate the desired state of a drone following a diamond-shaped trajectory.

    The function computes the drone’s position, velocity, and acceleration at
    any given time 't' while following a diamond-shaped trajectory.

    Parameters:
        t (float): Current time (in seconds).
        tfinal (float): Total trajectory duration.

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
    t = float(np.clip(t, 0.0, tfinal))

    def min_jerk_scalar(t_local, T_seg):
        if T_seg <= 0:
            return 0, 0, 0

        tau = np.clip(t_local / T_seg, 0, 1)
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        s_dot  = (30*tau**2 - 60*tau**3 + 30*tau**4) / T_seg
        s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / (T_seg**2)
        return s, s_dot, s_ddot

    def straight_segment_2d(t_local, T_seg, p_start, p_end):
        d = p_end - p_start
        s, s_dot, s_ddot = min_jerk_scalar(t_local, T_seg)

        pos = p_start + s * d
        vel = s_dot * d
        acc = s_ddot * d
        return pos, vel, acc

    ux = 1
    s_x, s_x_dot, s_x_ddot = min_jerk_scalar(t, tfinal)
    x  = s_x * ux
    vx = s_x_dot * ux
    ax = s_x_ddot * ux

    u = 1.0 / np.sqrt(2.0)
    p0 = np.array([0, 0])
    p1 = np.array([ u,  u])
    p2 = np.array([0, 2*u])
    p3 = np.array([-u,  u])
    p4 = np.array([0, 0])

    waypoints = [p0, p1, p2, p3, p4]

    T_seg = tfinal / 4

    seg = int(t // T_seg)
    if seg >= 4:
        seg = 3

    t_local = t - seg * T_seg

    p_start = waypoints[seg]
    p_end = waypoints[seg + 1]

    (yz_pos, yz_vel, yz_acc) = straight_segment_2d(t_local, T_seg, p_start, p_end)

    y, z = yz_pos
    vy, vz = yz_vel
    ay, az = yz_acc

    pos = np.array([x,  y,  z])
    vel = np.array([vx, vy, vz])
    acc = np.array([ax, ay, az])

    desired_state = {
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state
