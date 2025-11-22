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

    desired_state = {
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state
