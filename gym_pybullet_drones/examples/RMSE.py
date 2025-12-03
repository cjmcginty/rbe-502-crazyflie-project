"""
Script to numerically evaluate DSLPIDControl tracking performance
on circle / diamond trajectories.

This is designed to mirror the dynamics of `trajectory_tracking.py`
as closely as possible:

- Same CtrlAviary constructor arguments
- Same initial positions and orientations (INIT_XYZS, INIT_RPYS)
- Same call pattern to DSLPIDControl.computeControlFromState
- Same circle/diamond trajectory functions

We only add:
- Error logging
- RMS / max tracking error printout at the end for each trajectory
"""

import time
import math
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.trajectory import circle, diamond


# ----------------- CONFIG (mirrors trajectory_tracking.py) -----------------

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = False               # keep False for fast numerical runs
DEFAULT_RECORD_VISION = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 7

# same geometry constants
H = 0.1
H_STEP = 0.05
R = 0.3

# --------------------------------------------------------------------------


def make_inits(num_drones):
    """
    Reproduce INIT_XYZS and INIT_RPYS from trajectory_tracking.py
    for a given num_drones (assumed 1 in your use case).
    """
    init_xyzs = np.array([
        [
            R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
            R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R,
            H + i * H_STEP,
        ]
        for i in range(num_drones)
    ])
    init_rpys = np.array([
        [0, 0, i * (np.pi / 2) / num_drones]
        for i in range(num_drones)
    ])
    return init_xyzs, init_rpys


def sample_traj(t, name):
    """Return pos, vel, acc for the given trajectory name."""
    if name == "circle":
        st = circle.circle(t)
    elif name == "diamond":
        st = diamond.diamond(t)
    else:
        raise ValueError("Unknown trajectory name")
    return st["pos"], st["vel"], st["acc"]


def run_once(
    traj_name,
    drone_model=DEFAULT_DRONES,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
):
    """
    Run ONE trajectory (circle or diamond) using the same pattern
    as trajectory_tracking.py, but also compute tracking metrics.
    """

    # --- Initial states (same as trajectory_tracking.py) ---
    init_xyzs, init_rpys = make_inits(num_drones)

    # --- Create the environment (same args as trajectory_tracking.py) ---
    env = CtrlAviary(
        drone_model=drone_model,
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    # --- Controllers (same pattern, but we’ll only look at drone 0) ---
    controllers = [DSLPIDControl(drone_model=drone_model) for _ in range(num_drones)]

    # --- Set up loop variables ---
    action = np.zeros((num_drones, 4))
    num_steps = int(duration_sec * env.CTRL_FREQ)

    pos_err_log = []       # overall norm error for drone 0
    pos_err_abs_log = []   # per-axis abs error for drone 0

    start_wall = time.time()
    elapsed = 0.0

    for step in range(num_steps):
        # Step sim
        obs, reward, terminated, truncated, info = env.step(action)

        # Desired state from chosen trajectory
        pos_des, vel_des, acc_des = sample_traj(elapsed, traj_name)

        # Compute control exactly as in trajectory_tracking.py
        for j in range(num_drones):
            rpm, pos_e, des_rpy = controllers[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=pos_des,
                target_vel=vel_des,
                target_rpy=init_rpys[j, :],
                target_acc=acc_des,
            )
            action[j, :] = rpm

        # Log error for drone 0 only (that’s the one you care about)
        pos_err_log.append(np.linalg.norm(pos_e))      # pos_e from last j (j=0)
        pos_err_abs_log.append(np.abs(pos_e))

        # Optional GUI sync
        if gui:
            env.render()
            sync(step, start_wall, env.CTRL_TIMESTEP)

        elapsed += 1.0 / env.CTRL_FREQ

    # All done with this sim
    env.close()

    # --- Compute metrics from logs ---
    pos_err_arr = np.array(pos_err_log)            # (T,)
    pos_err_abs_arr = np.vstack(pos_err_abs_log)   # (T,3)

    rms_err = math.sqrt(np.mean(pos_err_arr ** 2))
    max_err = float(np.max(pos_err_arr))
    rms_xyz = np.sqrt(np.mean(pos_err_abs_arr ** 2, axis=0))
    max_xyz = np.max(pos_err_abs_arr, axis=0)

    return {
        "traj": traj_name,
        "rms": rms_err,
        "max": max_err,
        "rms_xyz": rms_xyz,
        "max_xyz": max_xyz,
    }


def main():
    # Run circle, then diamond, each in its own fresh sim
    circle_metrics = run_once("circle")
    diamond_metrics = run_once("diamond")

    print("\n================ Final Tracking Summary ================\n")

    def print_metrics(m):
        print(f"Trajectory: {m['traj']}")
        print(f"  RMS error: {m['rms']:.3f} m")
        print(f"  Max error: {m['max']:.3f} m")
        print(
            "  Per-axis RMS (x,y,z): "
            f"{m['rms_xyz'][0]:.3f}, "
            f"{m['rms_xyz'][1]:.3f}, "
            f"{m['rms_xyz'][2]:.3f}"
        )
        print(
            "  Per-axis Max (x,y,z): "
            f"{m['max_xyz'][0]:.3f}, "
            f"{m['max_xyz'][1]:.3f}, "
            f"{m['max_xyz'][2]:.3f}"
        )
        print()

    print_metrics(circle_metrics)
    print_metrics(diamond_metrics)

    print(
        "Use these numbers to compare gain sets. "
        "They should reflect what happens in trajectory_tracking.py "
        "with the same controller.\n"
    )


if __name__ == "__main__":
    main()
