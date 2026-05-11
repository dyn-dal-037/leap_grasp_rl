import numpy as np
import mujoco
from env.utils import (
    get_palm_position, get_object_pose,
    count_finger_contacts, get_fingertip_positions
)

TABLE_Z        = 0.455   # object center z when sitting on the table (measured, not guessed)
LIFT_THRESHOLD = 0.15    # how high the object needs to go before we call it a success


class RewardConfig:
    w_approach: float = 0.30
    w_contact:  float = 0.30
    w_lift:     float = 0.50
    w_smooth:   float = 0.01
    mode: str = "baseline"   # options: "baseline" | "fingertip" | "curriculum"


def compute_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    action: np.ndarray,
    cfg: RewardConfig,
    prev_obj_z: float,
) -> tuple[float, dict]:
    """
    Returns scalar reward + a dict of components for TensorBoard.

    Three modes ship here:
      baseline   — approach reward from palm centroid
      fingertip  — approach from mean fingertip distance (usually better signal)
      curriculum — lift is zero until 3+ fingers are in contact (reduces premature lifting)
    """
    obj_pos, _ = get_object_pose(model, data)
    n_contacts  = count_finger_contacts(model, data)

    # --- approach signal ---
    if cfg.mode == "fingertip":
        tips = get_fingertip_positions(model, data)
        approach = -float(np.mean(np.linalg.norm(tips - obj_pos, axis=1)))
    else:
        palm_pos = get_palm_position(model, data)
        approach = -float(np.linalg.norm(palm_pos - obj_pos))

    # --- contact signal (normalized 0→1) ---
    contact = float(n_contacts) / 4.0

    # --- lift signal ---
    lift_height = max(0.0, obj_pos[2] - TABLE_Z)
    if cfg.mode == "curriculum":
        # gate lift behind having 3+ fingers touching — otherwise the hand
        # just nudges the object upward without ever wrapping around it
        lift = lift_height if n_contacts >= 3 else 0.0
    else:
        lift = lift_height

    # --- action smoothness penalty (keeps joints from thrashing) ---
    smooth = -float(np.sum(action ** 2))

    r = (cfg.w_approach * approach
         + cfg.w_contact * contact
         + cfg.w_lift    * lift
         + cfg.w_smooth  * smooth)

    info = {
        "r_approach":  approach,
        "r_contact":   contact,
        "r_lift":      lift,
        "r_smooth":    smooth,
        "n_contacts":  n_contacts,
        "lift_height": lift_height,
    }
    return r, info
