import numpy as np
import mujoco

# body/geom names as they appear in right_hand.xml — had to grep the XML to confirm these
FINGERTIP_BODIES = ["if_ds", "mf_ds", "rf_ds", "th_ds"]
FINGERTIP_GEOMS  = ["if_tip", "mf_tip", "rf_tip", "th_tip"]
PALM_BODY        = "palm"
OBJECT_BODY      = "object"
OBJECT_GEOM      = "object_geom"


def get_fingertip_positions(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Returns (4, 3) — world-frame XYZ for each fingertip body."""
    tips = []
    for name in FINGERTIP_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        tips.append(data.xpos[bid].copy())
    return np.array(tips)


def get_palm_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Palm body centroid in world frame."""
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY)
    return data.xpos[palm_id].copy()


def get_object_pose(model: mujoco.MjModel, data: mujoco.MjData):
    """Returns (pos, quat) of the graspable object — both copied to avoid aliasing."""
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    return data.xpos[obj_id].copy(), data.xquat[obj_id].copy()


def count_finger_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """
    Returns how many distinct fingers (0–4) are currently touching the object.

    Walks the active contact list and checks if one geom is the object and the
    other is a known fingertip geom. Using a set means double-contacts on the
    same finger only count once.
    """
    obj_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    tip_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name): name
        for name in FINGERTIP_GEOMS
    }
    fingers_in_contact = set()

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if obj_geom_id in (g1, g2):
            other = g2 if g1 == obj_geom_id else g1
            if other in tip_geom_ids:
                fingers_in_contact.add(tip_geom_ids[other])

    return len(fingers_in_contact)
