import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from env.reward import RewardConfig, compute_reward, LIFT_THRESHOLD
from env.utils import get_object_pose, get_palm_position

SCENE_XML = os.path.join(os.path.dirname(__file__), "../assets/leap_grasp_scene.xml")

# sim runs at 200Hz, policy queries at 20Hz — so we step 10x per action
SIM_FREQ   = 200
POLICY_FREQ = 20
SIM_STEPS  = SIM_FREQ // POLICY_FREQ   # 10
MAX_STEPS  = 500
TABLE_Z    = 0.42


class LeapGraspEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, reward_mode: str = "baseline", render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None

        self.reward_cfg      = RewardConfig()
        self.reward_cfg.mode = reward_mode

        # pull joint limits once so we can reuse them in _map_action
        self._joint_lo = self.model.actuator_ctrlrange[:, 0]
        self._joint_hi = self.model.actuator_ctrlrange[:, 1]

        # 42-dim obs: 16 qpos + 16 qvel + 3 obj_pos + 4 obj_quat + 3 palm→obj
        obs_dim = 16 + 16 + 3 + 4 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )

        self._step_count  = 0
        self._prev_obj_z  = 0.455

    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[:16].copy()
        qvel = self.data.qvel[:16].copy()

        # reading from xpos/xquat is cleaner than poking at the freejoint qpos slice
        obj_pos, obj_quat = get_object_pose(self.model, self.data)
        palm_pos = get_palm_position(self.model, self.data)
        palm_to_obj = obj_pos - palm_pos

        return np.concatenate([qpos, qvel, obj_pos, obj_quat, palm_to_obj]).astype(np.float32)

    def _map_action(self, action: np.ndarray) -> np.ndarray:
        """Rescale [-1, 1] → actual joint position targets."""
        return 0.5 * (action + 1.0) * (self._joint_hi - self._joint_lo) + self._joint_lo

    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # small XY perturbation so the policy doesn't overfit to one exact object position
        dx, dy = rng.uniform(-0.03, 0.03, size=2)
        obj_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        qpos_adr = self.model.jnt_qposadr[obj_jnt_id]

        dx, dy = rng.uniform(-0.03, 0.03, size=2)
        self.data.qpos[qpos_adr + 0] = dx       # x
        self.data.qpos[qpos_adr + 1] = dy       # y
        self.data.qpos[qpos_adr + 2] = 0.455    # z — settled object center (verified)
        self.data.qpos[qpos_adr + 3] = 1.0      # qw (identity quaternion)
        self.data.qpos[qpos_adr + 4] = 0.0
        self.data.qpos[qpos_adr + 5] = 0.0
        self.data.qpos[qpos_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._prev_obj_z = self.data.qpos[qpos_adr + 2]

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        ctrl = self._map_action(np.clip(action, -1.0, 1.0))
        self.data.ctrl[:] = ctrl

        for _ in range(SIM_STEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = compute_reward(
            self.model, self.data, action, self.reward_cfg, self._prev_obj_z
        )

        obj_pos, _ = get_object_pose(self.model, self.data)
        self._prev_obj_z = obj_pos[2]
        self._step_count += 1

        dropped = obj_pos[2] < TABLE_Z - 0.05
        success  = info["lift_height"] > LIFT_THRESHOLD
        timeout  = self._step_count >= MAX_STEPS

        terminated = dropped or success
        truncated  = timeout

        info["success"] = success
        info["dropped"] = dropped

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer:
            self._renderer.close()
