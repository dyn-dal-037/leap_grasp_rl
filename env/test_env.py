import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.leap_grasp_env import LeapGraspEnv
import numpy as np

env = LeapGraspEnv(reward_mode="baseline")
obs, _ = env.reset(seed=42)

import mujoco
obj_id   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "object")
table_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "table")

print("Object z:", env.data.xpos[obj_id][2])
print("Table geom z:", env.data.geom_xpos[
    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface")
][2])

print(f"Obs shape: {obs.shape}")      # should be (42,)
print(f"Action space: {env.action_space}")

# quick sanity check — 50 random steps, just make sure nothing crashes
total_reward = 0
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Episode ended at step {i}: {info}")
        break

print(f"Total reward (random policy, {i+1} steps): {total_reward:.4f}")
print(f"Last info: {info}")
env.close()
