"""
eval/record_video.py
--------------------
Headless MuJoCo rollout → grasp_demo.gif (+ .mp4 as a bonus).

Tested on Colab T4 and over SSH. Uses mujoco.Renderer from the MuJoCo 3.x API,
so no display needed.

Usage:
    python eval/record_video.py \
        --model  leap_ppo_best.zip \
        --output assets/grasp_demo.gif \
        [--fps 20] [--seed 0] [--width 640] [--height 480]

Deps:
    pip install imageio[ffmpeg] imageio-ffmpeg
"""

import argparse
import os
import sys
import numpy as np
import mujoco
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.leap_grasp_env import LeapGraspEnv
from stable_baselines3 import PPO


def record_episode(
    model_path: str,
    output_path: str,
    fps: int,
    seed: int,
    width: int,
    height: int,
    max_frames: int,
) -> None:
    """One deterministic rollout, saved frame-by-frame as a GIF."""

    env   = LeapGraspEnv(render_mode=None)   # no interactive viewer
    model = PPO.load(model_path, env=None)

    obs, _ = env.reset(seed=seed)

    mj_model: mujoco.MjModel = env.model
    mj_data:  mujoco.MjData  = env.data

    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # fixed free camera — angle chosen to show palm approach and lift clearly
    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = 0.45
    cam.azimuth   = 160
    cam.elevation = -25
    cam.lookat[:] = [0.0, 0.0, 0.48]

    frames        = []
    done          = False
    step          = 0
    max_lift_seen = 0.0
    contacts_seen = 0

    print(f"Recording episode (seed={seed}, max_frames={max_frames}) …")

    while not done and step < max_frames:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()   # uint8 (H, W, 3)
        frames.append(frame)

        max_lift_seen = max(max_lift_seen, info.get("lift_height", 0.0))
        contacts_seen = max(contacts_seen, info.get("n_contacts",  0))
        step += 1

    renderer.close()
    env.close()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving {len(frames)} frames → {output_path} at {fps} fps …")

    imageio.mimsave(output_path, frames, fps=fps, loop=0)

    # MP4 for the report / anywhere GIFs look terrible
    mp4_path = output_path.replace(".gif", ".mp4")
    imageio.mimsave(mp4_path, frames, fps=fps, codec="libx264", quality=8)

    print(f"Done.  max_lift={max_lift_seen:.4f}m  contacts={contacts_seen}")
    print(f"  GIF → {output_path}")
    print(f"  MP4 → {mp4_path}")


# If you want a tracking camera instead of the hardcoded free cam, add this
# to assets/leap_grasp_scene.xml under <worldbody>:
#
#   <visual>
#     <global offwidth="640" offheight="480"/>
#   </visual>
#
#   <camera name="track_object" mode="trackcom" target="object"
#           pos="0.4 -0.4 0.7" xyaxes="1 0 0 0 0.5 1"/>
#
# Then swap cam= above for camera="track_object".
# If the named camera is missing, just call renderer.update_scene(mj_data)
# with no camera arg and it falls back to the default free cam.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="leap_ppo_best.zip")
    parser.add_argument("--output",     default="assets/grasp_demo.gif")
    parser.add_argument("--fps",        type=int, default=20)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--width",      type=int, default=640)
    parser.add_argument("--height",     type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Cap at this many frames (matches MAX_STEPS by default)")
    args = parser.parse_args()

    record_episode(
        model_path  = args.model,
        output_path = args.output,
        fps         = args.fps,
        seed        = args.seed,
        width       = args.width,
        height      = args.height,
        max_frames  = args.max_frames,
    )
