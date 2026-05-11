"""
eval/evaluate.py
----------------
10-episode rollout eval for leap_ppo_best.zip.

Important: use RAW observations. Don't load vec_normalize.pkl — the best checkpoint
was saved without normalization wrapping and loading one will break the obs scale.

Usage:
    python eval/evaluate.py --model leap_ppo_best.zip [--episodes 10] [--seed 42]
"""

import argparse
import time
import numpy as np
from stable_baselines3 import PPO
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.leap_grasp_env import LeapGraspEnv

TABLE_Z   = 0.455   # settled object center z, verified in sim
SUCCESS_H = 0.15    # lift threshold (metres above table)


def evaluate(model_path: str, n_episodes: int, seed: int, render: bool):
    env   = LeapGraspEnv(render_mode="human" if render else None)
    model = PPO.load(model_path, env=None)   # env=None → skip VecNorm

    ep_rewards     = []
    lift_heights   = []
    contact_counts = []
    success_count  = 0
    step_counts    = []

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _  = env.reset(seed=ep_seed)
        done     = False
        ep_reward    = 0.0
        max_lift     = 0.0
        max_contacts = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward    += reward
            lift_h        = info.get("lift_height", 0.0)
            n_contacts    = info.get("n_contacts",  0)
            max_lift      = max(max_lift,     lift_h)
            max_contacts  = max(max_contacts, n_contacts)

        success = max_lift >= SUCCESS_H
        success_count += int(success)

        ep_rewards.append(ep_reward)
        lift_heights.append(max_lift)
        contact_counts.append(max_contacts)
        step_counts.append(info.get("step", 0))

        status = "SUCCESS ✓" if success else "fail"
        print(f"  Ep {ep+1:02d}/{n_episodes} | reward={ep_reward:+8.3f} | "
              f"lift={max_lift:.4f}m | contacts={max_contacts} | {status}")

    env.close()

    print("\n" + "─" * 62)
    print(f"  Model              : {model_path}")
    print(f"  Episodes           : {n_episodes}")
    print(f"  Seed               : {seed}")
    print("─" * 62)
    print(f"  Mean episode reward: {np.mean(ep_rewards):+.3f}  ± {np.std(ep_rewards):.3f}")
    print(f"  Grasp success rate : {success_count}/{n_episodes}  "
          f"({100*success_count/n_episodes:.1f}%)")
    print(f"  Mean max lift      : {np.mean(lift_heights):.4f} m  ± {np.std(lift_heights):.4f}")
    print(f"  Best lift          : {max(lift_heights):.4f} m")
    print(f"  Mean contacts@lift : {np.mean(contact_counts):.2f}")
    print(f"  Mean episode steps : {np.mean(step_counts):.1f}")
    print("─" * 62)

    return {
        "model":         model_path,
        "n_episodes":    n_episodes,
        "mean_reward":   float(np.mean(ep_rewards)),
        "std_reward":    float(np.std(ep_rewards)),
        "success_rate":  float(success_count / n_episodes),
        "mean_lift":     float(np.mean(lift_heights)),
        "best_lift":     float(max(lift_heights)),
        "mean_contacts": float(np.mean(contact_counts)),
        "mean_steps":    float(np.mean(step_counts)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="leap_ppo_best.zip",
                        help="Path to SB3 checkpoint zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--render",   action="store_true",
                        help="Open viewer (local only — won't work on Colab)")
    args = parser.parse_args()

    print(f"\n{'═'*62}")
    print(f"  LEAP Hand PPO — Evaluation")
    print(f"{'═'*62}\n")
    t0 = time.time()
    results = evaluate(args.model, args.episodes, args.seed, args.render)
    print(f"\n  Wall time: {time.time()-t0:.1f}s\n")
