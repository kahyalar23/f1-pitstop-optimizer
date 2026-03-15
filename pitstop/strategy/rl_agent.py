"""
pitstop/strategy/rl_agent.py

PPO-based strategy agent using stable-baselines3.

Architecture:
    Actor-Critic MLP: [64, 64] hidden layers
    Policy: Discrete(4) — categorical softmax
    Value: Linear → scalar

Training loop:
    1. Collect rollouts in F1RaceEnv
    2. Compute advantages via GAE (Generalized Advantage Estimation)
    3. Update policy + value via clipped PPO objective
    4. Evaluate every N episodes

Proximal Policy Optimization objective:
    L_CLIP = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
    r_t = pi_theta(a|s) / pi_theta_old(a|s)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, StopTrainingOnRewardThreshold
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from pitstop.strategy.environment import F1RaceEnv


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

DEFAULT_PPO_KWARGS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,           # steps per rollout per env
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,             # discount factor
    "gae_lambda": 0.95,        # GAE lambda
    "clip_range": 0.2,         # PPO clipping epsilon
    "ent_coef": 0.01,          # entropy regularisation (encourages exploration)
    "vf_coef": 0.5,            # value function coefficient
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
    },
    "verbose": 1,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_agent(
    race: str = "generic",
    crew_name: str = "elite",
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    save_path: str = "./models/ppo_f1strategy",
    eval_freq: int = 10_000,
    seed: int = 42,
) -> Optional["PPO"]:
    """
    Train PPO agent on F1RaceEnv.

    Returns trained model (or None if stable-baselines3 not installed).
    """
    if not SB3_AVAILABLE:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3")
        return None

    os.makedirs(Path(save_path).parent, exist_ok=True)

    # Vectorised training envs
    env = make_vec_env(
        lambda: Monitor(F1RaceEnv(race=race, crew_name=crew_name, seed=None)),
        n_envs=n_envs, seed=seed
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Evaluation env
    eval_env = make_vec_env(
        lambda: Monitor(F1RaceEnv(race=race, crew_name=crew_name, seed=seed + 100)),
        n_envs=1, seed=seed + 100
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Callbacks
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path + "_logs",
        eval_freq=eval_freq,
        deterministic=True,
        callback_on_new_best=stop_cb,
    )

    model = PPO(
        env=env,
        seed=seed,
        **DEFAULT_PPO_KWARGS
    )

    print(f"\nTraining PPO agent on {race} ({total_timesteps:,} timesteps, {n_envs} envs)")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")

    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=True)

    # Save normalisation stats alongside model
    env.save(save_path + "_vec_normalize.pkl")
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    return model


# ---------------------------------------------------------------------------
# Evaluation & analysis
# ---------------------------------------------------------------------------

def evaluate_agent(
    model_path: str,
    race: str = "generic",
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 999,
) -> pd.DataFrame:
    """
    Evaluate trained agent across N episodes.
    Returns per-episode summary DataFrame.
    """
    if not SB3_AVAILABLE:
        print("stable-baselines3 not installed.")
        return pd.DataFrame()

    from stable_baselines3.common.vec_env import VecNormalize

    env = F1RaceEnv(race=race, seed=seed)
    model = PPO.load(model_path, env=env)

    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rows.append({
            "episode": ep,
            "total_reward": round(total_reward, 2),
            "final_position": info["position"],
            "pit_count": info["pit_count"],
            "total_time": info["total_time"],
        })

    return pd.DataFrame(rows)


def action_distribution_analysis(
    model_path: str,
    race: str = "generic",
    n_episodes: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Analyse what decisions the agent makes, by lap and tyre state.
    Useful for understanding learned strategy.
    """
    if not SB3_AVAILABLE:
        return pd.DataFrame()

    ACTION_NAMES = {0: "stay_out", 1: "pit_soft", 2: "pit_medium", 3: "pit_hard"}
    env = F1RaceEnv(race=race, seed=seed)
    model = PPO.load(model_path, env=env)

    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_name = ACTION_NAMES[int(action)]
            info_pre = env._get_info()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rows.append({
                "episode": ep,
                "lap": info_pre["lap"],
                "tyre_age": info_pre["tyre_age"],
                "compound": info_pre["compound"],
                "position": info_pre["position"],
                "gap_ahead": info_pre["gap_ahead"],
                "action": action_name,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline rule-based strategy (for comparison)
# ---------------------------------------------------------------------------

class RuleBasedStrategy:
    """
    Simple rule-based baseline: pit when tyre age exceeds compound threshold.
    Used to benchmark RL agent performance.
    """
    STOP_LAPS = {"soft": 18, "medium": 30, "hard": 45}

    def predict(self, obs: np.ndarray, **kwargs) -> tuple[int, None]:
        """Mimic SB3 predict() API."""
        tyre_age = obs[2] * 50     # denormalise
        compound_idx = round(obs[3] * 3)
        compound = ["soft", "medium", "hard"][min(compound_idx, 2)]
        threshold = self.STOP_LAPS[compound]

        if tyre_age > threshold:
            return 2, None  # pit for medium
        return 0, None      # stay out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate F1 RL strategy agent")
    parser.add_argument("mode", choices=["train", "eval", "baseline"])
    parser.add_argument("--race", default="generic")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--model", default="./models/ppo_f1strategy")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "train":
        train_agent(race=args.race, total_timesteps=args.steps, save_path=args.model)

    elif args.mode == "eval":
        df = evaluate_agent(args.model, race=args.race, n_episodes=args.episodes)
        print(df.describe().to_string())

    elif args.mode == "baseline":
        env = F1RaceEnv(race=args.race, seed=42)
        agent = RuleBasedStrategy()
        rewards = []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=42 + ep)
            total_r = 0
            done = False
            while not done:
                action, _ = agent.predict(obs)
                obs, r, term, trunc, _ = env.step(action)
                total_r += r
                done = term or trunc
            rewards.append(total_r)
        print(f"Baseline mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
