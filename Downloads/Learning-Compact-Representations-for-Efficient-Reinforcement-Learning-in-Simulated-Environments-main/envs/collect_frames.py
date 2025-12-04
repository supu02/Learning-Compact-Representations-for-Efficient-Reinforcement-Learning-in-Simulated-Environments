# envs/collect_frames.py
import os
import numpy as np
from tqdm import trange
from .make_env import make_minigrid_env

import torch
from torch.utils.data import Dataset, DataLoader

def collect_frames(
    env_id="MiniGrid-Empty-8x8-v0",
    num_frames=20000,
    save_path="data/frames/frames.npz"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = make_minigrid_env(env_id)
    obs, _ = env.reset()

    frames = []
    for _ in trange(num_frames, desc="Collecting frames"):
        # obs is HxWx3 uint8
        frames.append(obs)
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            obs, _ = env.reset()

    frames = np.stack(frames, axis=0)  # (N, H, W, C)
    np.savez_compressed(save_path, frames=frames)
    print(f"Saved {len(frames)} frames to {save_path}")


if __name__ == "__main__":
    collect_frames()