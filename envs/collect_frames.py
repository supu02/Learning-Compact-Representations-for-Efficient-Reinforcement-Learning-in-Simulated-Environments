import os

import numpy as np
from tqdm import trange

from .make_env import make_minigrid_env


def collect_frames(
    env_id: str = "MiniGrid-Empty-8x8-v0",
    num_frames: int = 20_000,
    save_path: str = "data/frames/frames.npz",
):
    """Collect image observations using a random policy."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = make_minigrid_env(env_id)
    obs, _ = env.reset()

    frames = []
    for _ in trange(num_frames, desc="Collecting frames"):
        frames.append(obs)
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    frames = np.stack(frames, axis=0)
    np.savez_compressed(save_path, frames=frames)
    print(f"Saved {len(frames)} frames to {save_path}")


if __name__ == "__main__":
    collect_frames()
