import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper


def make_minigrid_env(env_id: str = "MiniGrid-Empty-8x8-v0"):
    """Create a MiniGrid environment with image-only RGB observations."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env
