import gymnasium as gym
import minigrid

def make_minigrid_env(env_id: str = "MiniGrid-Empty-8x8-v0"):
    """
    Create a MiniGrid environment with RGB image observations.
    """
    env = gym.make(
        env_id,
        render_mode="rgb_array"
    )
    # Wrap to get only the image as observation
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

    # Get full RGB image
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)  # now env.observation_space is Box(H, W, 3)
    return env
