# rl/train_ppo_raw.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.make_env import make_minigrid_env

def make_env_fn():
    return make_minigrid_env("MiniGrid-Empty-8x8-v0")

def main():
    # create vectorized env
    env = make_vec_env(make_env_fn, n_envs=8)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="data/logs/ppo_raw"
    )

    model.learn(total_timesteps=200_000)
    model.save("saved_models/ppo_raw")

if __name__ == "__main__":
    main()