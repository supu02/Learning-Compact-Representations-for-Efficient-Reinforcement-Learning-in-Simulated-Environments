# rl/train_ppo_simclr.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.make_env import make_minigrid_env
from rl.features_simclr import SimCLRFeatureExtractor


def make_env_fn():
    return make_minigrid_env("MiniGrid-Empty-8x8-v0")


class SimCLRPolicyExtractor(SimCLRFeatureExtractor):
    def __init__(self, observation_space, feature_dim=128):
        super().__init__(observation_space, feature_dim)
        self.load_pretrained("saved_models/simclr_encoder.pt")


def main():
    env = make_vec_env(make_env_fn, n_envs=8)

    policy_kwargs = dict(
        features_extractor_class=SimCLRPolicyExtractor,
        features_extractor_kwargs=dict(feature_dim=128),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="data/logs/ppo_simclr"
    )

    model.learn(total_timesteps=200_000)
    model.save("saved_models/ppo_simclr")

if __name__ == "__main__":
    main()