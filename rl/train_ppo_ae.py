# rl/train_ppo_ae.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.make_env import make_minigrid_env
from rl.features_ae import AEFeatureExtractor

import torch
from torch import nn


def make_env_fn():
    return make_minigrid_env("MiniGrid-Empty-8x8-v0")


class AEPolicyExtractor(AEFeatureExtractor):
    def __init__(self, observation_space, latent_dim=64):
        super().__init__(observation_space, latent_dim)
        # load pretrained encoder
        self.load_pretrained("saved_models/ae_encoder.pt")


def main():
    env = make_vec_env(make_env_fn, n_envs=8)

    policy_kwargs = dict(
        features_extractor_class=AEPolicyExtractor,
        features_extractor_kwargs=dict(latent_dim=64),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="data/logs/ppo_ae"
    )

    model.learn(total_timesteps=200_000)
    model.save("saved_models/ppo_ae")

if __name__ == "__main__":
    main()