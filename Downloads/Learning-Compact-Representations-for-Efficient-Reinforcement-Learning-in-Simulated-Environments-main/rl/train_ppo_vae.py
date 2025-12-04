# rl/train_ppo_vae.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.make_env import make_minigrid_env
from rl.features_vae import VAEFeatureExtractor


def make_env_fn():
    """
    Factory function for MiniGrid environment.
    """
    return make_minigrid_env("MiniGrid-Empty-8x8-v0")


class VAEPolicyExtractor(VAEFeatureExtractor):
    """
    Wrapper autour du VAEFeatureExtractor qui charge directement
    le VAE pré-entraîné.
    """
    def __init__(self, observation_space, latent_dim: int = 64):
        super().__init__(observation_space, latent_dim)
        # charge l'encodeur VAE entraîné par rep_learning/train_vae.py
        self.load_pretrained("saved_models/vae_encoder.pt")


def main():
    # Environnement vectorisé (8 parallèles comme pour AE / SimCLR / raw)
    env = make_vec_env(make_env_fn, n_envs=8)

    policy_kwargs = dict(
        features_extractor_class=VAEPolicyExtractor,
        features_extractor_kwargs=dict(latent_dim=64),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="data/logs/ppo_vae",
    )

    # même budget de training que les autres pour une comparaison fair
    model.learn(total_timesteps=200_000)
    model.save("saved_models/ppo_vae")
    print("Saved PPO+VAE model to saved_models/ppo_vae")


if __name__ == "__main__":
    main()
