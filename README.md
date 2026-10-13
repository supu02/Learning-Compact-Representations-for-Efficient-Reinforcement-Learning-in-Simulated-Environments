# Learning Compact Visual Representations for Efficient Reinforcement Learning

A controlled MiniGrid study of **representation learning for pixel-based PPO**. The project compares PPO trained from raw RGB observations with PPO using frozen features learned by an **Autoencoder (AE)**, **Variational Autoencoder (VAE)**, or **SimCLR** encoder.

The central design separates perception from control:

```text
Random MiniGrid exploration
          |
          v
   20,000 RGB frames
          |
          +----------------+----------------+----------------+
          |                |                |
          v                v                v
         AE               VAE             SimCLR
       64-D μ?            64-D μ           128-D
          |                |                |
          +----------------+----------------+
                           |
                    freeze encoder
                           |
                           v
                     PPO policy
                           |
                           v
               MiniGrid-Empty-8x8
```

For the AE branch, PPO uses the encoder's 64-D latent vector. For VAE, PPO uses the deterministic 64-D mean vector `mu` rather than sampling from the latent distribution. For SimCLR, PPO uses the 128-D encoder output and discards the projection head after pretraining.

## Research question

Can compact representations learned before policy optimization provide useful state features for reinforcement learning from visual observations?

The repository tests this by holding the downstream algorithm and training budget broadly constant while changing the representation supplied to PPO.

## Environment and data collection

Experiments use `MiniGrid-Empty-8x8-v0`.

`envs/make_env.py` creates the environment with MiniGrid's `RGBImgPartialObsWrapper` followed by `ImgObsWrapper`, producing image-only RGB observations. With this environment/wrapper configuration, the model code operates on `56 x 56 x 3` observations.

Before RL training, `envs/collect_frames.py` collects **20,000 observations using a random policy** and stores them in:

```text
data/frames/frames.npz
```

The representation-learning scripts normalize the stored `uint8` frames to `[0, 1]`.

## Representation learning

The three methods use closely matched convolutional encoders with three stride-2 convolutional layers:

```text
RGB image
  |
Conv 3 -> 32
  |
Conv 32 -> 64
  |
Conv 64 -> 128
  |
Flatten
  |
Latent / feature vector
```

### Autoencoder

`models/autoencoder.py` implements a convolutional encoder-decoder.

- latent dimension: **64**
- objective: mean squared reconstruction error
- optimizer: Adam
- default learning rate: `1e-3`
- default batch size: `128`
- default training: `20` epochs

After training, only the encoder weights are saved to `saved_models/ae_encoder.pt`.

### Variational Autoencoder

`models/vae.py` implements a convolutional VAE with separate `mu` and `logvar` heads.

Its objective is:

```text
loss = reconstruction MSE + beta * KL divergence
```

Default `beta = 1.0`.

- latent dimension: **64**
- optimizer: Adam
- default learning rate: `1e-3`
- default batch size: `128`
- default training: `20` epochs

The VAE samples with the reparameterization trick during representation training. During PPO training, however, `VAEFeatureExtractor` returns **`mu` directly**, giving the policy a deterministic representation.

### SimCLR

`models/simclr.py` implements a convolutional encoder followed by a two-layer projection head.

Two independently augmented views are generated using:

- random resized crop;
- random horizontal flip;
- color jitter.

The model is trained with an InfoNCE-style contrastive objective at temperature `0.5`.

- encoder feature dimension: **128**
- projection dimension: **64**
- optimizer: Adam
- default learning rate: `1e-3`
- default batch size: `64`
- default training: `20` epochs

Only the encoder is saved for downstream RL; the projection head is not used by PPO.

## Frozen feature extractors

The files in `rl/features_*.py` integrate pretrained encoders with Stable-Baselines3.

Each extractor:

1. reconstructs the appropriate encoder architecture;
2. loads its pretrained weights;
3. sets `requires_grad = False`;
4. switches the encoder to evaluation mode;
5. runs feature extraction under `torch.no_grad()`.

This means PPO optimizes the policy on a **fixed representation** rather than fine-tuning the visual encoder.

## PPO experiments

Four training scripts are included:

| Experiment | Policy input | Representation size |
| --- | --- | ---: |
| `train_ppo_raw.py` | RGB observation via SB3 `CnnPolicy` | raw pixels |
| `train_ppo_ae.py` | frozen AE encoder | 64 |
| `train_ppo_vae.py` | frozen VAE mean `mu` | 64 |
| `train_ppo_simclr.py` | frozen SimCLR encoder | 128 |

All four scripts use:

- PPO from Stable-Baselines3;
- `MiniGrid-Empty-8x8-v0`;
- 8 vectorized environments;
- a training budget of **200,000 environment timesteps**;
- TensorBoard logging;
- saved PPO checkpoints under `saved_models/`.

The scripts otherwise rely on Stable-Baselines3 PPO defaults.

## Reported experiment results

The following values were recorded in the project README from the original experiments:

| Method | Reward ↑ | Episode length ↓ | FPS ↑ |
| --- | ---: | ---: | ---: |
| PPO, raw pixels | 0.94 | 14.7 | 392 |
| PPO + AE | 0.92 | 15.4 | 269 |
| PPO + VAE | 0.93 | 19.8 | **457** |
| PPO + SimCLR | **0.95** | **13.6** | 342 |

These values indicate that, in the recorded runs, SimCLR had the highest reward and shortest episode length, while the VAE run reported the highest FPS.

### Evidence boundary

The repository currently contains the training implementation but **does not contain the original evaluation script, TensorBoard logs, seeds, or raw result artifacts used to produce the table above**. The numbers should therefore be treated as **reported results from the original experiment**, not as a statistically validated benchmark.

The current repository also does not support a claim that one representation method is generally superior. Multi-seed evaluation is needed to quantify variance and make stronger comparisons.

## Repository structure

```text
.
├── envs/
│   ├── __init__.py
│   ├── collect_frames.py
│   └── make_env.py
├── models/
│   ├── autoencoder.py
│   ├── simclr.py
│   └── vae.py
├── rep_learning/
│   ├── train_autoencoder.py
│   ├── train_simclr.py
│   └── train_vae.py
├── rl/
│   ├── features_ae.py
│   ├── features_simclr.py
│   ├── features_vae.py
│   ├── train_ppo_ae.py
│   ├── train_ppo_raw.py
│   ├── train_ppo_simclr.py
│   └── train_ppo_vae.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated frames, model weights, and TensorBoard logs are intentionally not versioned.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the pipeline

Run commands from the repository root.

### 1. Collect random-policy frames

```bash
python -m envs.collect_frames
```

This creates `data/frames/frames.npz` with 20,000 observations.

### 2. Pretrain the encoders

```bash
python -m rep_learning.train_autoencoder
python -m rep_learning.train_vae
python -m rep_learning.train_simclr
```

This creates:

```text
saved_models/ae_encoder.pt
saved_models/vae_encoder.pt
saved_models/simclr_encoder.pt
```

### 3. Train PPO

Raw-pixel baseline:

```bash
python -m rl.train_ppo_raw
```

Frozen representation variants:

```bash
python -m rl.train_ppo_ae
python -m rl.train_ppo_vae
python -m rl.train_ppo_simclr
```

Each script trains for 200,000 timesteps and saves its PPO model under `saved_models/`.

### 4. Inspect training logs

The PPO scripts write TensorBoard logs below `data/logs/`.

```bash
tensorboard --logdir data/logs
```

## What the project demonstrates

- representation learning for reinforcement learning;
- convolutional autoencoders and variational autoencoders;
- SimCLR-style self-supervised contrastive learning;
- InfoNCE-style contrastive loss;
- pretrained and frozen visual feature extractors;
- Stable-Baselines3 custom feature-extractor integration;
- PPO from raw pixels versus learned latent representations;
- vectorized RL environments;
- separation of perception pretraining from downstream control.

## Limitations

The experiment uses a single, relatively simple MiniGrid environment.

The current code does not set or record random seeds, and the repository does not include multi-seed statistics. Differences in the reported table may therefore reflect run-to-run variance.

The representation methods do not have identical latent dimensionality: AE and VAE use 64 dimensions, while SimCLR uses 128. This prevents the experiment from isolating representation objective alone.

The augmentation choices used for SimCLR are generic image augmentations and were not ablated for MiniGrid.

The repository does not include the original evaluation artifacts behind the reported summary metrics.

## Next steps

The most useful extensions would be:

- add a deterministic evaluation script and save raw metrics;
- evaluate multiple seeds and report mean ± standard deviation;
- compare methods at matched latent dimensions;
- add learning curves rather than only final summary values;
- test harder MiniGrid tasks such as DoorKey or FourRooms;
- ablate SimCLR augmentations;
- compare frozen encoders with end-to-end fine-tuning.

## References

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Kingma and Welling, *Auto-Encoding Variational Bayes*, 2014.
- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, 2020.
- MiniGrid, Farama Foundation.
