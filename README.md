
🧠 Learning Compact Visual Representations for Efficient Reinforcement Learning

Perception–Control Decoupling for Pixel-Based PPO
MiniGrid Experiments (MiniGrid-Empty-8x8)

⸻

## Overview

Training reinforcement learning agents directly from raw pixels is challenging due to high-dimensional observations and unstable joint optimization of vision and control.

This project investigates whether learning compact visual representations before policy training improves sample efficiency and stability in pixel-based RL.

The approach decouples perception from control:

	1.	Learn a visual encoder using unsupervised/self-supervised learning.
	2.	Freeze the encoder.
	3.	Train a PPO agent on the latent representation.

We compare three representation learning objectives:

	•	Autoencoder (AE) — reconstruction-based
	•	Variational Autoencoder (VAE) — reconstruction + KL regularization
	•	SimCLR — contrastive self-supervised learning

All evaluations are conducted on MiniGrid-Empty-8x8.

⸻

## Core Idea

Instead of learning visual features and policy simultaneously, we:

	•	Collect unlabeled frames using random exploration
	•	Train an encoder on those frames
	•	Freeze the encoder
	•	Train PPO on the latent state

This isolates how the representation objective affects downstream RL performance.

⸻

## Environment

	•	Environment: MiniGrid-Empty-8x8
	•	Observation: RGB frames resized to 56×56
	•	Pretraining dataset: 20,000 random-policy frames
	•	RL budget: 200,000 environment timesteps

⸻

## Methods

Representation Learning

All encoders share the same CNN backbone for fair comparison.

Method	Latent Dim	Objective
AE	64	Reconstruction loss
VAE	64	Reconstruction + KL divergence
SimCLR	128	Contrastive loss

For VAE, the deterministic mean vector μ is used during PPO training.

Encoders are frozen during RL.

⸻

## Reinforcement Learning

	•	Algorithm: PPO
	•	Input: Frozen latent representation
	•	Training: 200k environment steps
	•	Evaluation metrics:
	•	Average reward
	•	Episode length
	•	Frames per second (FPS)

⸻

## Results (200k Environment Steps)

| Method            | Reward ↑ | Episode Length ↓ | FPS ↑ |
|------------------|---------:|-----------------:|------:|
| PPO (Raw Pixels) | 0.94     | 14.7             | 392   |
| PPO + AE         | 0.92     | 15.4             | 269   |
| PPO + VAE        | 0.93     | 19.8             | 457   |
| PPO + SimCLR     | **0.95** | **13.6**         | 342   |


⸻

## Key Findings

	•	SimCLR produced the most behaviorally efficient policies (shortest episode length).
	•	VAE achieved the highest inference speed (FPS).
	•	Reconstruction-based objectives (AE/VAE) do not necessarily produce control-optimal features.
	•	Contrastive learning appears better aligned with downstream decision-making.

⸻
## Project Structure

```
.
├── README.md
├── requirements.txt
├── models/        # Encoder architectures (AE / VAE / SimCLR)
├── rep_learning/  # Encoder training scripts
├── rl/            # PPO training scripts
└── envs/          # Environment utilities
```


⸻

## Design Rationale

This project evaluates a structured pipeline:

Unsupervised Representation → Frozen Latent → Policy Learning

By separating perception and control, we can directly analyze how representation objectives influence reinforcement learning performance.

⸻

## Future Work

	•	Multi-seed evaluation for statistical robustness
	•	Harder MiniGrid environments (FourRooms, DoorKey)
	•	Jointly trained world-model approaches
	•	Latent dimension ablations

⸻

## References

	•	Schulman et al., Proximal Policy Optimization, 2017
	•	Kingma & Welling, Variational Autoencoders, 2014
	•	Chen et al., SimCLR, 2020
	•	MiniGrid: Chevalier-Boisvert et al., 2018
