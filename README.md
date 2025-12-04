
Learning Compact Representations for Efficient Reinforcement Learning in Simulated Environments

This project investigates whether reinforcement learning (RL) agents can learn faster and more reliably when trained on compact latent representations rather than raw pixel observations.
We explore how representation learning—specifically using Convolutional Autoencoders—can improve sample efficiency, stability, and generalization in RL tasks.
The work is implemented in MiniGrid, a lightweight and interpretable RL environment.

Team Members: Meriem Jelassi · Supriya Malla · Sydney Nzunguli

⸻

1. 🔍 Motivation

Deep RL has achieved strong results on visual tasks, but raw observations are often high-dimensional and redundant.
This leads to:
	•	slow and unstable training
	•	poor sample efficiency
	•	limited generalization

Learning compact latent representations offers a potential solution.
Autoencoders or contrastive models can extract meaningful, low-dimensional features that capture the essential structure of an environment while removing irrelevant details.

This project explores whether using such latent embeddings can help RL agents train more efficiently and achieve more robust behavior.

⸻

2. 🎯 Problem Definition

Research Question:
Can an RL agent learn faster and generalize better when using compact latent representations instead of raw visual inputs?

Raw pixel frames often include unnecessary details.
Latent spaces, on the other hand:
	•	compress important features
	•	remove noise
	•	represent the underlying structure of the scene

Our goal is to evaluate whether RL performance improves when using these compact representations.

⸻

3. 🧠 Methodology

A. Representation Learning Stage
	1.	Collect state images from MiniGrid environments
	2.	Train a Convolutional Autoencoder to compress frames into low-dimensional vectors
	3.	Assess the quality of the latent space using:
	•	t-SNE / UMAP visualizations
	•	linear probing

B. Reinforcement Learning Stage
	1.	Freeze (or optionally fine-tune) the pretrained encoder
	2.	Use the latent vector output as input to a DQN agent
	3.	Train a baseline DQN directly on raw pixels for comparison
	4.	Analyze:
	•	sample efficiency
	•	stability across training
	•	final reward performance

C. Analysis
	•	Compare learning curves between the latent-based agent and the baseline
	•	Visualize the structure of learned latent spaces
	•	Interpret whether representation learning contributes to more stable policies

⸻

4. 📈 Expected Results

Agents trained on compact latent representations are expected to:
	•	converge faster
	•	achieve higher rewards with fewer interactions
	•	learn more stable and transferable policies
	•	exhibit structured latent spaces with semantically meaningful features
(e.g., agent position, object layout, navigational cues)

These findings would show how representation learning can improve the efficiency and robustness of RL systems.

⸻

5. 🛠️ Keywords

Convolutional Neural Networks · Autoencoder · Representation Learning · Reinforcement Learning · Contrastive Learning · Regularization · Generalization

⸻

6. 📂 Repository Structure

.
├── autoencoder/          # Model definition and training scripts
├── rl_agent/             # DQN agent trained on latent vectors
├── experiments/          # Evaluation, visualizations, tests
├── utils/                # Environment wrappers and helpers
└── README.md


⸻

7. 🚀 How to Run

Train the Autoencoder

python autoencoder/train_autoencoder.py

Train the DQN Agent

python rl_agent/train_rl.py

The RL script loads the pretrained encoder and trains the agent on latent states.

⸻

8. 👤 About the Project

This work was part of a research-oriented learning project focused on improving efficiency in reinforcement learning. Our aim was to combine ideas from deep representation learning and RL to study how learned latent spaces can support more efficient adaptive decision-making.
