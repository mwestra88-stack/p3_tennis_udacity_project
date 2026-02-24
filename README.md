# Collaboration and Competition – Tennis Environment (MADDPG)

## Project Details

This project solves the **Tennis** environment from the Udacity Deep Reinforcement Learning Nanodegree using **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**.

In this environment, two agents control rackets to bounce a ball over a net. The goal is to keep the ball in play cooperatively.

90% of this readme has been written by ChatGPT based on input and Python code provided by me

### Environment Description

- **Number of agents:** 2  
- **Observation space (per agent):** 24 continuous variables  
- **Action space (per agent):** 2 continuous actions  
  - Horizontal movement
  - Jump  

Each action dimension is continuous in the range **[-1, 1]**.

### Rewards

- +0.1 if an agent hits the ball over the net  
- -0.01 if the ball hits the ground or goes out of bounds  

The rewards are very small in magnitude. In this implementation, rewards are multiplied by 100 during training to stabilize gradients.

### Episode Termination

An episode ends when:
- The ball hits the ground, or
- A maximum timestep limit is reached.

### Solving Criterion

The environment is considered solved when:

> The **average over 100 consecutive episodes** of the **maximum score achieved by either agent** is **≥ +0.5**.

In this implementation, the environment was solved in **1143 episodes**, achieving an average score of **0.52** over 100 episodes.

---

## Implementation Overview

This project implements the **MADDPG algorithm**, where:

- Each agent has:
  - One Actor network
  - One Critic network
- Critics use **joint state and joint action spaces**
- Actors use local observations only
- Target networks are updated via soft updates
- Experience replay is shared across agents
- State normalization is applied per agent
- Rewards are scaled by a factor of 100

### Hyperparameters

```python
BUFFER_SIZE = 1e5
BATCH_SIZE = 128
GAMMA = 1
TAU = 1e-2
LRA = 1e-4
LRC = 3e-3
UPDATE_EVERY = 1
BUFFER_WARMUP = 10000
LEARN_STEPS = 2
HIDDEN_ACTOR = [400, 300]
HIDDEN_CRITIC = [400, 300]
NOISE_SIGMA = 0.3
INFLATE_REWARDS = 100

### Training Characteristics

- Ornstein-Uhlenbeck noise was replaced by Gaussian noise  
- Noise decays from 1.0 to 0.1  
- Two learning steps per environment step  
- Soft update coefficient τ = 1e-2  
- Reward scaling to counteract very small reward magnitudes  

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
```

### 2. Create Conda Environment

Open Anaconda Prompt and run:

```bash
py -3.6 -m venv unityagents36
unityagents36\Scripts\activate
```

Install dependencies (similar versions as used in the Banana Navigation project):

```bash
pip install torch==1.4.0
pip install numpy
pip install matplotlib
pip install jupyter
pip install unityagents==0.4.0
```

Note that newer versions of Python and ML-Agents are not compatible with this legacy environment.

### Register Jupyter kernel
To use this environment inside Jupyter Notebook:
```bash
python -m ipykernel install --user --name unityagents36 --display-name "Python 3.6 (unityagents)"
```
Then select **Python 3.6 (unityagents)** as the kernel in Jupyter.

### 3. Download the Tennis Environment
For this project, you will not need to install Unity - this is because Udacity has already built the environment for you, and you can download it from the link below (I used windows, it is also available for other operating systems)

* Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip
  
Then, place the file in the `p3_collab-compet/` folder in the course GitHub repository, and unzip (or decompress) the file.

## How to Run the Project

All training was performed inside a **Jupyter Notebook**.

### 1. Start Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook that contains the training code.

---

### 2. Initialize the Agent

```python
agent = MADDPG_Agent(
    state_size=24,
    action_size=2,
    num_agents=2,
    shared_actor=False,
    seed=1234
)
```

---

### 3. Train the Agent

```python
scores, scores_ma = maddpg()
```

Training parameters:

```python
def maddpg(
    n_episodes=2500,
    max_t=1000,
    noise_scale_start=1.0,
    noise_scale_end=0.1,
    noise_decay=0.999
):
```

During training:

- Rewards are accumulated per agent  
- The maximum of the two agent scores is stored  
- A moving average over 100 episodes is computed  
- Training stops once the average ≥ 0.5  

---

### 4. Plot Results

```python
plt.plot(episodes, scores, label="Max Score per Episode")
plt.plot(episodes, scores_ma, label="Moving Average (100)")
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend()
plt.show()
```

---

## Saved Models

Once solved, the following checkpoints are saved:

```
checkpoint_actor1.pth
checkpoint_actor2.pth
checkpoint_critic1.pth
checkpoint_critic2.pth
```

---

## Repository Structure

```
.
├── actor_model.py
├── critic_model.py
├── maddpg_agent.py
├── Tennis.ipynb
├── README.md
├── LICENSE
├── Report.pdf
├── checkpoint_actor1.pth
├── checkpoint_actor2.pth
├── checkpoint_critic1.pth
├── checkpoint_critic2.pth
```

---

## Summary

- Algorithm: MADDPG  
- Environment: Unity Tennis (2 agents)  
- State space: 24 (per agent)  
- Action space: 2 continuous (per agent)  
- Solved in: 1143 episodes  
- Average score achieved: 0.52  

This implementation demonstrates stable cooperative learning using centralized critics and decentralized actors.

---

## Future Improvements

- Parameter noise instead of action noise  
- Prioritized Experience Replay  
- Shared actor networks  
- Twin critics (TD3-style stabilization)  
- Curriculum learning  
- More systematic reward scaling  

---

**Author:** Martijn Westra  
Deep Reinforcement Learning Nanodegree  
