# Frozen Lake Reinforcement Learning Assignment

## First-Principles Q-Learning Implementation on an 8×8 Frozen Lake Environment

## 1.Introduction

### Reinforcement Learning



Reinforcement Learning (RL) is a machine learning paradigm in which an agent learns to make decisions by interacting with an environment. The agent takes actions, receives rewards or penalties, and gradually learns an optimal policy that maximizes cumulative rewards over time.



Key components include:



* Agent
* Environment
* State
* Action
* Reward
* Policy
* Value Function



### Frozen Lake

Frozen Lake is a classic reinforcement learning environment provided by OpenAI Gymnasium. The environment consists of a grid world containing:



* A starting position (S)
* Frozen tiles (F)
* Holes (H)
* A goal state (G)



### 2\. Environment Design

### State Representation

The environment uses a discrete state space.

For 8\*8 frozen lake:

S F F F F F F F

F F F F F F F F

F F F H F F F F

F F F H F F F F

F F F H F F F F

F H H F F F H F

F H F F H F H F

F F F H F F F G



Each state corresponds to the agent's location on the grid. The State space size is 64 states



#### Action Representation

The agent can perform four actions:

|Action|Meaning|
|-|-|
|0|Left|
|1|Down|
|2|Right|
|3|Up|



#### Reward Structure

|Event|Reward|Purpose|
|-|-|-|
|Goal Reached|+1.00|Encourages successful completion of the task|
|Fall into Hole|-1.00|Strong penalizes unsafe actions|
|Normal Movement|-0.01|Encourages shorter and more efficient paths|
|Boundary Collision|-0.05|Discourages invalid moves outside the grid|



The agent learns to maximize the probability of reaching the goal.

### 

### 3\. Q-Learning Algorithm



#### Description of Q-Learning

Q-Learning is a model-free reinforcement learning algorithm that learns the value of taking a specific action in a particular state.



The algorithm maintains a Q-table:

Q(State, Action), which stores expected future rewards



#### Q-Learning Update Equation

Q(s,a) ← Q(s,a) + α\[r + γmaxQ(s',a') − Q(s,a)]



where:

α = learning rate

γ = Discount Rate

r = Reward

s = Current State

a = Current Action

s' = Next Sate

### 

This update gradually improves the estimated action values.



#### Exploration Strategy



The project uses an ε-greedy strategy.



With probability ε:

Explore (random action)



With probability (1 − ε):

Exploit (best known action)



Epsilon is decayed throughout training to transition from exploration to exploitation.



### 4\. Training Procedure

#### Hyperparameters Used

|Hyperparameter|Value|
|-|-|
|Learning rate (α)|0.12|
|Discount Rate (γ)|0.99|
|Initial Epsilon|1.0|
|Minimum Epsilon|0.01|
|Epsilon Decay|0.9995|
|Episodes|20,000|
|Step per Episode|200|



#### Training Process

1. Initialize Q-table.
2. Start episode.
3. Select action using ε-greedy policy.
4. Execute action.
5. Observe reward and next state.
6. Update Q-value.
7. Continue until episode termination.
8. Repeat for all episodes.



### 5\. Results

Success Rate: 100%

Average Reward: 0.87

Successful Runs: 100

Failures: 0

Average Steps to Goal: 14



#### Learned Policy

↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

→ → → → ↓ ↓ ↓ ↓

↑ ↑ ↑ H ↓ ↓ → ↓

→ ↑ ↑ H → → → ↓

↑ ↑ ← H → → → ↓

↓ H H ↓ ↑ ↑ H ↓

↓ H → ↑ H ↓ H ↓

→ → ↑ H ↓ ↓ → G



#### Discussion of Performance

The agent successfully learned a near-optimal policy for navigating the Frozen Lake environment. Training rewards increased steadily while epsilon decreased over time, indicating a successful transition from exploration to exploitation. The final policy demonstrates that the agent learned safe paths that maximize the probability of reaching the goal while avoiding holes.



Include:



reward\_curve.png
Caption: Episode reward obtained during training over 20,000 episodes. The upward trend indicates that the agent gradually learned to avoid holes and reach the goal more consistently.

success\_rate.png
Caption: Rolling success rate during training. The success rate steadily increased and eventually approached 100%, demonstrating successful learning and policy convergence.

epsilon\_decay.png
Caption: Exploration rate (ε) throughout training. Epsilon decreases from 1.0 to 0.02, allowing the agent to transition from exploration to exploitation.

value\_heatmap.png
Caption: Learned state-value function V(s)=maxQ(s,a). Higher values appear along safe routes leading to the goal, while states near holes exhibit lower values.
policy.png

Policy.png
Caption: Final greedy policy extracted from the trained Q-table. Arrows represent the preferred action in each state, showing a safe path from the start state to the goal while avoiding holes.





### 6\. Execution Instructions

#### Install Dependencies

pip install -r requirements.txt



#### Train Agent



python train.py



#### Evaluate Agent

python evaluate.py



#### Visualize Results

python visualize.py



#### Generated Outputs

The following files are produced:

policy.txt

evaluation\_summary.txt

training\_metrics.csv

reward\_curve.png

success\_rate.png

epsilon\_decay.png

value\_heatmap.png

policy.png



### 7\. Limitations and Future Improvements



* Q-Learning scales poorly to large state spaces.
* Performance depends heavily on hyperparameter tuning.
* Frozen Lake is a tabular environment and does not require function approximation.



### 

