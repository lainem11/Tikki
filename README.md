# Fastest way to play Tikki

Run AI/main.py. Select enemy player model from the AI/model directory.

# Train AI

- 1. Train Proximal Policy Optimization (PPO) agent by running AI/Agent_V1-2_PPO.py.
  - 1. This runs series of training iterations to tune the model hyperparameters withing the specified search space. 
- 2. Choose the best agent (actor) with check_agent_performance.ipynb
- 3. Export model with model_to_mobile.ipynb
- 4. Move actor to Android\app\src\main\assets\model_x.pt
