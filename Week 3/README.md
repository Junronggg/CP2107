## Set Up
refer to hw_setup for environment set up

## Code modified
- cs285/scripts/run_hw2.py
- cs285/agents/pg_agent.py
- cs285/networks/policies.py
- cs285/infrastructure/utils.py
- cs285/scripts/plot_learning_curves.py

## Code explaination
- pg_agent.py: handles trajectory collection + calls policy update
  Manages policy training and trajectory collection
  Calls the policy’s update function using computed returns (reward-to-go or not)
  Stores collected experience (observations, actions, rewards)
  Handles training logic including Advantage computation & Value normalization (did not implement)
  - train(): Collects rollouts, computes Q-values, trains the policy.
  - calculate_q_vals(): Computes either total return or reward-to-go.
  - estimate_advantage(): (if advantage-based methods used, later HWs)
    
- policies.py: implements the neural network policy and gradient update logic
  Defines the policy network (actor)
  Implements the update() method
  Computes the policy gradient loss
  - get_action(obs): samples an action
  - update(observations, actions, q_vals): trains the policy via gradient ascent
    
- run_hw2.py: sets up everything and start training (main function)
  Parses command-line arguments
  Sets up: Environment using gym.make(), Agent (instance of class PGAgent), Logging via Logger, Starts the training loop
  - train_PG(): the main training loop used by run_hw2.py
  - Loops over iterations
  - Collects trajectories
  - Calls agent.train() and logs results

- utils.py: builds models, processes rollouts, computes helper stats
