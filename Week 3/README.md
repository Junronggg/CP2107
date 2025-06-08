## Set Up
refer to hw_setup for environment set up

## Code modified
- cs285/scripts/run_hw2.py
- cs285/agents/pg_agent.py
- cs285/networks/policies.py
- cs285/infrastructure/utils.py
- cs285/plot_batchsize.py
- cs285/plot_lb+rtg.py
- cs285/plot_sb+rtg.py

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

## Experiment results & plot graphs
- sb+no rtg: python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole --seed 10
- sb+rtg: python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg --seed 10
- lg+no rtg: python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb --seed 10
- lg+rtg: python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg --seed 10
- see effect of rtg (lb+no rtg vs lb+rtg; sb+no rtg vs sb+rtg):
  - python plot_sb+rtg.py
  - python plot_lb+rtg.py
- see effect of batch size: python plot_batchsize.py
