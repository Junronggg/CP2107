import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            """
            If discrete == True:
            The agent is acting in a discrete action space, like {left, right}, {0,1,2}
            For discrete actions: the policy outputs a categorical distribution (i.e., logits over possible actions)
            """
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            """
            For continuous actions, the policy outputs a Gaussian distribution (mean and standard deviation for each action dimension)
            """
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        if obs.ndim == 1:
            obs = obs[None]  # Convert to shape (1, ob_dim) for batching
        """
        Ensure the observation is 2D so it can be processed in a batch (even if the batch size is 1).
        If the input obs has shape (ob_dim,) 
        (i.e. it's a single observation like [0.1, -0.2, 0.05, 0.3]), this line reshapes it to (1, ob_dim), i.e. [[0.1, -0.2, 0.05, 0.3]].
        This is necessary because PyTorch models expect input with shape [batch_size, features].
        """
        obs_tensor = ptu.from_numpy(obs)  
        # Convert the NumPy array into a PyTorch tensor and move it to the correct device (CPU or GPU)
        action_distribution = self.forward(obs_tensor)  
        # Pass the observation through the policy network to get the action distribution.
        # Discrete case: returns a Categorical distribution over actions.
        # Continuous case: returns a Normal distribution with mean and std.
        action_sample = action_distribution.sample()  
        # Sample an action
        return ptu.to_numpy(action_sample[0])  
        # Convert to numpy, remove batch dimension

    def forward(self, obs: torch.FloatTensor):
        # defines how policy maps an observation to a distribution over actions
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            return distributions.Categorical(logits=logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            return distributions.Normal(mean, std)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        # Get the distribution over actions from the current policy
        distribution = self.forward(obs)  # Returns torch.distributions.Categorical for discrete

        # Compute log probabilities of the taken actions under current policy
        log_probs = distribution.log_prob(actions)  # Shape: (batch_size,)

        # Compute the policy gradient loss (negative of weighted log-probs)
        # We maximize expected reward => minimize negative
        loss = -(log_probs * advantages).mean()

        # Backprop and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
