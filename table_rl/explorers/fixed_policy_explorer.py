import numpy as np
from table_rl import explorer
from table_rl import utils

class FixedPolicyExplorer(explorer.Explorer):
    """Epsilon-greedy with epsilon decaying by a percentage

    Args:
      policy: a 2d numpy array representing the policy
    """

    def __init__(self, policy):
        utils.check_valid_policy(policy)
        self.policy = policy
        self.num_actions = self.policy.shape[1]


    def select_action(self, action_values) -> int:
        return np.random.choice(self.num_actions, p=self.policy[self.observation])

    def observe(self, obs, reward, terminated, truncated):
        """Select an action.

        Args:
          obs: next state/observation
          reward: reward received
          terminated: bool indicating environment termination
          truncated: bool indicating epsisode truncation
        """
        self.observation = obs 
