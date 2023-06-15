import numpy as np
from table_rl import explorer
from pdb import set_trace


class ConstantEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      logger: logger used
    """

    def __init__(self, epsilon, num_actions):
        self.epsilon = epsilon
        self.num_actions = num_actions

    def select_action(self, action_values) -> int:
        greedy = np.random.uniform() < 1 - self.epsilon
        action = np.argmax(action_values) if greedy else np.random.choice(self.num_actions)
        return action

    def observe(self, obs):
        """Select an action.

        Args:
          obs: Q-values
        """
        pass
