import numpy as np
from table_rl import explorer


class GreedyExplorer(explorer.Explorer):
    """Explorer that takes the greedy action always.

    Args:
      num_actions: number of actions
    """

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def select_action(self, action_values) -> int:
        greedy = np.random.uniform() < 1 - self.epsilon
        best_action_indices = np.flatnonzero(action_values == np.max(action_values))
        action = np.random.choice(best_action_indices) if greedy else np.random.choice(self.num_actions)
        return action

    def observe(self, obs):
        """Select an action.

        Args:
          obs: Q-values
        """
        pass