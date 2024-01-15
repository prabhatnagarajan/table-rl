import numpy as np
import scipy
from table_rl import explorer


class SoftmaxExploration(explorer.Explorer):
    """Softmax or Boltzmann exploration.

    Args:
      num_actions: integer indicating the number of actions
    """

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def select_action(self, preferences) -> int:
        probabilities = scipy.special.softmax(preferences)
        return np.random.choice()

    def observe(self, obs):
        pass
