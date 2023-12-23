import numpy as np
from table_rl import explorer


class ConstantEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, epsilon, num_actions):
        self.epsilon = epsilon
        self.num_actions = num_actions

    def select_action(self, action_values) -> int:
        greedy = np.random.uniform() < 1 - self.epsilon
        best_action_indices = np.flatnonzero(action_values == np.max(action_values))
        action = np.random.choice(best_action_indices) if greedy else np.random.choice(self.num_actions)
        return action

    def observe(self, obs):
        """Observes and observation and updates internal state

        Args:
          obs: state
        """
        pass


class LinearDecayEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with linear decay of epsilon.

    Args:
      epsilon_init: float indicating the value of epsilon
      epsilon_end: float indicating the final value of epsilon
      decay_steps: number of timesteps over which to decay epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, epsilon_init, epsilon_end, decay_steps, num_actions):
        assert 0 <= epsilon_init <= 1
        assert 0 <= epsilon_end <= 1
        assert epsilon_init >= epsilon_end
        self.epsilon_init = epsilon_init
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_init
        self.decay_value = (self.epsilon_init - self.epsilon_end) / decay_steps
        self.num_actions = num_actions

    def select_action(self, action_values) -> int:
        greedy = np.random.uniform() < 1 - self.epsilon
        best_action_indices = np.flatnonzero(action_values == np.max(action_values))
        action = np.random.choice(best_action_indices) if greedy else np.random.choice(self.num_actions)
        return action

    def observe(self, obs):
        """Observes an observation and updates internal state

        Args:
          obs: state
        """
        self.epsilon = max(self.epsilon_end, self.epsilon - self.decay_value)


class PercentageDecayEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with epsilon decaying by a percentage

    Args:
      epsilon_init: float indicating the value of epsilon
      decay_percentage: float indicating decay multiplier
      num_actions: integer indicating the number of actions
    """

    def __init__(self, epsilon_init, min_epsilon, decay_percentage, num_actions):
        self.epsilon = epsilon_init
        self.min_epsilon = min_epsilon
        assert 0 <= decay_percentage <= 1.
        self.decay_percentage = decay_percentage
        self.num_actions = num_actions

    def select_action(self, action_values) -> int:
        greedy = np.random.uniform() < 1 - self.epsilon
        best_action_indices = np.flatnonzero(action_values == np.max(action_values))
        action = np.random.choice(best_action_indices) if greedy else np.random.choice(self.num_actions)
        return action

    def observe(self, obs):
        """Observes an observation and updates internal state

        Args:
          obs: state
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_percentage)

