from table_rl import learner
import numpy as np


class QLearning(learner.Learner):
    """Abstract learner class."""

    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate,
                 explorer,
                 discount=0.99,
                 initial_val=0.):
        self.explorer = 
        self.learning_rate = learning_rate
        self.q = np.full((num_states, num_actions), initial_val)
        self.discount = discount

    def update_q(self, obs, action, reward, terminal):
        target = reward if terminal else reward + self.discount * np.max(self.q[obs])
        estimate = self.q[obs,action]
        self.q[obs,action] = estimate + self.learning_rate * (target - estimate)


    def act(self, obs: int, train: bool) -> int:
        """Returns an integer 
        """
        raise NotImplementedError()

    def observe(self, obs: int, action: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action.

        Returns:
            None
        """
        self.update_q()
        self.explorer.observe(obs, action, reward, terminated, truncated)

