from table_rl import learner
import numpy as np


class SARSA(learner.Learner):
    """Class that implements SARSA."""

    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate,
                 explorer,
                 discount=0.99,
                 initial_val=0.):
        self.explorer = explorer
        self.learning_rate = learning_rate
        self.q = np.full((num_states, num_actions), initial_val, dtype=float)
        self.discount = discount

    def update_q(self, obs, action, reward, terminated, next_obs, next_action):
        target = reward if terminated else reward + self.discount * self.q[next_obs, next_action] # where to get next action?
        estimate = self.q[obs, action]
        self.q[obs, action] = estimate + self.learning_rate * (target - estimate)
    
    def act(self, obs: int, train: bool) -> int:
        """Returns an integer 
        """
        if not train:
            return np.argmax(self.q[obs])
        if self.next_obs is not None:
            assert obs == self.next_obs
        if self.cached_action is None:
            action = self.explorer.select_action(q_values) if train else np.argmax(q_values)
        else:
            action = self.cached_action
        # TODO: Call update
        self.current_obs = obs
        q_values = self.q[obs]
        action = self.explorer.select_action(q_values) if train else np.argmax(q_values)
        self.last_action = action
        return action
        

    def observe(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        # self.update_q(self.current_obs, self.last_action, reward, terminated, obs)
        self.next_obs = obs
        next_obs_q_values = self.q[self.next_obs]
        self.cached_action = self.explorer.select_action(next_obs_q_values) if train else np.argmax(next_obs_q_values)
        self.explorer.observe(obs)
        if terminated or truncated:
            self.current_obs = None
            self.next_obs = None
            self.cached_next_action = None
            self.last_action = None

