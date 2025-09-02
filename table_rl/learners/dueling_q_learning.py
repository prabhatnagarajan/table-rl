from table_rl import learner
import numpy as np
from pdb import set_trace

class DuelingQLearning(learner.Learner):
    """Class that implements Dueling Q-Learning."""

    def __init__(self,
                 num_states,
                 num_actions,
                 step_size_schedule,
                 explorer,
                 identifiability_reconciler="max",
                 discount=0.99,
                 initial_val=0.):
        self.explorer = explorer
        self.step_size_schedule = step_size_schedule
        self.adv = np.full((num_states, num_actions), initial_val, dtype=float)
        self.v = np.full(num_states, initial_val, dtype=float)
        assert identifiability_reconciler in ["max", "mean"], "Invalid identifiability_reconciler"
        self.adv_subtractor = identifiability_reconciler
        self.discount = discount

    @property
    def q(self):
        if self.adv_subtractor == "max":
            adv_adjustment = np.max(self.adv, axis=1, keepdims=True)
        else:
            adv_adjustment = np.mean(self.adv, axis=1, keepdims=True)
        return self.v + self.adv - adv_adjustment

    def _update_v(self, obs, td_error, step_size):
        v_estimate = self.v[obs]
        self.v[obs] = v_estimate + step_size * td_error

    def _update_adv(self, obs, action, td_error, step_size):
        adv_estimate = self.adv[obs, action]
        self.adv[obs, action] = adv_estimate + step_size * td_error
        if self.adv_subtractor == "mean":
            self.adv[obs] -= np.mean(self.adv[obs]) * td_error
        else:
            max_action = np.argmax(self.adv[obs])
            self.adv[obs, max_action] -= step_size * td_error

    def update_q(self, obs, action, reward, terminated, next_obs):
        target = reward if terminated else reward + self.discount * np.max(self.q[next_obs])
        estimate = self.q[obs, action]
        step_size = self.step_size_schedule.step_size(obs, action)
        td_error = target - estimate
        self._update_v(obs, td_error, step_size)
        self._update_adv(obs, action, td_error, step_size)

    def act(self, obs: int, train: bool) -> int:
        """Returns an integer 
        """
        self.current_obs = obs
        q_values = self.q[obs]
        action = self.explorer.select_action(obs, q_values) if train else np.argmax(q_values)
        self.last_action = action
        return action
        
    def observe(self, obs: int, reward: float, terminated: bool, truncated: bool, training_mode: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        self.update_q(self.current_obs, self.last_action, reward, terminated, obs)
        self.explorer.observe(obs, reward, terminated, truncated, training_mode)
        self.step_size_schedule.observe(obs, reward, terminated, truncated, training_mode)
        if terminated or truncated:
            self.current_obs = None
            self.last_action = None

