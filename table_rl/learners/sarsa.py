from table_rl import learner
import numpy as np

class SarsaLambda(learner.Learner):
    """Class that implements SARSA(λ)."""

    def __init__(self,
                 num_states,
                 num_actions,
                 step_size_schedule,
                 explorer,
                 discount=0.99,
                 trace_lambda=0.9,
                 initial_val=0.):
        self.explorer = explorer
        self.step_size_schedule = step_size_schedule
        self.q = np.full((num_states, num_actions), initial_val, dtype=float)
        self.e = np.zeros((num_states, num_actions), dtype=float)  # Eligibility traces
        self.discount = discount
        self.trace_lambda = trace_lambda
        self.next_obs = None
        self.next_action = None


    def update_q(self, obs, action, reward, terminated, next_obs, next_action):
        if terminated:
            target = reward
        else:
            target = reward + self.discount * self.q[next_obs, next_action]
        estimate = self.q[obs, action]
        delta = target - estimate
        step_size = self.step_size_schedule.step_size(obs, action)
        self.e *= self.discount * self.trace_lambda  # Decay all eligibility traces
        self.e[obs, action] += 1  # Increment eligibility trace for the current state-action pair
        self.q += step_size * delta * self.e  # Update all Q-values based on eligibility traces
        self.e *= self.discount * self.trace_lambda  # Decay eligibility traces


    def act(self, obs: int, train: bool) -> int:
        """Returns an integer 
        """
        if not train:
            return np.argmax(self.q[obs])
        if self.next_obs is not None:
            assert obs == self.next_obs
        q_values = self.q[obs]
        if self.next_action is None:
            action = self.explorer.select_action(obs, q_values) if train else np.argmax(q_values)
        else:
            action = self.next_action
        self.current_obs = obs
        self.action = action
        return action

    def observe(self, obs: int, reward: float, terminated: bool, truncated: bool, training_mode: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        self.next_obs = obs
        next_obs_q_values = self.q[self.next_obs]
        # obs, action, reward, terminated, next_obs, next_action
        if terminated:
            self.next_action = None
        else:
            self.next_action = self.explorer.select_action(self.next_obs, next_obs_q_values) if training_mode else np.argmax(next_obs_q_values)
        self.update_q(self.current_obs, self.action, reward, terminated, obs, self.next_action)
        self.explorer.observe(obs, reward, terminated, truncated, training_mode)
        self.step_size_schedule.observe(obs, reward, terminated, truncated, training_mode)
        if terminated or truncated:
            self.current_obs = None
            self.next_obs = None
            self.next_action = None
            self.action = None


class Sarsa(learner.Learner):
    """Class that implements Sarsa."""

    def __init__(self,
                 num_states,
                 num_actions,
                 step_size_schedule,
                 explorer,
                 discount=0.99,
                 initial_val=0.):
        self.explorer = explorer
        self.step_size_schedule = step_size_schedule
        self.q = np.full((num_states, num_actions), initial_val, dtype=float)
        self.discount = discount
        self.next_obs = None
        self.next_action = None

    def update_q(self, obs, action, reward, terminated, next_obs, next_action):
        if terminated:
            target = reward
        else:
            target = reward + self.discount * self.q[next_obs, next_action]
        estimate = self.q[obs, action]
        step_size = self.step_size_schedule.step_size(obs, action)
        self.q[obs, action] = estimate + step_size * (target - estimate)
    
    def act(self, obs: int, train: bool) -> int:
        """Returns an integer 
        """
        if not train:
            return np.argmax(self.q[obs])
        if self.next_obs is not None:
            assert obs == self.next_obs
        q_values = self.q[obs]
        if self.next_action is None:
            action = self.explorer.select_action(obs, q_values) if train else np.argmax(q_values)
        else:
            action = self.next_action
        self.current_obs = obs
        self.action = action
        return action
        

    def observe(self, obs: int, reward: float, terminated: bool, truncated: bool, training_mode: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        self.next_obs = obs
        next_obs_q_values = self.q[self.next_obs]
        # obs, action, reward, terminated, next_obs, next_action
        if terminated:
            self.next_action = None
        else:
            self.next_action = self.explorer.select_action(self.next_obs, next_obs_q_values) if training_mode else np.argmax(next_obs_q_values)
        self.update_q(self.current_obs, self.action, reward, terminated, obs, self.next_action)
        self.explorer.observe(obs, reward, terminated, truncated, training_mode)
        self.step_size_schedule.observe(obs, reward, terminated, truncated, training_mode)
        if terminated or truncated:
            self.current_obs = None
            self.next_obs = None
            self.next_action = None
            self.action = None

