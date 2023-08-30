import gymnasium
from pdb import set_trace
import table_rl

from table_rl.learners import QLearning
from table_rl.explorers import ConstantEpsilonGreedy

class GridWorld(gymnasium.Env):

	def __init__(self, grid_length, max_episode_len=200):
		self.action_space = gymnasium.spaces.Discrete(4)
		self.observation_space = gymnasium.spaces.Discrete(grid_length * grid_length)
		self.grid_length = grid_length
		self.goal_x = grid_length - 1
		self.goal_y = grid_length - 1
		self.max_episode_len = max_episode_len

	def step(self, action):
		if action == 0: # left
			self.x = max(0, self.x - 1)
		elif action == 1: # right
			self.x = min(self.grid_length - 1, self.x + 1)
		elif action == 2: # up
			self.y = max(0, self.y - 1)
		elif action == 3: # down
			self.y = min(self.grid_length - 1, self.y + 1)
		observation = self.flatten_state(self.x, self.y)
		terminated = self.x == self.goal_x and self.y == self.goal_y
		reward = 1. if terminated else 0.
		truncated = self.timestep == self.max_episode_len
		info = {}
		self.timestep += 1
		return observation, reward, terminated, truncated, info


	def flatten_state(self, x, y):
		return x * self.grid_length + y


	def reset(self):
		self.timestep = 0
		self.x = 0
		self.y = 0
		return self.flatten_state(self.x, self.y), {}


def eval_loop(env, learner, num_eval_episodes):
	scores = []
	obs, info = env.reset()
	terminated = False
	truncated = False
	cumulative_reward = 0
	episodes = 0
	while episodes < num_eval_episodes:
		if terminated or truncated:
			episodes += 1
			scores.append(cumulative_reward)
			obs, info = env.reset()
			cumulative_reward = 0
			terminated = False
			truncated = False
		action = learner.act(obs, False)
		observation, reward, terminated, truncated, info = env.step(action)
		cumulative_reward += reward
	return scores

def training_loop(env, learner, steps, eval_interval):
	scores = []
	obs, info = env.reset()
	terminated = False
	truncated = False
	cumulative_reward = 0
	for timestep in range(steps):
		if terminated or truncated:
			scores.append(cumulative_reward)
			obs, info = env.reset()
			cumulative_reward = 0
			terminated = False
			truncated = False
		action = learner.act(obs, True)
		observation, reward, terminated, truncated, info = env.step(action)
		if reward != 0:
			print((observation, reward, terminated, truncated, info))
		cumulative_reward += reward
		learner.observe(observation, action, reward, terminated, truncated)
		# if timestep % eval_interval == 0:
		# 	eval_scores = eval_loop(env, learner, 1)
		# 	print(eval_scores)
	return scores

# env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env = GridWorld(10, max_episode_len=200)

num_actions = env.action_space.n
num_states = env.observation_space.n


explorer = ConstantEpsilonGreedy(0.2, num_actions)
learner = QLearning(num_states, num_actions, 0.01, explorer, discount=0.99, initial_val=1.)

scores = training_loop(env, learner, 900000, 5000)
set_trace()
