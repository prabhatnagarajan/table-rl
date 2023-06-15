import gymnasium as gym
from pdb import set_trace
import table_rl

from table_rl.learners import QLearning
from table_rl.explorers import ConstantEpsilonGreedy
env = gym.make('CliffWalking-v0')

num_actions = env.action_space.n
num_states = env.observation_space.n

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
		cumulative_reward += reward
		learner.observe(observation, action, reward, terminated, truncated)
		if timestep % eval_interval:
			eval_scores = eval_loop(env, learner, 1)
			print(eval_scores)
	return scores

explorer = ConstantEpsilonGreedy(0.2, num_actions)
learner = QLearning(num_states, num_actions, 0.01, explorer, discount=0.99, initial_val=1.)

scores = training_loop(env, learner, 1000000, 50000)
set_trace()
