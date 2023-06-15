import gymnasium as gym
from pdb import set_trace
import table_rl

from table_rl.learners import QLearning
from table_rl.explorers import ConstantEpsilonGreedy
env = gym.make('Taxi-v3')

num_actions = env.action_space.n
num_states = env.observation_space.n

explorer = ConstantEpsilonGreedy(0.2, num_actions)
learner = QLearning(num_states, num_actions, 0.01, explorer, discount=0.99, initial_val=0.)
obs, info = env.reset()

action = learner.act(obs, True)
set_trace()