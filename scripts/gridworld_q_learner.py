import gridworld_policy_eval
from pdb import set_trace
import gymnasium as gym
import table_rl

env = gridworld_policy_eval.BasicGridworld()

explorer = table_rl.explorers.ConstantEpsilonGreedy(0.1, 4)

agent = table_rl.learners.QLearning(15,
                  4,
                  0.1,
                  explorer,
                  discount=1.0,
                  initial_val=0.)

observation, info = env.reset()
for _ in range(10000):
    action = agent.act(observation, True)
    observation, reward, terminated, truncated, info = env.step(action)
    agent.observe(observation, action, reward, terminated, truncated)
    if terminated or truncated:
        observation, info = env.reset()

