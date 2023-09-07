import numpy as np
import table_rl
import table_rl.dp.dp as dp


env = table_rl.env.BasicGridworld()
policy = np.full((15, 4), 0.25)
value_function = dp.policy_v_evaluation(policy, env.R, env.T, 1.0, 10000)
optimal_values = dp.value_iteration(15, 4, env.R, env.T, 1.0, 10000)

