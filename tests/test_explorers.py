import pytest
import numpy as np
import math
import table_rl
import table_rl.dp.dp as dp

class TestEpsilonGreedyExplorers:

    def test_constant_epsilon_greedy(self):
        explorer = table_rl.explorers.ConstantEpsilonGreedy(0.1, 4)
        action_vals = np.array([1.0, 4.0, 4.0, 3.0])
        actions = [explorer.select_action(action_vals) for _ in range(500)]
        assert 1 in actions
        assert 2 in actions
        explorer = table_rl.explorers.ConstantEpsilonGreedy(0., 4)
        actions = [explorer.select_action(action_vals) for _ in range(500)]
        assert 0 not in actions, actions
        assert 3 not in actions
        action = explorer.select_action(np.array([1.0, 2.0, 4.0, 3.0]))
        assert action == 2

    def test_linear_decay_epsilon_greedy(self):
        explorer = table_rl.explorers.LinearDecayEpsilonGreedy(1.0, 0.1, 9, 4)
        expected_epsilons = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1]
        # TODO: Confirm all epsilons each step


    def test_pct_decay_epsilon_greedy(self):
        explorer = table_rl.explorers.PercentageDecayEpsilonGreedy(1.0, 0.2, 0.9, 4)
