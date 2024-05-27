
def assert_policy_shape(policy):
    assert len(policy.shape) == 2

def check_valid_policy(policy):
    assert_policy_shape(policy)
    states = range(policy.shape[0])
    actions = range(policy.shape[1])
    for s in states:
        np.sum(policy[s]) == 1.0
        for a in actions:
            assert 0 <= policy[s,a]<= 1.0
