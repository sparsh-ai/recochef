import numpy as np

def _build_transition_matrix(num_items,
                             concentration_parameter,
                             random_state,
                             atol=0.001):

    def _is_doubly_stochastic(matrix, atol):

        return (np.all(np.abs(1.0 - matrix.sum(axis=0)) < atol) and
                np.all(np.abs(1.0 - matrix.sum(axis=1)) < atol))

    transition_matrix = random_state.dirichlet(
        np.repeat(concentration_parameter, num_items),
        num_items)

    for _ in range(100):

        if _is_doubly_stochastic(transition_matrix, atol):
            break

        transition_matrix /= transition_matrix.sum(axis=0)
        transition_matrix /= transition_matrix.sum(1)[:, np.newaxis]

    return transition_matrix


def _generate_sequences(num_steps,
                        transition_matrix,
                        order,
                        random_state):

    elements = []

    num_states = transition_matrix.shape[0]

    transition_matrix = np.cumsum(transition_matrix,
                                  axis=1)

    rvs = random_state.rand(num_steps)
    state = random_state.randint(transition_matrix.shape[0], size=order,
                                 dtype=np.int64)

    for rv in rvs:

        row = transition_matrix[state].mean(axis=0)
        new_state = min(num_states - 1,
                        np.searchsorted(row, rv))

        state[:-1] = state[1:]
        state[-1] = new_state

        elements.append(new_state)

    return np.array(elements, dtype=np.int32)


def generate_sequential(num_users=100,
                        num_items=1000,
                        num_interactions=10000,
                        concentration_parameter=0.1,
                        order=3,
                        random_state=None):
    
    if random_state is None:
        random_state = np.random.RandomState()

    transition_matrix = _build_transition_matrix(
        num_items - 1,
        concentration_parameter,
        random_state)

    user_ids = np.sort(random_state.randint(0,
                                            num_users,
                                            num_interactions,
                                            dtype=np.int32))
    item_ids = _generate_sequences(num_interactions,
                                   transition_matrix,
                                   order,
                                   random_state) + 1
    timestamps = np.arange(len(user_ids), dtype=np.int32)
    ratings = np.ones(len(user_ids), dtype=np.float32)

    return (user_ids, item_ids, ratings, timestamps, num_users, num_items)