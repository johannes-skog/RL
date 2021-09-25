from typing import List
import logging
import numpy as np


class ReplayBuffer(object):
    """Replaybuffer for holding experience tuples"""

    def __init__(self, size: int, state_size: int, action_size: int):

        self._state = np.zeros(shape=(size, state_size), dtype=np.float)
        self._state_prime = np.zeros(shape=(size, state_size), dtype=np.float)
        self._action = np.zeros(shape=(size, action_size), dtype=np.float)
        self._reward = np.zeros(shape=size, dtype=np.float)
        self._weight = np.zeros(shape=size, dtype=np.float)
        self._done = np.zeros(shape=size, dtype=np.float)

        self._size = size
        self.log = logging.getLogger()

        self._seen = np.ones(shape=size, dtype=np.int)

        self._index = []

    def __len__(self):
        """Return the number of entries in the buffer"""

        return len(self._index)

    def _step_sample_prob(self, beta: float = 1, inverse=False):
        """Convert the weights to sample probabilies"""

        sample_prob = np.zeros_like(self._weight)

        weights = self._weight[:]

        if inverse:

            weights = np.abs(weights - weights.max()*0.999)

            sample_prob = (weights * (self._seen**beta))[np.array(self._index)]

        else:

            sample_prob = (weights / (self._seen**beta))[np.array(self._index)]

        sample_prob = sample_prob / sample_prob.sum()

        return sample_prob

    def add(
        self,
        state: np.array,
        state_prime: np.array,
        action: np.array,
        reward: np.array,
        done: np.array,
        weight: np.array = None,
        beta: float = 1,
    ):
        """Add experience tuples to the buffer"""

        assert state.shape[0] == state_prime.shape[0] == action.shape[0] == reward.shape[0] == done.shape[0]

        N = state.shape[0]

        if N + len(self) < self._size:

            _ind = [len(self) + i for i in range(N)]

            self._index.extend(_ind)

            indices = np.array(_ind)

        else:

            sampling_prob = self._step_sample_prob(beta, inverse=True)

            indices = np.random.choice(len(self), N, p=sampling_prob, replace=False)

        mask = np.zeros(self._size, dtype=bool)
        mask[indices] = True

        self._state[mask, :] = state
        self._state_prime[mask, :] = state_prime
        self._action[mask, :] = action
        self._reward[mask] = reward
        self._done[mask] = done
        self._seen[mask] = 1
        self._weight[mask] = weight if weight is not None else 1

    def update_weight(self, indices: List[int], weight: np.array, alpha: float = 0.5):
        """Update the experience tuple weights"""

        mask = np.zeros(self._size, dtype=bool)
        mask[indices] = True

        self._weight[mask] = self._weight[mask] * alpha + weight * (1 - alpha)

    def draw(self, size: int, replace: bool = False,  beta: float = 1, beta_2: float = 1):
        """Sample experience tuples from the buffer"""

        if len(self) < size:
            return [None] * 7

        # Convert the weights to sample probabilies
        sampling_prob = self._step_sample_prob(beta=beta) ** beta_2

        sampling_prob = sampling_prob / sampling_prob.sum()

        # Sample size number of entries according to the sample_prob
        indices = np.random.choice(len(self), size, p=sampling_prob, replace=replace)

        self._seen[indices] += 1

        return (
            self._state[indices, :],
            self._state_prime[indices, :],
            self._action[indices, :],
            self._reward[indices],
            self._done[indices],
            sampling_prob[indices],
            indices,
        )


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    replaybuffer = ReplayBuffer(
        5000,
        state_size=30,
        action_size=4,
    )

    import time
    start = time.time()

    for i in range(5000):

        state = np.random.rand(20, 30)
        state_prime = np.random.rand(20, 30)
        action = np.random.rand(20, 4)
        reward = np.random.rand(20)
        weight = np.random.rand(20)
        done = np.random.rand(20)

        replaybuffer.add(
            state=state,
            state_prime=state_prime,
            reward=reward,
            action=action,
            weight=weight,
            done=done
        )

        print(len(replaybuffer))

    print(time.time() - start)

    for i in range(12):

        replaybuffer.draw(
            size=1,
        )

    print(len(replaybuffer))
