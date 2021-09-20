import numpy as np


class OUNoise(object):
    """Generate Noise from a Ornstein–Uhlenbeck process"""

    def __init__(
        self,
        n: int,
        n_actions: int,
        mu: float = 0,
        sigma: float = 0.2,
        theta: float = 0.15,
        low: float = -1,
        high: float = 1,
    ):

        self._n = n
        self._n_actions = n_actions
        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._low = low
        self._high = high

        self.reset_noise()

    def reset_noise(self):

        self._noise = np.ones((self._n, self._n_actions)) * self._mu

    def generate(self):

        dn = self._theta * (self._mu - self._noise) + self._sigma * np.random.uniform(
            low=self._low, high=self._high, size=(self._n, self._n_actions)
        )

        self._noise = self._noise + dn

        return self._noise
