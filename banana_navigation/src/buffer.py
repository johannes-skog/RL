import pandas as pd
import torch
import logging
from dataclasses import dataclass
from typing import List

EPSILON = 10**-5


@dataclass
class Qentry(object):

    state: torch.Tensor
    state_prime: torch.Tensor
    action: int
    reward: float


@dataclass
class Qcollection(object):

    def __init__(self, entries: List[Qentry]):

        self.state = torch.cat([x.state for x in entries])
        self.state_prime = torch.cat([x.state_prime for x in entries])
        self.action = torch.cat([x.action for x in entries])
        self.reward = torch.cat([x.reward for x in entries])

        self.weights = None

    def add_weights(self, weights: torch.Tensor):

        self.weights = weights


class ReplayBuffer(object):

    def __init__(self, size: int):

        self._replay = pd.DataFrame(columns=["observation", "weight", "seen", "sample_prob"])
        self._size = size
        self.log = logging.getLogger()

    def inbuffer(self):

        return len(self)

    def __len__(self):

        return len(self._replay)

    def _step_sample_prob(self, beta: float = 1):

        self._replay["sample_prob"] = self._replay["weight"] / (self._replay["seen"]**beta)

        self._replay["sample_prob"] = self._replay["sample_prob"] / self._replay["sample_prob"].sum()

    def add_bulk(self, observations: List[Qentry], weights: List[float] = None, beta: float = 1):

        N = len(observations)

        if self.inbuffer() + N >= self._size:

            self._step_sample_prob(beta=beta)

            self.log.debug(f"Replay buffer is full, removing {N} entries")
            # Not super efficient, but works for now.
            # If the buffer is going to be full when we add the new entry, remove one
            # entry from the buffer according to the weigting scheme
            self._replay = self._replay.sample(n=self._size - N, weights="sample_prob", replace=False)

        for i in range(N):

            weight = weights[i] if weights is not None else 1

            self.add(observations[i], weight, beta)

    def add(self, observation: Qentry, weight: float = 1, beta: float = 1):

        new_entry = pd.Series(
            {
                "observation": observation,
                "weight": weight,
                "seen": 1
            }
        )

        if self.inbuffer() == self._size:

            self._step_sample_prob(beta=beta)

            self.log.debug("Replay buffer is full, removing one entry")
            # Not super efficient, but works for now.
            # If the buffer is going to be full when we add the new entry, remove one
            # entry from the buffer according to the weigting scheme
            self._replay = self._replay.sample(n=self._size - 1, weights="sample_prob", replace=False)

        self._replay = self._replay.append(new_entry, ignore_index=True)

    def draw(self, size: int, replace: bool = False, prune: bool = False, beta: float = 1):

        if self.inbuffer() == 0:
            return None

        self._step_sample_prob(beta=beta)

        replay_sample = self._replay.sample(n=size, weights="sample_prob", replace=replace)

        self._replay.at[replay_sample.index, "seen"] = self._replay.loc[replay_sample.index, "seen"] + 1

        if prune:
            self._replay = self._replay.drop(replay_sample.index)

        qcollection = Qcollection(
            replay_sample.observation
        )

        qcollection.add_weights(torch.from_numpy(replay_sample["sample_prob"].astype(float).values))

        return qcollection


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    replaybuffer = ReplayBuffer(5000)

    for i in range(100):

        qentry = Qentry(
            state=torch.rand(size=[10, 10]),
            state_prime=torch.rand(size=[10, 10]),
            action=torch.rand(size=[2]),
            reward=torch.rand(size=[1]),
        )

        replaybuffer.add(
            observation=qentry,
            weight=torch.rand(size=[1]),
        )

    for i in range(12):

        replaybuffer.draw(
            size=1,
            prune=True
        )

    qq = replaybuffer.draw(
        size=10,
        prune=True
    )

    print(qq.state.shape)
    print(qq.reward.shape)
