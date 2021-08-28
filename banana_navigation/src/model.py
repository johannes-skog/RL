from unityagents import UnityEnvironment
import torch
import numpy as np
import random
import copy

from typing import (
    Any,
    Dict
)

from buffer import (
    ReplayBuffer,
    Qentry,
)


class Qnet(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int, lr: float):

        super(Qnet, self).__init__()

        self._n_actions = out_dim

        self._model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.ELU(),
            torch.nn.Linear(32, out_dim),
        )

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

    def forward(self, s: torch.Tensor):
        """Forward the state through the network, return the qvalues and the greedy action"""

        qvalues = self._model.forward(s)
        action = torch.argmax(qvalues, dim=1)

        return action, qvalues

    def forward_epsilon(self, s: torch.Tensor, epsilon: float = 0.5):
        """Forward the state through the network, return the qvalues and the epsilon-greedy action"""

        bsize = s.shape[0]

        action, qvalues = self.forward(s)

        if random.random() < epsilon:
            action = torch.from_numpy(np.random.choice(np.arange(self._n_actions), size=bsize))

        return action, qvalues

    def copy(self):
        """Do a deepcopy of the network"""

        return copy.deepcopy(self)

    # qnet is another instance of Qnet
    def steal_weights(self, qnet: Any, tau: float = 1):
        """Steal the weights of qnet and use in this instance of Qnet, tau specifies how big part of qnet's we
        want to use in this instance"""

        for name_from, W_from in qnet.named_parameters():
            for name_to, W_to in self.named_parameters():
                if name_from == name_to:
                    W_to.data.copy_((1 - tau) * W_to.data + tau * W_from.data)

    def loss(
        self,
        actions: torch.Tensor,
        qvalues: torch.Tensor,
        qvalues_target: torch.Tensor,
        weights: torch.Tensor,
        reward: torch.Tensor,
        gamma: float,
    ):
        """Calculate the q-learning loss"""

        mask = torch.zeros_like(qvalues_target, dtype=torch.long)
        mask.scatter_(dim=1, index=actions.long().unsqueeze(1), src=torch.ones_like(actions).unsqueeze(1))

        loss = (reward + (gamma * torch.max(qvalues_target, 1)[0] - qvalues[mask == 1]))**2

        loss = loss * weights

        return loss.mean()

    def update(self, loss: torch.Tensor):
        """Update the network"""

        self.zero_grad()

        loss.backward()

        self._optimizer.step()


def _state_to_torch(state: np.array):
    """Convert a state vector returned by unity env to a torch tensor"""

    state = torch.from_numpy(state).float().unsqueeze(0)

    return state


def inference_episode_test(
        meta: Dict[str, Any],
        env: UnityEnvironment,
        qnet: Qnet,
        train_mode: bool = True
):
    """Run one episode of the env, store each state, action, reward tuple to replaybuffer"""

    scores = []

    for iterations in range(meta["iterations"]):

        # initialize the score
        score = 0

        env_info = env.reset(train_mode=train_mode)[meta["brain_name"]]  # reset the environment
        state = _state_to_torch(env_info.vector_observations[0])   # get the current state

        # Take an greedy action, get the qvalues
        action, qvalues = qnet.forward(state)

        # Send the action to the environment
        env_info = env.step(action.item())[meta["brain_name"]]

        # get the next state and covert it to a tensor
        state = _state_to_torch(env_info.vector_observations[0])

        while 1:

            # Take an greedy action, get the qvalues
            action, qvalues_prime = qnet.forward(state)

            # Take step
            # send the action to the environment
            env_info = env.step(action.item())[meta["brain_name"]]
            # get the next state and covert it to a tensor
            state_prime = _state_to_torch(env_info.vector_observations[0])

            # Get the reward
            reward_prime = env_info.rewards[0]

            state = state_prime

            # Keep track of the score for the episode
            score += reward_prime

            if env_info.local_done[0]:
                break

        scores.append(score)

    return np.array(scores).mean()


def inference_episode(
        meta: Dict[str, Any],
        env: UnityEnvironment,
        qnet: Qnet,
        replaybuffer: ReplayBuffer,
):
    """Run one episode of the env, store each state, action, reward tuple to replaybuffer"""

    # initialize the score
    score = 0
    # initialize the hits, i.e. number of bananas
    hits = 0

    env_info = env.reset(train_mode=True)[meta["brain_name"]]  # reset the environment
    state = _state_to_torch(env_info.vector_observations[0])   # get the current state

    # Take an epsilon-greedy action, get the qvalues
    action, qvalues = qnet.forward_epsilon(state, meta["epsilon"])

    # Send the action to the environment
    env_info = env.step(action.item())[meta["brain_name"]]

    # get the next state and covert it to a tensor
    state = _state_to_torch(env_info.vector_observations[0])

    qentris, weights = [], []

    while 1:

        action, qvalues_prime = qnet.forward_epsilon(state, meta["epsilon"])

        # Take step
        # send the action to the environment
        env_info = env.step(action.item())[meta["brain_name"]]
        # get the next state and covert it to a tensor
        state_prime = _state_to_torch(env_info.vector_observations[0])

        # Get the reward
        reward_prime = env_info.rewards[0]

        # Set a default reward that is different from zero
        reward_prime_mod = meta["reward_default"] if reward_prime == 0 else reward_prime

        # Prepare the entry that will be stored in the replay buffer
        qentry = Qentry(
            state=state,
            state_prime=state_prime,
            action=action,
            reward=torch.from_numpy(np.array([reward_prime_mod])),
        )

        qentris.append(qentry)

        # Temporal difference error between current state's q-value and the next.
        td_error = abs(
            reward_prime_mod + (meta["gamma"] * qvalues_prime.max().item() - qvalues[0, action].item())
        )

        weights.append(td_error)

        qvalues = qvalues_prime
        state = state_prime

        # Keep track of gotten bannanas during the episode
        hits += abs(reward_prime)
        # Keep track of the score for the episode
        score += reward_prime

        if env_info.local_done[0]:
            break

    # Add all entries that were produced during the episode
    replaybuffer.add_bulk(qentris, weights, beta=meta["beta"])

    return replaybuffer, score, hits


def train_episode(meta: Dict[str, Any], qnet_target: Qnet, qnet: Qnet, replaybuffer: ReplayBuffer):
    """Train the model using the stored experiences"""

    for i in range(meta["update_steps"]):

        batch = replaybuffer.draw(
            meta["batch_size"],
            replace=meta["replace_sampling"],
            prune=False,
            beta=meta["beta"]
        )

        _, qvalues_target = qnet_target.forward(batch.state_prime)
        _, qvalues = qnet.forward(batch.state)

        loss = qnet.loss(
            qvalues=qvalues,
            actions=batch.action,
            reward=batch.reward.float(),
            weights=((1/batch.weights) * (1 / len(replaybuffer))).float(),
            qvalues_target=qvalues_target.detach(),
            gamma=meta["gamma"],
        )

        qnet.update(loss)

    return qvalues_target, batch.action


if __name__ == "__main__":

    qnet = Qnet(100, 4)

    batch_size = 10

    s = torch.from_numpy(np.random.uniform(size=[batch_size, 100])).float()

    actions = qnet.forward_epsilon(s)

    actions_greedy = qnet.greedy(actions)

    qnet_copy = Qnet(100, 4)

    for name_from, W_from in qnet_copy.named_parameters():
        for name_to, W_to in qnet.named_parameters():
            if name_from == name_to:
                print(name_from, (W_to.data == W_from.data).float().mean())

    qnet.steal_weights(qnet_copy)

    for name_from, W_from in qnet_copy.named_parameters():
        for name_to, W_to in qnet.named_parameters():
            if name_from == name_to:
                print(name_from, (W_to.data == W_from.data).float().mean())
