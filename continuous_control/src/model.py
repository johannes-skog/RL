from typing import Any
import torch
import copy


class ResBlock(torch.nn.Module):

    def __init__(self, in_dim: int, inner_dim: int):

        super().__init__()

        self._block = torch.nn.Sequential(
            torch.nn.Linear(in_dim, inner_dim),
            torch.nn.LayerNorm(inner_dim),
            torch.nn.ELU(),
            torch.nn.Linear(inner_dim, in_dim),
            torch.nn.LayerNorm(in_dim),
        )

    def forward(self, x):

        y = self._block(x) + x

        return y


class GenericModel(torch.nn.Module):
    """The Generic model architecture used by the actor and critic"""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, squeeze_dim: int):

        super().__init__()
        layers = [
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, squeeze_dim),
            torch.nn.LayerNorm(squeeze_dim),
            torch.nn.ELU(),
            torch.nn.Linear(squeeze_dim, squeeze_dim),
            torch.nn.LayerNorm(squeeze_dim),
            torch.nn.ELU(),
            torch.nn.Linear(squeeze_dim, out_dim)
        ]

        self._model = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Do inference"""
        y = self._model(x)

        return y

    def copy(self):
        """Do a deepcopy of the network"""
        return copy.deepcopy(self)

    def steal_weights(self, model: Any, tau: float = 0):
        """Steal the weights of qnet and use in this instance of Qnet, tau specifies how big part of qnet's we
        want to use in this instance"""

        assert type(model) == type(self)

        for name_from, W_from in model.named_parameters():
            for name_to, W_to in self.named_parameters():
                if name_from == name_to:
                    W_to.data.copy_((1 - tau) * W_to.data + tau * W_from.data)


class Actor(GenericModel):
    """Actor network"""

    def __init__(
        self, in_dim: int,
        out_dim: int,
        hidden_dim: int,
        squeeze_dim: int,
    ):

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            squeeze_dim=squeeze_dim,
        )

    def forward(self, s: torch.Tensor):
        """Do inference"""

        y = super().forward(s)

        y = torch.tanh(y)

        return y


class Critic(GenericModel):
    """Critic network"""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, squeeze_dim: int):

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            squeeze_dim=squeeze_dim,
        )


class DDPG(torch.nn.Module):

    def __init__(
        self,
        critic: Critic,
        actor: Actor,
        critic_target: Critic,
        actor_target: Actor,
        tau_critic: float = 10**-3,
        tau_actor: float = 10**-3,
        lr_critic: float = 10**-4,
        lr_actor: float = 10**-4,
    ):

        super().__init__()

        self._critic = critic
        self._actor = actor

        self._critic_target = critic_target
        self._actor_target = actor_target

        self._optimizer_critic = torch.optim.Adam(
                self._critic.parameters(),
                lr=lr_critic
        )

        self._optimizer_actor = torch.optim.Adam(
                self._actor.parameters(),
                lr=lr_actor
        )

        self._tau_critic = tau_critic
        self._tau_actor = tau_actor

    def _forward(self, s: torch.Tensor, actor: Actor, critic: Critic, a: torch.Tensor = None):
        """Do inference w/ the networks if the action a is supplied, used that one as input to the critic,
        otherwise do inference on the actor network to generate the action
        """

        a = actor(s) if a is None else a

        sa = torch.cat((a, s), 1)

        q = critic(sa)

        return a, q

    def forward(self, s: torch.Tensor, a: torch.Tensor = None):
        """Do inference w/ the networks"""
        return self._forward(s=s, actor=self._actor, critic=self._critic, a=a)

    def forward_target(self, s: torch.Tensor, a: torch.Tensor = None):
        """Do inference w/ the target networks"""

        return self._forward(s=s, actor=self._actor_target, critic=self._critic_target, a=a)

    def loss(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        d: torch.Tensor,
        w: torch.Tensor,
        sprime: torch.Tensor,
        gamma: float,
    ):
        """Calculate the loss for the actor and critic network, also, calculat the new td error for the experience tuples
        """

        # --------------- Critic loss ---------------

        # Calculate the q-value using the stored action/state tuple
        _, q = self.forward(s, a)
        _, qprime = self.forward(sprime)

        # Calculate the q-value using the stored state after taking action a in state s
        _, q_target = self.forward_target(sprime)

        q, q_target, qprime = torch.squeeze(q), torch.squeeze(q_target),  torch.squeeze(qprime)

        # td error used to update the weight in replay buffer
        td_error = torch.abs(r + gamma * qprime - q)

        # Td loss
        cl = (w * ((q - (r + (gamma * (1-d) * q_target)))**2)).mean()

        # --------------- Actor loss ---------------

        # Calculate the q-value of taking the optimal action (according to the actor) when in state s
        a_opt, q = self.forward(s)

        q = torch.squeeze(q)

        al = (- w * q).mean()

        return cl, al, td_error

    def update(self, cl: torch.Tensor, al: torch.Tensor, grad_clip: float = 1):
        """Update the networks doing backpropagation on the losses"""

        self._critic.zero_grad()
        cl.backward(inputs=list(self._critic.parameters()), retain_graph=True)

        # clear out the grad from the cl loss
        self._actor.zero_grad()
        al.backward(inputs=list(self._actor.parameters()))

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm(self._critic.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm(self._actor.parameters(), grad_clip)

        self._optimizer_critic.step()
        self._optimizer_actor.step()

    def update_target_networks(self):
        """Update the target networks"""

        self._critic_target.steal_weights(
            self._critic,
            self._tau_critic
        )

        self._actor_target.steal_weights(
            self._actor,
            self._tau_actor,
        )
