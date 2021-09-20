import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import DDPG

from buffer import ReplayBuffer
from noise import OUNoise


def train(
    env,
    ddpg: DDPG,
    replaybuffer: ReplayBuffer,
    ounoise: OUNoise,
    writer: SummaryWriter,
    episodes: int,
    brain_name: str,
    device: str,
    batch_size: int,
    gamma: float,
    sigma_init: float,
    sigma_end: float,
    sigma_decay: float,
    scale_reward: float = 1,
    evaluation_rounds: int = 100,
    train_mode: bool = True,
    inference_steps: int = 20,
    update_steps: int = 20,
    beta: float = 1,
    beta_2: float = 2,
):

    best_val_score = -np.inf

    sigma = sigma_init

    env_info = env.reset(train_mode=train_mode)[brain_name]
    num_agents = len(env_info.agents)

    for episode in range(episodes):

        env_info = env.reset(train_mode=train_mode)[brain_name]
        states = env_info.vector_observations

        scores, timesteps = 0, 0

        losses_critic, losses_actor, rewards_in_batch = [], [], []

        q_old = None

        sigma = max(sigma_end, sigma_decay * sigma)

        ounoise.reset_noise()

        while True:

            timesteps += 1

            with torch.no_grad():

                state_device = torch.from_numpy(states).float().to(device)

                # Are we going to take an action from the noise generator or from the actor network?
                if sigma < np.random.uniform():
                    action = ddpg._actor.forward(s=state_device)
                    action = action.detach().cpu().numpy()
                else:
                    action = ounoise.generate()

                # Clip the actions to be in the correct range
                action = np.clip(action, -1, 1)

                # What is the q-value of taking the choosen action in the current state?
                _, q = ddpg.forward(s=state_device, a=torch.from_numpy(action).float().to(device))

            q = q.detach().cpu().numpy().flatten()

            # Take the action in the env
            env_info = env.step(action)[brain_name]
            next_states = env_info.vector_observations
            rewards = np.array(env_info.rewards)
            dones = np.array(env_info.local_done)

            scores += np.sum(rewards)

            # Convert the rewards to the correct one. All rewards > 0 should be 0.1
            rewards_scaled = rewards * 10  # (rewards > 0) * 0.1

            # print(rewards_scaled)

            if np.any(dones):
                break

            if q_old is not None:
                td_error = abs(rewards_scaled + (gamma * q - q_old))
            else:
                td_error = abs(rewards_scaled)

            replaybuffer.add(
                state=states,
                state_prime=next_states,
                reward=rewards_scaled,
                action=action,
                weight=td_error,
                done=dones,
                beta=beta,
            )

            # Update state and q-values
            states = next_states
            q_old = q

            if timesteps % inference_steps != 0:
                continue

            if (replaybuffer._reward > 0).sum() < 1000:
                continue

            for i in range(update_steps):

                (
                    bstate,
                    bstate_prime,
                    baction,
                    breward,
                    bdone,
                    bweight,
                    indices,
                ) = replaybuffer.draw(batch_size, replace=False, beta=beta, beta_2=beta_2)

                if bstate is not None:

                    rewards_in_batch.append(breward.mean())

                    cl, al, td_error = ddpg.loss(
                        gamma=gamma,
                        s=torch.from_numpy(bstate).float().to(device),
                        sprime=torch.from_numpy(bstate_prime).float().to(device),
                        r=torch.from_numpy(breward).float().to(device),
                        a=torch.from_numpy(baction).float().to(device),
                        d=torch.from_numpy(bdone).float().to(device),
                        # w=torch.from_numpy(((1/bweight) * (1 / len(replaybuffer)))).float().to(device),
                        w=torch.from_numpy(np.ones_like(bweight)).float().to(device),
                    )

                    # Update the networks
                    ddpg.update(cl=cl, al=al)

                    # Refresh the td-errors we have in replay buffer w/ the updated
                    replaybuffer.update_weight(indices, td_error.detach().cpu().numpy(), alpha=0)

                    losses_critic.append(cl.item())
                    losses_actor.append(al.item())

                    ddpg.update_target_networks()

        scores = scores / (num_agents)

        eval_score = evaluation(
            env=env,
            ddpg=ddpg,
            episodes=evaluation_rounds,
            device=device,
            brain_name=brain_name,
            train_mode=train_mode,
        )

        print("*" * 20 + f"  {episode}  " + "*" * 20)
        print(f"Critic Loss: {np.mean(losses_critic)}")
        print(f"Actor Loss: {np.mean(losses_actor)}")
        print(f"Training Score: {scores}")
        print(f"Validation Score: {eval_score}")
        print(f"Reward in batch: {np.mean(rewards_in_batch)/0.1}")

        writer.add_histogram("training/hist/actions", action.flatten(), episode)
        writer.add_scalar("training/sigma", sigma, episode)
        writer.add_scalar("training/kpi/score", scores, episode)
        writer.add_scalar("training/kpi/loss_actor", np.mean(losses_actor), episode)
        writer.add_scalar("training/kpi/loss_critic", np.mean(losses_critic), episode)
        writer.add_scalar("validation/kpi/score", eval_score, episode)

        # Save the best model
        if eval_score > best_val_score:

            best_val_score = eval_score
            torch.save(ddpg._actor.state_dict(), "continuous_control/models/actor.ckp")
            torch.save(ddpg._critic.state_dict(), "continuous_control/models/critic.ckp")


def evaluation(
    env,
    ddpg: DDPG,
    episodes: int,
    brain_name: str,
    device: str,
    train_mode: bool = True,
):

    timesteps = 0
    score = 0
    num_agents = None

    for i in range(episodes):

        env_info = env.reset(train_mode=train_mode)[brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations

        while True:

            timesteps += 1

            with torch.no_grad():

                state_device = torch.from_numpy(states).float().to(device)
                action = ddpg._actor.forward(s=state_device)
                action = action.detach().cpu().numpy()

            env_info = env.step(action)[brain_name]

            score += np.sum(np.array(env_info.rewards))
            states = env_info.vector_observations

            if np.any(np.array(env_info.local_done)):
                break

    score = score / (num_agents * episodes)

    return score
