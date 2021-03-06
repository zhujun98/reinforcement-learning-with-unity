"""
Copyright (c) 2020 Jun Zhu
"""
import copy

import numpy as np

import torch
from torch import optim
import torch.nn.functional as F

from agent_base import _AgentBase, Memory
from utilities import copy_nn, soft_update_nn, OUProcess

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


class DdpgAgent(_AgentBase):
    """Deep deterministic policy gradient agent.

    https://arxiv.org/pdf/1509.02971.pdf
    """
    def __init__(self, action_space, models, *,
                 brain_name="ReacherBrain",
                 model_file="ddpg_checkpoint.pth",
                 replay_memory_size=1000):
        """Initialization.

        :param int action_space: action space size.
        :param tuple models: Neural network models for actor and critic.
        :param int replay_memory_size: size of the replay buffer.
        """
        super().__init__(brain_name, model_file)
        self._action_space = action_space

        self._model_actor = models[0].to(device)
        self._model_actor_target = copy.deepcopy(models[0]).to(device)
        self._model_critic = models[1].to(device)
        self._model_critic_target = copy.deepcopy(models[1]).to(device)

        self._memory = Memory(replay_memory_size)

    def _act(self, state, noise=0.):
        """Override."""
        state = torch.from_numpy(state).float().to(device)
        self._model_actor.eval()  # set the module in evaluation mode
        with torch.no_grad():
            action_values = self._model_actor(state)
        self._model_actor.train()  # set the module in training mode

        action = action_values.cpu().detach().numpy()
        action += noise
        action = np.clip(action, -1., 1.)
        return action

    def _learn(self, experiences, opt_actor, opt_critic, gamma, tau):
        """Learn from a given trajectory.

        :param (Tuple[torch.Variable]) experiences: (s, a, r, s', done)
        :param Optimizer opt_actor: actor optimizer used for gradient ascend.
        :param Optimizer opt_critic: critic optimizer used for gradient ascend.
        :param float gamma: discount factor.
        :param float tau: soft update rate of the target network.
        """
        states, actions, rewards, next_states, dones = experiences

        # shape = (batch size, 1)
        q_next = self._model_critic_target(
            next_states, self._model_actor_target(next_states))
        q_targets = rewards + (gamma * q_next * (1 - dones))
        q_expected = self._model_critic(states, actions)

        # update critic by minimizing the loss
        opt_critic.zero_grad()
        critic_loss = F.mse_loss(q_targets.detach(), q_expected)
        critic_loss.backward()
        opt_critic.step()

        # update actor by minimizing the loss
        opt_actor.zero_grad()
        actor_loss = self._model_critic(
            states.detach(), self._model_actor(states))
        actor_loss = -actor_loss.mean()
        actor_loss.backward()
        opt_actor.step()

        # apply soft update
        soft_update_nn(
            self._model_actor, self._model_actor_target, tau)
        soft_update_nn(
            self._model_critic, self._model_critic_target, tau)

        return actor_loss.item(), critic_loss.item()

    def train(self, env, *,
              n_episodes=1000,
              theta=0.15,
              sigma=0.2,
              decay_rate=0.99,
              tau=0.001,
              gamma=1.0,
              learning_rate=(1e-3, 1e-3),
              weight_decay=(0., 0.),
              batch_size=16,
              replay_start_size=None,
              window=100,
              target_score=30,
              save_frequency=100,
              output_frequency=10):
        """Train the agent.

        :param gym.Env env: environment.
        :param int n_episodes: number of episodes.
        :param float theta: Ornstein-Uhlenbeck process constant.
        :param float sigma: Ornstein-Uhlenbeck process constant.
        :param float decay_rate: noise decay rate.
        :param float tau: soft update rate of the target network.
        :param double gamma: discount factor.
        :param tuple learning_rate: learning rates for actor and critic models.
        :param double weight_decay: L2 penalties for actor and critic models.
        :param int batch_size: mini batch size.
        :param int replay_start_size: a uniform random policy is run for this
            number of frames before training starts.
        :param int window: the latest window episodes will be used to evaluate
            the performance of the model.
        :param float target_score: the the average score of the latest window
            episodes is larger than the target score. The problem is considered
            solved.
        :param int save_frequency: the frequency of saving the states of the
            agent.
        :param int output_frequency: the frequency of summarizing the
            training result.
        """
        opt_actor = optim.Adam(self._model_actor.parameters(),
                               lr=learning_rate[0],
                               weight_decay=weight_decay[0])
        opt_critic = optim.Adam(self._model_critic.parameters(),
                                lr=learning_rate[1],
                                weight_decay=weight_decay[1])

        try:
            checkpoint = torch.load(self._model_file)
        except FileNotFoundError:
            checkpoint = None

        if checkpoint is None:
            i0 = 0
            scores = []
            losses_actor = []
            losses_critic = []
        else:
            i0 = checkpoint['epoch']
            scores = checkpoint['score_history']
            losses_actor = checkpoint['actor_loss_history']
            losses_critic = checkpoint['critic_loss_history']
            self._model_actor.load_state_dict(
                checkpoint['model_actor_state_dict'])
            copy_nn(self._model_actor, self._model_actor_target)
            self._model_critic.load_state_dict(
                checkpoint['model_critic_state_dict'])
            copy_nn(self._model_critic, self._model_critic_target)
            opt_actor.load_state_dict(
                checkpoint['optimizer_actor_state_dict'])
            opt_critic.load_state_dict(
                checkpoint['optimizer_critic_state_dict'])

            avg_score = np.mean(scores[-window:])
            print(f"Loaded existing model ended at epoch: {i0} with average"
                  f"score of {avg_score:8.2f}")

            if avg_score > target_score:
                print(f"Score of the current model {avg_score:8.2f} is already "
                      f"higher than the target score {target_score}!")
                return scores, losses_actor, losses_critic

        if replay_start_size is None:
            replay_start_size = batch_size * 2

        brain_name = self._brain_name
        i = i0
        decay = decay_rate ** i
        while i < n_episodes:
            i += 1

            random_process = OUProcess(
                self._action_space, theta=theta, sigma=sigma)

            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            # one episode has 1000 steps
            while True:
                action = self._act(state, random_process.next() * decay)
                env_info = env.step(action)[brain_name]
                reward = env_info.rewards[0]
                score += reward
                next_state = env_info.vector_observations[0]
                done = env_info.local_done[0]
                self._memory.append(state, action, reward, next_state, done)
                state = next_state

                if len(self._memory) > replay_start_size:
                    loss_actor, loss_critic = self._learn(
                        self._memory.sample(batch_size, device=device),
                        opt_actor,
                        opt_critic,
                        gamma,
                        tau
                    )

                    losses_actor.append(loss_actor)
                    losses_critic.append(loss_critic)

                if done:
                    break

            decay *= decay_rate

            scores.append(score)
            avg_score = np.mean(scores[-window:])

            if avg_score >= target_score:
                print(f"Epoch: {i:04d}, average score: {avg_score:8.2f}")
                self._save_model(i, opt_actor, opt_critic, scores,
                                 [losses_actor, losses_critic])
                break

            if i % output_frequency == 0:
                print(f"Epoch: {i:04d}, average score: {avg_score:8.2f}")

            if i % save_frequency == 0:
                self._save_model(i, opt_actor, opt_critic, scores,
                                 [losses_actor, losses_critic])

        return scores, losses_actor, losses_critic

    def _save_model(self, epoch, opt_actor, opt_critic, scores, losses):
        torch.save({
            'epoch': epoch,
            'optimizer_actor_state_dict': opt_actor.state_dict(),
            'optimizer_critic_state_dict': opt_critic.state_dict(),
            'score_history': scores,
            'actor_loss_history': losses[0],
            'critic_loss_history': losses[1],
            'model_actor_state_dict': self._model_actor.state_dict(),
            'model_critic_state_dict': self._model_critic.state_dict(),
        }, self._model_file)
        print(f"Model save in {self._model_file} after {epoch} epochs!")
