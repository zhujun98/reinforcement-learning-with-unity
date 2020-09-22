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


class DdpgAgent:
    def __init__(self, actor, critic):
        self.actor = actor.to(device)
        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(critic).to(device)

    def update_nn(self, tau):
        soft_update_nn(
            self.actor, self.actor_target, tau)
        soft_update_nn(
            self.critic, self.critic_target, tau)


class MaDdpgAgent(_AgentBase):
    """Multi-agent deep deterministic policy gradient agent.

    https://arxiv.org/abs/1706.02275.
    """
    def __init__(self, state_space, action_space, n_agents, models, *,
                 brain_name="TennisBrain",
                 model_file="maddpg_checkpoint.pth",
                 replay_memory_size=1000):
        """Initialization.

        :param int state_space: state space size.
        :param int action_space: action space size.
        :param int n_agents: number of agents.
        :param tuple models: Neural network classes for actor and critic.
        :param int replay_memory_size: size of the replay buffer.
        """
        super().__init__(brain_name, model_file)

        self._state_space = state_space
        self._action_space = action_space
        self._n_agents = n_agents

        self._agents = [DdpgAgent(models[0](), models[1]())
                        for _ in range(n_agents)]

        self._memory = Memory(replay_memory_size)

    def _act(self, states, noises=None):
        """Override."""
        if noises is None:
            noises = np.zeros((self._n_agents, self._action_space))

        actions = []
        for agent, state, noise in zip(self._agents, states, noises):
            state = torch.from_numpy(
                np.expand_dims(state, axis=0)).float().to(device)

            agent.actor.eval()  # set the module in evaluation mode
            with torch.no_grad():
                action_values = agent.actor(state)
            agent.actor.train()  # set the module in training mode

            action = np.squeeze(action_values.cpu().detach().numpy(), axis=0)
            action += noise
            action = np.clip(action, -1., 1.)
            actions.append(action)
        return actions

    def _learn(self, batch_size, opts_actor, opts_critic, gamma, tau):
        """Learn from a given trajectory.

        :param (Tuple[torch.Variable]) experiences: (s, a, r, s', done)
        :param Optimizer opts_actor: actor optimizers used for gradient ascend.
        :param Optimizer opts_critic: critic optimizers used for gradient ascend.
        :param float gamma: discount factor.
        :param float tau: soft update rate of the target network.
        """
        critic_losses = []
        actor_losses = []
        for i, (agent, opt_critic, opt_actor) \
                in enumerate(zip(self._agents, opts_critic, opts_actor)):

            # sample for each agent

            states, actions, rewards, next_states, dones = \
                self._memory.sample(batch_size, device=device)

            # update critic

            next_actions_target = []
            for j, _agent in enumerate(self._agents):
                next_actions_target.append(_agent.actor_target(
                    next_states[:, j * self._state_space:(j + 1) * self._state_space]))
            next_actions_target = torch.cat(next_actions_target, dim=-1)

            q_next = agent.critic_target(next_states, next_actions_target)
            q_targets = rewards[:, i, None] \
                        + gamma * q_next * (1 - dones[:, i, None])
            q_expected = agent.critic(states, actions)
            assert(q_expected.shape == q_targets.shape)

            critic_loss = F.mse_loss(q_targets.detach(), q_expected)

            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

            critic_losses.append(critic_loss.item())

            # update actor

            next_actions_pred = []
            for j, _agent in enumerate(self._agents):
                if j == i:
                    next_actions_pred.append(_agent.actor(
                        states[:, j * self._state_space:(j + 1) * self._state_space]))
                else:
                    next_actions_pred.append(
                        actions[:, j * self._action_space:(j + 1) * self._action_space])
            next_actions_pred = torch.cat(next_actions_pred, dim=-1)

            actor_loss = - agent.critic(states, next_actions_pred).mean()
            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()

            actor_losses.append(actor_loss.item())

        # apply soft update
        for agent in self._agents:
            agent.update_nn(tau)

        return actor_losses, critic_losses

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
              n_trainings_per_step=1,
              replay_start_size=None,
              window=100,
              target_score=30,
              continue_after_reaching_target=False,
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
        :param int n_trainings_per_step: number of trainings per time step.
        :param int replay_start_size: a uniform random policy is run for this
            number of frames before training starts.
        :param int window: the latest window episodes will be used to evaluate
            the performance of the model.
        :param float target_score: the the average score of the latest window
            episodes is larger than the target score. The problem is considered
            solved.
        :param bool continue_after_reaching_target: True for continuing the
            training after reaching the target score.
        :param int save_frequency: the frequency of saving the states of the
            agent.
        :param int output_frequency: the frequency of summarizing the
            training result.
        """
        opts_actor = [
            optim.Adam(agent.actor.parameters(),
                       lr=learning_rate[0],
                       weight_decay=weight_decay[0]) for agent in self._agents
        ]
        opts_critic = [
            optim.Adam(agent.critic.parameters(),
                       lr=learning_rate[1],
                       weight_decay=weight_decay[1]) for agent in self._agents
        ]

        try:
            checkpoint = torch.load(self._model_file)
        except FileNotFoundError:
            checkpoint = None

        if checkpoint is None:
            i0 = 0
            scores_hist = []
            best_saved_score = -np.inf
            losses_actor_hist = [[] for _ in range(self._n_agents)]
            losses_critic_hist = [[] for _ in range(self._n_agents)]
        else:
            i0 = checkpoint['epoch']
            scores_hist = checkpoint['score_history']
            losses_actor_hist = checkpoint['actor_loss_history']
            losses_critic_hist = checkpoint['critic_loss_history']
            for i, agent in enumerate(self._agents):
                agent.actor.load_state_dict(
                    checkpoint['model_actor_state_dict'][i])
                copy_nn(agent.actor, agent.actor_target)
                agent.critic.load_state_dict(
                    checkpoint['model_critic_state_dict'][i])
                copy_nn(agent.critic, agent.critic_target)
            for i, opt in enumerate(opts_actor):
                opt.load_state_dict(
                    checkpoint['optimizer_actor_state_dict'][i])
            for i, opt in enumerate(opts_critic):
                opt.load_state_dict(
                    checkpoint['optimizer_critic_state_dict'][i])

            # score is the maximum of the scores from all agents
            avg_score = np.mean(scores_hist[-window:])
            best_saved_score = avg_score
            print(f"Loaded existing model ended at epoch: {i0} with average"
                  f"score of {avg_score:8.4f}")

            if avg_score > target_score and not continue_after_reaching_target:
                print(f"Score of the current model {avg_score:8.4f} is already "
                      f"higher than the target score {target_score}!")
                return scores_hist, losses_actor_hist, losses_critic_hist

        if replay_start_size is None:
            replay_start_size = batch_size * 2

        brain_name = self._brain_name
        i = i0
        decay = decay_rate ** i
        while i < n_episodes:
            i += 1

            random_process = OUProcess(
                self._action_space * self._n_agents, theta=theta, sigma=sigma)

            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            scores = [0.] * self._n_agents
            # one episode has 14 ~ 50 steps
            while True:
                actions = self._act(
                    states,
                    random_process.next().reshape(self._n_agents, -1) * decay)
                env_info = env.step(actions)[brain_name]
                rewards = env_info.rewards
                for i_r, r in enumerate(rewards):
                    scores[i_r] += r
                    # It should be encouraged to hit the ball, even it is a
                    # "bad" hit. It is found that the result becomes more
                    # stable. Note that this change does not affect the
                    # calculation of the score.
                    if r < 0:
                        rewards[i_r] = -r
                next_states = env_info.vector_observations
                dones = env_info.local_done
                self._memory.append(
                    states, actions, rewards, next_states, dones)
                states = next_states

                if len(self._memory) > replay_start_size:
                    for _ in range(n_trainings_per_step):
                        losses_actor, losses_critic = self._learn(
                            batch_size,
                            opts_actor,
                            opts_critic,
                            gamma,
                            tau
                        )

                        for i_a in range(self._n_agents):
                            losses_actor_hist[i_a].append(losses_actor[i_a])
                            losses_critic_hist[i_a].append(losses_critic[i_a])

                if dones[0]:
                    break

            decay *= decay_rate

            scores_hist.append(max(scores))
            avg_score = np.mean(scores_hist[-window:])

            if i % output_frequency == 0:
                print(f"Epoch: {i:04d}, average score: {avg_score:8.4f}")
                # save if the target score has been achieved and the current score
                # is better than the saved score.
                if avg_score >= target_score and avg_score > best_saved_score:
                    self._save_model(i, opts_actor, opts_critic, scores_hist,
                                     [losses_actor_hist, losses_critic_hist])
                    best_saved_score = avg_score
                    if not continue_after_reaching_target:
                        break

            if i % save_frequency == 0 and \
                    (best_saved_score < target_score or avg_score > best_saved_score):
                # save if the target score has not been achieved or the current score
                # is better than the saved score.
                best_saved_score = avg_score
                self._save_model(i, opts_actor, opts_critic, scores_hist,
                                 [losses_actor_hist, losses_critic_hist])

        return scores_hist, losses_actor_hist, losses_critic_hist

    def _save_model(self, epoch, opts_actor, opts_critic, scores, losses):
        torch.save({
            'epoch': epoch,
            'optimizer_actor_state_dict':
                [opt.state_dict() for opt in opts_actor],
            'optimizer_critic_state_dict':
                [opt.state_dict() for opt in opts_critic],
            'score_history': scores,
            'actor_loss_history': losses[0],
            'critic_loss_history': losses[1],
            'model_actor_state_dict':
                [a.actor.state_dict() for a in self._agents],
            'model_critic_state_dict':
                [a.critic.state_dict() for a in self._agents],
        }, self._model_file)
        print(f"Model save in {self._model_file} after {epoch} epochs!")
