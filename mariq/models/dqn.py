import itertools
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from tqdm import trange
from ..utils.replay_buffer import ReplayBuffer
from .estimator import Estimator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(device) 

class DQN:
    def __init__(self, env, logger, batch_size=32, max_memory=50000):
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_memory)
        self.epsilon = 1
        self.epsilon_decay = 500
        self.epsilon_min = 0.1
        self.frames = 0
        self.copy_each = 10000
        self.save_each = 10000
        self.gamma = 0.90
        self.num_actions = env.action_space.n
        self.policy_agent = Estimator(env.observation_space.shape, self.num_actions).to(device)
        self.target_agent = Estimator(env.observation_space.shape, self.num_actions).to(device)
        self.sync_agents()

        self.optimizer = optim.Adam(self.policy_agent.parameters())
        self.env = env

        self.log = {
            'losses': [],
            'rewards': [],
            'accum_reward': [],
            'duration': []
        }

    def sync_agents(self):
        self.target_agent.load_state_dict(
            self.policy_agent.state_dict()
        )

    def act(self, state):
        state = Variable(torch.tensor(state).unsqueeze(0))
        if np.random.rand() < self.epsilon:
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            with torch.no_grad():
                action = self.policy_agent(state).max(1)[1].view(1, 1).item()
        '''
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        '''
        self.frames += 1
        self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * np.exp(-1. * self.frames / self.epsilon_decay)

        return action

    def compute_loss(self):
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = Variable(torch.tensor(np.float32(state)))
        next_state = Variable(torch.tensor(np.float32(next_state)))
        action = Variable(torch.tensor(action))
        reward = Variable(torch.tensor(np.float32(reward)))
        done = Variable(torch.tensor(np.float32(done)))

        q_policy = self.policy_agent(state).gather(1, action.unsqueeze(1)).squeeze(1)
        q_target = self.target_agent(next_state).max(1)[0]
        q_expected = reward + self.gamma * q_target * (1 - done)

        # Optimization
        loss = F.smooth_l1_loss(q_policy, q_expected.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self):
        torch.save(self.policy_agent.state_dict(), './checkpoints/policy_agent.pt')
        torch.save(self.target_agent.state_dict(), './checkpoints/target_agent.pt')

    def train(self, num_episodes):
        episode_range = trange(1, num_episodes + 1)
        accum_reward = 0
        for episode in episode_range:

            state = self.env.reset()
            episode_reward = 0
            for t in itertools.count():
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.append(
                    state, action, reward, next_state, done
                )

                state = next_state
                episode_reward += reward
                accum_reward += reward
                self.log['accum_reward'].append(accum_reward)


                # Update target every 10k frames
                if t % 10000 == 0:
                    self.sync_agents()

                # Optimize
                if len(self.memory) > self.batch_size:
                    loss = self.compute_loss()
                    self.log['losses'].append(loss)

                if done:
                    state = self.env.reset()
                    self.log['rewards'].append(episode_reward)
                    self.log['duration'].append(t + 1)
                    episode_range.set_description('| reward: {0} | dur: {1} | frames: {2} '.format(
                        episode_reward,
                        t + 1,
                        self.frames
                    ))
                    break



            if episode % 500 == 0:
                self.logger.info(
                    'episode: {0} | frame {1} | epslion: {2:.2f} | reward: {3:.2f} | duration: {4}'.format(
                        episode,
                        self.frames,
                        self.epsilon,
                        episode_reward,
                        t + 1,
                    )
                )
                self.plot(episode, 500)
                self.save()

        # save if done
        self.save()
