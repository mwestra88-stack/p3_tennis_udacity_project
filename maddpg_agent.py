import numpy as np
import random
from collections import namedtuple, deque

from critic_model import CriticNetwork
from actor_model import ActorNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)     # replay buffer size
BATCH_SIZE = 128           # minibatch size
GAMMA = 1               # discount factor
TAU = 1e-2                 # for soft update of target parameters
LRA = 1e-4                 # learning rate actor
LRC = 3e-3                 # learning rate critic
UPDATE_EVERY = 1          # how often to update the network
BUFFER_WARMUP = 10000      # Nr of samples in replay buffer before sampling and learning can start
LEARN_STEPS = 2           # Nr of learning steps every time an update of the neural nets is done
HIDDEN_ACTOR = [400, 300]  # hidden layers actor
HIDDEN_CRITIC = [400, 300] # hidden layers critic
NOISE_SIGMA = 0.3          # standard deviation of random standard normal gaussian noise added to actions for exploration purposes
INFLATE_REWARDS = 100      #Inflate level of rewards. They are very small leading to small gradients and losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, shared_actor, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            shared_actor (bool): Flag to indicate whether actor network of each agent should have the same starting weights
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.shared_actor = shared_actor
        self.seed = random.seed(seed)
        self.device = device
        
        joint_state_size = state_size * num_agents
        joint_action_size = action_size * num_agents

        # Initialize Actor Networks
        self.actors_local = []
        self.actors_target = []
        self.actor_optimizers = []

        if self.shared_actor:

            # Create ONE actor
            actor_local = ActorNetwork(state_size, action_size, seed, HIDDEN_ACTOR).to(device)
            actor_target = ActorNetwork(state_size, action_size, seed, HIDDEN_ACTOR).to(device)

            actor_target.load_state_dict(actor_local.state_dict())

            actor_optimizer = torch.optim.Adam(actor_local.parameters(), lr=LRA)

            # Store same reference for each agent
            for _ in range(num_agents):
                self.actors_local.append(actor_local)
                self.actors_target.append(actor_target)
                self.actor_optimizers.append(actor_optimizer)

        else:

            for _ in range(num_agents):

                actor_local = ActorNetwork(state_size, action_size, seed, HIDDEN_ACTOR).to(device)
                actor_target = ActorNetwork(state_size, action_size, seed, HIDDEN_ACTOR).to(device)

                actor_target.load_state_dict(actor_local.state_dict())

                actor_optimizer = torch.optim.Adam(actor_local.parameters(), lr=LRA)

                self.actors_local.append(actor_local)
                self.actors_target.append(actor_target)
                self.actor_optimizers.append(actor_optimizer)
        
        # Initialize Local and Target Critic-Networks
        self.critics_local = []
        self.critics_target = []
        self.critic_optimizers = []

        for _ in range(num_agents):
            critic_local = CriticNetwork(joint_state_size, joint_action_size, seed, HIDDEN_CRITIC).to(device)
            critic_target = CriticNetwork(joint_state_size, joint_action_size, seed, HIDDEN_CRITIC).to(device)
            critic_target.load_state_dict(critic_local.state_dict())
            critic_optimizer = torch.optim.Adam(critic_local.parameters(), lr=LRC)

            self.critics_local.append(critic_local)
            self.critics_target.append(critic_target)
            self.critic_optimizers.append(critic_optimizer)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Normalizing states
        self.state_norm = [RunningNorm(state_size) for _ in range(num_agents)]
        
    
    def step(self, states, actions, rewards, next_states, dones):
        """Stores and samples experiences. And, if conditions are fullfilled, the step function initiates a learning step for the actor and critic networks
        
        Params
        ======
            states (array_like): current state for each agent
            actions (array like): actions taken by each agent
            rewards (array like): rewards received by each agent
            next_states (array like): next state for each agent
            dones (array like): if game ended or not
            noise_scale (float): decay of random noise added to actions
        """
        # Save experience in replay memory
        # Update running stats using BOTH agents
        for i in range(self.num_agents):
            self.state_norm[i].update(states[i])
            self.state_norm[i].update(next_states[i])

         # Normalize per agent
        norm_states = np.array([self.state_norm[i].normalize(states[i]) for i in range(self.num_agents)])
        norm_next_states = np.array([self.state_norm[i].normalize(next_states[i]) for i in range(self.num_agents)])
        actions = np.array(actions)
        rewards = np.asarray(rewards, dtype=np.float32) * INFLATE_REWARDS
        self.memory.add(norm_states, actions, rewards, norm_next_states, dones)
        # Learn every UPDATE_EVERY time steps and execute 10 learning steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            if len(self.memory) > BUFFER_WARMUP: # warm-up buffer of 10K observations
                for _ in range(LEARN_STEPS):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)#, indices, priorities, weights, GAMMA)
                
    def act(self, states, noise_scale):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            states (array_like): current state for each agent
            noise_scale (float): decay of random noise added to actions
        """
        actions = []
        normalized_states = np.array([self.state_norm[i].normalize(states[i]) for i in range(self.num_agents)])
        normalized_states = torch.FloatTensor(normalized_states).to(self.device)   
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        for i in range(self.num_agents):
            state_i = normalized_states[i]
            self.actors_local[i].eval()
            with torch.no_grad():
                action_i = self.actors_local[i](state_i).cpu().numpy()
            self.actors_local[i].train()

            action_i = action_i.squeeze()
            action_i += np.random.normal(0, NOISE_SIGMA, size=self.action_size) * noise_scale
            action_i = np.clip(action_i, -1, 1) 

            actions.append(action_i)

        return actions

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        for agent_i in range(self.num_agents):

            # ---------------------- #
            #   UPDATE CRITIC
            # ---------------------- #

            #states = self.state_norm.normalize(states.cpu().numpy())
            #states = torch.from_numpy(states).float().to(device)

            #next_states = self.state_norm.normalize(next_states.cpu().numpy())
            #next_states = torch.from_numpy(next_states).float().to(device)
        
            # Target actions for all agents
            next_actions = []
            for i in range(self.num_agents):
                state_i = next_states[:, i*self.state_size:(i+1)*self.state_size]
                next_action = self.actors_target[i](state_i)
                next_actions.append(next_action)

            next_actions = torch.cat(next_actions, dim=1)
            
            with torch.no_grad():
                critic_target_next = self.critics_target[agent_i](next_states, next_actions)
                y = rewards[:, agent_i].unsqueeze(1) + (gamma * critic_target_next * (1 - dones[:, agent_i].unsqueeze(1)))
                
            critic_value_local = self.critics_local[agent_i](states, actions)
            #print('\ny:{}\tCtarget:{}'.format(y.mean(),critic_target_next.mean()),end="")
            critic_loss = F.mse_loss(y.detach(), critic_value_local, reduction='mean')
            self.critic_optimizers[agent_i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_local[agent_i].parameters(), 1)
            self.critic_optimizers[agent_i].step()

            # ---------------------- #
            #   UPDATE ACTOR
            # ---------------------- #

            actions_pred = []
            for i in range(self.num_agents):
                state_i = states[:, i*self.state_size:(i+1)*self.state_size]

                if i == agent_i:
                    actions_pred.append(self.actors_local[i](state_i))
                else:
                    actions_pred.append(
                        actions[:, i*self.action_size:(i+1)*self.action_size].detach()
                    )

            actions_pred = torch.cat(actions_pred, dim=1)
            
            actor_loss = -self.critics_local[agent_i](states, actions_pred).mean()
            
            #print("Q mean:", Q_values.mean().item(),end='\r')
            #if self.shared_actor:
                # Accumulate gradients for both agents first
            #    if agent_i ==0:
            #        self.actor_optimizers[0].zero_grad()

                #actor_loss.backward()

                #if agent_i == self.num_agents - 1:
                 #   self.actor_optimizers[0].step()
            #else:
            self.actor_optimizers[agent_i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_i].step()
            #print('\rCritic loss:{:.2f}\t Actor loss:{:.2f}'.format(critic_loss.item(), actor_loss.item()),end="")
            #for p in self.actors_local[0].parameters():
            #    print('\rGradient{:.10f}'.format(p.grad.abs().mean()),end="")
            #    break

        for agent_i in range(self.num_agents):
            # ------------------- update target network ------------------- #
            self.soft_update(self.critics_local[agent_i], self.critics_target[agent_i], TAU)
            self.soft_update(self.actors_local[agent_i], self.actors_target[agent_i], TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.FloatTensor(np.vstack([e.states.reshape(1, -1) for e in experiences if e is not None])).to(device)
        actions = torch.FloatTensor(np.vstack([e.actions.reshape(1, -1) for e in experiences if e is not None])).to(device)
        rewards = torch.FloatTensor(np.vstack([e.rewards for e in experiences if e is not None])).to(device)
        next_states = torch.FloatTensor(np.vstack([e.next_states.reshape(1, -1) for e in experiences if e is not None])).float().to(device)
        dones = torch.FloatTensor(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store prioritized experience tuples."""
    
    def __init__(self, buffer_size, batch_size, seed, alpha, beta_start, beta_frames, eps):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        
        self.frame = 1
        self.max_priority = 1.0
        
    # -------------------------
    # Beta annealing
    # -------------------------
    def beta(self):
        return min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames
        )
    
    # -------------------------
    # Add transition
    # -------------------------
    def add(self, state, action, reward, next_state, done):
        priority = max(self.max_priority, 1e-6) # New samples get maximum priority to prevent they are never sampled
        self.tree.add(priority, state, action, reward, next_state, done)

    # -------------------------
    # Sample batch
    # -------------------------    
    def sample(self):
        """Sample a batch of prioritized experiences from memory.
        When working with segments, every batch element will come from another interval. 
        This reduces variance compared to pure random samples
        
        """
        experiences = []
        idxs = []
        priorities = []

        total = self.tree.total()
        segment = total / self.batch_size

        beta = self.beta()
        self.frame += 1

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            experiences.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        probs = (np.array(priorities) / total) + 1e-8
        weights = (self.tree.size * probs) ** (-beta) #Importance sampling weights. Prevents sampling bias
        #print('\rframe {}\tbeta: {:.2f}'.format(self.frame, beta), end="")
        weights /= (weights.max()+1e-8)  # normalize. Prevents exploding gradients
        
        #uniform_sample = np.random.uniform(0, self.total(), self.batch_size)
        #results = [self.get(s) for s in uniform_sample]
        
        #indices = [r[0] for r in results]
        #priorities = [r[1] for r in results]
        #experiences = [r[2] for r in results]
        
        # DEBUG
        count=0
        for e in experiences:
            count += 1
            if isinstance(e, int):
                print("exp:{}".format(e))
                print("count:{}".format(count))
                print("Tree: {}".format(self.tree.tree))
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        experiences_2 = (states, actions, rewards, next_states, dones)
        
        return experiences_2, idxs, priorities, weights
    
    # -------------------------
    # Update priorities
    # -------------------------
    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            td_error = td_error.detach().cpu().item()
            priority = (abs(td_error) + self.eps) ** self.alpha
            priority = max(priority, 1e-6)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    
class SumTree:
    def __init__(self, buffer_size):
        """
        buffer_size = number of leaf nodes = max replay buffer size
        """
        self.buffer_size = buffer_size
        self.tree = np.zeros(2 * buffer_size - 1)
        self.data = np.zeros(buffer_size, dtype=object)
        self.write = 0
        self.size = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    # -------------------------
    # Private helpers
    # -------------------------
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # -------------------------
    # Public API
    # -------------------------
    def total(self):
        return self.tree[0]

    def add(self, priority, state, action, reward, next_state, done):
        idx = self.write + self.buffer_size - 1
        
        data = self.experience(state, action, reward, next_state, done)

        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.buffer_size + 1

        return idx, self.tree[idx], self.data[data_idx]
    
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        # Calculate the squared differences
        squared_diff = (predictions - targets) ** 2
        # Apply weights
        weighted_loss = self.weights * squared_diff
        # Return the mean loss
        return weighted_loss.mean()

class RunningNorm:
    def __init__(self, size, eps=1e-5):
        self.size = size
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(size)
        self.var = np.ones(size)

    def update(self, x):
        x = np.array(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)
