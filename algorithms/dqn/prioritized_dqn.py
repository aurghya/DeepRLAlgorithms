import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1)%self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios**self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total*probs[indices])**(-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = zip(*samples)
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)

beta_start = 0.4
beta_frames = 1000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
BUFFER_SIZE = 100000

class PERDQN(object):
    def __init__(self, state_size, action_size, model, seed):
        self.current_model = model(state_size, action_size)
        self.target_model = model(state_size, action_size)
        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()
        self.optimizer = optim.Adam(self.current_model.parameters())
        self.replay_buffer = NaivePrioritizedBuffer(100000)
    
    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict)
    
    def learn(self, batch_size, beta):
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size, beta)
        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))
        weights = Variable(torch.FloatTensor(weights))

        q_values = self.current_model(state)
        next_q_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma*next_q_value*(1-done)

        loss = (q_value - expected_q_value.detach()).pow(2)*weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy)
        self.optimizer.step()

        return loss
    