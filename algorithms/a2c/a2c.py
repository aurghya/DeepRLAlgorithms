import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
import os
import pickle
import time
from collections import deque

'''A2C Settings'''
TRAJ_LEN = 100
ENT_COEF = 1e-2
LAMBDA = 0.95

USE_GPU = torch.cuda.is_available()
NET_PATH = ''

class A2C(object):
    def __init__(self, model, lr):
        self.net = model()
        if USE_GPU:
            self.net = self.net.cuda()
        self.memory_counter = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def save_model(self):
        self.net.cpu()
        self.net.save(NET_PATH)
        if USE_GPU:
            self.net.cuda()
    
    def load_model(self):
        self.net.cpu()
        self.net.save(NET_PATH)
        if USE_GPU:
            self.net.cuda()
    
    def choose_action(self, state):
        self.memory_counter += 1
        x = torch.FloatTensor(state)
        if USE_GPU:
            x = x.cuda()
        action_log_probs, state_values = self.net(x)
        probs = F.softmax(action_log_probs, dim=1).data.cpu().numpy()