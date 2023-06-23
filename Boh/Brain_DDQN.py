import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import numpy as np

LEARNING_RATE = 0.00025
BATCH_SIZE = 256

GAMMA = 0.99
LAMBDA = 0.0001 

# TODO capire come fare il training e salvare il modello 
# 
# Utile?
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

def find_all_valid_actions(self, states):
        '''
        Function that given the player's state give back the cards that
        the player is allowed to play.
        '''

        valid = []
        for i, state in enumerate(states):
            # State 1 and 2 are the ones corresponding to card in the 
            # hand of the player. 1 not briscola, 2 briscola
            if state in [1,2]:
                valid.append(i)
            
        return valid


class Brain:

    def __init__(self, stateCnt, actionCnt, train = False):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        if train:
            self.model = DQN(self.stateCnt, self.actionCnt)
            self.model_ = DQN(self.stateCnt, self.actionCnt)

        # TODO come fare quando abbiamo già il modello?



    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, 
                    verbose=verbose)


    def evaluate(self, x, y):
        return self.model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=0)


    def predict(self, state, target=False):
        '''
        Function that performs the predictions given the current state.
        '''

        if target:
            return self.model_.predict(state.reshape(1, self.stateCnt), target)
        else:
            return self.model.predict(state.reshape(1, self.stateCnt), target)


    def predict_next_action(self, state, target=False): 
        '''
        Function that given the current state of the player gives back
        the next action using the predicted Q table
        '''

        # Select the actions that are valid given the state
        valid_actions = find_all_valid_actions(state)
        
        # Get the predictions of the nn
        next_Qs = self.predict(state, target).flatten()
        # Select the one that are actually valid
        next_Qs = next_Qs[valid_actions]
        # The best valid action
        idx = np.argmax(next_Qs)

        return valid_actions[idx]


    def updateTargetModel(self):

        self.model_.set_weights(self.model.get_weights())


class Memory:   # stored as ( s, a, r, s_ )

    def __init__(self, capacity):

        self.tree = SumTree(capacity)

        self.e = 0.01
        self.a = 0.6


    def get_priority(self, error):

        return (error + self.e) ** self.a


    def add(self, error, sample):

        p = self.get_priority(error)
        self.tree.add(p, sample)


    def sample(self, n):

        batch = []
        segment = self.tree.total() / n

        for i in range(n):

            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)

            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch


    def update(self, idx, error):
        p = self.get_priority(error)
        self.tree.update(idx, p)



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

