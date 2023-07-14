import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import numpy as np
import Train as Briscola
from tqdm import tqdm
#import wandb
BATCH_SIZE = 200
GAMMA = 0.99
LAMBDA = 0.0001 
LR = 0.00001 
TAU = 0.005
device = "cpu"

#wandb.login()
# start a new wandb run to track this script
#wandb.init(
#  set the wandb project where this run will be logged
#    project="briscola_game",
    
    # track hyperparameters and run metadata
#    config={
#    "learning_rate": 0.02,
#    "architecture": "QNN",
#    "dataset": "NO",
#    "epochs": 100,
#    }
#)

# TODO capire come fare il training e salvare il modello 
# 
# Utile?
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

def find_all_valid_actions(states):
        '''
        Function that given the player's state give back the cards that
        the player is allowed to play.
        '''

        valid = []

        for i, state in enumerate(states[0]):
            # State 1 and 2 are the ones corresponding to card in the 
            # hand of the player. 1 not briscola, 2 briscola
            if int(state) in [1,2]:
                valid.append(i)
            
        return valid


class Brain:

    def __init__(self, stateCnt, actionCnt, train = False):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        if train:
            self.model = DQN(self.stateCnt, self.actionCnt).to(device)
            self.model_ = DQN(self.stateCnt, self.actionCnt).to(device)

            self.model_.load_state_dict(self.model.state_dict())
            self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, 
                                    amsgrad=True)
            self.memory = ReplayMemory(10000)
            self.env = Briscola.Game_Train()
        else:
            self.model = torch.load('model.pt')


        # TODO come fare quando abbiamo già il modello?

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype= torch.float64)
        with torch.no_grad():
            next_state_values[non_final_mask] = (non_final_next_states.eq(1)*self.model_(non_final_next_states)).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()
        #wandb.log({"loss": loss})
        return loss


    def predict(self, state, target=False):
        '''
        Function that performs the predictions given the current state.
        '''

        if target:
            return self.model_.forward(state.reshape(1, self.stateCnt))
        else:
            return self.model.forward(state.reshape(1, self.stateCnt))


    def predict_next_action(self, state, target=False): 
        '''
        Function that given the current state of the player gives back
        the next action using the predicted Q table
        '''

        # Select the actions that are valid given the state
        valid_actions = find_all_valid_actions(state)
        
        # Get the predictions of the nn
        next_Qs = self.predict(state, target).flatten()
        #print(next_Qs)
        # Select the one that are actually valid
        next_Qs = next_Qs[valid_actions]

        # The best valid action
        idx = torch.argmax(next_Qs)
        
        if target:
            return next_Qs[idx]
        
        return valid_actions[idx]


    
    def train(self):

        if device == 'cuda0' :
            num_episodes = 1000
        else:
            num_episodes = 10000

        wins = []
        loss = []
        print(num_episodes)
        for i_episode in tqdm(range(num_episodes)):
            # Initialize the environment and get it's state
            self.env.reset()
            self.env.first_to_play = 1
            state = self.env.get_state_for_player(1)
            state = torch.tensor(state, dtype=torch.float64, 
                                device=device).unsqueeze(0)
            for _ in range(20):
                # Let the agent choose the action
                action = self.env.get_action_train(state, self.env.player_1, self)
                    #tensor_actions = torch.zeros(40, dtype=torch.int64)
                    #tensor_actions[action.id] = 1
                tensor_actions = torch.full((1,1), action.id, device = device,
                                            dtype=torch.int64)
                
                # Perform the action and see where it will lead to
                observation, reward, done = self.env.step(action)
                reward = torch.tensor([reward], device=device, dtype=torch.int64)
                
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float64, 
                                            device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, tensor_actions, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                loss.append(self.optimize_model())

                #print(self.model.state_dict())

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.model_.state_dict()
                policy_net_state_dict = self.model.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.model_.load_state_dict(target_net_state_dict)

                if done:
                    break
                
            wins.append(np.sign(reward[0]) if reward[0] else 0)
        torch.save(self.model, 'model.pt')
        return wins, loss
    



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
        self.layer1 = nn.Linear(n_observations, 128, dtype=torch.float64)
        self.layer2 = nn.Linear(128, 128, dtype=torch.float64)
        self.layer3 = nn.Linear(128, 128, dtype=torch.float64)
        self.layer4 = nn.Linear(128, n_actions, dtype=torch.float64)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return F.softmax(self.layer4(x))
    


