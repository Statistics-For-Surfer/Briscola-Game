from collections import namedtuple, deque
import torch.nn.functional as F
import torch.optim as optim
import config
from Train import GameTrain
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random
import torch
import wandb

# Define all the hyperparemeters for the network
BATCH_SIZE = 20
GAMMA = 0.99
LAMBDA = 0.001 
LR = 0.001 
TAU = 0.005

# Select the gpu as device if possible
device = 'cpu'

# Connect to the wandb.ai site to keep track of the results
### Check plots on 'https://wandb.ai/site'
wandb.login()
# start a new wandb run to track this script
wandb.init(
    #set the wandb project where this run will be logged
    project="briscola_game",
    # track hyperparameters and run metadata
    config={
    "win_ration": 0,
    "eps": .82,
    "Value Function": 0,
    "learning_rate": 0.01 ,
    "reward": 0, 
    "architecture": "QNN",
    "dataset": "NO",
    "epochs": 100,
    }
)


def find_all_valid_actions(states):
    '''
    Function that given the player's state give back the cards that
    the player is allowed to play.
    '''

    # To identify the valid card create a list with 0 if allowed else 1
    valid = np.ones(40)

    for i, state in enumerate(states[0][40:80]):
        if state:
            valid[i] = 0

    return np.where(valid == 1)[0]
    


class Brain:
    '''
    Class used to train the reinforced agent and later select the card to play
    using the trained network
    '''

    def __init__(self, stateCnt, actionCnt, train = False):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        # Agent has to be trained
        if train:
            # Define every object needed for the agent

            # networks
            self.model = DQN(self.stateCnt, self.actionCnt).to(device)
            self.model_ = DQN(self.stateCnt, self.actionCnt).to(device)
            self.model_.load_state_dict(self.model.state_dict())

            self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, 
                                    amsgrad=True)
            # Optimizer
            # Obeject to store and keep track of the actions
            self.memory = ReplayMemory(10000)
            # Environment
            self.env = GameTrain()

        # Agent was already trained
        else:
            # Load the trained network
            self.model = torch.load('model.pt')


    def optimize_model(self):
        '''
        Function used to update the parameters of the model during the training
        phase. 
        Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        '''
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
                                                    if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

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
        #print(non_final_next_states.eq(1).shape)
        #print(self.model_(non_final_next_states).shape)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.model_(non_final_next_states).max(1)[0]
            #(non_final_next_states.eq(1)*self.model_(non_final_next_states)).max(1)[0]
        
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

        wandb.log({"loss": loss , "reward": reward_batch , "eps" : config.eps})
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
        the next action using the predicted Q table.
        '''

        # Select the actions that are valid given the state
        non_valid_actions = find_all_valid_actions(state)
        
        # Get the predictions of the nn
        next_Qs = self.predict(state, target).flatten()

        # Mask the values for the cards that can't be played 
        next_Qs[non_valid_actions] = -1e8

        # The best valid action
        idx = torch.argmax(next_Qs)
    
        if target:
            return next_Qs[idx]
    
        return idx


    
    def train(self):
        '''
        Function that using the training environment simulate a high number of 
        games to train the agent.
        '''

        # Select the number of games to simulate
        num_episodes = 20000

        # Store during the simulation: number of lost and won games and the loss.
        wins = []
        w = 0
        p = 0
        loss = []
        print(num_episodes)
        for i_episode in tqdm(range(num_episodes)):
            # Initialize the environment
            self.env.reset()
            # Select who will be the fist to play
            self.env.first_to_play = random.randint(1,2) + 1

            # If the training agent is not the first to play, let the other make its play
            if self.env.first_to_play == 2:
                self.env.first_hand()

            # Get the state for the training agent
            state = self.env.get_state_for_player(1)
            state = torch.tensor(state, dtype=torch.float64, 
                                device=device).unsqueeze(0)
            
            # For each hand
            for _ in range(20):
                # Let the agent choose the action
                action = self.env.get_action_train(state, self.env.player_1, self)
                tensor_actions = torch.full((1,1), action.id, device = device,
                                            dtype=torch.int64)
                
                # Perform the action and find the reward obtained and the next state
                observation, reward, done = self.env.step(action)
                reward = torch.tensor([reward], device=device, dtype=torch.int64)
                
                # If the game is finished
                if done:
                    # There is no next state
                    next_state = None
                else:
                    # Store the new state
                    next_state = torch.tensor(observation, dtype=torch.float64, 
                                            device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, tensor_actions, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                loss.append(self.optimize_model())

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.model_.state_dict()
                policy_net_state_dict = self.model.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.model_.load_state_dict(target_net_state_dict)

                if done:
                    break

            if np.sign(reward[0].cpu()):
                w += 1
            p += 1
            
            # Store the win-ratio as the number of won games over the played ones.
            wandb.log({"wins_ratio": w / p})

                
            wins.append(np.sign(reward[0].cpu()) if reward[0] else 0)

        # Save the model
        torch.save(self.model, 'model.pt')
        return wins, loss
    



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    '''
    Class used to store all the touples ('state', 'action', 'next_state', 'reward')
    and retrieve them for the optimization step
    '''

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    '''
    Class to define the network structure.

    Dimensions:
    - 1x Input 162
    - 3x Hidden 256
    - 1x Output 40

    Activation functions:
    - ReLU, applied after the linear layers.
    - Softmax, applied to the output.
    '''

    def __init__(self, n_observations, n_actions, hidden = 256):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden, dtype=torch.float64)
        self.layer2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.layer3 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.layer4 = nn.Linear(hidden, n_actions, dtype=torch.float64)

    def forward(self, x):
        '''
        Called with either one element to determine next action, or a batch
        during optimization.
        '''
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        
        return F.softmax(self.layer4(x))
