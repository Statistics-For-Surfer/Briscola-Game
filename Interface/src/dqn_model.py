import torch.nn.functional as F
import torch.nn as nn
import torch


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden = 256):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden, dtype=torch.float64)
        self.layer2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.layer3 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.layer4 = nn.Linear(hidden, n_actions, dtype=torch.float64)


    def forward(self, x):
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        
        return F.softmax(self.layer4(x), dim=0)