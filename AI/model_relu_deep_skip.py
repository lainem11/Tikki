import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.distributions import Categorical
import math

class Actor(nn.Module):

    DICT_SIZE = 53+3+11   # 52 unique cards, word 0 for no info, and 11 for scores -5 -- 5
    def __init__(self, input_size, hidden_size, n_actions, emb_size):
        super(Actor, self).__init__()
        self.embedding_layer = nn.Embedding(self.DICT_SIZE, emb_size)
        self.hidden_layer1 = nn.Linear(input_size * emb_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size,hidden_size)
        self.hidden_layer3 = nn.Linear(hidden_size,hidden_size)
        self.hidden_layer4 = nn.Linear(hidden_size,hidden_size)
        self.hidden_layer5 = nn.Linear(hidden_size,hidden_size)
        self.output_layer = nn.Linear(hidden_size, n_actions)
        nn.init.orthogonal_(self.hidden_layer1.weight,math.sqrt(2.0))
        nn.init.orthogonal_(self.hidden_layer2.weight,math.sqrt(2.0))
        nn.init.orthogonal_(self.hidden_layer3.weight,math.sqrt(2.0))
        nn.init.orthogonal_(self.output_layer.weight,0.01)

    def forward(self, x):
        x = self.embedding_layer(x).view(x.shape[0], -1)
        x1 = F.relu(self.hidden_layer1(x))
        x2 = F.relu(self.hidden_layer2(x1))
        x3 = F.relu(self.hidden_layer3(x2))
        x4 = F.relu(self.hidden_layer4(x3))
        x5 = F.relu(self.hidden_layer5(x4)+self.hidden_layer2(x1))
        x6 = self.output_layer(x5)
        return x6
    
    def save(self,file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)
    
class Critic(nn.Module):
    DICT_SIZE = 53+3+11   # 52 unique cards, word 0 for no info, and 16 for scores -10 -- 5
    def __init__(self, input_size, hidden_size, emb_size):
        super(Critic, self).__init__()
        self.embedding_layer = nn.Embedding(self.DICT_SIZE, emb_size)
        self.hidden_layer1 = nn.Linear(input_size * emb_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size,hidden_size)
        self.hidden_layer3 = nn.Linear(hidden_size,hidden_size)
        self.hidden_layer4 = nn.Linear(hidden_size,hidden_size)
        self.hidden_layer5 = nn.Linear(hidden_size,hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.hidden_layer1.weight,math.sqrt(2.0))
        nn.init.orthogonal_(self.hidden_layer2.weight,math.sqrt(2.0))
        nn.init.orthogonal_(self.hidden_layer3.weight,math.sqrt(2.0))
        nn.init.orthogonal_(self.output_layer.weight,0.01)

    def forward(self, x):
        x = self.embedding_layer(x).view(x.shape[0], -1)
        x1 = F.relu(self.hidden_layer1(x))
        x2 = F.relu(self.hidden_layer2(x1))
        x3 = F.relu(self.hidden_layer3(x2))
        x4 = F.relu(self.hidden_layer4(x3))
        x5 = F.relu(self.hidden_layer5(x4)+self.hidden_layer2(x1))
        x6 = self.output_layer(x5)
        return x6
    
    def save(self,file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class DQN(nn.Module):
    EMB_DIM = 50
    DICT_SIZE = 53+3+16   # 52 unique cards, word 0 for no info, and 16 for scores -10 -- 5
    
    def __init__(self, input_size, hidden_size=60, output_size=52):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.embedding_layer = nn.Embedding(self.DICT_SIZE, self.EMB_DIM)
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_size * self.EMB_DIM, hidden_size*4),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_size*4,hidden_size*2),
            nn.ReLU()
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(hidden_size*2,hidden_size),
            nn.ReLU()
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size*2),
        )

        self.hidden_layer5 = nn.Sequential(
            nn.Linear(hidden_size*2,hidden_size*4),
        )
        
        self.output_layer = nn.Linear(hidden_size*4, output_size)
        
    
    def forward(self, x):
        x = x.long()  # Convert to long tensor once, at the beginning
        x = self.embedding_layer(x).view(x.shape[0], -1)
        x1 = self.hidden_layer1(x)
        x2 = self.hidden_layer2(x1)
        x3 = self.hidden_layer3(x2)
        x4 = F.relu(self.hidden_layer4(x3) + x2)
        x5 = F.relu(self.hidden_layer5(x4) + x1)
        x6 = self.output_layer(x5)
        return x6
    
    def save(self,file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)