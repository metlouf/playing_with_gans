import torch
import torch.nn as nn
import torch.nn.functional as F
from variables import *



class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

        # Initialize weights
        #self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with gain for LeakyReLU
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Special initialization for the final layer
        nn.init.xavier_uniform_(self.fc4.weight, gain=nn.init.calculate_gain('tanh'))
        if self.fc4.bias is not None:
            nn.init.constant_(self.fc4.bias, 0)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

        # Initialize weights
        #self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with gain for LeakyReLU
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Special initialization for the final layer
        nn.init.xavier_uniform_(self.fc4.weight, gain=nn.init.calculate_gain('sigmoid'))
        if self.fc4.bias is not None:
            nn.init.constant_(self.fc4.bias, 0)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))
