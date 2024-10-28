import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from variables import *



class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

    def make_hidden(self, batch_size):
        # Generate samples in the latent space
        return torch.rand(batch_size, 100).uniform_(-1, 1)

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = spectral_norm(nn.Linear(d_input_dim, 1024))
        self.fc2 = spectral_norm(nn.Linear(self.fc1.out_features, self.fc1.out_features//2))
        self.fc3 = spectral_norm(nn.Linear(self.fc2.out_features, self.fc2.out_features//2))
        self.fc4 = spectral_norm(nn.Linear(self.fc3.out_features, 1))

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))
