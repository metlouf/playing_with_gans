import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1024)