import torch
import torchvision
import os
import argparse


from model import Generator, Discriminator
from utils import load_model
from latent_space_OT import make_image
from variables import *
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("--n_samples", type=int, default=10000,)
    args = parser.parse_args()




    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    G = Generator(g_output_dim = mnist_dim).to(device)
    G.device = device
    G = load_model(G, 'checkpoints', mode = 'G_b')
    D = Discriminator(mnist_dim).to(device)
    def load_model(model, folder, mode='G'):
        ckpt = torch.load(os.path.join(folder, mode + '.pth'), map_location=torch.device(device))
        # Adjust key names if needed
        adjusted_ckpt = {}
        for k, v in ckpt.items():
            new_key = k
            # Example: Change key names if they differ
            if 'weight' in k:
                new_key = k.replace('weight', 'weight_orig')  # Modify this line as needed
            adjusted_ckpt[new_key] = v
        model.load_state_dict(adjusted_ckpt, strict=False)  # Use strict=False to ignore unexpected keys
        return model

    D = load_model(D, 'checkpoints', mode = 'D_b')
    D.device = device

    if device == 'cuda':
        G = torch.nn.DataParallel(G).to(device)
        G = torch.nn.DataParallel(G).to(device)
        D = torch.nn.DataParallel(D).to(device)
        D = torch.nn.DataParallel(D).to(device)
    else :
        G = G.to(device)
        D = D.to(device)
    print('Model loaded.')
    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = args.n_samples
    img = make_image(G = G, D= D, batchsize = n_samples, N_update= 6, ot=True, mode='dot', k=1, lr=0.01, optmode='adam')
    for k in range(n_samples):
        torchvision.utils.save_image(img[k, :, :], os.path.join('samples', f'{k}.png'))
