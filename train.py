import torch
import os
import numpy as np
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import Generator, Discriminator
from utils import D_train, G_train, save_models, generate_fake_samples, save_real_samples
from pytorch_fid.fid_score import calculate_fid_given_paths

from variables import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Size of mini-batches for SGD")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--save_metrics", type=bool, default=False)

    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')

    save_real_samples(args, train_loader, force_save=False)

    print("Real samples generated.")


    print('Model Loading...')
    mnist_dim = 784

    if device == 'mps':
        G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim).to(device)).to(device)
        D = torch.nn.DataParallel(Discriminator(mnist_dim).to(device)).to(device)
    else :
        G = Generator(g_output_dim = mnist_dim).to(device)
        D = Discriminator(mnist_dim).to(device)

    # model = DataParallel(model).to(device)
    print('Model loaded.')
    # Optimizer



    # define loss
    criterion = nn.BCELoss()

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

    print('Start Training :')

    n_epoch = args.epochs
    fid_values = []
    D_loss =[]
    D_real_loss = []
    D_fake_loss = []
    G_loss = []
    fid_min = np.inf
    for epoch in trange(1, n_epoch+1, leave=True):
        g_loss = 0
        d_loss = 0
        d_real_loss = 0
        d_fake_loss = 0
        delay_value = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            d_loss_batch, d_real_loss_batch, d_fake_loss_batch = D_train(x, G, D, D_optimizer, criterion)
            d_loss += d_loss_batch
            d_real_loss += d_real_loss_batch
            d_fake_loss += d_fake_loss_batch
            g_loss += G_train(x, G, D, G_optimizer, criterion)
        D_loss.append(d_loss / batch_idx)
        G_loss.append(g_loss / batch_idx)
        D_real_loss.append(d_real_loss / batch_idx)
        D_fake_loss.append(d_fake_loss / batch_idx)
        if epoch % 20 == 0:
            generate_fake_samples(args, G, args.n_samples, device)
            #Calculate the FID
            fid_value = calculate_fid_given_paths(['samples/real_samples', 'samples/fake_samples'],batch_size = args.batch_size,device = device,dims = 2048)
            print(f'Epoch {epoch}, FID: {fid_value:.2f}')
            if fid_value < fid_min:
                fid_min = fid_value
                fid_values.append(fid_value)
                save_models(G, D, 'checkpoints')
            elif (fid_value >= fid_min) and (delay_value == 3):
                print('Stopping training as FID is not improving')
                break
            else:
                delay_value += 1
    if args.save_metrics:
        print(D_loss)
        D_loss, G_loss, D_real_loss, D_fake_loss, fid_values = np.array(D_loss), np.array(G_loss), np.array(D_real_loss), np.array(D_fake_loss), np.array(fid_values)
        directory = f"metrics/epochs_{args.epochs}_lr_{args.lr}_batch_{args.batch_size}_spectral_normalized"
        os.makedirs(directory, exist_ok=True)
        np.save(directory + '/D_loss.npy', D_loss)
        np.save(directory + '/G_loss.npy', G_loss)
        np.save(directory + '/D_real_loss.npy', D_real_loss)
        np.save(directory + '/D_fake_loss.npy', D_fake_loss)
        np.save(directory + '/fid_values.npy', fid_values)

    print('Training done')
