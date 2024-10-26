import torch
import os
import numpy as np
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import datetime

from torch.utils.tensorboard import SummaryWriter

from model import Generator, Discriminator
from utils import D_train, G_train, save_models, generate_fake_samples, save_real_samples
from pytorch_fid.fid_score import calculate_fid_given_paths

from variables import *

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time + '/'
writer = SummaryWriter(log_dir)
print("run this for logs : tensorboard --logdir",log_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0001,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Size of mini-batches for SGD")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--save_metrics", type=bool, default=False)
    parser.add_argument("--early_stop", type=bool, default=False)
    parser.add_argument("--track_fid", type=bool, default=False)

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

    if device == 'cuda':
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
    fid_max = 0

    for epoch in trange(0, n_epoch+1, leave=True):
        g_loss = 0
        d_loss = 0
        d_real_loss = 0
        d_fake_loss = 0
        d_acc_real = 0
        d_acc_fake = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)

            d_loss_batch, d_rea_loss_batch, d_fake_loss_batch,d_acc_real_batch,d_acc_fake_batch = D_train(x, G, D, D_optimizer, criterion)
            d_loss += d_loss_batch
            d_real_loss += d_rea_loss_batch
            d_fake_loss += d_fake_loss_batch
            d_acc_real += d_acc_real_batch
            d_acc_fake += d_acc_fake_batch

            if epoch > 20:
                g_loss += G_train(x, G, D, G_optimizer, criterion)

        writer.add_scalars("train/Dloss",{
            "D_loss_total" : d_loss / batch_idx,
            "D_real_loss" : d_real_loss / batch_idx,
            "D_fake_loss" : d_fake_loss / batch_idx
            },epoch)

        writer.add_scalars("train/D_accuracy",{
            "D_accuracy_on_real" : d_acc_real / batch_idx,
            "D_accuracy_on_fake" : d_acc_fake / batch_idx,
            },epoch)

        writer.add_scalar("train/Gloss",g_loss / batch_idx,epoch)
        if epoch % 10 == 0:
            with torch.no_grad() :
                if args.track_fid :
                    generate_fake_samples(args, G, args.n_samples, device)
                    #Calculate and Save the FID
                    fid_value = calculate_fid_given_paths(['samples/real_samples', 'samples/fake_samples'],batch_size = args.batch_size,device = device,dims = 2048)
                    writer.add_scalar("train/FID",fid_value,epoch)
                    print(f'Epoch {epoch}, FID: {fid_value:.2f}')

                z = torch.randn(1, 100).to(device)
                G_output = (G(z)+1)/2
                writer.add_image("train/Fake_images", G_output.reshape(1,28,28), epoch)

                print("D_loss",d_loss / batch_idx)
                print("G_loss",g_loss / batch_idx)

        if args.early_stop:
            if fid_value > fid_max:
                fid_max = fid_value
                fid_values.append(fid_value)
            else:
                print('Stopping training as FID is not improving')
                break


    save_models(G, D, 'checkpoints')
    writer.close()
    print('Training done')
