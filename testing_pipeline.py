import torch
import torchvision
import os
import argparse
from improved_precision_recall import IPR
from model import Generator, Discriminator
from latent_space_OT import make_image
from utils import load_model
from torchvision import datasets, transforms

class TestingPipeline:
    def __init__(self, G, D, device):
        self.G = G
        self.D = D
        self.path_real = 'samples/real_samples'
        self.device = device
        self.G.device = device

    def compute_metrics(self, batch_size):
        print('Start Generating')
        os.makedirs('samples', exist_ok=True)

        try:
            os.makedirs(self.path_real, exist_ok= False)
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5), std=(0.5))])
            train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size, shuffle=True)
            compt = 0
            for images, _ in train_loader:
                if compt>= 79 * batch_size:
                        break
                else:
                    for i in range(images.shape[0]):
                        torchvision.utils.save_image(images[i:i+1], os.path.join(self.path_real, f'{compt}.png'))
                        compt += 1
                        if compt>= 79*batch_size:
                            break
        except:
            print('Real samples already exist')
        os.makedirs('samples/fake_samples', exist_ok=True)
        n_samples = args.n_samples
        for lr in args.lr:
            for N_update in args.N_update:
                print(N_update)
                print(f'lr = {lr}, N_update = {N_update}')
                img = make_image(G = self.G, D= self.D, batchsize = n_samples, N_update=N_update, ot=True, mode='dot', k=1, lr=lr, optmode='adam')
                for k in range(n_samples):
                    torchvision.utils.save_image(img[k, :, :], os.path.join('samples/fake_samples', f'{k}.png'))
                ipr = IPR(device = self.device, k = 5, batch_size= batch_size, num_samples = 5000)
                ipr.compute_manifold_ref(self.path_real)
                metric = ipr.precision_and_recall(self.path_real + '/../fake_samples')
                print('precision =', metric.precision)
                print('recall =', metric.recall)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=torch.device('cuda'))
    parser.add_argument('--model_type', type=str, default='vanilla_gan')
    parser.add_argument('--mnist_dim', type=int, default=784)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--lr', type=list, default=[0.05, 0.005, 0.0005, 0.00005])
    parser.add_argument('--N_update', type= list, default=[10, 20, 50, 100, 200])
    args = parser.parse_args()
    device = args.device
    if args.model_type == 'vanilla_gan':
        G = Generator(g_output_dim= args.mnist_dim)
        D = Discriminator(args.mnist_dim)
    G = load_model(G, args.model_path)
    D = load_model(D, args.model_path, mode='D')
    if device == 'cuda':
        G = torch.nn.DataParallel(G).to(device)
        G = torch.nn.DataParallel(G).to(device)
        D = torch.nn.DataParallel(D).to(device)
        D = torch.nn.DataParallel(D).to(device)
    else :
        G = G.to(device)
        D = D.to(device)
    pipeline = TestingPipeline(G, D, device)
    pipeline.compute_metrics(args.batch_size)
