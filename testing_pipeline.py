import torch
import torchvision
import os
import argparse
from improved_precision_recall import IPR
from model import Generator, Discriminator
from latent_space_OT import make_image
from utils import load_model
from torchvision import datasets, transforms
from pytorch_fid.fid_score import calculate_fid_given_paths

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
                if compt>= 10000:
                        break
                else:
                    for i in range(images.shape[0]):
                        torchvision.utils.save_image(images[i], os.path.join(self.path_real, f'{compt}.png'))
                        compt += 1
                        if compt>= 10000:
                            break
        except:
            print('Real samples already exist')
        os.makedirs('samples/fake_samples', exist_ok=True)
        ipr = IPR(device = self.device, k = 5, batch_size= batch_size, num_samples = 10000)
        ipr.compute_manifold_ref(self.path_real)
        n_samples = args.n_samples
        for lr in args.lr:
            for N_update in args.N_update:
                print(N_update)
                print(f'lr = {lr}, N_update = {N_update}')
                img = make_image(G = self.G, D= self.D, batchsize = n_samples, N_update=N_update, ot=True, space = args.space , k=1, lr=lr, optmode='adam')
                for k in range(n_samples):
                    torchvision.utils.save_image(img[k, :, :], os.path.join('samples/fake_samples', f'{k}.png'))
                metric = ipr.precision_and_recall(self.path_real + '/../fake_samples')
                fid_value = calculate_fid_given_paths(['samples/real_samples', 'samples/fake_samples'],batch_size = args.batch_size,device = device,dims = 2048)
                print('FID =', fid_value)
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
    parser.add_argument('--lr', type=list, default=[0.1, 0.01, 0.001, 0.0001])
    parser.add_argument('--N_update', type= list, default=[5, 10, 15,  20, 50, 70,  100, 150, 200])
    parser.add_argument('--space', type=str, default='target')
    args = parser.parse_args()
    device = args.device
    if args.model_type == 'vanilla_gan':
        G = Generator(g_output_dim= args.mnist_dim)
        D = Discriminator(args.mnist_dim)
    G = load_model(G, args.model_path, mode = 'G_b')
    D = Discriminator(784).to(device)
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
