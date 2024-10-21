import torch
import torchvision
import os
import argparse
from improved_precision_recall import IPR
from model import Generator
from utils import load_model
from torchvision import datasets, transforms

class TestingPipeline:
    def __init__(self, model, device):
        self.model = model
        self.path_real = 'samples/real_samples'
        self.device = device

    def compute_metrics(self, batch_size):
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.eval()
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
                for i in range(images.shape[0]):
                    n_image = compt * images.shape[0] + i
                    torchvision.utils.save_image(images[n_image:n_image+1], os.path.join(self.path_real, f'{n_image}.png'))
                    if n_image>=200:
                        break
        except:
            print('Real samples already exist')

        n_samples = 0
        with torch.no_grad():
            try:
                os.makedirs('samples/fake_samples', exist_ok=False)
            except:
                print('Fake samples directory already exist')
            images = torch.zeros((200, batch_size, 28, 28))
            while n_samples<200:
                z = torch.randn(batch_size, 100).to(self.device)
                x = self.model(z)
                x = x.reshape(batch_size, 28, 28)
                images[n_samples, :, :, :] = x
                for k in range(x.shape[0]):
                    if n_samples<200:
                        torchvision.utils.save_image(x[k:k+1], os.path.join(self.path_real + '/../fake_samples', f'{n_samples}.png'))
                        n_samples += 1

        print('Finish Generating')
        print('Start Computing Metrics')
        ipr = IPR(device = self.device, k = 1, batch_size= batch_size, num_samples = n_samples)
        ipr.compute_manifold_ref(self.path_real)
        images = images.reshape(-1, 28, 28).unsqueeze(1).repeat(1, 3, 1, 1)
        print(images.shape)
        metric = ipr.precision_and_recall(images)
        print('precision =', metric.precision)
        print('recall =', metric.recall)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default=torch.device('mps'))
    parser.add_argument('--model_type', type=str, default='vanilla_gan')
    parser.add_argument('--mnist_dim', type=int, default=784)
    args = parser.parse_args()
    device = args.device
    if args.model_type == 'vanilla_gan':
        model = Generator(g_output_dim= args.mnist_dim)
    model = load_model(model, args.model_path)
    pipeline = TestingPipeline(model, device)
    pipeline.compute_metrics(args.batch_size)
