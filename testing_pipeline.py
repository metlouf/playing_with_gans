import torch
import torchvision
import os
import argparse
from improved_precision_recall import IPR
from model import Generator, Discriminator
from latent_space_OT import make_image
from utils import load_model
from torchvision import datasets, transforms
import logging
import pandas as pd
from pytorch_fid.fid_score import calculate_fid_given_paths

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='logs/testing_pipeline.log',  # Specify the log file path
                    filemode='w')  # Use 'w' to overwrite or 'a' to append
logger = logging.getLogger(__name__)

class TestingPipeline:
    def __init__(self, G, D, device):
        self.G = G
        self.D = D
        self.path_real = 'samples/real_samples'
        self.device = device
        self.G.device = device
        self.results = []  # List to store results for DataFrame

    def compute_metrics(self, batch_size):
        logger.info('Start Generating')
        os.makedirs('samples', exist_ok=True)
        os.makedirs('logs', exist_ok=True)  # Create logs directory if it doesn't exist

        try:
            os.makedirs(self.path_real, exist_ok=False)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))
            ])
            test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=args.batch_size, shuffle=True)
            compt = 0
            for images, _ in test_loader:
                if compt >= 10000:
                    break
                else:
                    for i in range(images.shape[0]):
                        torchvision.utils.save_image(images[i], os.path.join(self.path_real, f'{compt}.png'))
                        compt += 1
                        if compt >= 10000:
                            break
        except FileExistsError:
            logger.warning('Real samples already exist, skipping creation.')

        os.makedirs('samples/fake_samples', exist_ok=True)
        ipr = IPR(device=self.device, k=5, batch_size=batch_size, num_samples=10000)
        ipr.compute_manifold_ref(self.path_real)
        n_samples = args.n_samples

        for lr in args.lr:
            for N_update in args.N_update:
                logger.info(f'Processing N_update = {N_update} with learning rate = {lr}')
                img = make_image(G=self.G, D=self.D, batchsize=n_samples, N_update=N_update,
                                 ot=args.ot, space=args.space, k=1, lr=lr, optmode='adam')

                for k in range(n_samples):
                    torchvision.utils.save_image(img[k, :, :], os.path.join('samples/fake_samples', f'{k}.png'))

                metric = ipr.precision_and_recall(self.path_real + '/../fake_samples')
                fid_value = calculate_fid_given_paths(['samples/real_samples', 'samples/fake_samples'],
                                                       batch_size=args.batch_size, device=self.device, dims=2048)

                # Log metrics
                logger.info('FID = %s', fid_value)
                logger.info('Precision = %s', metric.precision)
                logger.info('Recall = %s', metric.recall)

                # Store metrics for DataFrame
                self.results.append({
                    'learning_rate': lr,
                    'N_update': N_update,
                    'FID': fid_value,
                    'Precision': metric.precision,
                    'Recall': metric.recall
                })

        # Save results to a CSV file
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('logs/testing_pipeline_metrics.csv', index=False)  # Save the DataFrame as a CSV
        logger.info('Metrics saved to results/testing_pipeline_metrics.csv')


if __name__ == '__main__':
    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=torch.device('cuda'))
    parser.add_argument('--model_type', type=str, default='vanilla_gan')
    parser.add_argument('--mnist_dim', type=int, default=784)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--lr', type=list_of_floats, default=[0.1, 0.01, 0.001, 0.0001])
    parser.add_argument('--N_update', type= list_of_ints, default=[5, 10, 15,  20, 50, 70,  100, 150, 200])
    parser.add_argument('--space', type=str, default='target')
    parser.add_argument('--ot', type=bool, default=True)
    args = parser.parse_args()
    print(args.N_update)
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
