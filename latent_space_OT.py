import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import sqrt

def l2_norm(x):
    return torch.sqrt(torch.sum(x ** 2, dim=1))

def eff_k(G, D, trial=10):
    with torch.no_grad():
            # Generate latent vectors z1 and z2
            z1 = G.make_hidden(trial).to(G.device)
            z2 = G.make_hidden(trial).to(G.device)
            # Generate images x1 and x2 from the latent vectors
            x1 = G(z1)
            x2 = G(z2)
            # Get discriminator outputs f1 and f2
            f1 = D(x1)
            f2 = D(x2)
            # Compute the distance between discriminator outputs and latent vectors
            nu = l2_norm(f2 - f1) # L2 norm for distance of D outputs
            de_l = l2_norm(z2 - z1)  # L2 norm for distance of z vectors
            return torch.max(nu / de_l).item()

def eff_K(G, D, trial=10):
    with torch.no_grad():
            # Generate latent vectors z1 and z2
            z1 = G.make_hidden(trial).to(G.device)
            z2 = G.make_hidden(trial).to(G.device)
            # Generate images x1 and x2 from the latent vectors
            x1 = G(z1)
            x2 = G(z2)
            # Get discriminator outputs f1 and f2
            f1 = D(x1)
            f2 = D(x2)
            nu = l2_norm(f2 - f1) # L2 norm for distance of D outputs
            de_L = l2_norm(x2 - x1)  # L2 norm for distance of x vectors
    return torch.max(nu / de_L).item()



class Transporter_target_space():
    def __init__(self, G, D, k, y_xp, mode = 'dot', opt_mode = "adam") -> None:
        self.G = G
        self.D = D
        self.device = G.device
        self.y = self.G(self.G.make_hidden(10000).to(self.device)).detach()
        self.mode = mode
        self.lc = k
        self.dist = "uniform"
        self.opt_mode = opt_mode
        self.space = 'target'

    def get_x_va(self):
        return self.x.data

    def set_(self, y_xp, lr):
        self.x = y_xp.detach().clone().to(self.device)
        self.x.requires_grad = True
        if self.opt_mode == 'sgd':
            self.opt = torch.optim.SGD([self.x], lr)
        elif self.opt_mode == 'adam':
            self.opt = torch.optim.Adam([self.x], lr, betas=[0.0,0.9])
        else:
            print("Optimizer not implemented yet.")

    def H_y(self):
        if self.mode=='dot':
            return - self.D(self.x)/self.lc + torch.norm(self.x -self.y + 0.001, p = 2, dim = 1)
        else:
            return - self.D(self.x)/self.lc

    def step(self):
        self.opt.zero_grad()
        loss = self.H_y().mean()
        loss.backward()
        old_x = self.x.clone().detach()
        self.opt.step()


class Transporter_latent_space():
    def __init__(self, G, D, k, zy_xp, mode = 'dot', opt_mode = "adam"):
        self.G = G
        self.D = D
        self.device = G.device
        self.zy = self.G.make_hidden(10000).to(self.device).detach()
        self.mode = mode
        self.lc = k
        self.dist = "normal"
        self.opt_mode = opt_mode
        self.space = 'latent'


    def get_z_va(self):
        return self.z.data

    def modify_gradients(self, grad):
        return self.onegrads.to(G.device).to(grad.dtype)

    def set_(self, zy_xp, lr):
        self.z = nn.Parameter(zy_xp.detach().clone()).to(self.device)
        self.z.requires_grad = True
        if self.opt_mode == 'sgd':
            self.opt = torch.optim.SGD([self.z], lr)
            self.lr = lr
        elif self.opt_mode == 'adam':
            self.opt = torch.optim.Adam([self.z], lr, betas=(0.0, 0.8))
            self.lr = lr
        elif self.opt_mode == 'projected_GD':
            self.lr = lr
        else:
            raise NotImplementedError("Optimizer not implemented yet.")

    def H_zy(self):
        x = self.G(self.z)
        if self.mode=='dot':
            return - self.D(self.G(self.z))/self.lc + torch.norm(self.z - self.zy.clone().detach() + 0.001, p=2, dim=1)
        else:
            return - self.D(x)/self.lc

    def step(self):
        if self.opt_mode != 'projected_GD':
            self.opt.zero_grad()
        loss = self.H_zy().mean()
        loss.backward()
        if self.dist=='uniform':
            self.opt.step()
            self.z.data.clamp_(-1, 1)

        elif self.dist=='normal':
            if self.opt_mode == 'projected_GD':
                batch_size , D = self.z.shape[0], int(sqrt(self.z.shape[1]))
                g = self.z.grad.clone()  # Store the original gradient
                g_reshaped = g.view(batch_size, 1, D*D)
                z_reshaped = self.z.view(batch_size, D*D, 1)
                # Dot product to project g onto z
                dot_product = torch.bmm(g_reshaped, z_reshaped)
                projection = ((dot_product * z_reshaped) / torch.sqrt(torch.tensor(D, dtype=torch.float32))).squeeze(-1)
                # Calculate the modified gradient
                modified_grad = g - projection
                # Step 2: Update z manually using the modified gradient
                with torch.no_grad():  # Ensures no gradients are tracked for this update
                    self.z -= self.lr * modified_grad  # Apply the step

                self.z.grad.zero_()  # Clear the gradient
                self.z.requires_grad = True  # Re-enable gradient tracking
            else:
                self.opt.step()



def discriminator_optimal_transport_from(y_or_z_xp, transporter,G, N_update=10, lr = 0.05, show = False):
    transporter.set_(y_or_z_xp, lr = lr)
    if show:
        n_cols = 7  # Number of images per row
        fig, ax = plt.subplots(N_update // 10  , n_cols)  # Dynamic figure height
        # fig.subplots_adjust(wspace=0., hspace=0.)  # Minimal spacing between images
        ax = ax.flatten()
        old_image = torch.zeros(10000, 28, 28).data.cpu()
    for i in range(N_update):
        if transporter.space == 'latent':
            transporter.lc = 1#eff_k(transporter.G, transporter.D, trial = 200)
        else:
            transporter.lc = 1#eff_K(transporter.G, transporter.D, trial = 200)
        transporter.step()
        if show and i % 10 == 0:
            if transporter.space == 'target':
                image = transporter.get_x_va().reshape(-1, 28, 28).data.cpu()
            else:
                image = G(transporter.get_z_va()).reshape(-1, 28, 28).data.cpu()
            for j in range(min(n_cols, image.size(0))):  # Ensure only 6 images are plotted
                ax_idx = i // 10 * n_cols + j
                if ax_idx < len(ax):  # Ensure we don't exceed subplot limits
                    ax[ax_idx].imshow(image[j, :, :], cmap='gray')
                    ax[ax_idx].axis('off')
                    ax[ax_idx].set_title(f'Iteration {i}', fontsize=8)
    if show:
        plt.tight_layout()
        plt.show()




def make_image(G, D, batchsize, N_update=100, ot=True, space='target', k=1, lr=0.05, optmode='sgd', show = False):
    z = G.make_hidden(batchsize).to(G.device)
    #with torch.no_grad():
    if ot:
        z_xp = z
        if space == 'target':
            y_xp = G(z_xp)
            T = Transporter_target_space(G, D, k, y_xp, opt_mode= optmode)
            z_xp = y_xp
        elif space == 'latent':
            T = Transporter_latent_space(G, D, k, z_xp, opt_mode= optmode)
        else:
            raise NotImplementedError("Space not implemented yet.")
        discriminator_optimal_transport_from(z_xp, T, G, N_update, show= show)
        if space == 'target':
            y = T.get_x_va().data
        else:
            tz_y = T.get_z_va().data
            y = G(tz_y)
    else:
        y = G(z)
    return y.reshape(batchsize, 28, 28).data.cpu()
