import torch
import torch.nn as nn
import torch.nn.functional as F
#rom caca import pipi

def l2_norm(x):
    return torch.sqrt(torch.sum(x ** 2, dim=1))

def eff_k(G, D, trial=100):
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

def eff_K(G, D, trial=100):
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
    def __init__(self, G, D, k, zy_xp, mode, opt_mode = "adam") -> None:
        self.G = G
        self.D = D
        self.device = G.device
        self.zy = zy_xp.to(self.device)
        self.mode = mode
        self.onegrads = torch.ones(zy_xp.shape[0], 1, dtype=torch.float32)
        self.lc = k
        self.dist = "uniform"
        self.opt_mode = opt_mode

    def get_x_va(self):
        return self.x.data

    def set_(self, y_xp, lr):
        self.x = nn.Parameter(y_xp.clone()).to(self.G.device)
        if self.opt_mode == 'sgd':
            self.opt = torch.optim.SGD([self.x], lr)
        elif self.opt_mode == 'adam':
            self.opt = torch.optim.Adam([self.x], lr, betas=[0.0,0.9])
        else:
            print("Optimizer not implemented yet.")

    def H_y(self, x):
        if self.mode=='dot':
            return - self.D(x)/self.lc + torch.reshape(l2_norm(x - self.y + 0.001),self.D(x).shape)
        else:
            return - self.D(x)/self.lc

    def step(self):
        x = self.get_x_va()
        self.opt.zero_grad()
        loss = self.H_y(x)
        loss.backward(gradient= self.onegrads)
        self.opt.step()
        self.x.data.clamp_(-2, 2)

class Transporter_latent_space():
    def __init__(self, G, D, k, zy_xp, mode, opt_mode = "adam"):
        self.G = G
        self.D = D
        self.device = G.device
        self.zy = zy_xp.to(self.device)
        self.mode = mode
        self.onegrads = torch.ones(zy_xp.shape[0], 1, dtype=torch.float32).to(G.device)
        self.lc = k
        self.dist = "uniform"
        self.opt_mode = opt_mode


    def get_z_va(self):
        return self.z.data

    def modify_gradients(self, grad):
        return self.onegrads.to(G.device).to(grad.dtype)

    def set_(self, zy_xp, lr):
        self.z = nn.Parameter(zy_xp.detach().clone().to(self.G.device), requires_grad=True)
        if self.opt_mode == 'sgd':
            self.opt = torch.optim.SGD([self.z], lr)
        elif self.opt_mode == 'adam':
            self.opt = torch.optim.Adam([self.z], lr, betas=[0.0,0.9])
        else:
            print("Optimizer not implemented yet.")

    def H_zy(self, z):
        x = self.G(z)
        if self.mode=='dot':
            return - self.D(x)/self.lc + torch.reshape(l2_norm(z - torch.tensor(self.zy) + 0.001), self.D(x).shape)
        else:
            return - self.D(x)/self.lc

    def step(self):
        z = self.get_z_va()
        self.opt.zero_grad()
        loss = self.H_zy(z)
        loss.backward(gradient= self.onegrads)
        if self.dist=='uniform':
            self.opt.step()
            self.z.data.clamp_(-1, 1)

        elif self.dist=='normal':
            bs, dim, _, _  = z.shape
            prod = F.bmm(z.grad.view(bs, dim, 1), z.data.view(bs, 1, dim)).view(bs, 1, 1, 1)
            z.grad = z.grad - z.data * (prod / 11.31)  # 11.31 = sqrt(128)
            self.opt.step()


def discriminator_optimal_transport_from(y_or_z_xp, transporter, N_update=10, lr = 0.05):
    transporter.set_(y_or_z_xp, lr = lr)
    for i in range(N_update):
        transporter.step()


def make_image(G, D, batchsize, N_update=100, ot=True, mode='dot', k=1, lr=0.05, optmode='sgd'):
    z = G.make_hidden(batchsize)
    #with torch.no_grad():
    if ot:
        z_xp = z
        T = Transporter_latent_space(G, D, k, z_xp, mode=mode, opt_mode= optmode)
        discriminator_optimal_transport_from(z_xp, T, N_update)
        tz_y = T.get_z_va().data
        y = G(tz_y)
    else:
        y = G(z)
    return y.reshape(batch_size, 28, 28).data.cpu()
