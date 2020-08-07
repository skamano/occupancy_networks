import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from im2mesh.encoder.conv import Resnet18

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, c_dim, device=None, image_channels=3):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = Resnet18(c_dim, use_linear=False).to(device)  
        # 512 --> c_dim for use_linear=True, stays at 512 for use_linear=False 
        self.fc1 = nn.Linear(512, c_dim)
        self.fc2 = nn.Linear(512, c_dim)
        self.fc3 = nn.Linear(c_dim, 512)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 96, kernel_size=5, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 16, padding=2, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, padding=2, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_channels, padding=2, kernel_size=4, stride=2),
            # nn.BatchNorm2d(image_channels),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
