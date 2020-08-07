import os
from tqdm import trange
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch import distributions as dist
from torchvision.utils import save_image
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer


class Trainer(BaseTrainer):
    ''' Trainer object for image-to-image autoencoder.

    Args:
        model (nn.Module): Image-to-image autoencoder model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        eval_sample (bool): whether to evaluate samples
    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.eval_sample = eval_sample
        self.images_saved = 0
        self.input_img = None
        self.recon_img = None

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss, __, __ = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step. Evaluates image encoding loss for validation images 

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        loss, metric, KLD = self.compute_loss(data)
        device = self.device
        eval_dict = {'loss': loss, 'metric': metric, 'KLD': KLD}

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data. Should output side-by-side comparisons of reconstructed images and encoded images.
        Args:
            data (dict): data dictionary
        '''
        device = self.device

        # TODO: implement visualizations for input image and reconstructed image
        img = torch.cat([self.input_img, self.recon_img])
        path = os.path.join(self.vis_dir, 'sample_{}.png'.format(self.images_saved))
        self.images_saved += 1
        save_image(img.data.cpu(), path)

    def compute_loss(self, data, metric='MSE'):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
            data['inputs'] is a (B, C, H, W)-shape tensor
                B: batch size
                C: image channels
                H: image height
                W: image width
        '''
        device = self.device
        x = data['inputs'].to(device)
        recon_x, mu, logvar = self.model(x) 
        self.input_img = x  # save data for visualization
        self.recon_img = recon_x
        device = self.device
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if metric == 'BCE':
            BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
            loss = BCE + KLD
        elif metric == 'MSE':
            MSE = F.mse_loss(recon_x, x, size_average=False)
            loss = MSE + KLD
        else:
            raise NotImplementedError

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        if metric == 'BCE':
            return loss, BCE, KLD
        elif metric == 'MSE':
            return loss, MSE, KLD
        else:
            raise NotImplementedError
