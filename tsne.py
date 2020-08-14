import numpy as np
import matplotlib.pyplot as plt
from tsnecuda import TSNE
import torch
# import torch.distributions as dist
import os
import shutil
import argparse
import copy
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
from im2im.autoencoder import VAE
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
from im2mesh.utils.voxels import VoxelGrid


parser = argparse.ArgumentParser(
    description='Generate TSNE from test inputs.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
# generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(out_dir, 'time_tsne_full.pkl')
out_time_file_class = os.path.join(out_dir, 'time_tsne.pkl')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)
# model = config.get_model(cfg, device=device, dataset=train_dataset)
if 'encoder_path' in cfg['model'].keys():
    # load pre-trained encoder
    print('loading encoder from VAE')
    vae = VAE(c_dim=cfg['model']['c_dim'], device=device)
    vae_state_dict = torch.load(cfg['model']['encoder_path'])['model']
    vae.load_state_dict(vae_state_dict)
    model.encoder = copy.deepcopy(vae.encoder)
    for param in model.encoder.parameters():  # freeze encoder
        param.requires_grad = False

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Statistics
time_dicts = []

# Generate
model.eval()

# Count how many models already created
model_counter = defaultdict(int)

tsne_features = {}
print('Generating image embeddings...')
for it, data in enumerate(tqdm(test_loader)):
    # in_dir = os.path.join(out_dir, 'input')

    # Get index etc.
    idx = data['idx'].item()
    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    
    modelname = model_dict['model']
    category_id = model_dict.get('category', 'n/a')

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        # in_dir = os.path.join(in_dir, str(category_id))

        folder_name = str(category_id)
        if category_name != 'n/a':
            folder_name = str(folder_name) + '_' + category_name.split(',')[0]

    # Create directories if necessary

    # if not os.path.exists(in_dir):
        # os.makedirs(in_dir)
    
    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    # if cfg['generation']['copy_input']:
    # Save inputs
    if input_type == 'img':
        # inputs_path = os.path.join(in_dir, '%s.jpg' % modelname)
        with torch.no_grad():
            x = data['inputs'].to(device)
            z = model.encoder(x).squeeze(0).cpu().numpy()
            try:
                tsne_features[category_id].append(z)
            except KeyError:
                tsne_features[category_id] = []
                tsne_features[category_id].append(z)
            # tsne_features.append(model.encoder(data['inputs'].to(device)).squeeze(0).cpu().numpy())
        # inputs = data['inputs'].squeeze(0).cpu()
        # images.append(inputs)
        # visualize_data(inputs, 'img', inputs_path)
        # out_file_dict['in'] = inputs_path

    # Copy to visualization directory for first vis_n_output samples
    c_it = model_counter[category_id]
    # if c_it < vis_n_outputs:
    #     # Save output files
    #     img_name = '%02d.off' % c_it
    #     for k, filepath in out_file_dict.items():
    #         ext = os.path.splitext(filepath)[1]
    #         out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
    #                                 % (c_it, k, ext))
    #         shutil.copyfile(filepath, out_file)

    model_counter[category_id] += 1

# visualize t-SNE according to class
colors = ['r', 'b', 'g']
for category_id, color in zip(tsne_features.keys(), colors):
    X = np.array(tsne_features[category_id])
    X_embedded = TSNE().fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color)

plt.legend(list(tsne_features.keys()))
plt.savefig(os.path.join(out_dir, 'tsne.png'))

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
time_df_class = time_df.groupby(by=['class name']).mean()
time_df_class.to_pickle(out_time_file_class)

# Print results
time_df_class.loc['mean'] = time_df_class.mean()
print('Timings [s]:')
print(time_df_class)

