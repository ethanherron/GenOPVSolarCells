import os
import numpy as np
import torch
import models
from models import *
import trainers
from trainers import *
from data import *
from tqdm import tqdm

from torchvision.utils import save_image, make_grid


# Init device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load validation dataset for inference
data_path = '/data/EthanHerron/data/material_microstructures/validation_microstructures.npz'
dataset = MicrostructureDataset(data_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2)

# Init model and load trained weights
model = Generator_Normal()
model.load_state_dict(torch.load('/data/EthanHerron/GenerativeModeling/Conditional_Diffusion_MNIST/results/WGAN_Micro_v2i_01/generator_90.pth'))
model.eval()
model.to(device)

# Write inference loop
generate_morphologies = []
pbar = tqdm(dataloader)
with torch.no_grad():
    for x in pbar:
        noise = torch.randn(x.size(0), 128)
        noise = noise.to(device)
        x_gen = model(noise)
        generate_morphologies.extend(x_gen.cpu().numpy())


a = np.array(generate_morphologies, dtype=np.float32)


save_path = './results_inference/wgan'
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.savez_compressed(os.path.join(save_path,'validation_microstructures_wgan.npz'),np.array(a))

