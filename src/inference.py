import os
import argparse
import numpy as np
import torch
from networks import (
    DDPM,
    Unet,
    Generator
)

import models
from models import *

def infer_gan(args, device):
    print("Running GAN inference...")
    # Initialize the GAN model.
    model = Generator_Normal()
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.eval()
    model.to(device)

    # Generate noise input: shape [num_samples, 128]
    noise = torch.randn(args.num_samples, 128, device=device)
    with torch.no_grad():
        x_gen = model(noise)
    
    generated_samples = x_gen.cpu().numpy()

    # Prepare the save directory.
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    save_file = os.path.join(args.save_dir, "output.npz")
    np.savez_compressed(save_file, generated_samples=generated_samples)
    print(f"Saved GAN inference output to {save_file}.")

def infer_diffusion(args, device):
    print("Running Diffusion inference...")
    # Initialize the diffusion model.
    nn_model = Unet(in_channels=1, n_feat=128)
    model = models.DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=1000, device=device, drop_prob=0.1)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.eval()
    model.to(device)
    
    # Define a default sample shape (modify if necessary).
    sample_shape = (1, 64, 64)
    with torch.no_grad():
        x_gen, _ = model.sample(args.num_samples, sample_shape, device)
    
    generated_samples = x_gen.cpu().numpy()
    
    # Prepare the save directory.
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    save_file = os.path.join(args.save_dir, "output.npz")
    np.savez_compressed(save_file, generated_samples=generated_samples)
    print(f"Saved Diffusion inference output to {save_file}.")

def main():
    parser = argparse.ArgumentParser(description="Inference script for GAN and Diffusion models.")
    parser.add_argument("--mode", type=str, choices=["gan", "diffusion"], required=True,
                        help="Model type for inference: gan or diffusion.")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of samples to generate during inference.")
    parser.add_argument("--model_weights", type=str, required=True,
                        help="Path to the trained model weights.")
    parser.add_argument("--save_dir", type=str, default="./results_inference",
                        help="Directory to save the inference output.")
    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Append mode-specific subdirectory.
    if args.mode == "gan":
        args.save_dir = os.path.join(args.save_dir, "wgan")
        infer_gan(args, device)
    else:
        args.save_dir = os.path.join(args.save_dir, "diffusion")
        infer_diffusion(args, device)

if __name__ == "__main__":
    main()
