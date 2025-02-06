import os
import torch
from torch import autograd
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

def calc_gradient_penalty(netD, real_data, fake_data, device):
    """
    Computes the gradient penalty for WGAN-GP using a random interpolation
    between real_data and fake_data.
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_data.detach() + (1 - alpha) * fake_data.detach()
    interpolates.requires_grad_(True)
    
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

def adjust_lr(optimizer, initial_lr, epoch, total_epochs):
    """
    Linearly decays learning rate based on the current epoch.
    """
    lr = initial_lr * (1 - epoch / total_epochs)
    for group in optimizer.param_groups:
        group['lr'] = lr

def train_diffusion(n_epochs, dataloader, inf_samples, model, optimizer, initial_lr, device, save_dir, save_weights_freq=2):
    """
    Trains a diffusion model with AMP.
    
    Parameters:
      n_epochs (int): Number of epochs to train.
      dataloader (DataLoader): Training data loader.
      inf_samples (Tensor): Samples used for evaluation.
      model (nn.Module): Diffusion model.
      optimizer (Optimizer): Optimizer for model parameters.
      initial_lr (float): Initial learning rate (linearly decayed).
      device (torch.device): Device on which training occurs.
      save_dir (str): Directory to save images and model weights.
      save_weights_freq (int): Save frequency (in epochs) for model weights.
    """
    scaler = torch.cuda.amp.GradScaler()
    
    for ep in range(n_epochs):
        print(f"Epoch {ep}/{n_epochs}")
        model.train()
        adjust_lr(optimizer, initial_lr, ep, n_epochs)
        
        pbar = tqdm(dataloader, desc=f"Epoch {ep} Training")
        loss_ema = None
        
        for x in pbar:
            optimizer.zero_grad()
            x = x.to(device, non_blocking=True)
            
            # Forward pass under AMP autocast
            with torch.amp.autocast('cuda'):
                loss = model(x)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_val = loss.item()
            loss_ema = loss_val if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_val
            pbar.set_postfix(loss=f"{loss_ema:.4f}")
        
        # Evaluation: generate grid image using generated and real samples.
        model.eval()
        with torch.no_grad():
            x_real = inf_samples.to(device, non_blocking=True)
            # Optionally, you can use autocast here as well.
            with torch.amp.autocast('cuda'):
                x_gen, _ = model.sample(x_real.size(0), tuple(x_real[0].shape), device)
            x_all = torch.cat([x_gen.cpu(), x_real.cpu()])
            grid = make_grid(x_all * -1 + 1, nrow=10)
            image_path = os.path.join(save_dir, f"image_ep{ep}.png")
            save_image(grid, image_path)
            print(f"Saved image: {image_path}")
        
        if ep % save_weights_freq == 0:
            model_path = os.path.join(save_dir, f"model_{ep}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model weights: {model_path}")

def train_gan(n_epochs, dataloader, inf_samples, generator, discriminator, optimizers, initial_lr, device, save_dir, gan_type="standard", save_weights_freq=10):
    """
    Trains a GAN with AMP using either the standard (BCE loss) formulation or 
    the WGAN formulation with gradient penalty.
    
    Parameters:
      n_epochs (int): Number of epochs.
      dataloader (DataLoader): Training data loader.
      inf_samples (Tensor): Samples for evaluation.
      generator (nn.Module): Generator network.
      discriminator (nn.Module): Discriminator network.
      optimizers (list): [generator_optimizer, discriminator_optimizer].
      initial_lr (float): Initial learning rate (linearly decayed).
      device (torch.device): Training device.
      save_dir (str): Directory to save images and weights.
      gan_type (str): "standard" or "wgan" (determines loss formulation).
      save_weights_freq (int): Frequency (in epochs) to save model weights.
    """
    gen_opt, disc_opt = optimizers
    scaler = torch.cuda.amp.GradScaler()
    
    for ep in range(n_epochs):
        print(f"Epoch {ep}/{n_epochs}")
        adjust_lr(gen_opt, initial_lr, ep, n_epochs)
        adjust_lr(disc_opt, initial_lr, ep, n_epochs)
        
        pbar = tqdm(dataloader, desc=f"Epoch {ep} Training")
        idx = 0
        
        for x in pbar:
            x = x.to(device, non_blocking=True)
            
            if gan_type == "standard":
                # Create labels
                label_real = torch.ones(x.size(0), device=device).unsqueeze(-1)
                label_fake = torch.zeros(x.size(0), device=device).unsqueeze(-1)
                
                # Generator update
                for p in generator.parameters():
                    p.requires_grad = True
                for p in discriminator.parameters():
                    p.requires_grad = False
                generator.zero_grad()
                
                noise = torch.randn(x.size(0), 128, device=device)
                with torch.amp.autocast('cuda'):
                    gen_out = generator(noise)
                    fake_pred = discriminator(gen_out)
                    gen_loss = F.binary_cross_entropy_with_logits(fake_pred, label_real)
                scaler.scale(gen_loss).backward()
                scaler.step(gen_opt)
                scaler.update()
                
                # Discriminator update every 2 iterations
                if idx % 2 == 0:
                    for p in generator.parameters():
                        p.requires_grad = False
                    for p in discriminator.parameters():
                        p.requires_grad = True
                    discriminator.zero_grad()
                    
                    with torch.amp.autocast('cuda'):
                        real_pred = discriminator(x)
                        fake_pred = discriminator(gen_out.detach())
                        disc_loss = (F.binary_cross_entropy_with_logits(real_pred, label_real) +
                                     F.binary_cross_entropy_with_logits(fake_pred, label_fake))
                    scaler.scale(disc_loss).backward()
                    scaler.step(disc_opt)
                    scaler.update()
            
            elif gan_type == "wgan":
                # Generator update every 5 iterations.
                if idx % 5 == 0:
                    for p in generator.parameters():
                        p.requires_grad = True
                    for p in discriminator.parameters():
                        p.requires_grad = False
                    generator.zero_grad()
                    
                    noise = torch.randn(x.size(0), 128, device=device)
                    with torch.amp.autocast('cuda'):
                        gen_out = generator(noise)
                        fake_pred = discriminator(gen_out)
                        gen_loss = -fake_pred.mean()
                    scaler.scale(gen_loss).backward()
                    scaler.step(gen_opt)
                    scaler.update()
                
                # Discriminator update.
                for p in generator.parameters():
                    p.requires_grad = False
                for p in discriminator.parameters():
                    p.requires_grad = True
                discriminator.zero_grad()
                
                # Compute discriminator predictions under autocast.
                with torch.amp.autocast('cuda'):
                    real_pred = discriminator(x)
                    # If not updated above, compute a fresh gen_out.
                    if idx % 5 != 0:
                        noise = torch.randn(x.size(0), 128, device=device)
                        gen_out = generator(noise)
                    fake_pred = discriminator(gen_out.detach())
                    disc_loss_partial = fake_pred.mean() - real_pred.mean()
                # Compute gradient penalty in full precision for stability.
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    gp = calc_gradient_penalty(discriminator, x, gen_out, device)
                disc_loss = disc_loss_partial + gp
                scaler.scale(disc_loss).backward()
                scaler.step(disc_opt)
                scaler.update()
            
            idx += 1
        
        # Evaluation: generate sample grid image.
        with torch.no_grad():
            noise = torch.randn(inf_samples.size(0), 128, device=device)
            with torch.amp.autocast('cuda'):
                generated = generator(noise)
            x_real = inf_samples.to(device, non_blocking=True)
            x_all = torch.cat([generated.cpu(), x_real.cpu()])
            grid = make_grid(x_all * -1 + 1, nrow=10)
            image_path = os.path.join(save_dir, f"image_ep{ep}.png")
            save_image(grid, image_path)
            print(f"Saved image: {image_path}")
        
        if ep % save_weights_freq == 0:
            gen_path = os.path.join(save_dir, f"generator_{ep}.pth")
            disc_path = os.path.join(save_dir, f"discriminator_{ep}.pth")
            torch.save(generator.state_dict(), gen_path)
            torch.save(discriminator.state_dict(), disc_path)
            print(f"Saved generator weights: {gen_path}\nSaved discriminator weights: {disc_path}")
