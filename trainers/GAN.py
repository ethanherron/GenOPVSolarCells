import torch
from torch import autograd
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm



def calc_gradient_penalty(netD, real_data, fake_data, device):
    alpha = torch.rand(16, 1)
    alpha = alpha.expand(16, int(real_data.nelement() / 16)).contiguous()

    alpha = alpha.view(16, 1, 128, 128)              # Changed the 1 from 1
    alpha = alpha.to(device)

    fake_data = fake_data.view(16, 1, 128, 128)      # Changed the CATEGORY from 1
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty



def train_unconditional_GAN(n_epochs, dataloader, inf_samples, generator, discriminator, optimizers, lrate, device, save_dir, save_weights_freq=10):
    for ep in range(n_epochs):
            print(f'epoch {ep}')

            # linear lrate decay
            for opt in optimizers:
                opt.param_groups[0]['lr'] = lrate*(1-ep/n_epochs)

            pbar = tqdm(dataloader)
            idx = 0
            for x in pbar:
                x = x.to(device)

                # create labels
                label_real = torch.ones(x.size(0), device=device).unsqueeze(-1)
                label_fake = torch.zeros(x.size(0), device=device).unsqueeze(-1)

                if idx % 1 == 0:
                    for p in generator.parameters(): p.requires_grad_(True) # Train gen params
                    for p in discriminator.parameters(): p.requires_grad_(False) # Freeze disc params
                    for p in generator.parameters(): p.grad=None # Zero grads of generator for update
                    
                    noise = torch.randn(16, 128)
                    noise = noise.to(device)
                    
                    gen_density = generator(noise)

                    fake_pred = discriminator(gen_density)
                    gen_loss = F.binary_cross_entropy_with_logits(fake_pred, label_real)

                    gen_loss.backward(retain_graph=True)
                    optimizers[0].step()
                    
                if idx % 2 == 0 :
                    for p in generator.parameters(): p.requires_grad_(False) # Freeze gen params
                    for p in discriminator.parameters(): p.requires_grad_(True) # Train disc params
                    for p in discriminator.parameters(): p.grad=None # Zero grads of discriminator for update

                    real_pred = discriminator(x)
                    fake_pred = discriminator(gen_density.detach())
                    disc_loss = F.binary_cross_entropy_with_logits(real_pred, label_real) + F.binary_cross_entropy_with_logits(fake_pred, label_fake)

                    disc_loss.backward(retain_graph=True)
                    optimizers[1].step()
                idx += 1
            
            # for eval, save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            with torch.no_grad():
                noise = torch.randn(inf_samples.size(0), 128)
                noise = noise.to(device)
                x_gen = generator(noise)

                # append some real images at bottom, order by class also
                x_real = inf_samples

                x_all = torch.cat([x_gen.cpu(), x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}.png")

            if ep % save_weights_freq == 0:
                torch.save(generator.state_dict(), save_dir + f"generator_{ep}.pth")
                print('saved generator at ' + save_dir + f"generator_{ep}.pth")
                torch.save(discriminator.state_dict(), save_dir + f"discriminator_{ep}.pth")
                print('saved discriminator at ' + save_dir + f"discriminator_{ep}.pth")




def train_unconditional_WGAN(n_epochs, dataloader, inf_samples, generator, discriminator, optimizers, lrate, device, save_dir, save_weights_freq=10):
    for ep in range(n_epochs):
            print(f'epoch {ep}')

            # linear lrate decay
            for opt in optimizers:
                opt.param_groups[0]['lr'] = lrate*(1-ep/n_epochs)

            pbar = tqdm(dataloader)
            idx = 0
            for x in pbar:
                x = x.to(device)

                if idx % 5 == 0:
                    for p in generator.parameters(): p.requires_grad_(True) # Train gen params
                    for p in discriminator.parameters(): p.requires_grad_(False) # Freeze disc params
                    for p in generator.parameters(): p.grad=None # Zero grads of generator for update

                    noise = torch.randn(16, 128)
                    noise = noise.to(device)
                    
                    gen_density = generator(noise)

                    fake_pred = discriminator(gen_density)
                    gen_loss = - fake_pred.mean()

                    gen_loss.backward(retain_graph=True)
                    optimizers[0].step()

                for p in generator.parameters(): p.requires_grad_(False) # Freeze gen params
                for p in discriminator.parameters(): p.requires_grad_(True) # Train disc params
                for p in discriminator.parameters(): p.grad=None # Zero grads of discriminator for update

                real_pred = discriminator(x)
                fake_pred = discriminator(gen_density.detach())
                gradient_penalty = calc_gradient_penalty(discriminator, x, gen_density, device)
                
                disc_loss = fake_pred.mean() - real_pred.mean() + gradient_penalty

                disc_loss.backward(retain_graph=True)
                optimizers[1].step()

                idx += 1
            
            # for eval, save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            with torch.no_grad():
                noise = torch.randn(inf_samples.size(0), 128)
                noise = noise.to(device)
                x_gen = generator(noise)

                # append some real images at bottom, order by class also
                x_real = inf_samples

                x_all = torch.cat([x_gen.cpu(), x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}.png")

            if ep % save_weights_freq == 0:
                torch.save(generator.state_dict(), save_dir + f"generator_{ep}.pth")
                print('saved generator at ' + save_dir + f"generator_{ep}.pth")
                torch.save(discriminator.state_dict(), save_dir + f"discriminator_{ep}.pth")
                print('saved discriminator at ' + save_dir + f"discriminator_{ep}.pth")