import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm



def train_unconditional_diffusion(n_epochs, dataloader, inf_samples, model, optimizer, lrate, device, save_dir, save_weights_freq=2):
    for ep in range(n_epochs):
            print(f'epoch {ep}')
            model.train()

            # linear lrate decay
            optimizer.param_groups[0]['lr'] = lrate*(1-ep/n_epochs)

            pbar = tqdm(dataloader)
            loss_ema = None
            for x in pbar:
                optimizer.zero_grad()
                x = x.to(device)
                loss = model(x)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
                optimizer.step()
            
            # for eval, save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            model.eval()
            with torch.no_grad():
                # append some real images at bottom, order by class also
                x_real = inf_samples

                # gen images
                x_gen, _ = model.sample(x_real.size(0), tuple(x_real[0].shape), device)

                x_all = torch.cat([x_gen.cpu(), x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}.png")

            if ep % save_weights_freq == 0:
                torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")
                print('saved model at ' + save_dir + f"model_{ep}.pth")