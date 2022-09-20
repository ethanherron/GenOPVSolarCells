import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import save_image, make_grid
import acoustics
from acoustics.generator import *
from models import ContextUnet_3lvls



def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T))

    def sample(self, n_sample, size, device, noise_type):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        def gen_noise(size, type):
            noise = type(size)
            return rearrange(noise, '(b c h w) -> b c h w', c=1, h=128, w=128)

        # x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        x_i = gen_noise(n_sample*128*128,noise_type)
        x_i = torch.FloatTensor(x_i).to(device)
        # print(' ')
        # print('x_i - ', x_i.shape)
        # print(' ')
        # exit()

        x_i_store = [] # keep track of generated steps in case want to plot something 
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            # x_i = x_i.repeat(2,1,1,1)
            # t_is = t_is.repeat(2,1,1,1)

            # z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            z = gen_noise(n_sample*128*128,noise_type)
            z = torch.FloatTensor(z).to(device)

            # split predictions and compute weighting
            eps = self.nn_model(x_i, t_is)
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store




# Init device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load validation dataset for inference
data_path = '/data/EthanHerron/data/material_microstructures/validation_microstructures.npz'
dataset = MicrostructureDataset(data_path)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)

# Init model and load trained weights
nn_model = ContextUnet_3lvls(in_channels=1, n_feat=128)
model = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=1000, device=device, drop_prob=0.1)
model.load_state_dict(torch.load('/data/EthanHerron/GenerativeModeling/Conditional_Diffusion_MNIST/results/Diffusion_Micro/model_76.pth'))
model.eval()
model.to(device)

save_dir = './results_inference/diffusion_whitenoise'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Write inference loop
generate_morphologies = []
# pbar = tqdm(dataloader)
with torch.no_grad():
    x_real = dataset[20:40]
    x_gen, _ = model.sample(x_real.size(0), tuple(x_real[0].shape), device, noise_type=white)
    generate_morphologies.extend(x_gen.cpu().numpy())
    x_all = torch.cat([x_gen.cpu(), x_real])
    grid = make_grid(x_all*-1 + 1, nrow=10)
    save_image(grid, save_dir + f"image.png")
    print('saved image at ' + save_dir + f"image.png")

a = np.array(generate_morphologies, dtype=np.float32)

# save_path = './results_inference/diffusion_pinknoise'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# np.savez_compressed(os.path.join(save_path,'validation_microstructures_diffusion.npz'),np.array(a))