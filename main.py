import argparse, warnings
import numpy as np
import torch
import models
from models import *
import trainers
from trainers import *
from data import *


def main(args):
    # ------------------------
    # 1 MISC INITS
    # ------------------------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # ------------------------
    # 2 INIT DATA & NN
    # ------------------------
    data_path = '/data/EthanHerron/data/material_microstructures/microstructures.npz'
    dataset = MicrostructureDataset(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    # ------------------------
    # 3 INIT MODEL & TRAIN
    # ------------------------

    if args.method == 'Diffusion':
        nn_model = ContextUnet_3lvls(in_channels=1, n_feat=args.n_feat)
        model = models.DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=args.n_T, device=device, drop_prob=0.1)
        model.to(device)
        print("# of params in model: ", sum(a.numel() for a in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)

        train_unconditional_diffusion(n_epochs=args.n_epochs, 
                                      dataloader=dataloader, 
                                      inf_samples=dataset[80:100],
                                      model=model,
                                      optimizer=optimizer, 
                                      lrate=args.lrate, 
                                      device=device, 
                                      save_dir=args.save_dir
                                      )

    elif args.method == 'GAN':
        generator = Generator_Normal()
        discriminator = Discriminator_Micro()
        generator.to(device)
        discriminator.to(device)
        print("# of params in generator: ", sum(a.numel() for a in generator.parameters()))
        print("# of params in discriminator: ", sum(a.numel() for a in discriminator.parameters()))

        opt_g = torch.optim.Adam(generator.parameters(), lr=args.lrate)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lrate)
        optimizers = [opt_g, opt_d]

        train_unconditional_GAN(n_epochs=args.n_epochs, 
                                dataloader=dataloader, 
                                inf_samples=dataset[80:100],
                                generator=generator,
                                discriminator=discriminator,
                                optimizers=optimizers, 
                                lrate=args.lrate, 
                                device=device, 
                                save_dir=args.save_dir
                                )

    elif args.method == 'WGAN':
        generator = Generator_Normal()
        discriminator = Discriminator_WGAN_Micro()
        generator.to(device)
        discriminator.to(device)
        print("# of params in generator: ", sum(a.numel() for a in generator.parameters()))
        print("# of params in discriminator: ", sum(a.numel() for a in discriminator.parameters()))
        opt_g = torch.optim.RMSprop(generator.parameters(), lr=args.lrate)
        opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=args.lrate)
        optimizers = [opt_g, opt_d]

        train_unconditional_WGAN(n_epochs=args.n_epochs, 
                                dataloader=dataloader, 
                                inf_samples=dataset[80:100],
                                generator=generator,
                                discriminator=discriminator,
                                optimizers=optimizers, 
                                lrate=args.lrate, 
                                device=device, 
                                save_dir=args.save_dir
                                )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unconditional Diffusion Models')
    parser.add_argument('-m','--method', default='Diffusion', type=str,
                        help='Learning Methodology: Diffusion or GAN')
    parser.add_argument('-dt', '--data_type', default='Micro', type=str,
                        help='Topology Optimization or Material Microstructures')
    parser.add_argument('--n_feat',default=128, type=int,
                        help='Number of feature maps in network arch.')
    parser.add_argument('--n_T',default=1000, type=int,
                        help='Number of time steps in diffusion process.')
    parser.add_argument('--lrate',default=1e-4, type=float,
                        help='Model learning rate.')
    parser.add_argument('--save_dir', default='./Logs',
                        type=str,help='path to directory for storing the checkpoints etc.')
    parser.add_argument('-b','--batch_size', default=16, type=int,
                        help='Batch size')
    parser.add_argument('-ep','--n_epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('-g','--gpu', default=0, type=int,
                        help='gpu id to use from 0 to 3')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='num workers for data module.')
    parser.add_argument('--debug', default=False, type=bool,
                        help='fast_dev_run argument')
    hparams = parser.parse_args()
    main(hparams)