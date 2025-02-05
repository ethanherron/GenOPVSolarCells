"""
Train the JF_Net regressor for predicting J_sc and FF from microstructure images.

This script loads a NumPy dataset, defines the CNN regressor,
and trains it using an MSE loss (with an optional gradient penalty).
Only core training code and logging remain.
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
import models
from models import JF_Net
from torchvision import transforms
from tqdm import tqdm


def get_next_run(output_path):
	"""
	Create a new output subdirectory for this run.
	"""
	idx = 0
	run_path = os.path.join(output_path, "run_{:03d}".format(idx))
	while os.path.exists(run_path):
		idx += 1
		run_path = os.path.join(output_path, "run_{:03d}".format(idx))
	os.makedirs(run_path)
	return run_path


class NumpyDataset(Dataset):
	"""
	PyTorch Dataset for data stored in a NumPy file.
	Expects each entry to be a tuple (image, J, FF).
	"""
	def __init__(self, path):
		super().__init__()
		self.data = np.load(path, allow_pickle=True)
		self.transform = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, idx):
		img = self.transform(np.float32(self.data[idx, 0]))
		J = np.float32(self.data[idx, 1])
		FF = np.float32(self.data[idx, 2])
		return img, J, FF

	def __len__(self):
		return self.data.shape[0]


def get_dataloader(path, batchsize):
	"""
	Returns a DataLoader for the dataset.
	"""
	dataset = NumpyDataset(path)
	return DataLoader(dataset, batch_size=batchsize, drop_last=True, shuffle=True)


def compute_R2(pred, target):
	"""
	Compute the R^2 statistic.
	"""
	SS_res = np.sum((target - pred) ** 2)
	SS_tot = np.sum((target - np.mean(pred)) ** 2)
	return 1 - (float(SS_res) / SS_tot)


def train_JF(smooth, model, device, train_loader, optimizer, epoch, args):
	"""
	Train the regressor for one epoch.
	"""
	model.train()
	total_loss = 0
	total_J_R2 = 0
	total_ff_R2 = 0
	n_batches = 0

	for batch_idx, (data, J, ff) in enumerate(train_loader):
		data, J, ff = data.to(device), J.to(device), ff.to(device)
		optimizer.zero_grad()
		model.zero_grad()

		if smooth < 0:
			output = model(data)
			pred_J = output[:, 0]
			pred_ff = output[:, 1]
			loss = F.mse_loss(pred_J, J.float(), reduction='mean') + F.mse_loss(pred_ff, ff.float(), reduction='mean')
		else:
			LAMBDA = 1
			data.requires_grad_(True)
			output = model(data)
			gradients = autograd.grad(
				outputs=output,
				inputs=data,
				grad_outputs=torch.ones_like(output, device=device),
				create_graph=True,
				retain_graph=True,
				only_inputs=True
			)[0]
			gradients = gradients.view(gradients.size(0), -1)
			GP = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

			pred_J = output[:, 0]
			pred_ff = output[:, 1]
			loss = (F.mse_loss(pred_J, J.float(), reduction='mean') +
					F.mse_loss(pred_ff, ff.float(), reduction='mean') + GP)

		loss.backward()
		optimizer.step()

		# Compute R^2 scores.
		J_R2 = compute_R2(pred_J.cpu().detach().numpy(), J.float().cpu().detach().numpy())
		ff_R2 = compute_R2(pred_ff.cpu().detach().numpy(), ff.float().cpu().detach().numpy())

		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tJ_R^2: {:.4f}\tff_R^2: {:.4f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100 * batch_idx / len(train_loader), loss.item(), J_R2, ff_R2
			))
		total_loss += loss.item()
		total_J_R2 += J_R2
		total_ff_R2 += ff_R2
		n_batches += 1

	avg_loss = total_loss / n_batches
	avg_J_R2 = total_J_R2 / n_batches
	avg_ff_R2 = total_ff_R2 / n_batches
	print('Train Epoch: {} Average Loss: {:.4f}\tAverage J_R^2: {:.4f}\tAverage ff_R^2: {:.4f}'.format(
		epoch, avg_loss, avg_J_R2, avg_ff_R2
	))
	return avg_loss, avg_J_R2, avg_ff_R2


def test_JF(model, device, test_loader, epoch, args):
	"""
	Evaluate the model on the test set.
	"""
	model.eval()
	total_loss = 0
	total_J_R2 = 0
	total_ff_R2 = 0
	n_batches = 0

	with torch.no_grad():
		for batch_idx, (data, J, ff) in enumerate(test_loader):
			data, J, ff = data.to(device), J.to(device), ff.to(device)
			output = model(data)
			pred_J = output[:, 0]
			pred_ff = output[:, 1]
			loss = F.mse_loss(pred_J, J.float(), reduction='mean') + F.mse_loss(pred_ff, ff.float(), reduction='mean')
			total_loss += loss.item()

			J_R2 = compute_R2(pred_J.cpu().detach().numpy(), J.float().cpu().detach().numpy())
			ff_R2 = compute_R2(pred_ff.cpu().detach().numpy(), ff.float().cpu().detach().numpy())
			total_J_R2 += J_R2
			total_ff_R2 += ff_R2

			if batch_idx % args.log_interval == 0:
				print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tJ_R^2: {:.4f}\tff_R^2: {:.4f}'.format(
					epoch, batch_idx * len(data), len(test_loader.dataset),
					100 * batch_idx / len(test_loader), loss.item(), J_R2, ff_R2
				))
			n_batches += 1

	avg_loss = total_loss / n_batches
	avg_J_R2 = total_J_R2 / n_batches
	avg_ff_R2 = total_ff_R2 / n_batches
	print('Test Epoch: {} Average Loss: {:.4f}\tAverage J_R^2: {:.4f}\tAverage ff_R^2: {:.4f}'.format(
		epoch, avg_loss, avg_J_R2, avg_ff_R2
	))
	return avg_loss, avg_J_R2, avg_ff_R2


def main():
	"""
	Main training loop.
	"""
	parser = argparse.ArgumentParser(
		description='Train CNN model to predict properties (J_sc and FF) based on morphology.'
	)
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
	parser.add_argument('--epochs', type=int, default=25, help='Number of epochs (default: 25)')
	parser.add_argument('--lr', type=float, default=1E-4, help='Learning rate (default: 1E-4)')
	parser.add_argument('--gamma', type=float, default=0.7, help='LR scheduler gamma (default: 0.7)')
	parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
	parser.add_argument('--log_interval', type=int, default=10, help='Logging interval (default: 10)')
	parser.add_argument('--save_model', action='store_true', default=True, help='Save the trained model')
	parser.add_argument('--model_dir', default='./results/Jff_pretraining', help='Directory for model checkpoints')
	parser.add_argument('--smooth', type=int, default=0, help='Use Lipschitz constraint? (set <0 for no gradient penalty)')
	parser.add_argument('--gpu', type=int, default=0, help='GPU id (default: 0)')

	args = parser.parse_args()
	# Create a new run output directory.
	args.model_dir = get_next_run(args.model_dir)

	use_cuda = torch.cuda.is_available()
	torch.cuda.set_device(args.gpu)
	torch.manual_seed(args.seed)
	device = torch.device("cuda" if use_cuda else "cpu")

	# Load training and test dataloaders.
	train_loader = get_dataloader(
		path='/data/Joshua/DARPA_data/augmented_JF_filtered_norm_balanced_train.npy',
		batchsize=args.batch_size
	)
	test_loader = get_dataloader(
		path='/data/Joshua/DARPA_data/augmented_JF_filtered_norm_test.npy',
		batchsize=args.batch_size
	)

	model = JF_Net().to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

	for epoch in tqdm(range(1, args.epochs + 1)):
		train_JF(args.smooth, model, device, train_loader, optimizer, epoch, args)
		test_JF(model, device, test_loader, epoch, args)
		scheduler.step()

	if args.save_model:
		torch.save(model.state_dict(), os.path.join(args.model_dir, "regressor.pt"))
		print("Model saved to", os.path.join(args.model_dir, "regressor.pt"))

if __name__ == '__main__':
	main()