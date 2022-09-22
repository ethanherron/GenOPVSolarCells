from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import models
from models import JF_Net
import operator
import scipy
from scipy.stats import linregress
from tqdm import tqdm

# loads data and randomly visualizes microstructure and corresponding J value
def visualize_data(path, size=5):
	ds = np.load(path, allow_pickle=True)
	idx = np.random.choice(np.arange(len(ds)), size)
	for i in idx:
		img = ds[i][0]
		plt.imshow(img)
		plt.show()
		print(ds[i][1], '      ID:', i)

#generates experiment directory
def get_next_run(output_path):
    idx = 0
    path = os.path.join(output_path, "run_{:03d}".format(idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "run_{:03d}".format(idx))
    os.makedirs(path)
    return path

#loads dataset and convert to a iterable for JF data
class NumpyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = np.load(path, allow_pickle=True)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        return (self.transform(np.float32(self.data[idx,0])),
        		 np.float32(self.data[idx,1]),
        		 np.float32(self.data[idx,2]))

    def __len__(self):
        return self.data.shape[0]

#returns data loader
def get_dataloader(path, batchsize):
    ds = NumpyDataset(path)
    dl = torch.utils.data.DataLoader(ds, batch_size=batchsize, drop_last=True, shuffle=True)
    return dl

#compute R^2 statistic
def compute_R2(pred, target, mode):
    SS_Residual = sum((target-pred)**2)
    SS_Total = sum((target-np.mean(pred))**2)
    # print()
    # print('res dims: ', SS_Residual.shape, ' total dims: ', SS_Total.shape)
    # print()
    R2 = 1 - (float(SS_Residual))/SS_Total
    return R2

# train high-fidelity model to predict J_sc and FF
def train_JF(smooth, model, device, train_loader, optimizer, epoch, argsp):
	model.train()
	total_loss = 0
	total_JR = 0
	total_ffR = 0
	n = 0
	for batch_idx, (data, J, ff) in enumerate(train_loader):
		data, J, ff = data.to(device), J.to(device), ff.to(device)
		optimizer.zero_grad()
		model.zero_grad()

		if smooth < 0:
			output = model(data)
			pred_J= output[:,0]
			pred_ff = output[:,1]
			J_loss = F.mse_loss(pred_J, J.float(), reduction='mean')
			ff_loss = F.mse_loss(pred_ff, ff.float(), reduction='mean')
			loss  =  J_loss + ff_loss

		else:
			LAMBDA = 1
			data.requires_grad_(True)
			output = model(data)
			#compute gradient penalty to ensure norm of gradients are at 1
			#returns a tuple, taking first element
			gradients = autograd.grad(outputs=output, inputs=data, grad_outputs=torch.ones(output.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
			gradients = gradients.view(gradients.size(0), -1)
			GP = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

			#compute MSE loss
			pred_J = output[:,0]
			pred_ff = output[:,1]
			J_loss = F.mse_loss(pred_J, J.float(), reduction='mean')
			ff_loss = F.mse_loss(pred_ff, ff.float(), reduction='mean')
			#compute total loss
			loss = J_loss + ff_loss + GP

		loss.backward()
		J_R2 = compute_R2(pred_J.cpu().detach().numpy(), J.float().cpu().detach().numpy(), mode='train')
		ff_R2 = compute_R2(pred_ff.cpu().detach().numpy(), ff.float().cpu().detach().numpy(), mode='train')
		optimizer.step()
		if batch_idx % argsp.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}    J_R^2:{:.4f}  ff_R^2:{:.4f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), 
				loss.item(),
				J_R2, ff_R2))
		total_loss += loss.item()
		total_JR += J_R2
		total_ffR += ff_R2
		n+=1

	total_loss /= n
	total_JR /=n
	total_ffR /=n
	print('\nTrain set: Average loss: {:.4f} Average J_R^2: {:.4f} Average ff_R^2: {:.4f}'.format(total_loss, total_JR, total_ffR))
	return total_loss, total_JR, total_ffR

# test high-fidelity model to predict J_sc and FF
def test_JF(model, device, test_loader, epoch, argsp):
	#sets evaluation mode to 
	model.eval() 
	test_loss = 0
	total_test_loss = 0 
	total_test_JR = 0
	total_test_ffR = 0
	J_true = []
	J_pred = []
	ff_true = []
	ff_pred = []

	n = 0
	with torch.no_grad():
		for batch_idx, (data, J, ff) in enumerate(test_loader):
			data, J, ff = data.to(device), J.to(device), ff.to(device)
			output = model(data)
			pred_J = output[:,0]
			pred_ff = output[:,1]

			J_loss = F.mse_loss(pred_J, J.float(), reduction='mean')
			ff_loss = F.mse_loss(pred_ff, ff.float(), reduction='mean')

			#compute total loss
			loss = J_loss + ff_loss

			J_test_R2 = compute_R2(pred_J.cpu().detach().numpy(), J.float().cpu().detach().numpy(), mode='test')
			ff_test_R2 = compute_R2(pred_ff.cpu().detach().numpy(), ff.float().cpu().detach().numpy(), mode='test')

			if batch_idx % argsp.log_interval == 0:
				print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}   J_R^2:{:.4f}  ff_R^2:{:.4f}'.format(
				epoch, batch_idx * len(data), len(test_loader.dataset),
				100. * batch_idx / len(test_loader), 
				loss.item(),
				J_test_R2,
				ff_test_R2))

			J_true.extend(J.cpu().detach().numpy())
			ff_true.extend(ff.cpu().detach().numpy())

			J_pred.extend(pred_J.cpu().detach().numpy())
			ff_pred.extend(pred_ff.cpu().detach().numpy())

			total_test_loss += loss.item()
			total_test_JR += J_test_R2
			total_test_ffR += ff_test_R2
			
			n += 1
	total_test_loss /= n
	total_test_JR/= n
	total_test_ffR/= n

	print('\nTest set: Average loss: {:.4f} Average Test J R^2:{:.4f} Average Test ff R^2:{:.4f}'.format(total_test_loss, total_test_JR, total_test_ffR))
	print(' ')

	if epoch == argsp.epochs:
		#predict J, compute R^2 and save
		J_pred = np.asarray(J_pred).astype('float32')
		J_true = np.asarray(J_true).astype('float32')
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(J_true, J_pred)
		plt.figure(figsize=(10,10))
		a = plt.axes(aspect='equal')
		plt.scatter(J_true, J_pred)
		plt.xlabel('J-True Values')
		plt.ylabel('J-Predictions')
		lims = [0, max(J_pred)+1]
		plt.xlim(lims)
		plt.ylim(lims)
		plt.plot(J_true, (intercept + slope*J_true[:]), label = "R-squared for fill: %f" % r_value**2)
		plt.legend()
		_ = plt.plot(lims, lims)
		plt.savefig(os.path.join(argsp.model_dir, 'J_R2_plot.jpg'))

		ff_pred = np.asarray(ff_pred).astype('float32')
		ff_true = np.asarray(ff_true).astype('float32')
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ff_true, ff_pred)
		plt.figure(figsize=(10,10))
		a = plt.axes(aspect='equal')
		plt.scatter(ff_true, ff_pred)
		plt.xlabel('$FF$-True Values')
		plt.ylabel('$FF$-Predictions')
		lims = [0, max(ff_pred)+1]
		plt.xlim(lims)
		plt.ylim(lims)
		plt.plot(J_true, (intercept + slope*J_true[:]), label = "R-squared for fill: %f" % r_value**2)
		plt.legend()
		_ = plt.plot(lims, lims)
		plt.savefig(os.path.join(argsp.model_dir, 'FF_R2_plot.jpg'))

	return total_test_loss, J_true, J_pred, ff_true, ff_pred, total_test_JR, total_test_ffR

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='CNN model to predict property (J and FF) based on morphology')
	parser.add_argument('--batch_size', type=int, default=64,
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000,
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=25,
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=1E-4,
						help='learning rate (default: 1E-4)')
	parser.add_argument('--gamma', type=float, default=0.7,
						help='Learning rate step gamma (default: 0.7)')
	parser.add_argument('--seed', type=int, default=0,
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10,
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=True,
						help='For Saving the current Model')
	parser.add_argument('--model_dir', default='./results/Jff_pretraining',
						help='Directory to save the current Model')
	parser.add_argument('--smooth', type=int, default=0,
						help='Trained with Lipschitz constraint?')
	parser.add_argument('--gpu', type=int, default=0)
	argsp = parser.parse_args()
	argsp.model_dir = get_next_run(argsp.model_dir)

	torch.cuda.set_device(argsp.gpu)        
	use_cuda = torch.cuda.is_available()
	torch.manual_seed(argsp.seed)
	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	train_loader = get_dataloader(path='/data/Joshua/DARPA_data/augmented_JF_filtered_norm_balanced_train.npy', batchsize=argsp.batch_size)
	test_loader = get_dataloader(path='/data/Joshua/DARPA_data/augmented_JF_filtered_norm_test.npy', batchsize=argsp.batch_size)

	model = JF_Net().to(device)
	optimizer = optim.Adam(model.parameters(), lr=argsp.lr)
	scheduler = StepLR(optimizer, step_size=1, gamma=argsp.gamma)

	test_losses = []
	train_losses = []
	train_JRs = []
	train_ffRs = []
	test_JRs = []
	test_ffRs = []
	
	#training loop
	if argsp.smooth > -1:
		print('Training with GP')
	else:
		print('Training without GP')

	for epoch in tqdm(range(1, argsp.epochs + 1)):
		train_loss, train_JR, train_ffR = train_JF(argsp.smooth, model, device, train_loader, optimizer, epoch, argsp)
		test_loss, J_true, J_pred, ff_true, ff_pred, J_test_R, ff_test_R = test_JF(model, device, test_loader, epoch, argsp)

		train_losses.append(train_loss)
		test_losses.append(test_loss)
		train_JRs.append(train_JR)
		train_ffRs.append(train_ffR)
		test_JRs.append(J_test_R)
		test_ffRs.append(ff_test_R)

		scheduler.step()

	#save model after training
	if argsp.save_model:
		torch.save(model.state_dict(), os.path.join(argsp.model_dir, "regressor.pt"))

	#plotting results
	plt.figure()
	plt.plot(train_losses, label='train loss')
	plt.plot(test_losses, label='test loss')
	plt.plot(train_JRs, label='train $J_{sc}$ $R^2$')
	plt.plot(test_JRs, label='test $J_{sc}$ $R^2$')
	plt.plot(train_ffRs, label='train $FF$ $R^2$')
	plt.plot(test_ffRs, label='test $FF$ $R^2$')
	plt.ylabel('Losses (MSE)')
	plt.xlabel('Epoch')
	plt.legend()
	plt.savefig(os.path.join(argsp.model_dir, 'loss.png'))

	#plot errors for J
	diff = list(map(operator.sub, J_true, J_pred))
	avg = np.mean(diff)
	plt.figure()
	plt.hist(diff, label='mean: ' + str(avg))
	plt.ylabel('Density')
	plt.xlabel('$J_{sc}$ Error')
	plt.legend()
	plt.savefig(os.path.join(argsp.model_dir, 'J_error.png'))

	#plot error for ff
	#scaling factors used to scale ff lables
	J_scale = 20.855220777251184
	min_ff = 0.400146484375

	#scale ff back to 0.4 and 0.8
	#norm_ff = (ff - min(ff)) * J_scale
	ff_true = list(np.asarray(ff_true)/J_scale + min_ff)
	ff_pred = list(np.asarray(ff_pred)/J_scale + min_ff)

	diff = list(map(operator.sub, ff_true, ff_pred))
	avg = np.mean(diff)
	plt.figure()
	plt.hist(diff, label='mean: ' + str(avg))
	plt.ylabel('Density')
	plt.xlabel('$FF$ Error')
	plt.legend()
	plt.savefig(os.path.join(argsp.model_dir, 'ff_error.png'))

if __name__ == '__main__':
	main()