import sys
import torch
import argparse
import config
from wandb_logging import WandBLogger
from scripts import vae
import scripts.dataloader as dataloader
from scripts.loss import TripletLoss, CustomLoss
from scripts.helpers import EarlyStopping


def main(params):
	# init wandb
	wandb_logger = WandBLogger()

	# get run name
	name = wandb_logger.run_name

	wandb_logger.log_config(params)

	if params.use_early_stop:
		early_stopping = EarlyStopping(args, verbose=True, maximize=False)

	# check device
	if params.device == 'cuda':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	else:
		device = torch.device('cpu')

	# load data
	train_dataset = dataloader.TripletDataset(label_col=params.target_label, setting='train')
	test_dataset = dataloader.TripletDataset(label_col=params.target_label, setting='test')
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=2)

	input_dim = len(train_dataset.data.columns) - 1

	model = vae.VAE(input_dim, params.hidden_dim, params.latent_dim, device)

	# optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

	# loss
	criterion = CustomLoss().to(device)
	criterion_triplet = TripletLoss(params.margin).to(device)

	# training
	col_names = train_loader.dataset.data.columns

	best_current_loss = None
	for epoch in range(params.epochs):
		model.train()
		running_loss = 0.0
		running_triplet = 0.0
		running_contrastive = 0.0

		for i, (anchor, positive, negative, labels) in enumerate(train_loader):
			anchor = anchor.to(device)
			positive = positive.to(device)
			negative = negative.to(device)

			optimizer.zero_grad()

			# stack anchor, positive and negative samples
			inputs = torch.cat((anchor, positive, negative), 0)

			outputs, mu, logvar = model(inputs)

			# Encode inputs and compute cosine embedding loss
			a = model.encode(anchor)
			p = model.encode(positive)
			n = model.encode(negative)

			loss_contrastive = torch.sum(criterion_triplet(a, p, n)) * 3 / anchor.shape[0]
			loss_bce = torch.sum(criterion(outputs, inputs, mu, logvar, col_names)) / outputs.shape[0]
			# harmonic mean of both losses
			h = 2 / (1 / loss_contrastive + 1 / loss_bce)

			loss = params.weight_triplet_loss * loss_contrastive + params.weight_custom_loss * loss_bce
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			running_triplet += loss_contrastive.item()
			running_contrastive += loss_bce.item()

		# log loss
		wandb_logger.log_metrics({'train_loss': running_loss / len(train_loader),
		                          'train_loss_triplet': running_triplet / len(train_loader),
		                          'train_loss_contrastive': running_contrastive / len(train_loader),
		                          'h': h}, epoch)

		# test model
		model.eval()
		for i, (anchor, positive, negative, labels) in enumerate(test_loader):
			anchor = anchor.to(device)
			positive = positive.to(device)
			negative = negative.to(device)

			# stack anchor, positive and negative samples
			inputs = torch.cat((anchor, positive, negative), 0)

			outputs, mu, logvar = model(inputs)

			# Encode inputs and compute cosine embedding loss
			a = model.encode(anchor)
			p = model.encode(positive)
			n = model.encode(negative)

			loss_contrastive = torch.sum(criterion_triplet(a, p, n)) * 3 / anchor.shape[0]
			loss_bce = torch.sum(criterion(outputs, inputs, mu, logvar, col_names)) / outputs.shape[0]
			h = 2 / (1 / loss_contrastive + 1 / loss_bce)

			loss = params.weight_triplet_loss * loss_contrastive + params.weight_custom_loss * loss_bce

			running_loss += loss.item()
			running_triplet += loss_contrastive.item()
			running_contrastive += loss_bce.item()

			# log test loss
			wandb_logger.log_metrics({'test_loss': running_loss / len(test_loader),
			                          'test_loss_triplet': running_triplet / len(test_loader),
			                          'test_loss_contrastive': running_contrastive / len(test_loader),
			                          'h': h
			                          }, epoch)

			if params.use_early_stop:
				early_stopping(h, model, args)
				if early_stopping.early_stop:
					wandb_logger.log_config({'early_stop': True})
					wandb_logger.finish()
					print("Early stopping")
					sys.exit()

		if params.verbose:
			if epoch % 2 == 0:
				print("Epoch {} loss: {:.4f}".format(epoch + 1, running_loss / len(train_loader)))
				print("Contrastive loss: {:.4f}".format(running_contrastive/len(train_loader)))
				print("Triplet loss: {:.4f}".format(running_triplet/len(train_loader)))

		if epoch % 100 == 0:
			if best_current_loss is None:
				best_current_loss = running_loss / len(train_loader)

			if best_current_loss > running_loss / len(train_loader):
				best_current_loss = running_loss / len(train_loader)
				torch.save(model.state_dict(), paths.model_dir + f'checkpoints/{name}.pt')
				print('Model saved')


if __name__ == '__main__':
	base_config = config.Config()
	paths = config.Paths()
	hp = config.Hyperparameters()

	# use argument parser to update config
	parser = argparse.ArgumentParser(description="PyTorch implementation of Contrastive Learning, Hyperparemters")
	parser.add_argument('--base', type=str, default=paths.working_dir, help='path to base')
	parser.add_argument('--project_name', type=str, default=base_config.project_name, help='name of project')
	parser.add_argument('--random_state', type=int, default=base_config.random_state, help='random state')

	# path related config
	parser.add_argument('--data_dir', type=str, default=paths.data_dir, help='path to data')
	parser.add_argument('--model_dir', type=str, default=paths.model_dir, help='path to model')
	parser.add_argument('--working_dir', type=str, default=paths.working_dir, help='path to log')

	# hyperparameters
	parser.add_argument('--hidden_dim', type=int, default=hp.hidden_dim, help='hidden dimension')
	parser.add_argument('--latent_dim', type=int, default=hp.latent_dim, help='latent dimension')
	parser.add_argument('--batch_size', type=int, default=hp.batch_size, help='batch size')
	parser.add_argument('--epochs', type=int, default=hp.epochs, help='number of epochs')
	parser.add_argument('--learning_rate', type=float, default=hp.learning_rate, help='learning rate')
	parser.add_argument('--weight_triplet_loss', type=float, default=hp.weight_triplet_loss, help='weight of triplet loss')
	parser.add_argument('--weight_custom_loss', type=float, default=hp.weight_custom_loss, help='weight of custom loss')
	parser.add_argument('--margin', type=float, default=hp.margin, help='margin for triplet loss')
	parser.add_argument('--verbose', type=bool, default=hp.verbose, help='verbose')
	parser.add_argument('--device', type=str, default=hp.device, help='device to use')

	# early stopping
	parser.add_argument('--use_early_stop', action='store_true', default=True)
	parser.add_argument('--early_stop_patience', type=int, default=100)
	parser.add_argument('--early_stop_min_delta', type=float, default=1.0)
	parser.add_argument('--early_stop_metric', type=str, default='h', help='metric to use for early stopping')
	parser.add_argument('--early_stop_mode', type=str, default='min', help='mode for early stopping (min or max)')
	parser.add_argument('--save_best_model', action='store_true', default=False)

	args = parser.parse_args()

	for key in paths.__dict__.keys():
		if key in args.__dict__.keys():
			setattr(paths, key, args.__dict__[key])

	for key in hp.__dict__.keys():
		if key in args.__dict__.keys():
			setattr(hp, key, args.__dict__[key])

	main(hp)
