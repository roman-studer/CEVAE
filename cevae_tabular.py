import torch
import os
import config
from wandb_logging import WandBLogger
from scripts import vae
import scripts.dataloader as dataloader
from scripts.loss import TripletLoss, CustomLoss


def main(hp):
	# init wandb
	wandb_logger = WandBLogger()
	wandb_logger.log_config(hp)

	# check device
	if hp.device == 'cuda':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	else:
		device = torch.device('cpu')

	# load data
	train_dataset = dataloader.TripletDataset(label_col=hp.target_label, setting='train')
	test_dataset = dataloader.TripletDataset(label_col=hp.target_label, setting='test')
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=2)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=2)

	input_dim = len(train_dataset.data.columns) - 1

	model = vae.VAE(input_dim, hp.hidden_dim, hp.latent_dim, device)

	# optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

	# loss
	criterion = CustomLoss().to(device)
	criterion_triplet = TripletLoss(hp.margin).to(device)

	# training
	col_names = train_loader.dataset.data.columns

	best_current_loss = None
	for epoch in range(hp.epochs):
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

			loss = hp.weight_triplet_loss * loss_contrastive + hp.weight_custom_loss * loss_bce
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			running_triplet += loss_contrastive.item()
			running_contrastive += loss_bce.item()

		# log loss
		wandb_logger.log_metrics({'train_loss': running_loss / len(train_loader),
		                          'train_loss_triplet': running_triplet / len(train_loader),
		                          'train_loss_contrastive': running_contrastive / len(train_loader)
		                          }, epoch)


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

			loss = hp.weight_triplet_loss * loss_contrastive + hp.weight_custom_loss * loss_bce

			running_loss += loss.item()
			running_triplet += loss_contrastive.item()
			running_contrastive += loss_bce.item()

			# log test loss
			wandb_logger.log_metrics({'test_loss': running_loss / len(test_loader),
			                          'test_loss_triplet': running_triplet / len(test_loader),
			                          'test_loss_contrastive': running_contrastive / len(test_loader)
			                          }, epoch)

		if hp.verbose:
			if epoch % 2 == 0:
				print("Epoch {} loss: {:.4f}".format(epoch + 1, running_loss / len(train_loader)))
				print("Contrastive loss: {:.4f}".format(running_contrastive/len(train_loader)))
				print("Triplet loss: {:.4f}".format(running_triplet/len(train_loader)))

		if epoch % 100 == 0:
			if best_current_loss is None:
				best_current_loss = running_loss / len(train_loader)

			if best_current_loss > running_loss / len(train_loader):
				best_current_loss = running_loss / len(train_loader)
				torch.save(model.state_dict(), paths.model_dir + 'checkpoints/best_model.pt')
				print('Model saved')


if __name__ == '__main__':
	paths = config.Paths()
	hp = config.Hyperparameters()

	main(hp)
