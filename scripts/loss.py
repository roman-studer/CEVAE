import torch
import torch.nn as nn
import torch.nn.functional as F


def get_one_hot_index(cols):
	"""
	Generates a list of indexes (start, end) for one hot encoded columns based on prefix of column names
	:param cols: list of column names
	:return: list of (start, end) tuples
	"""
	one_hot_index = []
	start = 0
	end = 0
	current = cols[0].split('_')[0]
	for i in cols:
		if i.split('_')[0] == current:
			end += 1
		else:
			one_hot_index.append((start, end))
			start = end
			end += 1
			current = i.split('_')[0]

	return one_hot_index


class TripletLoss(nn.Module):
	def __init__(self, margin):
		"""
		Triplet loss function for contrastive learning
		:param margin: margin between positive and negative sample
		"""
		super(TripletLoss, self).__init__()
		self.margin = margin

	def forward(self, a, p, n):
		"""
		Calculates the triplet loss for a batch of anchor, positive and negative samples.
		Tip: hard examples (i.e. positive and negative samples are close), speed up training
		Loss cannot be below 0 and is only positive if distance between anchor and positive sample is larger than
		distance between anchor and negative sample plus margin.
		:param a: anchor samples
		:param p: positive samples
		:param n: negative samples
		:return: triplet loss
		"""
		distance_positive = torch.norm(a - p, dim=1)
		distance_negative = torch.norm(a - n, dim=1)
		loss_triplet = torch.clamp(distance_positive - distance_negative + self.margin, min=0.0)
		return loss_triplet


class ContrastiveLoss(nn.Module):
	def __init__(self, margin):
		"""
		Contrastive loss function for contrastive learning
		:param margin: margin between positive and negative sample
		"""
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, z1, z2, label):
		"""
		Calculates the contrastive loss for a batch of samples.
		:param z1: samples 1
		:param z2: samples 2
		:param label: label 1 if samples are similar, 0 if samples are different
		:return: contrastive loss
		"""
		distance = torch.norm(z1 - z2, dim=1)
		loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
		return loss_contrastive


class CustomLoss(nn.Module):
	def __init__(self):
		"""
		Custom loss function for variational autoencoder
		"""
		super(CustomLoss, self).__init__()

	# TODO split loss_catgorical into loss per one-hot encoded feature (due to codependecy of features)
	def forward(self, input, target, mu, logvar, cols):
		"""
		Calculates the custom loss for a batch of samples. Loss for the reconstruction of numerical features is calculated using
		cosine embedding loss. Loss for the reconstruction of categorical features is calculated using binary cross entropy loss.
		:param input: reconstructed samples
		:param target: original samples
		:param mu: mean of latent space
		:param logvar: log variance of latent space
		:param cosine_label: label 1 if samples are similar, -1 if samples are different
		:return: custom loss
		"""

		# numerical features cosine similarity loss
		# loss_numerical = 1 - F.cosine_similarity(input[:, :4], target[:, :4], dim=1)
		loss_numerical = 1 - F.cosine_similarity(input, target, dim=1)

		# one-hot-encoded features binary cross entropy loss
		# loss_categorical = 0.
		# for start, end in get_one_hot_index(cols):
		# 	start += 4
		# 	end += 4
		# 	loss_numerical += F.binary_cross_entropy(input[:, start:end], target[:, start:end])

		# kullback-leibler divergence loss
		loss_kdl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

		return loss_numerical + loss_kdl