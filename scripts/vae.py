import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self, input_dim, hidden_dim, latent_dim):
		super(Encoder, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		#self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc31 = nn.Linear(hidden_dim, latent_dim)
		self.fc32 = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x):
		h = F.relu(self.fc1(x))
		h = F.relu(self.fc2(h))
		mu = self.fc31(h)
		sigma = self.fc32(h)
		return mu, sigma


class Decoder(nn.Module):
	def __init__(self, latent_dim, hidden_dim, output_dim):
		super(Decoder, self).__init__()
		self.fc1 = nn.Linear(latent_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		h = F.relu(self.fc1(x))
		h = F.relu(self.fc2(h))
		return torch.sigmoid(self.fc3(h))


class VAE(nn.Module):
	def __init__(self, input_dim, hidden_dim, z_dim, device):
		super(VAE, self).__init__()
		self.encoder = Encoder(input_dim, hidden_dim, z_dim).to(device)
		self.decoder = Decoder(z_dim, hidden_dim, input_dim).to(device)

	@staticmethod
	def reparameterize(mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def forward(self, x):
		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)
		return self.decoder(z), mu, logvar

	def encode(self, x):
		mu, logvar = self.encoder(x)
		return self.reparameterize(mu, logvar)

	def decode(self, z):
		return self.decoder(z)
