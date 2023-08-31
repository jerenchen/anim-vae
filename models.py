import torch


class Encoder(torch.nn.Module):
  def __init__(self, out_dims, dims = 128, dim2 = 64, dim3 = 16):
    super().__init__()

    self.encoder = torch.nn.Sequential(
      torch.nn.Linear(out_dims, dims),
      torch.nn.ReLU(),
      torch.nn.Linear(dims, dim2),
      torch.nn.ReLU(),
      torch.nn.Linear(dim2, dim3),
    )

  def forward(self, x):
    return self.encoder(x)

  def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    self.encoder = self.encoder.to(*args, **kwargs)

class Decoder(torch.nn.Module):
  def __init__(self, out_dims, dims = 128, dim2 = 64, dim3 = 16):
    super().__init__()

    self.decoder = torch.nn.Sequential(  
      torch.nn.Linear(dim3, dim2),
      torch.nn.ReLU(),
      torch.nn.Linear(dim2, dims),
      torch.nn.ReLU(),
      torch.nn.Linear(dims, out_dims),
    )

  def forward(self, x):
    return self.decoder(x)

class AE(torch.nn.Module):
  def __init__(self, out_dims, dims = 128, dim2 = 64, dim3 = 16):
    super(AE, self).__init__()
    self.encoder = Encoder(out_dims, dims, dim2, dim3)
    self.decoder = Decoder(out_dims, dims, dim2, dim3)

  def forward(self, x):
    z = self.encoder(x)
    return self.decoder(z)

class VariationalEncoder(torch.nn.Module):
  def __init__(self, out_dims, dims = 128, dim2 = 64, dim3 = 16):
    super().__init__()

    self.encode0 = torch.nn.Linear(out_dims, dims)
    self.encode1 = torch.nn.Linear(dims, dim2)
    self.encodemu = torch.nn.Linear(dim2, dim3)
    self.encodesigma = torch.nn.Linear(dim2, dim3)

    self.mu = 0
    self.logvar = 1
    self.N = torch.distributions.Normal(0, 1)

  def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    self.N.loc = self.N.loc.to(*args, **kwargs)
    self.N.scale = self.N.scale.to(*args, **kwargs)

  def encode(self, x):
    y = torch.nn.functional.relu(self.encode0(x))
    z = torch.nn.functional.relu(self.encode1(y))
    return self.encodemu(z), self.encodesigma(z)

  def forward(self, x):
    self.mu, self.logvar = self.encode(x)
    self.logvar = self.logvar.exp()
    return self.mu + self.logvar * self.N.sample(self.mu.shape)
    """self.mu, self.logvar = self.encode(x)
    std = torch.exp(0.5*self.logvar)
    eps = torch.randn_like(std)
    return self.mu + eps * std"""

class VAE(torch.nn.Module):
  def __init__(self, out_dims, dims = 128, dim2 = 64, dim3 = 16):
    super().__init__()
    self.encoder = VariationalEncoder(out_dims, dims, dim2, dim3)
    self.decoder = Decoder(out_dims, dims, dim2, dim3)

  def forward(self, x):
    z = self.encoder(x)
    return self.decoder(z)
  
  def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    self.encoder.to(*args, **kwargs)
    self.decoder.to(*args, **kwargs)