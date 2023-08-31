import torch


class Normalizer(torch.nn.Module):
  def __init__(self, X):
    super().__init__()
    self._offset = torch.min(X, dim=0)[0]
    self._scale = torch.max(X, dim=0)[0] - self._offset
  def forward(self, x):
    return (x - self._offset) / self._scale

class Denormalizer(torch.nn.Module):
  def __init__(self, X):
    super().__init__()
    self._offset = torch.min(X, dim=0)[0]
    self._scale = torch.max(X, dim=0)[0] - self._offset
  def forward(self, x):
    return x * self._scale + self._offset
