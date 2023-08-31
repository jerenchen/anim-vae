import numpy
import torch
import time
import os
import glob
import argparse

from layers import Normalizer, Denormalizer
from models import AE, VAE


parser = argparse.ArgumentParser(description='Animation Autoencoders.')
parser.add_argument("input", help="Input training data", type=str)
parser.add_argument("-p", "--prefix", help="Output file prefix", type=str, default="default")
parser.add_argument("-o", "--out-dir", help="Output directory", type=str, default="./")
parser.add_argument("--no-gpu", help="Do not use hardware-accelerated device", action='store_true')
parser.add_argument("--no-vae", help="Do not use variational autoencoder", action='store_true')
parser.add_argument("-d", "--dims", help="L1 L2 L3 dimensions", type=int, nargs=3, default=[128, 64, 16])
parser.add_argument("-s", "--skip", help="Number of inbetween samples to skip", type=int, default=0)
parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=5000)
parser.add_argument("-u", "--update", help="Frequency (epochs) to update training", type=int, default=100)
parser.add_argument("-lr", "--learning-rate", help="Learning rate", type=float, default=0.0001)

args = parser.parse_args()
prefix = args.prefix
in_path = args.input
out_dir = args.out_dir
use_vae = False if args.no_vae else True
dims = args.dims
skip = args.skip
freq = args.update
epochs = args.epochs
lr = args.learning_rate

device = torch.device('cpu')
if not args.no_gpu:
  device = torch.device("mps") if torch.backends.mps.is_available() else device
  device = torch.device("cuda") if torch.cuda.is_available() else device

nettype = 'vae' if use_vae else 'ae'
wd = 1e-8


def load_numpy_array():
  X = numpy.load(in_path)
  if os.path.splitext(in_path)[1] == '.npz':
    X = X[X.files[0]]
    return torch.tensor(X).float().to(device)

def main():

  X_ = load_numpy_array()
  n_sams, n_dims = X_.shape

  # Data normalization runtime layers for ONNX export
  normalizer = Normalizer(X_)
  denormalizer = Denormalizer(X_)
  # Now, normalize the data
  X_ = normalizer(X_)

  Model = VAE if use_vae else AE
  params = dims

  model = Model(n_dims, *params)
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)

  # For VAE, loss has 2 implmentations
  #   otherwise, loss = `torch.nn.MSELoss()`(model(X_), Y_)
  compute_loss = None
  if use_vae:
    """def vae_loss_(x, y):
      # https://github.com/pytorch/examples/blob/main/vae/main.py
      BCE = torch.nn.functional.binary_cross_entropy(model(x), y, reduction='sum')
      KLD = -0.5 * torch.sum(1 + model.logvar - model.mu.pow(2) - model.logvar.exp())
      return BCE + KLD"""
    def vae_loss_(x):
      # https://avandekleut.github.io/vae/
      SE = ((x - model(x))**2).sum()
      KL = (model.encoder.logvar**2 + model.encoder.mu**2 - torch.log(model.encoder.logvar) - 1/2).sum()
      return SE + KL
    compute_loss = vae_loss_
  else:
    mse_loss_func_ = torch.nn.MSELoss()
    def mse_loss_(x):
      return mse_loss_func_(model(x), x)
    compute_loss = mse_loss_
    
  state_dict = {}
  epoch = 0
  loss = None
  accumtime = 0.0

  infixes = [nettype]
  infixes += ['{}n{}'.format(i,v) for i,v in enumerate(dims)]
  infixes += ['{}x{}'.format(n_sams, n_dims)]
  out_prefix = '{}.{}'.format(prefix, '_'.join(infixes))

  ckpt_dir = os.path.join(out_dir, "checkpoints")
  def save_state_():
    if state_dict.get('epoch') is None:
      # state not initialized; possibly resumed but not trained...
      return
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)
    ckpt_file = os.path.join(ckpt_dir, "{}.{}.pt".format(out_prefix, state_dict['epoch']))
    torch.save(state_dict, ckpt_file)

  # Resume previous training?
  checkpoint = None
  ckpt_files = glob.glob(os.path.join(ckpt_dir, "{}.*.pt".format(out_prefix)))
  if ckpt_files:
    checkpoint = max(ckpt_files, key=os.path.getmtime)
  if checkpoint:
    if not input('Resume checkpoint "{}"? [Y/n]'.format(checkpoint)) in ['N', 'n', 'no', 'No', 'NO']:
      data_dict = torch.load(checkpoint)
      model.load_state_dict(data_dict['model'])
      optimizer.load_state_dict(data_dict['optimizer'])
      epoch = data_dict['epoch']
      loss = data_dict['loss']
      accumtime = data_dict['accumtime']

  t0 = t = time.time()
  err = 0

  print('Begin/Resume training from epoch', epoch, 'with learning rate', lr, '...')
  print('  Input:', n_sams, 'x', n_dims)
  print('  Model:', nettype.upper(), '-', ' '.join(['L{}:{}'.format(i,v) for i,v in enumerate(dims)]))
  print(' Output:', out_prefix)

  try:
    while epoch < epochs:

      model.train()
      loss = compute_loss(X_)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch += 1
      if epoch % freq == 0:
        t1 = time.time()
        err = loss.detach().cpu().numpy()
        if use_vae:
          err /= n_dims
        elapsedtime = t1 - t
        accumtime += elapsedtime
        print(epoch, err, '({}s)'.format(round(elapsedtime, 3)))
        t = t1

      state_dict['epoch'] = epoch
      state_dict['model'] = model.state_dict()
      state_dict['optimizer'] = optimizer.state_dict()
      state_dict['loss'] = loss
      state_dict['accumtime'] = accumtime

      # TODO: maintain a state_dict snapshot for the lowest so far

  except KeyboardInterrupt:
    save_state_()
  else:
    save_state_()

    """torch.onnx.export(
      torch.nn.Sequential(normalizer ,model, denormalizer),
      X_[0, :],
      os.path.join(out_dir, "{}.{}.onnx".format(out_prefix, epoch)),
      input_names=['U'],
      output_names=['V']
    )"""
    torch.save(state_dict, os.path.join(out_dir, '{}.{}.pt'.format(out_dir, epoch)))

    t1 = time.time()
    print('Completed @ epoch', epoch, err, '({}s)'.format(round(t1 - t0, 3)))

if __name__ == "__main__":
  main()