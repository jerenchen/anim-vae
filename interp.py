from pxr import Usd, UsdGeom
from pxr.Vt import Vec3fArray

import os
import torch
import numpy

from layers import Normalizer, Denormalizer
from models import AE, VAE

import argparse


parser = argparse.ArgumentParser(description='Anim AE Interpolator.')
parser.add_argument("aefile", help="Checkpoint to the autoencoders", type=str)
parser.add_argument("npzfile", help="Training data for normalization", type=str)
parser.add_argument("usdfile", help="Input USD scene", type=str)
parser.add_argument("usdprimpath", help="USD prim path pointing to the mesh to be overriden", type=str)
parser.add_argument("output", help="Output USD anim", type=str)
parser.add_argument("--steps", help="Steps for interpolation", type=int, default=24)
parser.add_argument("--start", help="Start sample for interpolation", type=int, default=-1)
parser.add_argument("--end", help="End sample for interpolation", type=int, default=0)
parser.add_argument("-vae", help="VAE or not (AE)", type=bool, default=True)
parser.add_argument("-d", "--device", help="cpu, cuda, or mps", type=str, choices=['cpu','mps','cuda'], default='cpu')
args = parser.parse_args()

use_vae = args.vae
steps = args.steps
device = args.device
start = args.start
end = args.end

def data_normalizers(filename):
  X = numpy.load(filename)
  if os.path.splitext(filename)[1] == '.npz':
    X = X[X.files[0]]
  X_ = torch.tensor(X).float().to(device)
  normalizer = Normalizer(X_)
  denormalizer = Denormalizer(X_)
  return normalizer, denormalizer
  

def interp_anim(aefile, indata, ref, primpath, output, use_vae = True):
  """
  """

  stage = Usd.Stage.CreateNew(output)
  prim = stage.OverridePrim(primpath)
  prim.GetReferences().AddReference(ref)

  mesh = UsdGeom.Mesh(prim)
  attr = mesh.GetPointsAttr()
  T = attr.GetTimeSamples()
  P = attr.Get(T[start])
  Q = attr.Get(T[end])

  x0 = torch.tensor([p[i] for p in P for i in range(3)]).to(device)
  x1 = torch.tensor([p[i] for p in Q for i in range(3)]).to(device)

  normalizer, denormalizer = data_normalizers(indata)

  x0 = normalizer(x0)
  x1 = normalizer(x1)

  state_dict = torch.load(aefile)
  model = (VAE if use_vae else AE)(len(x0))
  model.load_state_dict(state_dict['model'])
  model.eval()
  model.to(device)

  z0 = model.encoder(x0)
  z1 = model.encoder(x1)

  attr.Clear()

  steps = 4
  for t in range(steps):
    print('Interpolating frame {}/{}...'.format(t+1,steps))
    w = ((t+1)/float(steps)) * torch.ones(z0.shape).to(device)
    z = z1 * w + z0 * (1 - w)
    X_ = denormalizer(model.decoder(z).cpu().detach()).view(-1, 3)
    P_ = Vec3fArray(X_.shape[0])
    for i, x_ in enumerate(X_):
      P_[i] = [v.item() for v in x_]
    attr.Set(P_, 13.0 + t)

  stage.Save()

if __name__ == "__main__":
  interp_anim(
    args.aefile,
    args.npzfile,
    args.usdfile,
    args.usdprimpath,
    args.output
  )