from pxr import Usd, UsdGeom

import numpy

import argparse


parser = argparse.ArgumentParser(description='(USD) Anim Data Exporter.')
parser.add_argument("usdfile", help="Input USD scene file", type=str)
parser.add_argument("usdprimpath", help="USD prim path pointing to the animated mesh", type=str)
parser.add_argument("npzfile", help="Output training data", type=str)
args = parser.parse_args()


def export(usdfile, primpath, npzfile):
  """
  """
  stage = Usd.Stage.Open(usdfile)
  usdprimpath = stage.GetPrimAtPath(primpath)

  if not UsdGeom.Mesh(usdprimpath):
    return None

  attr = UsdGeom.Mesh(usdprimpath).GetPointsAttr()
  T = attr.GetTimeSamples()
  def read_row_(t):
    P = attr.Get(t)
    return [p[i] for p in P for i in range(3)]

  x = read_row_(T[0])
  X = numpy.zeros((len(T), len(x)), dtype=numpy.float32)
  X[0, :] = x
  for ri, ts in enumerate(T[1:]):
    X[ri, :] = read_row_(ts)

  numpy.savez(npzfile, X)

if __name__ == "__main__":
  export(args.usdfile, args.usdprimpath, args.npzfile)