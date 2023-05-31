import numpy as np
import sys, warnings, argparse

from sigpyproc.readers import FilReader as F
from Furby_p3.Furby_reader import Furby_reader as Fr

def get_injected_data(furby_f, filt, samp):
  ff = furby_f
  ff_data = ff.read_data()
  filt_data = filt.read_block(samp, ff.header.NSAMPS)

  if ff.header.BW_MHZ * filt.header.bandwidth <0:
    ff_data = ff_data[::-1]   #Flip the band of the furby data

  rms_of_filt_data_per_chan = filt_data.std(axis=1)
  added = filt_data + (ff_data * rms_of_filt_data_per_chan[:, None])
  return added.astype(filt.header.dtype)

def write_to_filt(data, out):
  if data.dtype != out.bitsinfo.dtype:
     warnings.warn("Given data (dtype={0}) will be unasfely cast to the requested dtype={1} before being written out".format(data.dtype, o.dtype), stacklevel=1)
  out.cwrite(data.T.ravel().astype(out.bitsinfo.dtype, casting='unsafe'))

def copy_filts(inp, out, start, end, gulp=8192):
  for nsamps, ii, d in inp.read_plan(gulp, start=start, nsamps=end-start, verbose=False):
    write_to_filt(d, out)

def assert_isamp_validity(isamps, furby_nsamps, file_end_samp):
  #x = [0]
  x = []
  x.extend(list(isamps))
  x.extend([file_end_samp])
  x = np.array(x)
  if x[0] < 0:
    raise ValueError(f"The injection timestamp cannot be negative, provided - {x[0]}")
  inj_separations = x[1:] - x[:-1]
  print(x, isamps, inj_separations)

  for ii in range(len(inj_separations)):
    if inj_separations[ii] < furby_nsamps[ii]:
      raise ValueError(f"The {ii}th furby cannot be injected as the tstamp separation ({inj_separations[ii]}) is not enough. Required at least {furby_nsamps[ii]}")

def load_furby_headers(furbies):
  ffs = []
  furby_nsamps = []
  for ifurby in furbies:
    ff = Fr(ifurby)
    nsamps = ff.header.NSAMPS
    ffs.append(ff)
    furby_nsamps.append(nsamps)

  return ffs, furby_nsamps


def main():
  filt = args.filt
  furbies = args.furbies
  f = F(filt)
  if f.header.nifs !=1:
    raise NotImplementedError("nifs !=1 are currently not supported")

  if args.isamps:
    isamps = np.array(args.isamps)
  elif args.isamps_s:
    isamps = np.array(args.isamps / f.header.tsamp).astype(int)

  assert len(isamps) == len(furbies)
  
  isamps = sorted(isamps)
  furby_fs, furby_nsamps = load_furby_headers(furbies)

  assert_isamp_validity(isamps, furby_nsamps, f.header.nsamples)

  o = f.header.prep_outfile("injected_{}".format(f.filename))

  for ii, isamp in enumerate(isamps):
    if ii ==0:
      copy_filts(inp = f, out = o, start= 0, end = isamp)
    
    injected_data = get_injected_data(furby_fs[ii], filt = f, samp = isamp)
    write_to_filt(data = injected_data, out=o)
    sys.stdout.write("Injected {0} in {1} at {2:.2f}\n".format(furbies[ii], f.filename, isamp*f.header.tsamp))
    
    if ii == len(isamps)-1:
      copy_filts(inp=f, out=o, start = isamp+Fr(furbies[ii]).header.NSAMPS, end = f.header.nsamples)
    else:
      copy_filts(inp=f, out=o, start = isamp+Fr(furbies[ii]).header.NSAMPS, end = isamps[ii+1])

if __name__ == '__main__':
  a = argparse.ArgumentParser()
  a.add_argument('-filt', type=str, help="Filterbank to inject in")
  a.add_argument('-furbies', type=str, nargs='+', help="Path to furby files")
  g = a.add_mutually_exclusive_group()
  g.add_argument('-isamps_s', type=float, nargs='+', help="Injection time stamps in seconds")
  g.add_argument('-isamps', type=int, nargs='+', help="Injection time stamps in samples")
  
  args = a.parse_args()
  main()