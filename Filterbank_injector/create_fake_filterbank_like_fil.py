from sigpyproc.readers import FilReader as F
import argparse
import numpy as np

def get_min_max_values(dtype, nbits):

    if dtype.char == 'f':
        min_value = np.finfo(dtype).min
        max_value = np.finfo(dtype).max
    elif dtype.char == 'i':
        min_value = np.iinfo(dtype).min
        max_value = np.iinfo(dtype).max

        if nbits < 8:
            max_value = 2**(nbits - 1) - 1
            min_value = -1 * max_value - 1

    elif dtype.char == 'u':
        min_value = np.iinfo(dtype).min
        max_value = np.iinfo(dtype).max

        if nbits < 8:
            max_value = 2**nbits -1

    else:
        raise ValueError(f"Unknown dtype provided - {dtype}")

    return min_value, max_value    



def main(args):
    f = F(args.fil)
    if f.header.nifs !=1:
        raise NotImplementedError("Cannot handle nifs !=1 yet")

    if args.nsamples:
        tot_nsamps = args.nsamples
    elif args.dur:
        tot_nsamps = int(args.dur // f.header.tsamp)
    else:
        raise ValueError("Either nsamples or dur need to be specified")
    
    if tot_nsamps < 1:
        raise ValueError(f"Nsamples need to be at least 1, provided - {tot_nsamps}")
    
    
    file_size = tot_nsamps * f.header.nchans * f.header.nbits / 8 
    if file_size > 10e9 and not args.allow_unlimited_file_size:
        raise ValueError("The output file is going to be {0:.2f} GB in size. Please specify 'allow_unlimited_file_size' if this is desired".format(file_size/1e9))

    new_header = {}
    new_header['nsamples'] = tot_nsamps
    o = f.header.prep_outfile(filename=args.outname, update_dict = new_header)
    
    samp_size = f.header.nchans
    gulp = 1024
    ngulp = int(tot_nsamps // gulp)
    remaining_samps = tot_nsamps - ngulp * gulp
    if remaining_samps > 0:
        ngulp += 1
    
    if args.seed:
        np.random.seed(args.seed)


    min_value, max_value = get_min_max_values(f.header.dtype, f.header.nbits)
    
    if args.mean:
        assert min_value < args.mean < max_value, "The desired mean is outside the dynamic range"
        mean = args.mean
    else:
        mean = (max_value + min_value) / 2

    if args.rms:
        assert mean + args.rms * 2 < max_value, "The requested rms is too high, there will be excessive 2-sigma clipping"
        rms = args.rms
    else:
        rms = (max_value - mean) / 6


    for igulp in range(ngulp):
        if igulp==ngulp -1:
            gulp = remaining_samps
        sim_data = np.random.normal(mean, rms, samp_size * gulp)
        o.cwrite(sim_data.astype(f.header.dtype, casting='unsafe'))

    o.close()


if __name__=='__main__':
    a = argparse.ArgumentParser()
    a.add_argument("outname", type=str, help="Name of the output filterbank")
    a.add_argument("-fil", type=str, help="Path to the filterbank that has to be emulated", required=True)
    g = a.add_mutually_exclusive_group()
    g.add_argument("-nsamples", type=int, help="No. of samples to simulate")
    g.add_argument("-dur", type=float, help="Length of the simulated filterbank in seconds")

    a.add_argument("-rms", type=float, help="The per channel rms to simulate (def = (max_value_nbits - mean)/6)", default=None)
    a.add_argument("-mean", type=float, help="Per channel mean to simulate (def = max_value_nbits /2)", default=None)
    a.add_argument("-seed", type=int, help="Seed for generating the simulated data")
    a.add_argument("-allow_unlimited_file_size", action='store_true', help="Allow an arbitrarily large file to be written to disk (def = False)", default=False)

    args = a.parse_args()
    main(args)