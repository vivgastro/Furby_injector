import argparse
import numpy as np
import yaml

def main(args):
    seed = args.seed
    const_value = args.const_value
    add_noise = args.add_noise
    add_const = args.add_const 

    if args.injection_upix is None:
        injection_ra = np.linspace(args.ra[0], args.ra[1], args.num, endpoint=True)
    else:
        injection_upixs = np.linspace(args.injection_upix[0], args.injection_upix[1], args.num, endpoint=True)

    if args.injection_vpix is None:
         injection_dec = np.linspace(args.dec[0], args.dec[1], args.num, endpoint=True)
    else:
        injection_vpixs = np.linspace(args.injection_vpix[0], args.injection_vpix[1], args.num, endpoint=True)
    
    if (args.injection_upix is None) or (args.injection_vpix is None):
        injection_coords = np.array([  [injection_ra[ii], injection_dec[ii]]  for ii in range(args.num)])
        coord_key = 'injection_coords'
        coord_val = injection_coords
    else:
        injection_pixels = np.array([   [injection_upixs[ii], injection_vpixs[ii]] for ii in range(args.num)  ])
        coord_key = 'injection_pixels'
        coord_val = injection_pixels

    injection_taus = np.zeros(args.num) + 1e-16
    injection_spectra = np.array([r'flat'] * args.num)
    injection_noise_per_sample = np.ones(args.num) * args.noise_per_sample
    injection_shapes = np.array([r'tophat'] * args.num)

    #injection_tstamps = np.linspace(args.tstamp[0], args.tstamp[1], args.num, endpoint = True)
    injection_tstamps = (1 + np.arange(args.num) * args.blockstep) * args.nt - args.toff + args.blockoff * args.nt
    #injection_tstamps = (1 + np.arange(args.num)*2) * 256 - 4
    #injection_tstamps = (1 + np.arange(args.num)*2) * 256 - np.arange(args.num)
    injection_snrs = np.linspace(args.snr[0], args.snr[1], args.num, endpoint = True)

    if args.width_samps is None:
        injection_widths = np.linspace(args.width[0], args.width[1], args.num, endpoint = True)
        width_key = 'width'
        width_val = injection_widths
    else:
        injection_width_samps = np.linspace(args.width_samps[0], args.width_samps[1], args.num, endpoint=True)
        width_key = 'width_samps'
        width_val = injection_width_samps
        

    if args.dm_samps is None:
        injection_dms = np.linspace(args.dm[0], args.dm[1], args.num, endpoint = True)
        dm_key = 'dm'
        dm_val = injection_dms
    else:
        injection_dm_samps = np.linspace(args.dm_samps[0], args.dm_samps[1], args.num, endpoint=True)
        dm_key = 'dm_samps'
        dm_val = injection_dm_samps

    injection_subsample_phases = np.linspace(args.subsample_phase[0], args.subsample_phase[1], args.num, endpoint=True)

    params = {
            'seed': seed,
            'const_value': const_value,
            'add_noise' : add_noise,
            'add_const': add_const,
            'injection_tsamps' : injection_tstamps.tolist(),
            coord_key : coord_val.tolist(),
            
            'furby_props' : [
                                {'snr': float(injection_snrs[ii]),
                                 width_key: float(width_val[ii]),
                                 dm_key: float(dm_val[ii]),
                                 'tau0': float(injection_taus[ii]),
                                 'spectrum' : str(injection_spectra[ii]),
                                 'noise_per_sample': float(injection_noise_per_sample[ii]),
                                 'shape': str(injection_shapes[ii]),
                                 'subsample_phase':float(injection_subsample_phases[ii])
                                 } for ii in range(args.num)
                            ]
            
            }

    with open(args.outfile, 'w') as f:
        yaml.dump(params, f)

    if yaml.safe_load(open(args.outfile, 'r')) == params:
        print("YAYY")
    else:
        print("Booo")

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('-num', type=int, help="No of injections required (def = 1)", default = 1)
    a.add_argument('-snr', type=float, nargs=2, help = "SNR range endpoints (e.g. 10, 12; def=20,20)", default=[20, 20])
    
    g1 = a.add_mutually_exclusive_group()
    g1.add_argument('-dm', type=float, nargs=2, help = "DM range endpoints [pc/cc] (e.g. 10, 12; def=0,0)", default=[0, 0])
    g1.add_argument('-dm_samps', type=float, nargs=2, help = "DM range endpoints [samps] (e.g. 0, 1.5; def=0,0)")
    
    g2 = a.add_mutually_exclusive_group()
    g2.add_argument('-width', type=float, nargs=2, help = "width range endpoints [s] (e.g. 0.0001, 0.0002; def=0.0017,0.0017)", default=[0.0017, 0.0017])
    g2.add_argument('-width_samps', type=float, nargs=2, help = "width range endpoints [samps] (e.g. 0.1, 2; def=1,1)")
    
    a.add_argument('-tstamp', type=int, nargs = 2, help = "Injection tstamp [samples] range endpoints (e.g. 100, 200; def = 100, 1000)", default =[100, 1000])
    a.add_argument('-blockoff', type=int, help="Offset the start of injection by these many blocks (def =0)", default=0)
    a.add_argument('-blockstep', type=int, help="How many blocks to increment between injections. 1 means injections will be put in adjacent blocks (def =1)", default=1)
    a.add_argument('-toff', type=int, help="Time offset of the injection within the block (from right edge) (def=4)", default=4)
    a.add_argument('-nt', type=int, help="nt for each block (def = 256)", default=256)
    a.add_argument('-subsample_phase', type=float, nargs=2, help="Subsample phase of the centre of the FRB (def:0.5)", default=[0.5, 0.5])
    
    g3 = a.add_mutually_exclusive_group()
    g3.add_argument('-ra', type=float, nargs=2, help="Injection RA range", default=[0.0, 0.0])
    g3.add_argument('-injection_upix', type=float, nargs = 2, help="Injection upix range (e.g. 128, 130.5); def=[128, 128]")

    g4 = a.add_mutually_exclusive_group()
    g4.add_argument('-dec', type=float, nargs=2, help="Injection DEC range", default=[-30.0, -30.0])    
    g4.add_argument('-injection_vpix', type=float, nargs = 2, help="Injection vpix range (e.g. 128, 130.5); def=[128, 128]")

    a.add_argument('-add_noise', action='store_true', help="Add noise? (def = False)", default= False)
    a.add_argument('-add_const', action='store_true', help="Add const? (def = False)", default= False)
    a.add_argument('-seed', type = int, help="Seed (def = 777)", default=777)
    a.add_argument('-const_value', type = int, help="Constant value (def = 1)", default=1)
    a.add_argument('-noise_per_sample', type=float, help='Noise per sample in the data to which the FRB will be added; def=1.0', default=1.0)
    a.add_argument('-outfile', type=str, help = "Path to the output file (def = './auto_inj_params.yml')", default = "./auto_inj_params.yml")

    args = a.parse_args()
    main(args)



