import argparse
import numpy as np
import yaml

def main(args):
    seed = args.seed
    add_noise = args.add_noise

    injection_ra = np.linspace(args.ra[0], args.ra[1], args.num, endpoint=True)
    injection_dec = np.linspace(args.dec[0], args.dec[1], args.num, endpoint=True)
    injection_coords = np.array([  [injection_ra[ii], injection_dec[ii]]  for ii in range(args.num)])

    injection_taus = np.zeros(args.num) + 1e-16
    injection_spectra = np.array([r'flat'] * args.num)
    injection_noise_per_sample = np.ones(args.num)
    injection_shapes = np.array([r'tophat'] * args.num)

    #injection_tstamps = np.linspace(args.tstamp[0], args.tstamp[1], args.num, endpoint = True)
    #injection_tstamps = (1 + np.arange(args.num)) * 256 - 20
    injection_tstamps = (1 + np.arange(args.num)*2) * 256 - 4
    #injection_tstamps = (1 + np.arange(args.num)*2) * 256 - np.arange(args.num)
    injection_snrs = np.linspace(args.snr[0], args.snr[1], args.num, endpoint = True)
    injection_widths = np.linspace(args.width[0], args.width[1], args.num, endpoint = True)
    injection_dms = np.linspace(args.dm[0], args.dm[1], args.num, endpoint = True)
    injection_subsample_phases = np.linspace(args.subsample_phase[0], args.subsample_phase[1], args.num, endpoint=True)

    params = {
            'seed': seed,
            'add_noise' : add_noise,
            'injection_tsamps' : injection_tstamps.tolist(),
            'injection_coords' : injection_coords.tolist(),
            
            'furby_props' : [
                                {'snr': float(injection_snrs[ii]),
                                 'width': float(injection_widths[ii]),
                                 'dm': float(injection_dms[ii]),
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
    g1.add_argument('-dm_samps', type=float, nargs=2, help = "DM range endpoints [samps] (e.g. 0, 1.5; def=0,0)", default=[0, 0])
    
    g2 = a.add_mutually_exclusive_group()
    g2.add_argument('-width', type=float, nargs=2, help = "width range endpoints [s] (e.g. 0.0001, 0.0002; def=0.0017,0.0017)", default=[0.0017, 0.0017])
    g2.add_argument('-width_samps', type=float, nargs=2, help = "width range endpoints [samps] (e.g. 0.1, 2; def=1,1)", default=[1, 1])
    
    a.add_argument('-tstamp', type=int, nargs = 2, help = "Injection tstamp [samples] range endpoints (e.g. 100, 200; def = 100, 1000)", default =[100, 1000])
    a.add_argument('-subsample_phase', type=float, nargs=2, help="Subsample phase of the centre of the FRB (def:0.5)", default=[0.5, 0.5])
    
    g3 = a.add_mutually_exclusive_group()
    g3.add_argument('-ra', type=float, nargs=2, help="Injection RA range", default=[0.0, 0.0])
    g3.add_argument('-injection_upix', type=float, nargs = 2, help="Injection upix range (e.g. 128, 130.5); def=[128, 128]", default=[128, 128])

    g4 = a.add_mutually_exclusive_group()
    g4.add_argument('-dec', type=float, nargs=2, help="Injection DEC range", default=[-30.0, -30.0])    
    g4.add_argument('-injection_vpix', type=float, nargs = 2, help="Injection vpix range (e.g. 128, 130.5); def=[128, 128]", default=[128, 128])

    a.add_argument('-add_noise', action='store_true', help="Add noise? (def = False)", default= False)
    a.add_argument('-seed', type = int, help="Seed (def = 777)", default=777)
    a.add_argument('-outfile', type=str, help = "Path to the output file (def = './auto_inj_params.yml')", default = "./auto_inj_params.yml")

    args = a.parse_args()
    print(args)
    main(args)



