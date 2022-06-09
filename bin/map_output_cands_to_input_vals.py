import yaml
import numpy as np
import pandas as pd 
import argparse
from Furby_p3.Furby_reader import Furby_reader as F

class Injection(object):
    '''
    Injetion class that holds all properties of a given injection
    '''

    def __init__(self, tstamp):
        self.tstamp = tstamp
        self.was_injected = False

def parse_injection_params(inj_params, tobs_samp):
    tsamps = inj_params['injection_tsamps']
    N_inj = len(tsamps)
    coords = None
    pixels = None
    if 'injection_coords' in inj_params.keys():
        if type(inj_params['injection_coords'][0]) == float:
            upixs = np.array([128] * N_inj)
            vpixs = np.array([128] * N_inj)
            pixels = list(zip(upixs, vpixs))
        elif (type(inj_params['injection_coords'][0]) == list) and (len(inj_params['injection_coords'][0]) == 2):
            ras = np.array([ii[0] for ii in inj_params['injection_coords']])
            decs = np.array([ii[1] for ii in inj_params['injection_coords']])
            coords = list(zip(ras, decs))
    else:
        upixs = np.array([ii[0] for ii in inj_params['injection_pixels']])
        vpixs = np.array([ii[1] for ii in inj_params['injection_pixels']])
        pixels = list(zip(upixs, vpixs))
    
    injections = []
    for ii in range(N_inj):
        inj = Injection(tsamps[ii])
        if coords is not None:
            inj.pos_key = 'coords'
            inj.pos_val = coords[ii]
        elif pixels is not None:
            inj.pos_key = 'pixels'
            inj.pos_val = pixels[ii]

        if inj.tstamp < tobs_samp:
            inj.was_injected = True
        else:
            inj.was_injected = False

        inj.header_dict = inj_params['furby_props'][ii]
        
        if 'dm_samps' in inj.header_dict:
            inj.dm_key = 'dm_samps'
            inj.cand_dm_key = 'dm'
        else:
            inj.dm_key = 'dm'
            inj.cand_dm_key = 'dm_pccm3'
        
        if 'width' in inj.header_dict:
            inj.header_dict['width_samps'] = inj.header_dict['width'] / 0.017

        injections.append(inj)
    return injections

def read_injections_yml(inj_file):
    with open(inj_file, 'r') as f:
        inj_params = yaml.safe_load(f)

    return inj_params


def find_cand(all_cands, injection):
    print("--------->\n", all_cands)
    if not injection.was_injected:
        return None
    if injection.pos_key is 'coords':
        ras = np.array([ii[0] for ii in all_cands['coords']])
        decs = np.array([ii[1] for ii in all_cands['coords']])
        pos_mask = (injection.pos_val[0] - 0.5 < ras) & (ras < injection.pos_val[0] + 0.5) & (injection.pos_val[1] - 0.5 < decs) & (decs < injection.pos_val[1] + 0.5)
    elif injection.pos_key is 'pixels':
        upixs = np.array([ii[0] for ii in all_cands['pixels']])
        vpixs = np.array([ii[1] for ii in all_cands['pixels']])
        pos_mask = (injection.pos_val[0] - 1 <= upixs) & (upixs <= injection.pos_val[0] + 1) & (injection.pos_val[1] - 1 <= vpixs) & (vpixs <= injection.pos_val[1] + 1)

    time_mask = ((injection.tstamp - 8) <= all_cands['total_sample']) & (all_cands['total_sample'] <= (injection.tstamp + 10))

    if injection.dm_key is 'dm':
        cand_dm_key = 'dm_pccm3'
    elif injection.dm_key is 'dm_samps':
        cand_dm_key = 'dm'
    
    dm_mask = ((injection.header_dict[injection.dm_key] - 10) <= all_cands[cand_dm_key]) & (all_cands[cand_dm_key] <= (injection.header_dict[injection.dm_key] + 10))

    print("POS MASK :::::::::::::\n\n---------------=================-------------\n", pos_mask)
    selected_cand = all_cands[pos_mask & time_mask & dm_mask]
    
    print(f"MASK_SUMS = {pos_mask.sum()}, {time_mask.sum()}, {dm_mask.sum()}")
    print("==========>>\n", selected_cand)

    if len(selected_cand) == 0:
        return None

    elif len(selected_cand) == 1:
        return selected_cand

    else:
        selected_cand = selected_cand.iloc[ np.argsort(selected_cand['snr']).iloc[-1] ]
        return selected_cand

def main(args):
    HDR = ['snr', 'upix', 'vpix', 'boxc_width', 'time', 'dm', 'iblk', 'rawsn', 'total_sample', 'obstime_sec', 'mjd', 'dm_pccm3', 'ra_deg', 'dec_deg']
    nt = args.nt
    
    results = pd.read_csv(args.cand_file, sep='\s+', skipfooter =1, skiprows=1, names = HDR)
    tobs_samp = np.max(results['iblk'] + 1) * nt

    results['coords'] = [(results['ra_deg'][i], results['dec_deg'][i]) for i in range(len(results)) ]
    results['pixels'] = [(results['upix'][i], results['vpix'][i]) for i in range(len(results)) ]
    inj_params_from_file = read_injections_yml(args.inj_file)
    N_injections = len(inj_params_from_file['injection_tsamps'])

    rf = open(args.output_file, 'w')
    rf.write("#Writing results of cross-match between inj-file: {0} and cand-file: {1}\n\n".format(args.inj_file, args.cand_file))
    rf.write("Injection.snr \t Injection.dm \t Injection.width \t Injection.tstamp \t Injection.coord \t Injection.subsample_phase \t Pipeline.snr \t Pipeline.dm \t Pipeline.width \t Pipeline.tstamp \t Pipeline.coord\n")

    injections = parse_injection_params(inj_params_from_file, tobs_samp)
    '''
    injections = []
    for ii in range(N_injections):
        if 'furby_file' in inj_params_from_file.keys():
            furby_header = F(inj_params_from_file['furby_file'][ii]).header
            inj_header = {}
            inj_header['snr'] = furby_header.BOXCAR_SNR
            inj_header['width'] = furby_header.BOXCAR_WIDTH_SAMPS
            inj_header['dm'] = furby_header.DM
            inj_header['shape'] = furby_header.SHAPE
            inj_header['noise_per_sample'] = furby_header.NOISE_PER_SAMPLE
            inj_header['subsample_phase'] = furby_header.SUBSAMPLE_PHASE
                
        elif 'furby_props' in inj_params_from_file.keys():
            print("Reading furby_props from file")
            inj_header = inj_params_from_file['furby_props'][ii]    

        inj_tsamp = inj_params_from_file['injection_tsamps'][ii]
        inj_coord = inj_params_from_file['injection_coords'][ii]

        injection = Injection(inj_tsamp, inj_coord, inj_header)

        if inj_tsamp < tobs_samp:
            injection.was_injected = True

            injections.append(injection)

    '''
    for ii, injection in enumerate(injections):
        inj_part = f"{injection.header_dict['snr']:.2f} \t {injection.header_dict[injection.dm_key]:.2f} \t {injection.header_dict['width_samps']:.2f} \t {injection.tstamp} \t {injection.pos_val} \t {injection.header_dict['subsample_phase']:.2f} \t "
        cand = find_cand(all_cands = results, injection = injection)

        if cand is None:
            cand_part = "-1 \t\t -1 \t\t -1 \t\t -1 \t\t -1"
        else:
            cand_part = f"{cand['snr']} \t\t {cand[injection.cand_dm_key]} \t\t {cand['boxc_width']} \t\t {cand['total_sample']} \t\t {cand[injection.pos_key]}"

        full_line = inj_part + cand_part
        rf.write(full_line + "\n")

    
if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("-inj_file", type=str, help="Path to injection file", required=True)
    a.add_argument("-cand_file", type=str, help="Path to cand file", required=True)
    a.add_argument("-output_file", type=str, help='Path to the cross-matching output file', required=True)
    a.add_argument("-nt", type=int, help="No of time samples in each block (def = 256)", default=256)
    a.add_argument("-nf", type=int, help="No of channels (def = 256)", default=256)
    a.add_argument("-fmax", type=float, help="Center freq of highest channel (MHz, def = 990)", default=990.0)
    a.add_argument("-fmin", type=float, help="Center freq of bottom channel (MHz, def = 736)", default=736.0)
    a.add_argument("-tsamp_ms", type=float, help="Sampling time in ms (def = 1.7ms)", default = 1.7)
    
    args = a.parse_args()
    main(args)

        

        





