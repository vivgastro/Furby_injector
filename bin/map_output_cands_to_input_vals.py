import yaml
import numpy as np
import pandas as pd 
import argparse
from Furby_p3.Furby_reader import Furby_reader as F

class Injection(object):
    '''
    Injetion class that holds all properties of a given injection
    '''

    def __init__(self, inj_tsamp, inj_coord, header_dict):
        self.header = header_dict
        self.tstamp = inj_tsamp
        self.coord = inj_coord

def read_injections_yml(inj_file):
    with open(inj_file, 'r') as f:
        inj_params = yaml.safe_load(f)

    return inj_params


def find_cand(all_cands, snr, dm, width, tstamp, coord):
    print("--------->\n", all_cands)
    coord_mask = (all_cands['coord'] == coord)
    time_mask = ((tstamp - 8) <= all_cands['total_sample']) & (all_cands['total_sample'] <= (tstamp + width + 8))
    dm_mask = ((dm - 10) <= all_cands['dm_pccm3']) & (all_cands['dm_pccm3'] <= (dm + 10))

    selected_cand = all_cands[coord_mask & time_mask & dm_mask]
    print(f"MASK_SUMS = {coord_mask.sum()}, {time_mask.sum()}, {dm_mask.sum()}")
    print("==========>>\n", selected_cand)

    if len(selected_cand) == 0:
        return None

    elif len(selected_cand) == 1:
        return selected_cand

    else:
        selected_cand = selected_cand.iloc[ np.argsort(selected_cand['snr']).iloc[-1] ]
        return selected_cand
    

def make_dm_pccc_column(r, tsamp, fmax, fmin, nf):
    delay_samps = r['dm']
    chw = (fmax - fmin) / (nf - 1.) /1e6  #MHz
    BW = chw * nf                         #MHz
    cfreq = (fmax + fmin) / 2 / 1e9       #GHz
    delay_us = delay_samps * tsamp *1e6    #usec

    #Now since we know that delay_us = 8.3 * DM(pc/cc) * BW(MHz) * cfreq(GHz)**-3
    #We can calculate DM(pc/cc) = delay_us / (8.3 * BW(MHz) * cfreq(GHz)**-3)

    dm_pccc = delay_us / (8.3 * BW * cfreq**-3)
    
    r['dm_pccc'] = dm_pccc
    print(f"DM(pc/cc) = {dm_pccc} == DM(samples) = {delay_samps}")
    return r
    

def main(args):
    HDR = ['snr', 'upix', 'vpix', 'boxc_width', 'time', 'dm', 'iblk', 'rawsn', 'total_sample', 'obstime_sec', 'mjd', 'dm_pccm3', 'ra_deg', 'dec_deg']
    nt = args.nt
    
    results = pd.read_csv(args.cand_file, sep='\s+', skipfooter =1, skiprows=1, names = HDR)
    tobs_samp = np.max(results['iblk'] + 1) * nt
    #results = make_dm_pccc_column(results, args.tsamp_ms*1e-3, args.fmax*1e6, args.fmin*1e6, args.nf)
    #results['tstamp'] = results['time'] + results['iblk'] * args.nt
    results['coord'] = results['upix']-128 + results['vpix']-128

    inj_params_from_file = read_injections_yml(args.inj_file)
    N_injections = len(inj_params_from_file['injection_tsamps'])

    rf = open(args.output_file, 'w')
    rf.write("#Writing results of cross-match between inj-file: {0} and cand-file: {1}\n\n".format(args.inj_file, args.cand_file))
    rf.write("Injection.snr \t Injection.dm \t Injection.width \t Injection.tstamp \t Injection.coord \t Injection.subsample_phase \t Pipeline.snr \t Pipeline.dm \t Pipeline.width \t Pipeline.tstamp \t Pipeline.coord\n")

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
            inj_header['width'] = inj_header['width'] / (args.tsamp_ms * 1e-3)

        inj_tsamp = inj_params_from_file['injection_tsamps'][ii]
        inj_coord = inj_params_from_file['injection_coords'][ii]

        injection = Injection(inj_tsamp, inj_coord, inj_header)

        if inj_tsamp < tobs_samp:
            injection.was_injected = True

            injections.append(injection)

    
    for ii, injection in enumerate(injections):
        inj_part = f"{injection.header['snr']:.2f} \t {injection.header['dm']:.2f} \t {injection.header['width']:.2f} \t {injection.tstamp} \t {injection.coord} \t {injection.header['subsample_phase']:.2f} \t "
        cand = find_cand(all_cands = results, snr=injection.header['snr'], width = injection.header['width'], dm = injection.header['dm'], tstamp = injection.tstamp, coord = injection.coord)

        if cand is None:
            cand_part = "-1 \t\t -1 \t\t -1 \t\t -1 \t\t -1"
        else:
            cand_part = f"{cand['snr']} \t\t {cand['dm_pccm3']} \t\t {cand['boxc_width']} \t\t {cand['total_sample']} \t\t {cand['coord']}"

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

        

        





