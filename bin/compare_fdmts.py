from craft.craco_plan import PipelinePlan
from craft import uvfits
from get_dm_mask import get_dm_mask
from Furby_p3.miniFDMT import get_dm_mask as get_dm_mask_vg
import sys
sys.path.append("/home/gup037/Codes/craft/src/")
from craft import fdmt 

import matplotlib.pyplot as plt
from Visibility_injector.inject_in_fake_data import FakeVisibility
import numpy as np
import argparse
import yaml

def do_fdmt(block, Ndm, Sdm, Ddm, Nt, St, chan_freqs, method):
    '''
    Does a brute force DMT-transform of the block of data for
    Ndm DM trials starting at Sdm DM with Ddm DM steps,
    and for Nt samples starting at St

    block: np.ndarray
        A block of data of the shape (freq, time) to run
        FDMT on
    Ndm: int
        Number of DM trials to run
    Sdm: int
        Starting DM trial in samples
    Ddm: int
        DM step in samples
    Nt: int
        No of time samples to run FDMT on
    St: int
        Starting sample to run the FDMT on
    chan_freqs: np.ndarray or list
        A list or numpy array containing a list of chan center freqs in GHz
    method: str
        Name of the method used to obtain the dm_mask-
        'bf' for brute-fore (VG's version)
        'f' for Fast (Keith's version)
    '''

    #HERE I AM MAKING THE ASSUMPTION THAT THE DM (in samples) IS 
    #CALCULATED USING THE CENTER FREQUENCIES OF THE HIGHEST AND 
    #THE LOWEST CHANNEL, AND THE FIRST (0th) CHANNEL HAS THE HIGHEST 
    #FREQUENCY

    dm_trials = np.arange(Ndm) * Ddm + Sdm
    samp_trials = np.arange(Nt) + St
    assert St - (Sdm + Ndm) > 0, "Not enough samples at the start to\
        dedipserse out to the maximum DM trial"
    assert (St + Nt) < block.shape[2], f"Not enough samples at the end to\
        accomodate all the requested time samples trials. St={St}, Nt={Nt}, block.shape = {block.shape}"

    assert method.lower() in ['bf', 'f'], f"Unknown method specified {method}"
    b = np.abs(block).sum(axis=0)
    chan_freqs = chan_freqs

    foff = np.abs(chan_freqs[0] - chan_freqs[1])
    #print(chan_freqs, foff)
    time_serieses = []
    for idm in dm_trials:
        if method.lower() == 'bf':
            dm_mask = get_dm_mask_vg(idm, chan_freqs)[::-1]
        elif method.lower() == 'f':
            dm_mask = get_dm_mask(f_lower=chan_freqs[-1] - foff/2, 
                                  chanbwGhz=foff,
                                  Nchan = len(chan_freqs),
                                  NDMtrials = 1024,
                                  TimeBlockSize = 256,
                                  wanted_idm = int(idm))
        else:
            raise RuntimeError("Hallelujah!")
        #plt.figure()
        #plt.imshow(np.abs(block).sum(axis=0), aspect='auto', interpolation='None')
        #plt.figure()
        #plt.imshow(dm_mask, aspect='auto', interpolation='None')
        #plt.title(method)
        #plt.show(block=False)
        #_ = input()
        #plt.close('all')
        time_series = []
        for isamp in samp_trials:
            #print(idm, dm_mask.shape, isamp)
            frb_ex = b[:, isamp - dm_mask.shape[1]+1: isamp+1]
            time_series.append( np.sum(b[:, isamp - dm_mask.shape[1]+1: isamp+1]  *   dm_mask ) / np.sum(b) / np.sqrt(256/np.sum(dm_mask)))

        time_serieses.append(time_series)
    return np.array(time_serieses)

def parse_inj_file(inj_file):
    with open(inj_file) as f:
        inj_params = yaml.safe_load(f)
    return inj_params


def main(args):
    injection_params = parse_inj_file(args.inj_params)
    fits = '/home/gup037/tmp/frb_d0_t0_a1_sninf_lm00.fits'
    f = uvfits.open(fits)

    plan = PipelinePlan(f, "--ndm 512")
    FV = FakeVisibility(plan,injection_params_file = args.inj_params)

    k_best_peaks = []
    v_best_peaks = []
    xes = []
    jj = -1
    try:
        for ii, block in enumerate(FV.get_fake_data_block()):
            print(f"Got block ID = {ii}")
            if np.sum(np.abs(block)) > 0:
                jj+=1
                print(f"Working on block ID = {ii}, jj={jj}")
                xes.append(injection_params['furby_props'][jj]['dm_samps'])
                fdmt_out_k = do_fdmt(block,
                                   Ndm = 10, 
                                   Sdm = max([xes[-1] -5, 0] ),
                                   Ddm = 1,
                                   Nt = 7,
                                   St = injection_params['injection_tsamps'][jj]%256 - 4,
                                   chan_freqs = plan.freqs[::-1] * 1e-9,
                                   method ='f')
                fdmt_out_v = do_fdmt(block,
                                   Ndm = 10, 
                                   Sdm = max([xes[-1] -5, 0] ),
                                   Ddm = 1,
                                   Nt = 7,
                                   St = injection_params['injection_tsamps'][jj]%256 - 4,
                                   chan_freqs = plan.freqs[::-1] * 1e-9,
                                   method ='bf')

                k_best_peaks.append(np.max(fdmt_out_k))
                v_best_peaks.append(np.max(fdmt_out_v))
    except RuntimeError as re:
        print(f"Caught a runtime error:{re}\n\nIgnoring and exiting the loop")
        pass
        
    plt.figure()
    plt.plot(xes, k_best_peaks, label="K")
    plt.plot(xes, v_best_peaks, label="V")
    plt.xlabel("DM [samps]")
    plt.ylabel("Peak of FDMT butterfly")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("inj_params", type=str, help="Path to the file containing injection parameters")
    a.add_argument("-pall", dest='plot_all', action='store_true', help="Plot all blocks?")
    args = a.parse_args()
    main(args)


