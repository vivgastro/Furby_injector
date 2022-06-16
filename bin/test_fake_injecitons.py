from craft.craco_plan import PipelinePlan
from craft import uvfits
import sys
sys.path.append("/home/gup037/Codes/craft/src/craft")

from get_dm_mask import get_dm_mask
from Furby_p3.miniFDMT import get_dm_mask as get_dm_mask_vg

from craft import fdmt

import matplotlib.pyplot as plt
from Visibility_injector.inject_in_fake_data import FakeVisibility
import numpy as np
import argparse
import yaml
N = np

def dedisperse(data, dm_samps = None): 
    BW_MHZ = 256
    NCHAN = 256
    chw = BW_MHZ / NCHAN 
    foff = chw / 2. 
    f0 = 735.0
    fmax = 992.0
    
    tsamp = 1.7e-3                      #Dada header has to have tsamp in usec

    fch = (f0 + foff) + N.arange(NCHAN) * chw   #(f0 + foff) becomes the centre frequency of the first channel

    delays_in_samples = N.rint(( fch**-2 - (f0)**-2 ) / (fmax**-2 - f0**-2) * dm_samps).astype('int')

    #delays = dm * 4.14881e3 * ( fch**-2 - (f0+foff)**-2 )   #in seconds
    #delays -= delays[int(self.header.NCHAN/2)]
    #delays_in_samples = N.rint(delays / tsamp).astype('int')

    d_data = []
    for i, row in enumerate(data):
        d_data.append(N.roll(row, -1*delays_in_samples[i]))
    d_data = N.array(d_data)
    return d_data


def parse_inj_file(inj_file):
    with open(inj_file) as f:
        inj_params = yaml.safe_load(f)
    return inj_params


def convert_DM_from_pccc_to_samp(dm, plan):
    D = 4.14881e6       #ms; from Pulsar Handbook 4.1.1
    ftop_MHz = (plan.fmax + np.abs(plan.foff/2.) ) * 1e-6
    fmin_MHz = (plan.fmin - np.abs(plan.foff/2.) ) * 1e-6
    delays = D * np.array([ftop_MHz, fmin_MHz])**(-2) * dm    *1e-3  #Only works if freq in MHz and D in ms. Output delays in ms, multiply by 1e-3 to conver to 's'
    delays_samples = np.round(delays / plan.tsamp_s).astype('int')
    dm_samples = np.abs(delays_samples[-1] - delays_samples[0])
    return dm_samples.value


def main(args):
    plot_all = True
    injection_params = parse_inj_file(args.inj_params)

    fits = '/home/gup037/tmp/frb_d0_t0_a1_sninf_lm00.fits'
    f = uvfits.open(fits)

    injections_params_file = args.inj_params
    plan = PipelinePlan(f, "--ndm 512")
    #print("Plan.freqs = ", plan.freqs)
    FV = FakeVisibility(plan, tot_nsamps = 256 * 200 * 50,  injection_params_file = args.inj_params)
    sums_m1 = []
    sums_0 = []
    sums_1 = []
    sums_1_vg = []
    true_sums = []
    dms = []
    sub=0
    try:
        iblock = np.zeros((190, 256, 1))
        for ii, block in enumerate(FV.get_fake_data_block()):
            if ii > 200:
                break
            print(ii, len(injection_params['furby_props']), "<<<<<<")
            #print(injection_params['furby_props'][int(ii)]['dm_samps'])
            iblock = np.concatenate([iblock, block], axis=2)
            if ii%3 == 0:
                if np.max(np.abs(iblock)) == 0:
                    sub += -1
                    continue
                jj = int(ii/3) + sub
                #dm_pccc = injection_params['furby_props'][int(ii/2)]['dm']
                #dms.append(dm_pccc)
                #dms.append(dm_pccc)
                #dm_samps = convert_DM_from_pccc_to_samp(dm_pccc, plan)
                dm_samps = injection_params['furby_props'][jj]['dm_samps']
                dms.append(dm_samps)

                #print("Sum of the data in block {1} = {0}".format(np.sum(block.real[:, :, -4]), ii))
                #get_dm_mask_call = f"get_dm_mask((f_lower = {(plan.fmin - np.abs(plan.foff)/2.)*1e-9},  chanbwGhz={np.abs(plan.foff)*1e-9}, Nchan={plan.nf}, NDMtrials = 1024, TimeBlockSize = {plan.nt}, wanted_idm = {dm_samps})"
                dm_mask = get_dm_mask(f_lower = (plan.fmin - np.abs(plan.foff)/2.)*1e-9,  chanbwGhz=np.abs(plan.foff)*1e-9, Nchan=plan.nf, NDMtrials = 1024, TimeBlockSize = plan.nt, wanted_idm = int(dm_samps))
                dm_mask_vg = get_dm_mask_vg(int(dm_samps), plan.freqs[::-1])[::-1]
                #print("DM_MASK_VG =", dm_mask_vg, "for dm =", int(dm_samps))

                if args.plot_all:
                    '''
                    software FDMT
                    vfdmt = fdmt.Fdmt((plan.fmin - np.abs(plan.foff)/2.)*1e-9,  np.abs(plan.foff)*1e-9, plan.nf, 1024, iblock.shape[2])
                    dout = vfdmt(np.abs(iblock).sum(axis=0))
                    print("sum= ", np.abs(iblock).sum())
                    '''

                    '''
                    Custom dedispersion
                    plt.figure()
                    plt.imshow(dedisperse(np.abs(iblock).sum(axis=0), dm_samps = -300), aspect='auto', interpolation="None")
                    plt.title(f"Dedispersed block at De-DM = -300.0")
                    '''
                    
                    '''
                    Butterfly plot

                    plt.figure()
                    plt.imshow(dout, aspect='auto', interpolation="None")
                    plt.title("Butterfly plot from the python pipeline")
                    '''

                nsamps_in_mask = dm_mask.shape[1]
                nsamps_in_mask_vg = dm_mask_vg.shape[1]

                frb_block = np.abs(iblock).sum(axis=0)
                frb_loc = np.argmax(frb_block[0])
                print(ii, iblock.shape)
                print("shape of dm_mask = ", dm_mask.shape, frb_loc)
                frb_ex_1 = frb_block[:,frb_loc - nsamps_in_mask+1:frb_loc+1]
                frb_ex_1_vg = frb_block[:,frb_loc - nsamps_in_mask_vg+1:frb_loc+1]
                
                cross_p_1 = frb_ex_1 * dm_mask / np.sqrt(dm_mask.sum())
                cross_p_1_vg = frb_ex_1_vg * dm_mask_vg / np.sqrt(dm_mask_vg.sum())
                
                
                sum_of_cross_p_1 = np.sum(cross_p_1)
                sum_of_cross_p_1_vg = np.sum(cross_p_1_vg)
                
                true_sums.append(np.sum(frb_block))

                sums_1.append(sum_of_cross_p_1)
                sums_1_vg.append(sum_of_cross_p_1_vg)
                print(f"{ii}")
                if args.plot_all:
                    plt.figure()
                    plt.imshow(dm_mask, aspect='auto', interpolation="None", alpha=0.5)
                    plt.imshow(frb_ex_1, aspect='auto', interpolation="None", alpha=0.5)
                    plt.title("DM mask Keith for dm_samps = {1}, block = {2}".format(dm_samps, dm_samps, ii))

                    plt.figure()
                    plt.imshow(dm_mask_vg, aspect='auto', interpolation="None", alpha=0.5)
                    plt.imshow(frb_ex_1_vg, aspect='auto', interpolation="None", alpha=0.5)
                    #plt.title("DM mask for dm_pccc = {0}, dm_samps = {1}, block = {2}".format(dm_pccc, dm_samps, ii))
                    plt.title("DM mask VG for dm_samps = {1}, block = {2}".format(dm_samps, dm_samps, ii))
            
                if args.plot_all:
                    plt.figure()
                    plt.imshow(cross_p_1, aspect='auto', interpolation="None")
                    plt.title("Common part of the sweep (Keith)")
                    
                    plt.figure()
                    plt.imshow(cross_p_1_vg, aspect='auto', interpolation="None")
                    plt.title("Common part of the sweep (VG)")
                if args.plot_all: 
                    plt.figure()
                    plt.imshow(np.abs(iblock).sum(axis=0), aspect='auto', interpolation='None')
                    plt.title(f"Block = {ii}")
                    plt.show(block=False)
                    _ = input()
                    plt.close('all')
                iblock = np.zeros((190, 256, 1))
    except RuntimeError as re:
        print(re)
        pass
        
    plt.figure()
    plt.plot(dms, np.array(sums_1) /np.array(true_sums) , label='K')
    plt.plot(dms, np.array(sums_1_vg) /np.array(true_sums) , label='V')
    plt.xlabel("DM [samps]")
    plt.ylabel("Sum of frb sweep * FDMT dm mask / sqrt(sum(mask))")
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("inj_params", type=str, help="Path to the file containing injection parameters")
    a.add_argument("-pall", dest='plot_all', action='store_true', help="Plot all blocks?")
    args = a.parse_args()
    main(args)
