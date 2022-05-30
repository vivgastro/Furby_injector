from craft.craco_plan import PipelinePlan
from craft import uvfits
import sys
sys.path.append("/home/gup037/Codes/craft/src/craft/")
sys.path.append("/home/gup037/Codes/")
from get_dm_mask import get_dm_mask

import matplotlib.pyplot as plt
from Visibility_injector.inject_in_fake_data import FakeVisibility
import numpy as np
import argparse
import yaml

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
    injection_params = parse_inj_file(args.inj_params)

    fits = '/home/gup037/tmp/frb_d0_t0_a1_sninf_lm00.fits'
    f = uvfits.open(fits)

    injections_params_file = args.inj_params
    plan = PipelinePlan(f, f"--ndm 512")

    FV = FakeVisibility(plan, tot_nsamps = 256 * 2 * 50,  injection_params_file = args.inj_params)
    sums = []
    dms = []
    for ii, block in enumerate(FV.get_fake_data_block()):
        if ii%2 == 0:
            dm_pccc = injection_params['furby_props'][int(ii/2)]['dm']
            dms.append(dm_pccc)
            dm_samps = convert_DM_from_pccc_to_samp(dm_pccc, plan)

            #print("Sum of the data in block {1} = {0}".format(np.sum(block.real[:, :, -4]), ii))
            get_dm_mask_call = f"get_dm_mask(f_lower = {plan.fmin - np.abs(plan.foff)/2.},  chanbwGhz={np.abs(plan.foff)*1e-9}, Nchan={plan.nf}, NDMtrials = 1024, TimeBlockSize = {plan.nt}, wanted_idm = {dm_samps})"

            dm_mask = get_dm_mask(f_lower = plan.fmin - np.abs(plan.foff)/2.,  chanbwGhz=np.abs(plan.foff)*1e-9, Nchan=plan.nf, NDMtrials = 1024, TimeBlockSize = plan.nt, wanted_idm = dm_samps)
            #if True:
                #plt.figure()
                #plt.imshow(dm_mask, aspect='auto', interpolation="None")
                #plt.title("DM mask for dm_pccc = {0}, dm_samps = {1}, block = {2}".format(dm_pccc, dm_samps, ii))
        
            nsamps_in_mask = dm_mask.shape[1]
            frb_block = np.abs(block).sum(axis=0)
            frb_loc = np.argmax(frb_block[0])

            frb_ex = frb_block[:,frb_loc - nsamps_in_mask+1:frb_loc+1]
            #print("frb_ex.shape = {0}, dm_mask.shape = {1}".format(frb_ex.shape, dm_mask.shape))
            cross_p = frb_ex * dm_mask
            sum_of_cross_p = np.sum(cross_p)
            sums.append(sum_of_cross_p)
            print(f"{ii}")
            #if True:
                #plt.figure()
                #plt.imshow(cross_p, aspect='auto', interpolation="None")
        #if True: 
            #plt.figure()
            #plt.imshow(np.abs(block).sum(axis=0), aspect='auto', interpolation='None')
            #plt.title(f"Block = {ii}")
            #plt.show(block=False)
            #_ = input()
            #plt.close('all')
    #'''
    plt.figure()
    plt.plot(dms, sums)
    plt.xlabel("DM [pc/cc]")
    plt.ylabel("Sum of frb sweep * FDMT dm mask")
    #'''
    plt.show()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("inj_params", type=str, help="Path to the file containing injection parameters")
    args = a.parse_args()
    main(args)
