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

from get_dm_mask import get_dm_mask

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

    foff_GHz = np.abs(plan.freqs[1] - plan.freqs[0]) * 1e-9
    flower_GHz = plan.freqs[-1]*1e-9 - foff_GHz/2.

    this_inj = -1
    #next_inj = 0
    cumulative_block = np.empty((plan.nf, 0))
    for ii, iblock in enumerate(FV.get_fake_data_block()):
        print(f"Got block {ii}, this_inj is {this_inj}")
        block = np.abs(iblock).sum(axis=0)
        next_inj_samp = injection_params['injection_tsamps'][this_inj + 1]
        next_inj_dm = injection_params['furby_props'][this_inj + 1]['dm_samps']

        if ii * plan.nt <= next_inj_samp < (ii+1)*plan.nt:
            print(f"This block is the last block for the current inj ->{this_inj}")
            cumulative_block = np.concatenate([cumulative_block, block], axis=1)

            dm_mask_vg, ss, es, ms = get_dm_mask_vg(next_inj_dm, plan.freqs[::-1], return_edges=True)
            dm_mask_k = get_dm_mask(f_lower = flower_GHz, chanbwGhz = foff_GHz, Nchan = plan.nf, NDMtrials = 1024, TimeBlockSize = plan.nt, wanted_idm = int(next_inj_dm))
            
            frb_loc = np.argmax(cumulative_block[0])
            frb_ex = cumulative_block[:, frb_loc-int(np.floor(next_inj_dm)) - 2: frb_loc+3]

            if args.plot_all:
                plt.figure()
                plt.imshow(frb_ex, aspect='auto', interpolation='None', extent=[0, frb_ex.shape[1], plan.nf, 0])
                print(f"The shape of the cumulative block is {cumulative_block.shape}")
                plt.title("Cumul block")
                plt.figure()
                plt.imshow(dm_mask_vg[::-1], aspect='auto', interpolation='None', extent=[0, dm_mask_vg.shape[1], plan.nf, 0])
                plt.plot(ss, np.arange(plan.nf)[::-1], '.-', label="Start")
                plt.plot(ms, np.arange(plan.nf)[::-1], '.-', label="Mid")
                plt.plot(es, np.arange(plan.nf)[::-1], '.-', label="End")
                plt.title("DM mask VG")
                plt.legend()
                plt.figure()
                plt.imshow(dm_mask_k, aspect='auto', interpolation='None', extent=[0, dm_mask_k.shape[1], plan.nf, 0])
                plt.title("DM mask K")

                plt.figure()
                plt.imshow(frb_ex, aspect='auto', interpolation='None', extent=[0, frb_ex.shape[1], plan.nf, 0])
                #plt.imshow(dm_mask_vg[::-1], aspect='auto', interpolation='None')
                offset = frb_ex.shape[1] - 2.5 - es[-1]
                plt.plot(ss + offset, np.arange(plan.nf)[::-1], '.-', label="Start")
                plt.plot(ms + offset, np.arange(plan.nf)[::-1], '.-', label="Mid")
                plt.plot(es + offset, np.arange(plan.nf)[::-1], '.-', label="End")
                plt.show(block=False)
                _ = input("Hit Enter to continue to next iteration")
                plt.close('all')

            this_inj += 1
            cumulative_block = np.empty((plan.nf, 0))

        elif ii * plan.nt <= next_inj_samp - next_inj_dm < (ii+1)*plan.nt:
            print(f"This block is a precursor to the next injection this_inj ={this_inj}")
            cumulative_block = np.concatenate([cumulative_block, block], axis=1)
        else:
            print(f"This is an empty block")

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("inj_params", type=str, help="Path to the file containing injection parameters")
    a.add_argument("-pall", dest='plot_all', action='store_true', help="Plot all blocks?")
    args = a.parse_args()
    main(args)


