import matplotlib.pyplot as plt
from Visibility_injector.inject_in_fake_data import FakeVisibility
import numpy as np
import argparse
from craft.craco_plan import PipelinePlan
from craft import uvfits



def main(args):
    injections_params_file = args.inj_params
    fits = '/home/gup037/tmp/frb_d0_t0_a1_sninf_lm00.fits'
    f = uvfits.open(fits) 
    plan = PipelinePlan(f, f"--ndm 512")

    injections_params_file = args.inj_params

    FV = FakeVisibility(plan, tot_nsamps = None,  injection_params_file = injections_params_file)
    
    block_gen = FV.get_fake_data_block

    for ii, block in enumerate(FV.get_fake_data_block()):
        plt.figure()
        plt.imshow(np.abs(block).sum(axis=0), aspect='auto')
        plt.title(f"Block = {ii}")
        plt.show()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("inj_params", type=str, help="Path to the file containing injection parameters")
    args = a.parse_args()
    main(args)