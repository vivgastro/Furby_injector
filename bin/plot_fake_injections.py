import matplotlib.pyplot as plt
from Visibility_injector.inject_in_fake_data import FakeVisibility
import numpy as np
import argparse
from craft.craco_plan import PipelinePlan
from craft import uvfits



def main(args):
    injections_params_file = args.inj_params
    fits = args.uv
    f = uvfits.open(fits) 
    plan = PipelinePlan(f, f"--ndm 250")

    injections_params_file = args.inj_params

    FV = FakeVisibility(plan, tot_nsamps = None,  injection_params_file = injections_params_file)
    
    block_gen = FV.get_fake_data_block

    for ii, block in enumerate(FV.get_fake_data_block()):
        print(f"For block {ii}: Sum of abs(block) is {np.sum(np.abs(block))}, mean of abs(block) is {np.mean(np.abs(block.ravel()))}, rms is {np.std(block.ravel())}" )
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(221)
        ax1.imshow(block.real.sum(axis=0), aspect='auto', interpolation='None')
        ax1.set_title("Real")
        ax2 = fig.add_subplot(222, sharex = ax1, sharey=ax1)
        ax2.imshow(block.imag.sum(axis=0), aspect='auto', interpolation='None')
        ax2.set_title("Imag")
        ax3 = fig.add_subplot(223, sharex = ax1, sharey=ax1)
        ax3.imshow(np.abs(block).sum(axis=0), aspect='auto', interpolation='None')
        ax3.set_title("Abs")
        plt.title(f"Block = {ii}")
        plt.show()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("inj_params", type=str, help="Path to the file containing injection parameters")
    a.add_argument("-uv", type=str, help="Path to the uvfits file to make a Plan from (def = /home/gup037/tmp/frb_d0_t0_a1_sninf_lm00.fits)", default="/home/gup037/tmp/frb_d0_t0_a1_sninf_lm00.fits")
    args = a.parse_args()
    main(args)
