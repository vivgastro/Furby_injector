import matplotlib.pyplot as plt
from Visibility_injector.inject_in_fake_data import FakeVisibility
import numpy as np
import argparse

class FakePlan():
    pass

plan = FakePlan
plan.nt = 300
plan.nf = 256
plan.nbl = 190
#The following three params are needed for furby simulation
plan.ftop = 991
plan.fbottom = 735
plan.tsamp = 0.001

def main(args):
    injections_params_file = args.inj_params
    FV = FakeVisibility(plan, tot_nsamps = int(0.6e4),  injection_params_file = injections_params_file)

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
