import matplotlib.pyplot as plt
from Visibility_injector.inject_in_fake_data import FakeVisibility
import numpy as np
import argparse

class FakePlan(object):
    
    def __init__(self, nt=300, nbl=190, tsamp=0.001, freqs = np.arange(735, 991, 1) + 0.5):
        self.nt = nt
        self.nbl = nbl
        self.freqs = freqs
        self.tsamp_s = tsamp

    @property
    def tsamp_s(self):
        '''
        Returns the tsamp (in sec)
        '''
        return self.tsamp_s
        
    @property
    def nf(self):
        '''
        Returns the number of channels
        '''
        return len(self.freqs)

    @property
    def foff(self):
        '''
        Returns the channel width (in MHz)
        '''
        return np.abs(self.freqs[1] - self.freqs[0])

    @property
    def fmax(self):
        '''
        Returns the max freq (in MHz)
        '''
        return self.freqs[-1]

    @property
    def fmin(self):
        '''
        Returns the minimum freq (in MHz)
        '''
        return self.freqs[0]


plan = FakePlan()

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
