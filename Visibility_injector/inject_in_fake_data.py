import numpy as np
from Furby_p3.sim_furby import get_furby
from Furby_p3.Telescope import Telescope
from Furby_p3.Furby_reader import Furby_reader
import yaml
import matplotlib.pyplot as plt

class FakeVisibility(object):
    '''
    Simulates fake visibilities 
    '''
    
    def __init__(self, plan, tot_nsamps, injection_params_file):
        self.plan = plan
        self.nblk = tot_nsamps // plan.nt
        self.tot_nsamps = self.nblk * plan.nt
        self.read_in_runtime = False
        self.simulate_in_runtime = False

        if self.nblk < 1:
            raise ValueError(f"Too few tot_nsamps:{tot_nsamps}. We need to generate at least one block")
        self.blk_shape = (plan.nbl, plan.nf, plan.nt)
        self.amplitude_ratio =  1/np.sqrt(self.blk_shape[0])
        
        self.get_injection_params(injection_params_file)
        self.n_injections = len(self.injection_params['injection_tsamps'])
        assert self.injection_params['min_gap_samps'] > self.blk_shape[2], "Minimum gap b/w injections needs to be >= plan.nt"
        
    def get_injection_params(self, injection_params_file):
        with open(injection_params_file) as f:
            self.injection_params = yaml.safe_load(f)

        if 'furby_file' in self.injection_params:
            self.read_in_runtime = True
        elif 'furby_props' in self.injection_params:
            self.simulate_in_runtime = True
        else:
            raise ValueError("The injection params file needs to specify either 'furby_file' or 'furby_props'")

        self.tel_props_dict = {'ftop': self.plan.ftop,
                                'fbottom': self.plan.fbottom,
                                'nch': self.plan.nf,
                                'tsamp': self.plan.tsamp,
                                'name': "FAKE"                                
                            }

    def get_next_furby(self, iFRB):
        if self.read_in_runtime:
            print("Reading fuby from file: {0}".format(self.injection_params['furby_file'][iFRB]))
            furby = Furby_reader(self.injection_params['furby_file'][iFRB])
            furby_data = furby.read_data()

            if (
                (furby.header.NCHAN == self.plan.nf) and
                (furby.header.TSAMP == self.plan.tsamp) and
                (furby.header.FTOP == self.plan.ftop) and
                (furby.header.FBOTTOM == self.plan.fbottom)  ):

                    if furby.header.BW < 0:
                        furby_data = furby_data[::-1, :].copy()
                    furby_data *= self.amplitude_ratio
                    return furby_data, furby.header.NSAMPS
            else:
                raise ValueError("Params for furby_{0} do not match the requested telescope params".format(furby.header.ID))

        elif self.simulate_in_runtime:

            P = self.injection_params['furby_props']
            print("Simulating {ii}th furby from params:\n{params}".format(ii=iFRB, params=P))
            furby_data, _, _, furby_nsamps = get_furby(
                P['dm'][iFRB],
                P['snr'][iFRB],
                P['width'][iFRB],
                P['tau0'][iFRB],
                self.tel_props_dict,
                P['spectrum'][iFRB],
                P['noise_per_sample'],            
                 )
            furby_data = furby_data[::-1, :].copy() * self.amplitude_ratio

            import matplotlib.pyplot as plt
            plt.imshow(furby_data, aspect='auto')
            plt.title("simulated FRB")
            plt.show()

            return furby_data, furby_nsamps

    def add_fake_noise(self, data_block, seed = None):
        if seed is None:
            seed = np.random.randint(100)
        np.random.seed(seed)
        data_block.real = np.random.randn(*self.blk_shape)
        np.random.seed(seed+1)
        data_block.imag = np.random.randn(*self.blk_shape)

    def get_fake_data_block(self):
        injecting_here = False
        iFRB = 0
        current_mock_FRB_data, current_mock_FRB_NSAMPS = self.get_next_furby(iFRB)

        samps_added = 0
        injection_samp = self.injection_params['injection_tsamps'][iFRB]

        data_block = np.zeros(self.blk_shape, dtype=np.complex64)
        for iblk in range(self.nblk):
            self.add_fake_noise(data_block)

            print(f"Block ID: {iblk}")
            
            if injection_samp >= iblk * self.blk_shape[2] and injection_samp < (iblk + 1) * self.blk_shape[2]:
                print(f"Injection will start in this block")
                injecting_here = True
            if injecting_here:
                injection_start_samp_within_block = max([0, injection_samp - iblk * self.blk_shape[2]])
                print(f"injection_start_samp_within_block = {injection_start_samp_within_block}")

                samps_to_add_in_this_block = min([self.blk_shape[2] - injection_start_samp_within_block, injection_samp + current_mock_FRB_NSAMPS - iblk * self.blk_shape[2]])

                print(f"samps_to_add_in_this_block = {samps_to_add_in_this_block}")

                data_block[:, :, injection_start_samp_within_block : injection_start_samp_within_block + samps_to_add_in_this_block].real += \
                    current_mock_FRB_data[:, samps_added : samps_added + samps_to_add_in_this_block]

                samps_added += samps_to_add_in_this_block
                print(f"samps_added = {samps_added}")
            
            if (injection_samp + current_mock_FRB_NSAMPS) >= iblk * self.blk_shape[2] and (injection_samp + current_mock_FRB_NSAMPS) < (iblk + 1) * self.blk_shape[2]:
            #if samps_added == current_mock_FRB_NSAMPS:
                print("This was the last block which had a section of the frb, onto the next one")
                injecting_here = False
                iFRB += 1
                if iFRB < self.n_injections:
                    current_mock_FRB_data, current_mock_FRB_NSAMPS = self.get_next_furby(iFRB)
                    samps_added = 0
                    injection_samp = self.injection_params['injection_tsamps'][iFRB]
                    print(f"New injection samp will be {injection_samp}")

            
            yield data_block




