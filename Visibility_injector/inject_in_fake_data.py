import numpy as np
from Furby_p3.sim_furby import get_furby
from Furby_p3.Furby_reader import Furby_reader
from Visibility_injector.simulate_vis import gen_dispersed_vis_1PS_from_plan as gen_vis

import yaml
#import matplotlib.pyplot as plt
import logging, sys
from logging.handlers import RotatingFileHandler

#TODO: Remove this line later
import astropy.units as u

def setUpLogging(logfile=None):
    '''
    Sets up and returns a logger
    '''

    logger = logging.getLogger("Visbility_injector")

    stderr_formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
    logfile_formatter = logging.Formatter("%(asctime)s, %(levelname)s: %(message)s")

    consoleHandler = logging.StreamHandler(sys.stderr)
    consoleHandler.setFormatter(stderr_formatter)
    consoleHandler.setLevel(logging.DEBUG)
    logger.addHandler(consoleHandler)

    if logfile:
        try:
            fileHandler = RotatingFileHandler(logfile, maxBytes=2.0E7, backupCount=2)
            fileHandler.setFormatter(logfile_formatter)
            fileHandler.setLevel(logging.DEBUG)
            logger.addHandler(fileHandler)
        except IOError as E:
            raise IOError("Could not enable logging to file: {0}\n".format(logfile), E)
    
    return logger

    
class FakeVisibility(object):
    '''
    Simulates fake visibilities 
    '''
    
    def __init__(self, plan, injection_params_file, tot_nsamps=None, logfile = None):
        '''
        Initialises all the parameters required for simulating fake
        visibilities, and parses the injection parameters provided.
        Also sets the seed for random number generator

        Params
        ------
        plan : object
            An object of the craft.craco_plan.PipelinePlan() or a
            FakePlan() class. The object must contain the following
            attributes:
            - nt: Number of time samples in each block
            - nf: Number of freq channels
            - nbl: Number of baselines
            - fmax: Center freq of highest channel (in Hz)
            - fmin: Center freq of lower channel (in Hz)
            - tsamp_s: Sampling time (in sec)
        
        injection_params_file : str
            Path to the yaml file containing params of the required
            injections
        
        tot_nsamps : int, optional
            Total number of samples that need to be simulated.
            Note - If the requested number of samples is not an
            integral multiple of the plan.nt, then the last block 
            containing a fraction of plan.nt will not be generated.
        
        logfile : str, optional
            Log file to which the injection would be logged
            Give None to disable logging to a file
            Def = None
        
        Raises
        ------
        ValueError :
            If any of the parameters don't make sense or are
            incompatible with one another.

        '''
        self.log = setUpLogging(logfile)
        self.plan = plan
        self.ftop_MHz = (self.plan.fmax + self.plan.foff/2) / 1e6
        self.fbottom_MHz = (self.plan.fmin - self.plan.foff/2) / 1e6
        self.get_injection_params(injection_params_file)
        #Adding an extra line to make sure I parse only floats from plan
        #TODO remove this later
        if isinstance(self.plan.tsamp_s, u.Quantity):
            self.tsamp_s = self.plan.tsamp_s.to('s').value
        else:
            self.tsamp_s = self.plan.tsamp_s
        self.set_furby_gen_mode()

        
        if tot_nsamps is None:
            self.max_nblk = np.inf
        else:
            self.max_nblk = tot_nsamps // plan.nt

        if self.max_nblk < 1:
            raise ValueError(f"Too few tot_nsamps:{tot_nsamps}. We need to generate at least one block")
        self.blk_shape = (plan.nbl, plan.nf, plan.nt)

        self.amplitude_ratio =  1/np.sqrt(self.blk_shape[0])
        self.log.info(f"plan.shape = {self.blk_shape}, Amp ratio = {self.amplitude_ratio}")
        self.n_injections = len(self.injection_params['injection_tsamps'])

        self.sort = np.argsort(self.injection_params['injection_tsamps'])

        self.tel_props_dict = {'ftop': self.ftop_MHz,
                                'fbottom': self.fbottom_MHz,
                                'nch': self.plan.nf,
                                'tsamp': self.tsamp_s, #TODO - change this back to self.plan.tsamp_s
                                'name': "FAKE"                                
                            }
        np.random.seed(self.injection_params['seed'])

        
    def convert_dm_samps_to_pccc(self, dm_samps):
        '''
        Converts DM from samp units to pc/cc
        '''
        #DM equation => delta_t = D * dm * (1 / f1**2 - 1 / f2**2)
        #So, dm = delta_t / (D * (f1**-2  - f2**-2)

        delta_t = dm_samps * self.plan.tsamp_s
        D = 4.14881e6    #ms, needs freqs to be in MHz, output delays in ms   #From Pulsar Handbook 
        DM_pccc = (delta_t * 1e3) / (D * (self.fbottom_MHz**-2 - self.ftop_MHz**-2))
        return DM_pccc

    def convert_dm_pccc_to_samps(self, dm_pccc):
        '''
        Converts DM from pccc to samps
        '''
        D = 4.14881e6   #ms, needs freqs to be in MHz, output delays in ms #From Pulsat Handbook
        delta_t_ms = dm_pccc * D * (self.fbottom_MHz**-2 - self.ftop_MHz**-2)      #ms
        delta_t_s = delta_t_ms * 1e-3
        delta_t_samps = np.rint(delta_t_s / self.plan.tsamp_s).astype('int')

        return delta_t_samps


    def convert_inj_tstamp_s_to_samps(self, tstamp):
        '''
        Converts injection tstamp (in seconds) to samps
        '''
        injection_samp = np.rint(tstamp / self.plan.tsamp).astype('int')
        return injection_samp

    def convert_inj_tstamp_samp_to_s(self, tstamp_samp):
        '''
        Converts injection tstamp in samps to seconds
        '''
        injection_samp_s = tstamp_samp * self.plan.tsamp_s
        return injection_samp_s

    def convert_inj_ra_dec_to_upix_vpix(self, ra, dec):
        '''
        Converts requested RA and DEC (in degrees) to expected
        pixel coordinate in the image
        '''
        upix, vpix = self.plan.wcs.world_to_pixel(self, ra,dec)
        return upix, vpix
    
    def convert_inj_upix_vpix_to_ra_dec(self, upix, vpix):
        '''
        Converts requested pixel upix and vpix coordinates to
        the sky values (RA and DEC)
        '''
        ra, dec = self.plab.wcs.pixel_to_world(ra, dec)
        return ra, dec

    def convert_inj_width_s_to_samps(self, w_s):
        '''
        Converts the requested inj width (in s) to samps
        '''
        return w_s / self.plan.tsamp_s

    def convert_inj_width_samps_to_s(self, w_samps):
        '''
        Converts the requested inj width (in samps) to s
        '''
        return w_samps * self.plan.tsamp_s

    def convert_tau0_samps_to_s(self, tau0_samps):
        '''
        Converts the requested tau0 (in samps) to seconds
        '''
        return tau0_samps * self.plan.tsamp_s

    def get_injection_params(self, injection_params_file):
        '''
        Parses the injection parametes yaml file 

        Params
        ------
        injection_params_file : str
            Path to the yaml file containing injection params
        '''
        self.log.info("Loading the injection param file")
        with open(injection_params_file) as f:
            self.injection_params = yaml.safe_load(f)

        if 'injection_pixels' in self.injection_params:
            self.injection_params['injection_coords'] = [self.convert_inj_upix_vpix_to_ra_dec(ii[0], ii[1]) for ii in self.injection_params['injection_pixels']]

        #This is a patch to support older injection yaml files where injection coord was specified as 0.0
        for ii, coord_pair in enumerate(self.injection_params['injection_coords']):
            if isinstance(coord_pair, 'float') and coord_pair == 0.0:
                    self.injection_params['injection_coords'][ii] == [self.plan.ra.to('deg').value, self.plan.dec.to('deg').value]

        
        prop_defaults = {
                            'shape': 'tophat',
                            'spectrum_type': 'flat',
                            'dmsmear': True,
                            'subsample_phase': 0.5,
                            'noise_per_sample': 1.0,
                            'tfactor': 100,
                            'tot_nsamps': None,
                            'scattering_index': 4.4
        }
        self.injection_params['parsed_furby_props'] = []

        for prop in self.injection_params['furby_props']:
            
            parsed_prop = prop_defaults.copy()
            parsed_prop['snr'] = prop['snr']
            
            if 'width_samps' in prop:
                parsed_prop['width'] = self.convert_inj_width_samps_to_s(prop['width_samps'])
            else:
                parsed_prop['width'] = prop['width']
            if 'dm_samps' in prop:
                parsed_prop['dm'] = self.convert_dm_samps_to_pccc(prop['dm_samps'])
            else:
                parsed_prop['dm'] = prop['dm']
            if 'tau0_samps' in prop:
                parsed_prop['tau0'] = self.convert_tau0_samps_to_s(prop['tau0_samps'])
            else:
                parsed_prop['tau0'] = prop['tau0']

            for iprop in prop_defaults.keys():
                if iprop in prop.keys():
                    parsed_prop[iprop] = prop[iprop]
            
            self.injection_params['parsed_furby_props'].append(parsed_prop)
            



    def set_furby_gen_mode(self):
        '''
        Sets the furby generation mode based on the injection params

        Raises
        ------
        ValueError:
            If the necessary options are not provided in injection params
        '''
        self.read_in_runtime = False
        self.simulate_in_runtime = False
        if 'furby_files' in self.injection_params:
            self.read_in_runtime = True
        elif 'furby_props' in self.injection_params:
            self.simulate_in_runtime = True
        else:
            raise ValueError("The injection params file needs to specify either 'furby_files' or 'furby_props'")


    def get_ith_furby(self, iFRB):
        '''
        Gets the block of data containing the Furby that is at the ith
        position in the list of injections. It will either read the 
        furby_file from disk, or call the get_furby() function to 
        generate one in real-time dependining upon the specified params
        in the yaml file.

        Params
        ------
        iFRB : int
            Index of the Furby which needs to be injected.

        Returns
        -------
        furby_data : numpy.ndarray
            2-D block of numpy array containing the time-freq profile
            of the mock FRB
        furby_nsamps : int
            Number of time samples in that furby
        location_of_frb : int
            The samp numnber of the peak of the frb in the lowest
            freq channel

        Raises
        ------
        ValueError:
            If the header params of the furby read from disk do not 
            match the params in the plan.
        
        '''
        if self.read_in_runtime:
            self.log.info("Reading fuby from file: {0}".format(self.injection_params['furby_files'][iFRB]))
            furby = Furby_reader(self.injection_params['furby_files'][iFRB])
            furby_data = furby.read_data()

            if (
                (furby.header.NCHAN == self.plan.nf) and
                (furby.header.TSAMP_US * 1e6 == self.plan.tsamp_s) and
                (furby.header.FTOP_MHZ == self.ftop_MHz) and
                (furby.header.FBOTTOM_MHZ == self.fbottom_MHz)  or
                True):

                    if furby.header.BW_MHZ < 0:
                        furby_data = furby_data[::-1, :].copy()
                    furby_data *= self.amplitude_ratio
                    location_of_frb = np.argmax(furby_data[0])
                    return furby_data, furby.header.NSAMPS, location_of_frb
            else:
                self.log.info("furby_header = {0}".format(furby.header), "nf, tsamp_s, ftop_MHz, fbottom_MHz = ",
                self.plan.nf, self.plan.tsamp_s, self.ftop_MHz, self.fbottom_MHz)
                raise ValueError("Params for furby_{0} do not match the requested telescope params".format(furby.header.ID))

        elif self.simulate_in_runtime:

            P = self.injection_params['parsed_furby_props'][iFRB]
            self.log.info("Simulating {ii}th furby with params:\n{params}".format(ii=iFRB, params=P))

            furby_data, furby_header = get_furby(**P)
            furby_data = furby_data[::-1, :].copy() * self.amplitude_ratio
            location_of_frb = np.argmax(furby_data[0])
            return furby_data, furby_header['NSAMPS'], location_of_frb

        else:
            raise ValueError("Neither asked to read nor to simulate")

    def add_fake_noise(self, data_block):
        '''
        Adds fake noise to the real and imag part of the 
        provided data_block

        Params
        ------
        data_block : numpy.ndarray
            Numpy array of dtype complex64 to which noise has to be 
            added
        '''
        data_block.real = np.random.randn(*data_block.real.shape)
        data_block.imag = np.random.randn(*data_block.imag.shape)


    def get_fake_data_block(self):
        '''
        Gets data blocks containing fake noise and injected furbys.
        It calls the add_fake_noise() and get_ith_furby() functions,
        adds the noise and the furby at appropriate time stamps
        and yields one data block at a time.

        If a furby is asked to be injected before the last injection
        has finished, it raises a Warning and ignored the next
        injection while continuing to finish the current one. Future
        injections remain unaffected.
        '''
        breaking_point = False
        injecting_here = False
        i_inj = 0
        iFRB = self.sort[i_inj]
        iblk = -1
        current_mock_FRB_data, current_mock_FRB_NSAMPS, location_of_peak = self.get_ith_furby(iFRB)
        current_mock_FRB_vis = gen_vis(self.plan, src_ra_deg = self.injection_params['injection_coords'][iFRB][0], 
                                        src_dec_deg = self.injection_params['injection_coords'][iFRB][1], 
                                        dynamic_spectrum=current_mock_FRB_data.T)
        samps_added = 0
        injection_samp = self.injection_params['injection_tsamps'][iFRB] - location_of_peak

        #for iblk in range(self.nblk):
        while True:
            iblk +=1
            self.log.info(f"Block ID: {iblk}, start_samp = {iblk * self.blk_shape[2]}, end_samp = {(iblk+1) * self.blk_shape[2]}")
            data_block = np.zeros(self.blk_shape, dtype=np.complex64)

            if self.injection_params['add_noise']:
                self.add_fake_noise(data_block)

            if injection_samp + samps_added < iblk * self.blk_shape[2]:
                raise RuntimeError(f"The requested injection samp {injection_samp} is too soon.")
            
            if injection_samp >= iblk * self.blk_shape[2] and injection_samp < (iblk + 1) * self.blk_shape[2]:
                self.log.info(f"Injection will start in this block")
                injecting_here = True
            if injecting_here:
                injection_start_samp_within_block = max([0, injection_samp - iblk * self.blk_shape[2]])

                injection_end_samp_within_block = min([self.blk_shape[2], 
                    injection_samp + current_mock_FRB_NSAMPS - iblk * self.blk_shape[2]])

                self.log.info(f"injection_start_samp_within_block = {injection_start_samp_within_block}")
                self.log.info(f"injection_end_samp_within_block = {injection_end_samp_within_block}")

                samps_to_add_in_this_block = injection_end_samp_within_block - injection_start_samp_within_block

                data_block[:, :, injection_start_samp_within_block : injection_end_samp_within_block] += \
                    current_mock_FRB_vis[:, :, samps_added : samps_added + samps_to_add_in_this_block]

                samps_added += samps_to_add_in_this_block
            
            if (injection_samp + current_mock_FRB_NSAMPS-1) >= iblk * self.blk_shape[2] and (injection_samp + current_mock_FRB_NSAMPS-1) < (iblk + 1) * self.blk_shape[2]:
            #if samps_added == current_mock_FRB_NSAMPS:
                self.log.info("This was the last block which had a section of the frb, now onto the next one")
                injecting_here = False
                i_inj += 1
                if i_inj >= self.n_injections or iblk >= self.max_nblk:
                    self.log.info("This was also the last FRB, so this will be the last block I will yield")
                    breaking_point = True

                else:
                    iFRB = self.sort[i_inj]

                    if i_inj < self.n_injections:
                        current_mock_FRB_data, current_mock_FRB_NSAMPS, location_of_peak = self.get_ith_furby(iFRB)
                        current_mock_FRB_vis = gen_vis(self.plan, src_ra_deg = self.injection_params['injection_coords'][iFRB][0], 
                                                    src_dec_deg = self.injection_params['injection_coords'][iFRB][1], 
                                                    dynamic_spectrum=current_mock_FRB_data.T)

                        samps_added = 0
                        injection_samp = self.injection_params['injection_tsamps'][iFRB] - location_of_peak
                        self.log.info(f"New injection samp will be {injection_samp}")


    
            yield data_block

            if breaking_point:
                break




