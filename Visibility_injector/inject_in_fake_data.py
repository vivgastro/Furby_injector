import numpy as np
from Furby_p3.sim_furby import get_furby
from Furby_p3.Furby_reader import Furby_reader
from Visibility_injector.simulate_vis import gen_dispersed_vis_1PS_from_plan as gen_vis
from Visibility_injector.simulate_vis import convert_cube_to_dict
from craft import uvfits, craco

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
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Visbility_injector")

    stderr_formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
    logfile_formatter = logging.Formatter("%(asctime)s, %(levelname)s: %(message)s")

    #consoleHandler = logging.StreamHandler(sys.stdout)
    #consoleHandler.setFormatter(stderr_formatter)
    #consoleHandler.setLevel(logging.DEBUG)
    #logger.addHandler(consoleHandler)

    if logfile:
        try:
            fileHandler = RotatingFileHandler(logfile, maxBytes=2.0E7, backupCount=2)
            fileHandler.setFormatter(logfile_formatter)
            fileHandler.setLevel(logging.DEBUG)
            logger.addHandler(fileHandler)
        except IOError as E:
            raise IOError("Could not enable logging to file: {0}\n".format(logfile), E)
    
    
    return logger

class CurrentInjection(object):
    def __init__(self, iFRB, 
                 injection_start_samp, 
                 injection_end_samp, 
                 furby_nsamps, 
                 location_of_peak):
        self.iFRB = iFRB
        self.furby_samps_added = 0
        self.injection_start_samp = injection_start_samp
        self.injection_end_samp = injection_end_samp
        self.furby_nsamps = furby_nsamps
        self.location_of_peak = location_of_peak
        self._furby_data = None
        self._furby_vis = None
        self.injecting_here = False

    @property
    def furby_data(self):
        if self._furby_data is None:
            raise RuntimeError("Furby_data has not been set yet!")
        return self._furby_data

    @property
    def furby_vis(self):
        if self._furby_vis is None:
            raise RuntimeError("Furby_vis has not been set yet!")
                
    def set_furby_data(self, furby_data_block):
        self._furby_data = furby_data_block

    def set_furby_vis(self, furby_vis_block):
        self._furby_vis = furby_vis_block

class FakeVisibility(object):
    '''
    Simulates fake visibilities 
    '''
    
    def __init__(self, plan, injection_params_file, tot_nsamps=None, vis_source=None, outblock_type=np.ndarray, logfile = None):
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

        vis_source : str, optional
            File path from which vis data has to be read. If None,
            then vis data would be simulated on the fly. Def = None
            (Reading from the file is not implemented yet)

        outblock_type : type, optional
            Format of the output block desired. Options are:
            np.ndarray or dict. Default = np.ndarray

        
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
        #print("Plan.nt IS ", plan.nt)
        self.ftop_MHz = (self.plan.fmax + self.plan.foff/2) / 1e6
        self.fbottom_MHz = (self.plan.fmin - self.plan.foff/2) / 1e6
        #Adding an extra line to make sure I parse only floats from plan
        #TODO remove this later
        if isinstance(self.plan.tsamp_s, u.Quantity):
            self.tsamp_s = self.plan.tsamp_s.to('s').value
        else:
            self.tsamp_s = self.plan.tsamp_s
        self.get_injection_params(injection_params_file)

        if outblock_type in [np.ndarray, dict]:
            self.outblock_type = outblock_type
        else:
            raise ValueError(f"Unknown outblock_type specified: {outblock_type}")

        self.set_furby_gen_mode()

        self.vis_source = self.get_vis_data(vis_source)
        
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
        
        self.injections_added = 0   #Added for use in add_frb_to_data_block() only
        self.injecting_here = False #Added for use in add_frb_to_data_block() only

        self.sort = np.argsort(self.injection_params['injection_tsamps'])

        self.tel_props_dict = {'ftop': self.ftop_MHz,
                                'fbottom': self.fbottom_MHz,
                                'nch': self.plan.nf,
                                'tsamp': self.tsamp_s, #TODO - change this back to self.plan.tsamp_s
                                'name': "FAKE"                                
                            }
        np.random.seed(self.injection_params['seed'])


    def get_vis_data(self, fname):
        '''
        Either creates fake vis data (zeros or random noise)
        Or reads Visibility data from a file on disk

        If fname is None, then it generates fake vis data

        fname: str
            Path to the file on disk which contains the visibities to read
        '''
        self.log.debug(f"Vis_source is {fname}")
        if fname == None:
            while True:
                data_block = np.zeros(self.blk_shape, dtype=np.complex64)
                if self.injection_params['add_noise']:
                    self.log.debug("Adding fake noise")
                    self.add_fake_noise(data_block)
                yield data_block
        else:
            self.log.debug("Reading vis from file")
            f = uvfits.open(fname)
            blocker = f.time_blocks(nt = self.plan.nt)
            while True:
                block = next(blocker)
                yield craco.bl2array(block)
            #return f.time_blocks(nt = self.plan.nt)
            #Open the file
            #Read the data
            #Make a data cube (nbl, nch, nt)
            #Assert that the shape (nbl, nch, nt) matches (plan.nbl, plan.nf, plan.nt)
            #Reorder the baselines to match plan.baseline_order
            #Return the cube
            #raise NotImplementedError("Not yet implemented reading of visibilities from a file")

    def convert_dm_samps_to_pccc(self, dm_samps):
        '''
        Converts DM from samp units to pc/cc
        '''
        #DM equation => delta_t = D * dm * (1 / f1**2 - 1 / f2**2)
        #So, dm = delta_t / (D * (f1**-2  - f2**-2)

        delta_t = dm_samps * self.tsamp_s
        D = 4.14881e6    #ms, needs freqs to be in MHz, output delays in ms   #From Pulsar Handbook 
        ftop = self.ftop_MHz - (self.plan.foff / 2e6)
        #print("THIS IS THE CHANGED CODEEEEE")
        fbottom = self.fbottom_MHz + (self.plan.foff / 2e6)
        DM_pccc = (delta_t * 1e3) / (D * (fbottom**-2 - ftop**-2))
        return DM_pccc

    def convert_dm_pccc_to_samps(self, dm_pccc):
        '''
        Converts DM from pccc to samps
        '''
        D = 4.14881e6   #ms, needs freqs to be in MHz, output delays in ms #From Pulsat Handbook
        ftop = self.ftop_MHz - (self.plan.foff / 2e6)
        fbottom = self.fbottom_MHz + (self.plan.foff / 2e6)
        delta_t_ms = dm_pccc * D * (fbottom**-2 - ftop**-2)      #ms
        delta_t_s = delta_t_ms * 1e-3
        #print("THIS IS THE CHANGED CODEEEEE")
        delta_t_samps = np.rint(delta_t_s / self.tsamp_s).astype('int')

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
        injection_samp_s = tstamp_samp * self.tsamp_s
        return injection_samp_s

    def convert_inj_ra_dec_to_upix_vpix(self, ra, dec):
        '''
        Converts requested RA and DEC (in degrees) to expected
        pixel coordinate in the image
        '''
        upix, vpix = self.plan.wcs.world_to_pixel(ra,dec)
        return upix, vpix
    
    def convert_inj_upix_vpix_to_ra_dec(self, upix, vpix):
        '''
        Converts requested pixel upix and vpix coordinates to
        the sky values (RA and DEC)
        '''
        coords = self.plan.wcs.pixel_to_world(upix, vpix)
        return coords.ra.deg, coords.dec.deg

    def convert_inj_width_s_to_samps(self, w_s):
        '''
        Converts the requested inj width (in s) to samps
        '''
        return w_s / self.tsamp_s

    def convert_inj_width_samps_to_s(self, w_samps):
        '''
        Converts the requested inj width (in samps) to s
        '''
        return w_samps * self.tsamp_s

    def convert_tau0_samps_to_s(self, tau0_samps):
        '''
        Converts the requested tau0 (in samps) to seconds
        '''
        return tau0_samps * self.tsamp_s

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
            if isinstance(coord_pair, float) and coord_pair == 0.0:
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
                (furby.header.TSAMP_US * 1e6 == self.tsamp_s) and
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
                self.plan.nf, self.tsamp_s, self.ftop_MHz, self.fbottom_MHz)
                raise ValueError("Params for furby_{0} do not match the requested telescope params".format(furby.header.ID))

        elif self.simulate_in_runtime:

            P = self.injection_params['parsed_furby_props'][iFRB]
            self.log.info("Simulating {ii}th furby with params:\n{params}".format(ii=iFRB, params=P))

            furby_data, furby_header = get_furby(telescope_params=self.tel_props_dict, **P)
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


    def gen_no_vis(self, dyn_spectrum):
        outblock = np.zeros((self.blk_shape[0], dyn_spectrum.shape[0], dyn_spectrum.shape[1]), dtype=np.complex64)
        outblock[:].real = dyn_spectrum
        return outblock

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
        #current_mock_FRB_vis = self.gen_no_vis(current_mock_FRB_data)
        current_mock_FRB_vis = gen_vis(self.plan, src_ra_deg = self.injection_params['injection_coords'][iFRB][0], 
                                       src_dec_deg = self.injection_params['injection_coords'][iFRB][1], 
                                       dynamic_spectrum=current_mock_FRB_data.T)
        samps_added = 0
        injection_samp = self.injection_params['injection_tsamps'][iFRB] - location_of_peak

        #for iblk in range(self.nblk):
        while True:
            iblk +=1
            self.log.info(f"Block ID: {iblk}, start_samp = {iblk * self.blk_shape[2]}, end_samp = {(iblk+1) * self.blk_shape[2]}")

            data_block = next(self.vis_source)


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
                print("I've just injected something")

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
                        #current_mock_FRB_vis = self.gen_no_vis(current_mock_FRB_data)
                        current_mock_FRB_vis = gen_vis(self.plan, src_ra_deg = self.injection_params['injection_coords'][iFRB][0], 
                                                    src_dec_deg = self.injection_params['injection_coords'][iFRB][1], 
                                                    dynamic_spectrum=current_mock_FRB_data.T)

                        samps_added = 0
                        injection_samp = self.injection_params['injection_tsamps'][iFRB] - location_of_peak
                        self.log.info(f"New injection samp will be {injection_samp}")


            if self.outblock_type == np.ndarray:
                yield data_block
            elif self.outblock_type == dict:
                yield convert_cube_to_dict(self.plan, data_block)

            if breaking_point:
                break
    
    def load_next_injection(self):
        iFRB = self.sort[self.injections_added]
        current_mock_FRB_data, current_mock_FRB_NSAMPS, location_of_peak = self.get_ith_furby(iFRB)
        injection_start_samp = self.injection_params['injection_tsamps'][iFRB] - location_of_peak
        injection_end_samp = injection_start_samp + current_mock_FRB_NSAMPS - 1
        next_injection = CurrentInjection(
            iFRB=iFRB,
            injection_start_samp=injection_start_samp,
            injection_end_samp=injection_end_samp,
            furby_nsamps=current_mock_FRB_NSAMPS,
            location_of_peak=location_of_peak)
        next_injection.set_furby_data(current_mock_FRB_data)
        return next_injection

    def inject_frb_in_data_block(self, data_block, iblk, current_plan):
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
        if self.injections_added == self.n_injections:
            return data_block
        
        assert data_block.shape == self.blk_shape
        nt = self.blk_shape[2]
        samps_passed = iblk * nt
        
        if iblk == 0 and self.n_injections > 0 and self.injections_added == 0:
            #Load the details of the first injection -- because you need a few values to determine the appropriate injection_start
            self.current_injection = self.load_next_injection()

        if self.current_injection.injection_start_samp < samps_passed and not self.injecting_here:
            raise RuntimeError(f"The requested injection samp {self.current_injection.injection_start_samp} is in the past!")
            
        if self.current_injection.injection_start_samp >= samps_passed and self.current_injection.injection_start_samp < (samps_passed + nt):
            self.log.info(f"{self.injections_added + 1}th injection will start in this block")
            self.injecting_here = True
            #current_plan = create_plan(self.fname, iblk)
            current_mock_FRB_vis = gen_vis(current_plan, src_ra_deg = self.injection_params['injection_coords'][self.current_injection.iFRB][0], 
                                       src_dec_deg = self.injection_params['injection_coords'][self.current_injection.iFRB][1], 
                                       dynamic_spectrum=self.current_injection.furby_data.T)
            self.current_injection.set_furby_vis(current_mock_FRB_vis)
        
        if self.injecting_here:
            
            injection_start_samp_within_block = max([0, self.current_injection.injection_start_samp - samps_passed])
            injection_end_samp_within_block = min([nt, self.current_injection.injection_end_samp - samps_passed])
            self.log.info(f"injection_start_samp_within_block = {injection_start_samp_within_block}")
            self.log.info(f"injection_end_samp_within_block = {injection_end_samp_within_block}")
            
            samps_to_add_in_this_block = injection_end_samp_within_block - injection_start_samp_within_block
    
            data_block[:, :, injection_start_samp_within_block : injection_end_samp_within_block] += \
                self.current_injection.furby_vis[:, :, self.current_injection.furby_samps_added : self.current_injection.furby_samps_added + samps_to_add_in_this_block]
            
            self.current_injection.furby_samps_added += samps_to_add_in_this_block
            
            if (self.current_injection.injection_end_samp) >= iblk * nt and (self.current_injection.injection_end_samp < (iblk + 1) * nt):
            #if samps_added == current_mock_FRB_NSAMPS:
                self.log.info("This was the last block which had a section of the current frb, now onto the next one")
                self.injecting_here = False
                self.injections_added += 1
                if self.injections_added == self.n_injections:
                    self.log.info("This was also the last FRB I was asked to add.. Yayy")
                else:
                    #Now load the next FRB
                    self.current_injection = self.load_next_injection()
                    data_block = self.inject_frb_in_data_block(data_block, iblk, current_plan)

        return data_block
            