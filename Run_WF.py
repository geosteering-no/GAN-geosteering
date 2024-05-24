from pathOPTIM import pathfinder
import numpy as np
from write_data_var import new_data
from pipt.loop.assimilation import Assimilate
from GAN import GanLog
from input_output import read_config
from pipt import pipt_init

def main():
    num_decissions = 1 #64 # number of decissions to make

    start_position = (32, 0) # initial position of the well
    state = np.load('orig_prior.npz')['m'] # the prior latent vector
    np.savez('prior.npz', **{'m': state}) # save the prior as a file

    kf = {'bit_pos': [start_position],
          'vec_size': 60,
          'reporttype': 'pos',
          'reportpoint': [int(el) for el in range(1)],
          'datatype': [f'res{el}' for el in range(1, 14)],
          'parallel': 1}

    sim = GanLog(kf)

    kd, kf = read_config.read_txt('es.pipt')  # read the config file.

    for i in range(num_decissions):
        # start by assimilating data at the current position

        #make a set of syntetic data for the current position
        new_data({'bit_pos': [start_position],
                  'vec_size': 60,
                  'reporttype': 'pos',
                  'reportpoint': [int(el) for el in range(1)],
                  'datatype': [f'res{el}' for el in range(1,14)]})
        # do inversion
        sim.update_bit_pos([start_position])
        analysis = pipt_init.init_da(kd, kf, sim)  # Re-initialize the data assimilation to read the new data
        assimilation = Assimilate(analysis)
        assimilation.run()

        state = np.load('posterior_state_estimate.npz')['m'] # import the posterior state estimate
        np.savez('prior.npz', **{'m': state}) # save the posterior state estimate as the new prior
        # todo here we are saving the decision files - use them
        np.savez(f'estimate_decission_{i}.npz', **{'m': state, 'pos': start_position}) # save the posterior state estimate and the current position of the well

        # find the next position along the optimal path given the state
        next_position = pathfinder().run(state, start_position)[0]
        # todo modify path-finder output here

        start_position = next_position # update the start position for the next decission

main()
