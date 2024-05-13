"""
Wrap the GAN earth model parametrization and the ML logging tool into a PET compatible framework
"""
import numpy as np
import os,sys,threading
from copy import deepcopy
home = os.path.expanduser("~") # os independent home

# local load of additional modules.
sys.path.append(os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/deep-borehole-inverse-problem/KERNEL'))
sys.path.append(os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/deep-borehole-inverse-problem/USER_SERGEY'))
sys.path.append(os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/gan-geosteering'))
import mcwd_converter
from vector_to_image import GanEvaluator
from run_model_clean import DnnEvaluatorMcwd
from resitivity import get_resistivity_default
import warnings
# Ignore FutureWarning and UserWarning
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

class GanLog:
    def __init__(self,input_dict=None):
        self.input_dict = input_dict
        self.get_resistivity_default = get_resistivity_default
        self.convert = mcwd_converter.convert_to_mcwd_input
        if input_dict is not None and "file_name" in input_dict:
            self.file_name = input_dict["file_name"]
        else:
            self.file_name = os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth')

        if input_dict is not None and "vec_size" in input_dict:
            self.vec_size = int(input_dict["vec_size"])
        else:
            self.vec_size = 60

        if input_dict is not None and "bit_pos" in input_dict:
            self.pos = [int(el) for el in input_dict['bit_pos']]
        else:
            self.pos = [0]

        #print(self.vec_size)

        self.true_prim = [self.input_dict['reporttype'], self.input_dict['reportpoint']]

        self.true_order = [self.input_dict['reporttype'], self.input_dict['reportpoint']]
        self.all_data_types = self.input_dict['datatype']
        self.l_prim = [int(i) for i in range(len(self.true_prim[1]))]

    def generate_earth_model(self, input):
        if input is None:
            input = self.m
        if len(input.shape) == 1:
            image = self.gan_evaluator.eval(input_vec=input)
            return image
        elif len(input.shape) == 2:
            images = self.gan_evaluator.eval(input_ensemble=input)
            return images

    def _convert_to_resistivity_single(self, earth_model):
        my_shape = earth_model.shape
        result = np.zeros((my_shape[1], my_shape[2]))
        for i in range(my_shape[1]):
            for j in range(my_shape[2]):
                result[i, j] = self.get_resistivity_default(earth_model[:, i, j])
        return result

    def convert_to_resistivity_poor(self, earth_model):
        if len(earth_model.shape) == 3:
            return self._convert_to_resistivity_single(earth_model)
        elif len(earth_model.shape) == 4:
            my_shape = earth_model.shape
            result = np.zeros((my_shape[0], my_shape[2], my_shape[3]))
            for i in range(my_shape[0]):
                result[i, :, :] = self._convert_to_resistivity_single(earth_model[i, :, :, :])
            return result
        return None

    def setup_fwd_run(self,**kwargs):
        self.__dict__.update(kwargs)  # parse kwargs input into class attributes

        assimIndex = [i for i in range(len(self.l_prim))]
        trueOrder = self.true_order

        self.pred_data = [deepcopy({}) for _ in range(len(assimIndex))]
        # for ind in self.l_prim:
        #     for key in self.all_data_types:
        #         self.pred_data[ind][key] = np.zeros((1, 1))

        if isinstance(trueOrder[1], list):  # Check if true data prim. ind. is a list
            self.true_prim = [trueOrder[0], [x for x in trueOrder[1]]]
        else:  # Float
            self.true_prim = [trueOrder[0], [trueOrder[1]]]

    def run_fwd_sim(self, state,member_index=0):
        """
        Method for calling the simulator, handling restarts (it necessary) and retriving data

        ----
        Parameters:
            state : dictionary of system states
        """

        self.gan_evaluator = GanEvaluator(self.file_name, self.vec_size)
        self.mcwd_evaluator = DnnEvaluatorMcwd(
            trained_model_directory=os.path.join(home,
                                                 'OneDrive/DISTINGUISH/ECMOR_study/deep-borehole-inverse-problem/USER_SERGEY/Adaptive_architecture_2_dataset84599_11746'),
            experiment_name="Adaptive_architecture_2")

        for ind in self.l_prim:
            for key in self.all_data_types:
                self.pred_data[ind][key] = np.zeros((1, 1))

        success = False
        while not success:
            success,response = self.call_sim(state=state)

        self.extract_data(response)

        return deepcopy(self.pred_data)


    def call_sim(self,state,output_img = False):
        """
        Run the GAN model

        Optional input:
                - input dictionary with input state:

        Output:
            - d:                    Predicted data
        """

        success = True # this can be used to signal if run has failed

        # if path is not None:
        #     filename = path
        # else:
        #     filename = ''
        image = self.generate_earth_model(*state.values())
        #
        input = []
        output = []
        for p in self.pos:
            column_of_pixels = image[:, :, p]
            mcwd_input = self.convert(column_of_pixels, 32)
            # print("Column of pixels {}".format(column_of_pixels))
            # print("MCWD input {}".format(mcwd_input))
            input.append(mcwd_input)
            output.append(self.mcwd_evaluator.eval(mcwd_input, scale_output=False))
        # print("output {}".format(mcwd_input))
        #log_result = {'res': np.array(output)}
        # print(log_result)
        modelresponse = {'res':np.array(output)}

        if output_img:
            return success,modelresponse,image
        else:
            return success, modelresponse

    def extract_data(self,modelresponse=None):
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key_inx,key in enumerate(self.all_data_types):
                if self.pred_data[prim_ind][key] is not None:  # Obs. data at assim. step
                    self.pred_data[prim_ind][key] = np.array([modelresponse['res'][prim_ind,key_inx]])


# if __name__ == '__main__':
#
#     keys = {'bit_pos': [0, 1, 2, 3, 4, 5, 6, 7, 8],
#             'vec_size': 60}
#
#     my_gan = logGan(keys)
#     numpy_input = np.load('../gan-geosteering/saves/chosen_realization_C1.npz')
#     numpy_single = numpy_input['arr_0']
#
#
#     my_gan.call_sim({'m':numpy_single})