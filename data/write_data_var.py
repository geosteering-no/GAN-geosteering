import numpy as np
import csv
from scipy.linalg import block_diag
import os,sys

# adjust relative path
if __name__ == '__main__':
    prefix = '../'
else:
    prefix = ''

sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/KERNEL')
sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/USER_SERGEY')
sys.path.append(prefix + '../gan-geosteering')

from GAN import GanLog

from vector_to_image import GanEvaluator
from run_model_clean import DnnEvaluatorMcwd

keys = {'bit_pos': [int(el) for el in range(64)],
        'vec_size': 60,
        'reporttype': 'pos',
        'reportpoint': [int(el) for el in range(64)],
        'datatype': [f'res{el}' for el in range(1,14)],
        'file_name': prefix+'../gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth'}

my_gan = GanLog(keys)
numpy_input = np.load(prefix + '../gan-geosteering/saves/chosen_realization_C1.npz')
numpy_single = numpy_input['arr_0']

my_gan.gan_evaluator = GanEvaluator(my_gan.file_name, my_gan.vec_size)
my_gan.mcwd_evaluator = DnnEvaluatorMcwd(
            trained_model_directory=prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/USER_SERGEY/Adaptive_architecture_2_dataset84599_11746',
            experiment_name="Adaptive_architecture_2")

modelresponse = my_gan.call_sim(state={'m':numpy_single})[1]

with open('true_data.csv', 'w') as f:
        writer = csv.writer(f)
        for el in range(len(keys['bit_pos'])):
            writer.writerow([str(elem) for elem in modelresponse['res'][el, :]])

with open('data_index.csv', 'w') as f:
    writer = csv.writer(f)
    for el in keys['bit_pos']:
        writer.writerow([str(el)])

with open('data_types.csv', 'w') as f:
    writer = csv.writer(f)
    for el in keys['datatype']:
        writer.writerow([str(el)])

with open('bit_pos.csv', 'w') as f:
    writer = csv.writer(f)
    for el in keys['bit_pos']:
        writer.writerow([str(el)])

### Write correlated variance ####
rel_var = 0.01
corrRange = 10
nD=13 # only correlation along point data
sub_covD = np.zeros((len(keys['bit_pos']),nD,nD))
for k in range(len(keys['bit_pos'])):
    sub_data = np.array([modelresponse['res'][k,el] for el in range(13)]).flatten()
    for i in range(nD):
        for j in range(nD):
            sub_covD[k,i,j] = (sub_data[i]*rel_var)*(sub_data[j]*rel_var) * np.exp(-3.*(np.abs(i - j)/corrRange)**1.9)
Cd = block_diag(sub_covD[0,:,:],sub_covD[1,:,:], sub_covD[2,:,:],sub_covD[3,:,:],sub_covD[4,:,:],
                             sub_covD[5,:,:],sub_covD[6,:,:], sub_covD[7,:,:],sub_covD[8,:,:])

np.savez('cd.npz',Cd)

m_true = numpy_single.copy()
mean_f = m_true * 0.25
mean_f[20:44] = 0.
np.savez('mean_field.npz', mean_f)
