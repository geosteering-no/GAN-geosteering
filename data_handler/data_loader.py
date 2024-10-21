import time
import grdecl_raw


# data_file = r'T:\600\60010\FakeImageDataset\Distinguish\NOFAULT_MODEL_R1_remove_header.grdecl'
# data_file = r'T:\600\60010\FakeImageDataset\Distinguish\NOFAULT_MODEL_R1.grdecl'
data_file = r'C:\Users\juyo\Downloads\NOFAULT_MODEL_R1.grdecl'

t1 = time.time()
grid = grdecl_raw.read(data_file)
t2 = time.time()
print('Elapsed time: {:.3f}'.format(t2 - t1))

t = 0