## This code is to load MPPT Value using the Pretrained dataset from TensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


## Testing The MPPT Value Generated From Python
def test_MPPT(filepath, Irradiance, Temperature):

    model = load_model(filepath)

    Xtest= np.array([Irradiance, Temperature]).reshape((1, 2))
    Y_pred = model.predict(Xtest)
    MPP = Y_pred[0]
    MPP = MPP.tolist()
    #print('MPP Value = ', MPP[0])
    return MPP[0]

## For testing the code
#filepath= 'pre_train_MPPT_05-14-20-31.h5'
#test_MPPT(filepath, 600, 25)