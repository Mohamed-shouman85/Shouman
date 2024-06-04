import numpy as np
import pandas as pd
import keras
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Input, Dense
from numpy.polynomial.polynomial import Polynomial
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt



def Train_Dataset(data_instance, Gen_h5name):

    ## Inputs:
    # Data_instance : Class of Training and Tested Datasets.
    # Eqn_degree    : Curve Fitting Polynomial Equation Degree.
    # Gen_h5name    : Name of the created "*.h5" with the whole directory if required
    

    # Assuming the class has been populated with data already
    # Creating a DataFrame from the data stored in the class

    # Generate synthetic data
    num_samples = len(data_instance.Irradiance)
    #print(num_samples)
    ## Generate Variable to be used for polynomial fitting of V-I and V-P Curves
    Irradiance=data_instance.Irradiance # Irradiance
    Temperature=data_instance.Temperature # Temperature
    MPP = data_instance.MPP 
    

    # Prepare the inputs and outputs for the model
    X = np.column_stack([Irradiance, Temperature])
    Y = np.column_stack([MPP])

    # Define the regression model
    inputs = Input(shape=(2,))
    dense = Dense(128, activation='relu')(inputs)
    outputs = Dense(1)(dense)  # Predicting 3 coefficients for voltage-current curves
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=MeanSquaredError())


    # Split data into training and testing sets
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Train the model
    model.fit(X_train, Y_train, epochs=100, batch_size=50, verbose=1)
    
    model.evaluate(X_train, Y_train)

    print(model.metrics_names)
    print(model.evaluate(X_train, Y_train))

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])

    loss, accuracy = model.evaluate(X_train, Y_train)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    
    print('file name: ', Gen_h5name)
    #model.save(Gen_h5name)

    # Save the model in TensorFlow SavedModel format
    model.save(Gen_h5name)
    model.save(Gen_h5name.replace("pre_train_data", "pre_train_data_tf"), save_format='tf')

    print('file name should be saved')
    Y_pred= Test_target(model, X_test, Y_test, data_instance)

    print("Tested Output = ", Y_test, " , Predicted Value = ", Y_pred)
    return model
    

def Test_target(model, X_test, Y_test, data_instance):

    Irradiance=data_instance.Irradiance # Irradiance
    Temperature=data_instance.Temperature # Temperature
    MPP = data_instance.MPP
    # Predict with the model
    Y_pred = model.predict(X_test)

    print("Shape of real Tested X:" , X_test.shape)

    Xtest0 = X_test[0]
    Ytest0 = Y_test[0]
    Ypred0 = Y_pred[0]
    return Ypred0
                
def Test_req(model, X_test):
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])
    Y_pred = model.predict(X_test)
    MPP = Y_pred[0]
    print('Predicted MPP', MPP)
    return MPP

